# V-JEPA 2 evidence accumulation (CS 238 / AA 228 project)

How efficiently does a video encoder let an agent accumulate evidence? We cast it as a POMDP
(see `CS_238_Final_Proj_Proposal.pdf`, Fig. 1):

- **hidden state:** the SSv2 action class y
- **observation o_t:** a short window of the video
- **belief:** b_t = p(y | o_0..o_t)
- **actions:** STOP (reward 1 if argmax b_t = y) or CONTINUE (cost c, see the next window)

The observation model is a frozen V-JEPA 2 encoder plus **our own linear probe**, which gives p(y | o_t) per window. The probe reads the mean of the encoder's output tokens and is trained on SSv2 train windows. It is deliberately weak and identical for every encoder, so differences between encoders aren't absorbed by a powerful readout. Meta's pretrained SSv2 attentive probe (`--obs meta`) is kept only as a sanity check and a ceiling for the baseline encoder.

## Files (read in this order)
| file | what it does |
|---|---|
| `config.py` | every path and constant: data root, model id, window size K, frame step, stride |
| `prepare_ssv2.sh` | downloads SSv2 (public HF mirror of the official archives, 19.4 GB) and unpacks it |
| `extract_features.py` | video → sliding windows → V-JEPA 2 + probe → `features_*.h5` (logits + embeddings per window) |
| `extract.sbatch` | runs the extraction as a SLURM array of 1-GPU jobs |
| `train_probe.py` | trains the linear probe on SSv2-train window embeddings, then writes its logits for every validation window |
| `pomdp.py` | belief update (tempered naive Bayes), rollout loop, agents (FixedBudget, Threshold, Lookahead, FittedQ) |
| `run_pomdp.py` | fits agents on half of SSv2-val and evaluates them on the other half; writes the table + figures |
| `test_pomdp.py` | tests on a synthetic POMDP whose exact posterior is known |

**Adding an agent:** write a class with `act(belief, t, T) -> bool` (True = stop). Optionally give it `fit(Bs, logq, y, cost)`. Then add it to the `agents` dict in `run_pomdp.py`.

**Another encoder** (larger V-JEPA 2, or the neural fine-tuned one): change `MODEL_ID` / `MODEL_TAG` in `config.py`, or add a loading branch in `extract_features.py`. Nothing downstream changes.

## Setup (any cluster)
```bash
export EVACC_ROOT=/path/to/big/scratch   # default: /scratch/users/liubr/aa228
micromamba create -f environment.yml && micromamba activate evacc
```

## Run
```bash
sbatch prepare_ssv2.sh                     # 1. data (or: bash prepare_ssv2.sh)
python extract_features.py --limit 8 --device cpu --num-workers 2   # 2. smoke test (slow on CPU)
sbatch extract.sbatch                      # 3a. features for SSv2 validation (~127k windows, 4x 1-GPU jobs, ~16 min)
SPLIT=train MAX_VIDEOS=50000 sbatch extract.sbatch   # 3b. features for a 50k-video train subset (probe training)
python train_probe.py                      # 4. our linear probe (train-50000 -> validation logits)
python run_pomdp.py                        # 5. agents -> $EVACC_ROOT/results/<model>_<split>_fs<step>_linear/
python run_pomdp.py --obs meta             #    same, using Meta's pretrained probe (reference)
pytest test_pomdp.py
```

## Design notes
- **Windows.** Window t is K=16 frames taken every `FRAME_STEP`=2 raw frames, starting at raw frame 4t. A window runs about 2.7 s, and windows start every 0.33 s (SSv2 is 12 fps). The probe was trained on 16 frames at step 4 that span the whole clip, so single windows are deliberately partial evidence. Each video also gets one "full clip" forward pass, which serves as a sanity check against published probe accuracy.
- **Splits.** Probes are trained on SSv2 *train* (ours on a fixed random 50k-video subset). All agent fitting and testing uses SSv2 *validation*, split 50/50 by class (agent-fit / agent-test).
- **Belief.** log b_t = log p(y) + α Σ_τ [log q_τ(y) − log p(y)]. α < 1 discounts correlated windows (it is the naive-Bayes independence correction). α is fit on agent-fit.
- **Lookahead.** This is the one-step tree in Fig. 1. The expectation over o_{t+1} samples y ~ b_t, then the window t+1 of a random agent-fit video of class y.
