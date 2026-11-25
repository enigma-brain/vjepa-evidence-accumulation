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
| `pomdp.py` | belief update (tempered naive Bayes), rollout loop, agents (FixedBudget, Threshold, Lookahead, FittedQ, LSM = Longstaff-Schwartz least-squares Monte Carlo) |
| `run_pomdp.py` | fits agents on half of SSv2-val and evaluates them on the other half; writes the table + figures |
| `train_e2e.py` | end-to-end: linear probe + α + a learned stopping rule, trained together on the exact expected reward (no RL sampling needed) |
| `compare_models.py` | paired comparison of observation models on the same videos: calibrated confidence per window, time to reach confidence θ, commitment time, stopping time; bootstrap CIs |
| `probe_analysis.py` | per-window accuracy by position in the video, calibration (ECE) of windows and beliefs, Diving48 per-attribute accuracy |
| `belief_models.py` | cross-fitted comparison of belief models: one α vs a per-window temperature and/or a step-dependent weight α·(t+1)^(γ−1) |
| `extract_pixels.py` | negative control: the same windows as raw 8×8 RGB pixels (CPU only), same layout as a features file |
| `export_comparison.py` | one encoder's side of a cross-model comparison: per-window / per-video tables for A1–A4 under `comparison/<label>/` |
| `representation_analysis.py` | label-free backbone analyses B6–B8 (temporal predictability, trajectory curvature, dimensionality); with two encoders also B9 (CKA) |
| `paired_comparison.py` | joins two exports on video_id: paired differences with bootstrap CIs, accumulation curves |
| `check_encoder_weights.py` | are two checkpoints' encoders identical (e.g. task checkpoint vs pretrained, or where a fine-tune started)? |
| `test_pomdp.py`, `test_windows.py` | tests on a synthetic POMDP whose exact posterior is known; 30 Hz resampling and window layout |

**Adding an agent:** write a class with `act(belief, t, T) -> bool` (True = stop). Optionally give it `fit(Bs, logq, y, cost)`. Then add it to the `agents` dict in `run_pomdp.py`.

**Another encoder:** any HF V-JEPA 2 checkpoint works. Set `EVACC_MODEL_ID` / `EVACC_MODEL_TAG` (or `MODEL_ID` / `MODEL_TAG` in `config.py`). Crop/resize and embedding width follow the checkpoint config. Checkpoints without an SSv2 head (e.g. `facebook/vjepa2-vitg-fpc64-384`) only get `emb_mean`, which is all our linear probe needs (`--obs meta` needs the head). A non-HF encoder (e.g. the neural fine-tuned one) still needs a loading branch in `extract_features.py`.

## Setup (any cluster)
```bash
export EVACC_ROOT=/path/to/big/scratch   # default: /scratch/users/liubr/aa228
micromamba create -f environment.yml && micromamba activate evacc
```

## SkyPilot (lab `at` cluster)
GPU steps run as managed jobs; everything durable lands in `~/persist/aa228` on the `at-liubr-workspace` volume.
SSv2 is kept there as its archive parts, and each job unpacks it onto its own local disk.
```bash
sky jobs launch -d -n liubr-ssv2-download sky/prepare_ssv2.yaml    # once
sky jobs launch -d -n liubr-vitl-val sky/extract.yaml              # see the header of sky/extract.yaml for more
sky jobs launch -d -n liubr-vitg-val sky/extract.yaml --env MODEL_ID=facebook/vjepa2-vitg-fpc64-384-ssv2 --env MODEL_TAG=vitg16_384_ssv2 --env BATCH_SIZE=16
# Diving48
sky jobs launch -d -n liubr-diving48-download sky/prepare_diving48.yaml   # once
# (bash; in zsh write ${=D48} so the variable splits into words)
D48="--env DATASET=diving48 --env MODEL_ID=facebook/vjepa2-vitl-fpc32-256-diving48 --env MODEL_TAG=vitl16_diving48 --env BATCH_SIZE=32"
sky jobs launch -d -n liubr-d48-test  sky/extract.yaml $D48 --env SPLIT=test
sky jobs launch -d -n liubr-d48-train sky/extract.yaml $D48 --env SPLIT=train
sky jobs launch -d -n liubr-d48-analysis sky/analysis.yaml --env DATASET=diving48 --env MODEL_TAG=vitl16_diving48 --env STEPS="test probe pomdp"
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
python train_e2e.py                        # 6. end-to-end variants (CPU, ~1h) -> e2e_summary.csv, fig4_e2e.png
python compare_models.py vitl16_ssv2:linear vitl16_ssv2:meta pixels8:linear   # 7. which model is confident earlier?
python extract_pixels.py --split validation && python extract_pixels.py --split train --max-videos 50000 \
  && python train_probe.py --tag pixels8   #    pixel negative control (CPU), then include pixels8:linear above
pytest test_pomdp.py
```

## Cross-model comparison (baseline vs neural fine-tuned V-JEPA 2)
Every encoder goes through one fixed protocol, so only the encoder weights differ:
- **Windows:** each video is resampled to **30 Hz** by timestamp, then cut into **16-frame** windows (533 ms; the neural model's input: 16 frames in, 16 firing-rate bins out), stride 16 (no overlap), plus one end-aligned last window. 256 px center crop, same normalisation. File names carry `30hz_k16s16`.
- **Baseline:** frozen pretrained ViT-L. `check_encoder_weights.py` confirmed that `vjepa2-vitl-fpc16-256-ssv2` and `vjepa2-vitl-fpc32-256-diving48` carry the encoder of `vjepa2-vitl-fpc64-256` bit for bit (388/388 tensors), so the task checkpoints give the pretrained encoder's features plus Meta's probe logits (a baseline-only reference).
- **Readouts:** our linear probes on `emb_mean`, `emb_ml`, `emb_ml+emb_std`, trained the same way on the same train windows. Predicted firing rates of the neural model become one extra observation model for it.
- **Folds:** `export_comparison.folds_by_id` depends only on (video_id, label), so every encoder gets the same 2-fold cross-fit.
- **Analyses:** A1 per-window informativeness (raw and temperature-calibrated), A2 accumulation curves and per-window information gain, A3 fitted α/γ, A4 agents' reward / windows per cost and time to calibrated confidence, A5 position / attributes / direction pairs (from the A1 table), B6 temporal predictability, B7 trajectory curvature, B8 effective dimensionality, B9 CKA between encoders.
- **Positive control:** ViT-g (`vjepa2-vitg-fpc64-256`, same 256 px input and token grid). It counts as a positive control only if it beats ViT-L significantly on A1/A2 under this protocol; checked first with probes trained on one third of Diving48 train (`--train-shards shard0of3 --suffix=-third`).
- **Not controlled:** there is no same-recipe fine-tune without neural targets, so a difference means "fine-tuned on neural data from those stimuli", not "neural data" per se.
```bash
# per encoder (after extraction + train_probe.py): tables under $EVACC_ROOT/comparison/<label>/<dataset>/30hz_k16s16/
sky jobs launch -d -n liubr-d48-export sky/analysis.yaml --env DATASET=diving48 --env MODEL_TAG=vitl16_diving48 \
    --env PROBE_KEYS="emb_mean emb_ml emb_ml+emb_std" --env LABEL=baseline --env STEPS="probe export represent"
# two encoders: paired differences (+ CKA)
sky jobs launch -d -n liubr-d48-paired sky/analysis.yaml --env DATASET=diving48 --env PAIR="baseline neural" \
    --env PAIR_TAGS="vitl16_diving48 <neural tag>" --env STEPS="paired cka"
```

## Design notes
- **Windows (legacy native-fps protocol, `EVACC_FPS=native`, files `*_fs<step>_*`; the current protocol is above).** Window t is K=16 frames taken every `FRAME_STEP`=2 raw frames, starting at raw frame 4t. A window runs about 2.7 s, and windows start every 0.33 s (SSv2 is 12 fps). The probe was trained on 16 frames at step 4 that span the whole clip, so single windows are deliberately partial evidence. Each video also gets one "full clip" forward pass, which serves as a sanity check against published probe accuracy.
- **Splits.** Probes are trained on the dataset's *train* split (SSv2: ours on a fixed random 50k-video subset). All agent fitting and testing uses the eval split (SSv2 *validation*, Diving48 *test*), cross-fitted in `run_pomdp.py` and `direction_check.py`: split 50/50 by class into halves A and B, fit on A and score B, then the reverse, so every video is scored once by parameters fitted on the other half. `summary.csv` gives 95% bootstrap CIs over videos for each agent's reward, and paired CIs for its reward minus FixedBudget's on the same videos. (`compare_models.py` and `train_e2e.py` still use the single agent-fit / agent-test split.)
- **Belief.** log b_t = log p(y) + α Σ_τ [log q_τ(y) − log p(y)]. α < 1 discounts correlated windows (it is the naive-Bayes independence correction). α and p(y) are fit on the fit half. The subtracted p(y) must be the class distribution the probe was trained on, so a probe trained on a rebalanced set needs a uniform p(y) there.
- **Datasets.** `EVACC_DATASET=ssv2` (default) or `diving48` (V2 cleaned annotations; 48 classes; Meta's probe `facebook/vjepa2-vitl-fpc32-256-diving48`). Windows and the full-clip check follow `config.DATASETS`: Diving48 uses non-overlapping windows of 16 consecutive frames (stride 16, plus one end-aligned last window so the entry is always seen; ~7 windows per test clip). Meta's probe was trained on 32 frames at step 2, so its per-window accuracy is below its training setting; our linear probe is trained on these same windows. (Earlier `*_fs2_*` runs used 32 frames at step 2, stride 8: 7/8 overlap, and the fitted step-dependent weight was close to averaging.) The full-clip check is unchanged and the full-clip check averages 4 evenly spaced clips like Meta's eval (one spatial view, so expect slightly below the reported accuracy).
- **End-to-end.** Stopping never changes what the agent sees, so P(stop at t) = h_t ∏_{s<t}(1 − h_s) makes the expected reward an exact, differentiable sum. The reward uses a sharpened soft-argmax of b_t (k=10), because b_t(y) itself keeps rising with more windows and pays the agent to over-watch. Three variants: `frozen` (train-split probe fixed), `ce` (probe learns only by cross-entropy), `full` (probe also learns from the reward). Comparing `full` with `ce` isolates the effect of end-to-end training on the probe.
- **Comparing confidence across models.** An overconfident model looks confident earlier for free, so `compare_models.py` recalibrates each model's belief per window (temperature τ_t fit on agent-fit) before measuring confidence. ECE before/after is in `per_window.csv`. Stopping time under the cost-optimal Threshold agent is reported too, but it is confounded: a more informative model has more to gain from watching, so it can rationally watch *longer*.
- **Lookahead.** This is the one-step tree in Fig. 1. The expectation over o_{t+1} samples y ~ b_t, then the window t+1 of a random agent-fit video of class y.
