# Handoff: baseline done, neural model next

Status as of 2026-11-25. The baseline side of the baseline-vs-neural comparison is finished. This file says what exists,
what must not change, and the exact steps to add the neural fine-tuned V-JEPA 2 on another cluster.
Background and design rationale: `README.md` → "Cross-model comparison".

## 1. Protocol invariants (do not change, or the paired comparison breaks)
- **Frames:** resample every video to **30 Hz** by timestamp (`extract_features.resample_index`).
- **Windows:** K=16 frames, stride 16, plus an end-aligned last window. 256 px center crop, HF V-JEPA 2 normalisation.
  The file-name tag is `30hz_k16s16` (`config.WINDOW`). Leave `EVACC_FPS` unset; `native` is the legacy protocol.
- **Features:** `emb_mean`, `emb_std`, `emb_ml` (token means of 4 upper layers: 17/19/21/23 for ViT-L).
  The neural model must be ViT-L, so it gets the same layers.
- **Readouts:** linear probes `emb_mean`, `emb_ml`, `emb_ml+emb_std`, trained by `train_probe.py` on the same train windows.
  - SSv2: `train-50000`, a fixed random 50k subset (`MAX_VIDEOS=50000`).
  - Diving48: `train`, 3 shards.
- **Folds:** `export_comparison.folds_by_id` depends only on (video_id, label), so any encoder gets the same folds.
- **Agents in the comparison:** FixedBudget, Threshold, Lookahead, FittedQ.
  - LSM (Longstaff–Schwartz) is implemented and tested but excluded. It stopped too early on both datasets:
    on SSv2 it was −0.046 to −0.063 reward vs FixedBudget at c=0.02.
  - `paired_comparison.py` drops the LSM rows still present in the old baseline exports.
- **Primary metrics:** A1 (per-window NLL/acc) and A2 (accumulation curves, mean NLL over t), plus the FixedBudget/Threshold frontier.
  FittedQ is secondary.

## 2. What exists (on the `at` volume `at-liubr-workspace`, under `~/persist/aa228` = `$EVACC_ROOT`)
| Path | Size | What |
|---|---|---|
| `comparison/baseline/{diving48,ssv2}/30hz_k16s16/<obs>/` | 76 MB | **The baseline results.** A1–A4 tables, `summary.csv`, `manifest.json`, `representation/` (B6–B8), code snapshot `.tgz` |
| `comparison/{vitl_third,vitg_third}/diving48/`, `comparison/paired/vitg_third_vs_vitl_third/` | 6 MB | ViT-g positive control (probes on 1/3 of Diving48 train) |
| `features/features_vitl16_diving48_{test,train}_30hz_k16s16_*.h5` | ~2 GB | baseline features, Diving48 (task ckpt `vjepa2-vitl-fpc32-256-diving48`, encoder = `vjepa2-vitl-fpc64-256`) |
| `features/features_vitl16_ssv2_{validation,train-50000}_30hz_k16s16_*.h5` | ~7 GB | baseline features, SSv2 (task ckpt `vjepa2-vitl-fpc16-256-ssv2`, same encoder) |
| `features/probe-linear*_30hz_k16s16*` | ~0.5 GB | baseline probe weights (`.pt`) and eval logits (`.h5`) |
| `features/features_vitg16_256_*` | ~1 GB | ViT-g features (Diving48 test + train shard 0) |
| `features/*_fs1_*`, `*_fs2_*` | — | legacy native-fps protocol; not needed |
| `data/diving48` (9.6 GB), `data/ssv2` (19 GB) | 28 GB | raw videos and annotations |

The obs names in the exports are:
- `linear`: the `emb_mean` probe;
- `linear-emb_ml`;
- `linear-emb_ml+emb_std`;
- `meta`: Meta's probe, baseline only and never paired.

### Baseline headline numbers (30 Hz; per-window accuracy / final accumulated accuracy)
| Readout | Diving48 test | SSv2 validation |
|---|---|---|
| emb_mean | 0.167 / 0.290 | (see `summary.csv`) |
| emb_ml | 0.222 / 0.380 | 0.243 / 0.305 |
| emb_ml+emb_std | 0.221 / 0.399 | (see `summary.csv`) |
| Meta's probe | 0.309 / 0.611 | 0.245 / 0.305 |

Other baseline results:
- Belief exponent γ ≈ 0.2–0.5.
- B6 R²: Diving48 0.53–0.57, SSv2 0.76–0.78.
- B7 curvature: 99–113°.
- B8 participation ratio: Diving48 13–17, SSv2 56–99.
- Positive control, ViT-g − ViT-L (third-data probes):
  - per-window accuracy +0.013 / +0.018;
  - mean NLL −0.13 / −0.033;
  - all significant, so the protocol can detect a better encoder.

## 3. Steps for the neural model
1. **Code:** `git clone` the repo and check out branch `comparison-30hz` (until it is merged).
   - Then set up the environment: `environment.yml`, or `sky/common_setup.sh` for a venv.
   - Set `EVACC_ROOT` to big scratch storage.
2. **Data and baseline** on the new cluster (see §4).
3. **Loader** (not written yet): add a branch in `extract_features.py`, around `model_cls.from_pretrained(config.MODEL_ID…)` at line ~216.
   - It must return a model whose `encode(model, clips, layers)` yields per-layer tokens like the HF `VJEPA2Model`.
     Simplest: load the neural checkpoint's encoder state dict into `VJEPA2Model.from_pretrained("facebook/vjepa2-vitl-fpc64-256")`, after mapping key names.
   - Things to find out from the neural repo:
     - checkpoint format and key names;
     - whether the readout head (predicted firing rates) is included;
     - its preprocessing. Must be 256 px RGB with the same normalisation; if it differs, that is a confound to record.
   - Give it no task head. `meta` is then absent, which is correct.
   - Equivalence test: on a few clips, our forward and the neural repo's forward must give the same tokens.
4. **Weights check:** `python check_encoder_weights.py facebook/vjepa2-vitl-fpc64-256 <neural .safetensors>`.
   - Expect "DIFFERENT" with a per-layer diff (fine-tuning changed the weights).
   - A key mismatch means the key mapping is wrong.
   - Also confirm it started from that checkpoint.
5. **Extract**, same protocol, `EVACC_MODEL_TAG=vitl16neural_<dataset>` (unique per dataset), four runs:
   - Diving48 `test`;
   - Diving48 `train`, 3 shards;
   - SSv2 `validation`, 4 shards;
   - SSv2 `train`, 4 shards, with `MAX_VIDEOS=50000`.
   - Check that window counts equal the baseline's: Diving48 test 17,020; SSv2 validation 197,820.
6. **Probes and export** per dataset, matching the `PROBE_KEYS` and `LABEL` of the baseline:
   ```bash
   export EVACC_DATASET=diving48 EVACC_MODEL_TAG=vitl16neural_diving48
   for k in emb_mean emb_ml emb_ml+emb_std; do python train_probe.py --key $k; done
   python export_comparison.py --label neural --obs linear linear-emb_ml linear-emb_ml+emb_std
   python representation_analysis.py --labels neural
   ```
   On `at` this is `sky/analysis.yaml` with these envs:
   - `--env DATASET=diving48 --env MODEL_TAG=vitl16neural_diving48 --env META=0`
   - `--env PROBE_KEYS="emb_mean emb_ml emb_ml+emb_std" --env LABEL=neural --env STEPS="probe export represent"`
7. **Paired comparison and CKA:**
   ```bash
   python paired_comparison.py baseline neural          # -> comparison/paired/neural_vs_baseline/<dataset>/30hz_k16s16/
   python representation_analysis.py --tags vitl16_diving48 vitl16neural_diving48 --labels baseline neural   # B9 CKA, paired B8
   ```
   - SSv2 baseline tag: `vitl16_ssv2`.
   - Sanity check: `paired_comparison.py baseline baseline` must give 0 everywhere.
8. **Neural-rate observation model** (not implemented):
   - Store the predicted firing rates per window (16 bins × neurons) as an extra h5 key in step 5.
   - Train a linear probe on them (`train_probe.py --key <that key>`).
   - Use it as one more obs for the neural model only. It has no baseline counterpart, so report it unpaired.
9. **Interpretation:** compare the size of any neural − baseline effect with the ViT-g − ViT-L control (§2).
   - There is no same-recipe fine-tuning control, so a difference means "fine-tuned on neural data from those stimuli".

## 4. Moving to the other cluster
- **Code: git.** The work is on branch `comparison-30hz`. `initial-pipeline` only accepts pull requests, and the repo moved to `enigma-brain/vjepa-evidence-accumulation`.
- **Results and features: copy only what the comparison needs.** About 2.5 GB without the train features, or about 10 GB with them:
  - `comparison/` (83 MB). This is all `paired_comparison.py` needs.
  - The baseline eval features, needed for B9 CKA and paired B8:
    - `features/features_vitl16_diving48_test_30hz_k16s16_shard0of1.h5`;
    - `features/features_vitl16_ssv2_validation_30hz_k16s16_shard*of4.h5`.
  - Optional:
    - the baseline train features, only to retrain baseline probes;
    - `features/probe-linear*_30hz_k16s16*`, to rerun the baseline export there.
- **Raw videos:** re-download on the new cluster with the prepare scripts (`sky/prepare_{ssv2,diving48}.yaml`; the URLs are inside) instead of copying 28 GB.
  The neural extraction needs the raw videos for both train and eval splits.
- **How to copy:**
  - Another SkyPilot cluster (morpho/DO): a small `at` job that tars the files above and uploads them to the lab's Ceph S3, then pull from the other side.
  - Outside SkyPilot: an `at` dev box with the volume mounted, then `rsync`/`scp` from it.
  - Nothing at `at` is deleted, so the originals stay as a backup.
- **Paths:** everything resolves under `$EVACC_ROOT`, so put the copied `comparison/` and `features/` under the new `$EVACC_ROOT` with the same structure.
