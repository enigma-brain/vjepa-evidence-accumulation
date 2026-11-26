# Notes for Claude

- Read `HANDOFF.md` first. It holds the current status, the protocol invariants, and the step list for adding the neural model.
- The cross-model comparison only holds if every encoder goes through the same protocol:
  - 30 Hz;
  - 16-frame stride-16 windows (`30hz_k16s16`);
  - the same probes and the same folds.
  Do not change `config.py` windowing, `extract_features.py` preprocessing, or `export_comparison.folds_by_id` without saying so explicitly.
- LSM is excluded from the comparison on purpose (see `HANDOFF.md`).
- Tests: `pytest -q test_pomdp.py test_windows.py`.
- Do not commit to `initial-pipeline` directly; it requires pull requests.
