# OBELISK
 
Code and released assets for the paper *"Windows Malware Detector as a Compound AI System: Trade-Offs in Accuracy, Efficiency, and Adversarial Robustness"* (A. Ponte, L. Demetrio, L. Oneto, B. Biggio, F. Roli).
 
OBELISK is a Compound AI System for Windows malware detection: a three-level pipeline combining YARA-based signature matching, static analysis with an XGBoost model trained on EMBER features, and dynamic analysis with [Nebula](https://github.com/dtrizna/nebula). Filtering thresholds between levels let you trade detection performance for training and inference cost. The repository also includes the black-box evasion attacks (GAMMA section injection and padding) used in the paper to evaluate the system under four threat models of increasing attacker knowledge (TM1–TM4).
 
## Requirements
 
- Python 3.9+
- [`secml`](https://github.com/pralab/secml) and [`secml-malware`](https://github.com/pralab/secml_malware)
- [`ember`](https://github.com/elastic/ember) (EMBER static feature extraction)
- `xgboost`, `lightgbm`
- `yara-python`
- [Nebula](https://github.com/dtrizna/nebula) (only needed for the dynamic-analysis level — see below)
## Modules
 
- ### YARA Signatures
    The pool of YARA signatures used for the blocklist and the allowlist rule. We also provide the subset of blocklist rules that produce no false positives on our training set — the rules actually deployed in OBELISK.
- ### Static Module with XGBoost
    The static models deployed inside OBELISK: `xgb_no_filters` (used in STND and OBV1, trained on the full dataset) and `xgb_with_filters` (used in OBV2 and OBV3, trained only on samples not resolved by the signature level).
- ### Dynamic Module with Nebula
    The Nebula models trained for each value of δ used in the ablation study, plus the model trained on the full dataset (NEBULA-ALL). To use them, install the original Nebula repository: **https://github.com/dtrizna/nebula**.
## Attack Interfaces
 
Wrappers implementing the black-box evasion attacks (GAMMA section injection and padding) used to build the A1–A4 attack sets described in the paper:
 
- **`gbdt_transfer.py`** (`OpenGbdt`) — attacks the open-source EMBER-based surrogate model of [Anderson et al.](https://arxiv.org/abs/1804.04637); used for A1 and A2. Model weights: **https://github.com/endgameinc/malware_evasion_competition/tree/master/models/ember**.
- **`xgb_wrappers.py`** (`CClassifierXGBoost` / `CXGBWrapperPhi`) — attacks the static XGBoost model deployed inside OBELISK; used for A3 and A4.
- **`ai_system_wrapper.py`** (`AISystemWrapper`) — attacks a surrogate Compound AI System; used for A2 and A4. It can also be used for A1 and A3 by initializing it with `filter=False`, which disables the signature level.

## Running OBELISK
 
`ai_sys_inference.py` instantiates the five representative Compound AI System configurations analyzed in the paper (Sect. V-D) — **STND**, **OBV1@δ6**, **OBV2@δ6**, **OBV3@δ6**, **OBV3@δ11** — and runs inference with each of them on a sample.
 
`AISystem` has no built-in threshold defaults: `static_malware_threshold` and `dynamic_threshold` must always be supplied (a missing one raises `ValueError`), and in non-baseline mode `static_goodware_threshold` is required as well. Before running the script, edit `ai_sys_inference.py` to:
 
1. set `sample_path` to the PE file you want to analyze;
2. fill in the tuned ≈1% FPR operating point for each system's `static_malware_threshold`/`dynamic_threshold` (currently left as `None` placeholders);
3. point `XGB_NO_FILTERS`, `XGB_WITH_FILTERS`, `NEBULA_ALL_DIR`, and `NEBULA_DELTA_DIR` at your actual `data/models/...` layout.
```bash
python ai_sys_inference.py
```
 
The script prints, for each of the five systems, which level took the final decision and the resulting verdict, plus the per-stage scores for one of them.

## Running the Attacks
 
`attack_on_tm.py` initializes each of the four threat models (TM1–TM4) with `AISystemWrapper` and runs the GAMMA and padding attacks against it.

## Citation
 
The paper is currently under review. We published the [preprint](https://arxiv.org/pdf/2609.08394) on arXiv.
