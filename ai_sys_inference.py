"""
Example script: instantiate the 5 representative Compound AI System
configurations analyzed in Sect. V-D of the paper (STND, OBV1@d6, OBV2@d6,
OBV3@d6, OBV3@d11) and run inference on one or more samples.

System definitions (paper, Sect. V-A "Considered Compound AI Systems"):
  - STND    : YARA + GBDT-ALL    (single threshold)  + NEBULA-ALL
  - OBV1    : YARA + GBDT-ALL    (two thresholds, d)  + NEBULA-ALL
  - OBV2    : YARA + GBDT-YARA   (two thresholds, d)  + NEBULA-ALL
  - OBV3    : YARA + GBDT-YARA   (two thresholds, d)  + NEBULA-GBDT-YARA (retrained per delta)

NOTE: adjust the import path to match where the AISystem class actually
      lives in the repository.
"""

from src.modules.ai_system.ai_system import AISystem

# The 12 delta values from the ablation study (Sect. V-B), d0 (lowest) to
# d11 (highest), each already x10^-3. For OBV1/OBV2/OBV3, a sample with
# static score p <= delta is labeled goodware, p >= 1 - delta is labeled
# malware, and anything in between is forwarded to the dynamic level.
DELTA_VALUES = [
    0.1e-3, 0.27e-3, 0.44e-3, 0.72e-3, 1.18e-3, 1.93e-3,
    3.16e-3, 5.18e-3, 8.48e-3, 13.89e-3, 22.76e-3, 37.28e-3,
]

# Model paths — adjust to your actual data/models layout.
XGB_NO_FILTERS = "data/models/xgb_no_filters.json"      # GBDT-ALL   (STND, OBV1)
XGB_WITH_FILTERS = "data/models/xgb_with_filters.json"  # GBDT-YARA  (OBV2, OBV3)

# NEBULA-ALL is the same trained model shared by STND/OBV1/OBV2 (it does not
# depend on delta). NEBULA-GBDT-YARA is retrained per delta and is only used
# by OBV3 — this assumes one model folder per delta index, matching the
# "delta_11" layout already used by AISystem's own defaults; adjust if your
# directory layout differs.
NEBULA_ALL_DIR = "data/models/nebula_ablation_data/baseline"
NEBULA_DELTA_DIR = "data/models/nebula_ablation_data/delta_{}"


def _nebula_paths(directory: str) -> dict:
    return dict(
        dynamic_model_path=f"{directory}/dynamic_model.pt",
        bpe_model_path=f"{directory}/bpe.model",
        vocab_path=f"{directory}/bpe_vocab.json",
    )


def stnd_example(static_malware_threshold: float, dynamic_threshold: float) -> AISystem:
    """STND: YARA + GBDT-ALL (single threshold) + NEBULA-ALL.
    """
    return AISystem(
        baseline=True,
        static_malware_threshold=static_malware_threshold,
        dynamic_threshold=dynamic_threshold,
    )


def obv1_delta6_example(dynamic_threshold: float = None) -> AISystem:
    """OBV1 @ delta_6: YARA + GBDT-ALL (two thresholds) + NEBULA-ALL."""
    delta = DELTA_VALUES[6]
    return AISystem(
        baseline=False,
        static_model_path=XGB_NO_FILTERS,
        static_goodware_threshold=delta,
        static_malware_threshold=1 - delta,
        dynamic_threshold=dynamic_threshold,  #set the ~1% FPR operating point tuned for OBV1@d6
        **_nebula_paths(NEBULA_ALL_DIR),
    )


def obv2_delta6_example(dynamic_threshold: float = None) -> AISystem:
    """OBV2 @ delta_6: YARA + GBDT-YARA (two thresholds) + NEBULA-ALL."""
    delta = DELTA_VALUES[6]
    return AISystem(
        baseline=False,
        static_model_path=XGB_WITH_FILTERS,
        static_goodware_threshold=delta,
        static_malware_threshold=1 - delta,
        dynamic_threshold=dynamic_threshold,  #set the ~1% FPR operating point tuned for OBV2@d6
        **_nebula_paths(NEBULA_ALL_DIR),
    )


def obv3_delta6_example(dynamic_threshold: float = None) -> AISystem:
    """OBV3 @ delta_6: YARA + GBDT-YARA (two thresholds) + NEBULA-GBDT-YARA (trained for delta_6)."""
    delta = DELTA_VALUES[6]
    return AISystem(
        baseline=False,
        static_model_path=XGB_WITH_FILTERS,
        static_goodware_threshold=delta,
        static_malware_threshold=1 - delta,
        dynamic_threshold=dynamic_threshold,  #set the ~1% FPR operating point tuned for OBV3@d6
        **_nebula_paths(NEBULA_DELTA_DIR.format(6)),
    )


def obv3_delta11_example(dynamic_threshold: float = None) -> AISystem:
    """OBV3 @ delta_11: YARA + GBDT-YARA (two thresholds) + NEBULA-GBDT-YARA (trained for delta_11).
    """
    delta = DELTA_VALUES[11]
    return AISystem(
        baseline=False,
        static_model_path=XGB_WITH_FILTERS,
        static_goodware_threshold=delta,      # optional: matches AISystem's own default
        static_malware_threshold=1 - delta,   # optional: matches AISystem's own default
        dynamic_threshold=dynamic_threshold,  #set the ~1% FPR operating point tuned for OBV3@d11
        **_nebula_paths(NEBULA_DELTA_DIR.format(11)),
    )


def main():
    # Fill in the tuned thresholds for each system before running this
    # (Sect. V-B: each system's threshold is tuned at ~1% FPR).
    # STND requires both thresholds explicitly (AISystem has no built-in
    # default); OBV1/OBV2/OBV3 can leave dynamic_threshold=None,
    # which makes predict() return the raw Nebula score for that branch.
    systems = {
        "STND": stnd_example(static_malware_threshold=0.964414, dynamic_threshold=0.999991),
        "OBV1-d6": obv1_delta6_example(dynamic_threshold=0.992648),
        "OBV2-d6": obv2_delta6_example(dynamic_threshold=0.993269),
        "OBV3-d6": obv3_delta6_example(dynamic_threshold=0.999967),
        "OBV3-d11": obv3_delta11_example(dynamic_threshold=0.999963),
    }

    sample_path = "path/to/sample.exe"

    print(f"Running inference on {sample_path} with all 5 configurations:\n")
    for name, system in systems.items():
        module_used, verdict = system.predict(sample_path)
        print(f"[{name}] deciding module: {module_used} -> verdict: {verdict}")

    print("\n[OBV3-d11] per-stage scores:")
    scores = systems["OBV3-d11"].predict_separate(sample_path)
    for i in range(0, len(scores), 2):
        stage, value = scores[i], scores[i + 1]
        print(f"  {stage}: {value}")


if __name__ == "__main__":
    main()