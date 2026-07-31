from src.modules.ai_system.ai_system import AISystem


def stnd_example():
    """Instantiate the system in baseline mode (uses default models/thresholds)."""
    system = AISystem(baseline=True)
    return system


def custom_example():
    """Instantiate the system with custom models and thresholds (baseline=False)."""
    system = AISystem(
        baseline=False,
        static_model_path="data/models/xgb_with_filters.json",
        dynamic_model_path="data/models/nebula_ablation_data/delta_11/dynamic_model.pt",
        bpe_model_path="data/models/nebula_ablation_data/delta_11/bpe.model",
        vocab_path="data/models/nebula_ablation_data/delta_11/bpe_vocab.json",
        static_malware_threshold=0.9,  # optional, overrides the default
        static_goodware_threshold=0.05,  # optional, overrides the default
        dynamic_threshold=0.99,  # optional
    )
    return system


def main():
    # Choose which system to instantiate
    stnd = stnd_example()
    baseline = custom_example()

    # The sample format depends on what YaraMatcher / StaticModule / DynamicModule
    # expect internally (e.g. path to a PE file, raw bytes, or a feature vector)
    sample_path = "path/to/sample.exe"

    module_used, verdict = stnd.predict(sample_path)
    print(f"[predict] deciding module: {module_used} -> verdict: {verdict}")
    # verdict: 0 = goodware, 1 = malware, -1 = dynamic unavailable/error

    # --- Inference with separate scores for each stage (useful for analysis/debugging) ---
    scores = stnd.predict_separate(sample_path)
    print("[predict_separate] per-stage scores:")
    for i in range(0, len(scores), 2):
        stage, value = scores[i], scores[i + 1]
        print(f"  {stage}: {value}")

    # --- Inference on multiple samples ---
    samples = ["sample1.exe", "sample2.exe", "sample3.exe"]
    results = [stnd.predict(s) for s in samples]
    for s, (module, v) in zip(samples, results):
        print(f"{s}: {module} -> {v}")


if __name__ == "__main__":
    main()


