from src.modules.ai_system.ai_system import AISystem
import sys


if __name__ == "__main__":

    # Initialize the AI System with default settings, hence with the best delta from the ablation study
    obelisk = AISystem(default=True)
    baseline = AISystem(baseline=True)

    x_path = sys.argv[1]

    y_obelisk = obelisk.predict(x_path)
    y_baseline = baseline.predict(x_path)

    print(f"OBELISK prediction: {y_obelisk}")
    print(f"Baseline prediction: {y_baseline}")



