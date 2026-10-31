"""CI quality gate: train models and enforce a minimum test R2 score."""
from train_model import train_and_evaluate

MIN_R2 = 0.90


def main():
    report = train_and_evaluate()
    score = report["best_r2"]
    print(f"\nQuality gate threshold: R2 >= {MIN_R2:.3f}")
    print(f"Best model test R2: {score:.4f}")
    if score < MIN_R2:
        raise SystemExit(
            f"FAILED: model R2 {score:.4f} is below required {MIN_R2:.3f}"
        )
    print("PASSED: Check ML quality gate")


if __name__ == "__main__":
    main()
