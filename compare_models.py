import argparse
from utils import load_artifact

def main(baseline_model_path: str, tuned_model_path: str):
    """Compare baseline vs tuned models."""
    print("=" * 70)
    print("BASELINE vs TUNED MODEL COMPARISON")
    print("=" * 70)
    
    # Load baseline model
    print("\n[BASELINE MODEL]")
    baseline_artifact = load_artifact(baseline_model_path)
    baseline_model = baseline_artifact['model']
    baseline_name = baseline_artifact['model_name']
    
    baseline_score = baseline_artifact.get('score', 'N/A')
    
    print(f"Model: {baseline_name}")
    print(f"ROC AUC: {baseline_score:.4f}" if isinstance(baseline_score, float) else f"ROC AUC: {baseline_score}")
    print(f"Hyperparameters: {baseline_artifact.get('hyperparameters', 'N/A')}")
    
    # Load tuned model
    print("\n[TUNED MODEL]")
    tuned_artifact = load_artifact(tuned_model_path)
    tuned_model = tuned_artifact['model']
    tuned_name = tuned_artifact['model_name']
    
    tuned_score = tuned_artifact.get('score', 'N/A')
    
    print(f"Model: {tuned_name}")
    print(f"ROC AUC: {tuned_score:.4f}" if isinstance(tuned_score, float) else f"ROC AUC: {tuned_score}")
    print(f"Hyperparameters: {tuned_artifact.get('hyperparameters', 'N/A')}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("IMPROVEMENT")
    print("=" * 70)
    
    if isinstance(baseline_score, float) and isinstance(tuned_score, float):
        auc_diff = tuned_score - baseline_score
        auc_pct = (auc_diff / baseline_score) * 100 if baseline_score > 0 else 0
        
        print(f"ROC AUC Change: {baseline_score:.4f} → {tuned_score:.4f} ({auc_diff:+.4f}, {auc_pct:+.1f}%)")
        
        if tuned_score > baseline_score:
            print(f"\n✓ Tuned model IMPROVED ROC AUC by {auc_pct:.1f}%")
        else:
            print(f"\n✗ Tuned model performed worse")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default='models/best_model.joblib', help='Baseline model path')
    parser.add_argument('--tuned', default='models/tuned_model.joblib', help='Tuned model path')
    args = parser.parse_args()
    main(args.baseline, args.tuned)
