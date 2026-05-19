import argparse
import json
from utils import load_artifact
from preprocess import preprocess
from data_loader import load_data
import pandas as pd
import numpy as np

def infer_from_sample(artifact, sample: dict):
    """Run inference on a single sample (raw feature dict)."""
    scaler = artifact['scaler']
    model = artifact['model']
    
    # Convert raw sample dict to DataFrame for preprocessing
    sample['target'] = 0  # Add dummy target for preprocess function
    sample_df = pd.DataFrame([sample])
    
    # Apply same preprocessing pipeline
    X_processed, _, _ = preprocess(sample_df)
    
    # Apply scaler
    x_scaled = scaler.transform(X_processed)
    
    prob = model.predict_proba(x_scaled)[0,1]
    pred = int(prob >= 0.5)
    return {'probability': float(prob), 'prediction': pred}

def main(model_path: str, sample_json: str = None):
    artifact = load_artifact(model_path)
    if sample_json:
        sample = json.loads(sample_json)
    else:
        # Build a dummy sample of zeros for raw features (before feature engineering)
        # Use only the original raw features, not engineered ones
        original_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        sample = {f: 0 for f in original_features}
    
    result = infer_from_sample(artifact, sample)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--sample', help='JSON string with feature values')
    parser.add_argument('--sample-file', help='Path to JSON file with sample')
    parser.add_argument('--sample-stdin', action='store_true', help='Read sample JSON from stdin')
    parser.add_argument('--sample-quick', dest='quick', action='store_true', help='Use zero-valued quick sample')
    args = parser.parse_args()
    sample = None
    if args.sample:
        sample = args.sample
    elif args.sample_file:
        with open(args.sample_file,'r') as f:
            sample = f.read()
    elif args.sample_stdin:
        import sys
        sample = sys.stdin.read()
    elif args.quick:
        sample = None
    if sample is None and not args.quick:
        print('No sample provided; use --sample-quick for a zero sample or --sample JSON')
    main(args.model, sample)
