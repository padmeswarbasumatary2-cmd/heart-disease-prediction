import argparse
from data_loader import load_data
from preprocess import preprocess
from utils import load_artifact
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import os

def evaluate(data_path: str, model_path: str, out_dir: str = 'results'):
    df = load_data(data_path)
    artifact = load_artifact(model_path)
    scaler = artifact['scaler']
    model = artifact['model']
    
    # Apply same preprocessing (feature engineering) used during training
    X, y, _ = preprocess(df)
    
    # Apply the saved scaler (already fit)
    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)[:,1]
    preds = model.predict(X_scaled)

    print(classification_report(y, preds))

    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
    print('ROC AUC:', roc_auc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', default='results')
    args = parser.parse_args()
    evaluate(args.data, args.model, args.out)
