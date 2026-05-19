import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score
from data_loader import load_data
from preprocess import preprocess
from utils import save_artifact
import os

def main(data_path: str, out_path: str):
    df = load_data(data_path)
    X, y, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'logreg': LogisticRegression(max_iter=1000, random_state=42),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'svm': SVC(kernel='rbf', probability=True, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'nb': GaussianNB()
    }

    best_name = None
    best_score = -1
    best_model = None
    results = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:,1]
            acc = accuracy_score(y_test, model.predict(X_test))
            score = roc_auc_score(y_test, probs)
            results[name] = {'roc_auc': score, 'accuracy': acc}
            print(f'{name:8s} ROC AUC: {score:.4f} | Accuracy: {acc:.4f}')
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        except Exception as e:
            print(f'{name:8s} failed: {e}')

    artifact = {
        'model': best_model, 
        'scaler': scaler, 
        'features': list(X.columns), 
        'score': best_score, 
        'model_name': best_name,
        'all_results': results
    }
    save_artifact(artifact, out_path)
    print(f'\nSaved best model {best_name} (ROC AUC={best_score:.4f}) -> {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV data (data/heart.csv)')
    parser.add_argument('--out', default='models/best_model.joblib', help='Output artifact path')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    main(args.data, args.out)
