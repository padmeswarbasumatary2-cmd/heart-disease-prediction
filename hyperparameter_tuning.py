import argparse
import json
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from data_loader import load_data
from preprocess import preprocess
from utils import save_artifact
import os

def tune_svm(X_train, y_train, X_test, y_test):
    """Hyperparameter tuning for SVM using GridSearchCV."""
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 0.01]
    }
    
    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    print(f"\nSVM Best Params: {grid_search.best_params_}")
    print(f"SVM Best CV ROC AUC: {grid_search.best_score_:.4f}")
    
    probs = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))
    roc_auc = roc_auc_score(y_test, probs)
    print(f"SVM Test ROC AUC: {roc_auc:.4f} | Accuracy: {acc:.4f}")
    
    return grid_search.best_estimator_, roc_auc, grid_search.best_params_

def tune_random_forest(X_train, y_train, X_test, y_test):
    """Hyperparameter tuning for Random Forest using RandomizedSearchCV."""
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }
    
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=5, scoring='roc_auc', 
                                       n_jobs=-1, verbose=0, random_state=42)
    random_search.fit(X_train, y_train)
    
    print(f"\nRandom Forest Best Params: {random_search.best_params_}")
    print(f"Random Forest Best CV ROC AUC: {random_search.best_score_:.4f}")
    
    probs = random_search.best_estimator_.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, random_search.best_estimator_.predict(X_test))
    roc_auc = roc_auc_score(y_test, probs)
    print(f"Random Forest Test ROC AUC: {roc_auc:.4f} | Accuracy: {acc:.4f}")
    
    return random_search.best_estimator_, roc_auc, random_search.best_params_

def tune_gradient_boosting(X_train, y_train, X_test, y_test):
    """Hyperparameter tuning for Gradient Boosting using GridSearchCV."""
    param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    print(f"\nGradient Boosting Best Params: {grid_search.best_params_}")
    print(f"Gradient Boosting Best CV ROC AUC: {grid_search.best_score_:.4f}")
    
    probs = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))
    roc_auc = roc_auc_score(y_test, probs)
    print(f"Gradient Boosting Test ROC AUC: {roc_auc:.4f} | Accuracy: {acc:.4f}")
    
    return grid_search.best_estimator_, roc_auc, grid_search.best_params_

def tune_logistic_regression(X_train, y_train, X_test, y_test):
    """Hyperparameter tuning for Logistic Regression."""
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    print(f"\nLogistic Regression Best Params: {grid_search.best_params_}")
    print(f"Logistic Regression Best CV ROC AUC: {grid_search.best_score_:.4f}")
    
    probs = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))
    roc_auc = roc_auc_score(y_test, probs)
    print(f"Logistic Regression Test ROC AUC: {roc_auc:.4f} | Accuracy: {acc:.4f}")
    
    return grid_search.best_estimator_, roc_auc, grid_search.best_params_

def main(data_path: str, out_path: str):
    df = load_data(data_path)
    X, y, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    tuned_models = {}
    best_score = -1
    best_model = None
    best_name = None
    best_params = None
    
    # Tune each model
    print("=" * 60)
    print("HYPERPARAMETER TUNING IN PROGRESS")
    print("=" * 60)
    
    try:
        svm_model, svm_score, svm_params = tune_svm(X_train, y_train, X_test, y_test)
        tuned_models['svm'] = {'model': svm_model, 'score': svm_score, 'params': svm_params}
        if svm_score > best_score:
            best_score = svm_score
            best_model = svm_model
            best_name = 'svm'
            best_params = svm_params
    except Exception as e:
        print(f"SVM tuning failed: {e}")
    
    try:
        rf_model, rf_score, rf_params = tune_random_forest(X_train, y_train, X_test, y_test)
        tuned_models['rf'] = {'model': rf_model, 'score': rf_score, 'params': rf_params}
        if rf_score > best_score:
            best_score = rf_score
            best_model = rf_model
            best_name = 'rf'
            best_params = rf_params
    except Exception as e:
        print(f"Random Forest tuning failed: {e}")
    
    try:
        gb_model, gb_score, gb_params = tune_gradient_boosting(X_train, y_train, X_test, y_test)
        tuned_models['gb'] = {'model': gb_model, 'score': gb_score, 'params': gb_params}
        if gb_score > best_score:
            best_score = gb_score
            best_model = gb_model
            best_name = 'gb'
            best_params = gb_params
    except Exception as e:
        print(f"Gradient Boosting tuning failed: {e}")
    
    try:
        lr_model, lr_score, lr_params = tune_logistic_regression(X_train, y_train, X_test, y_test)
        tuned_models['lr'] = {'model': lr_model, 'score': lr_score, 'params': lr_params}
        if lr_score > best_score:
            best_score = lr_score
            best_model = lr_model
            best_name = 'lr'
            best_params = lr_params
    except Exception as e:
        print(f"Logistic Regression tuning failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TUNING SUMMARY")
    print("=" * 60)
    for name, info in tuned_models.items():
        print(f"{name:15s} ROC AUC: {info['score']:.4f} | Params: {info['params']}")
    
    # Save best model
    artifact = {
        'model': best_model,
        'scaler': scaler,
        'features': list(X.columns),
        'score': best_score,
        'model_name': best_name,
        'hyperparameters': best_params,
        'all_tuned_results': {k: {'score': v['score'], 'params': v['params']} for k, v in tuned_models.items()}
    }
    save_artifact(artifact, out_path)
    print(f"\n✓ Saved best tuned model {best_name} (ROC AUC={best_score:.4f}) -> {out_path}")
    print(f"  Hyperparameters: {best_params}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV data')
    parser.add_argument('--out', default='models/tuned_model.joblib', help='Output artifact path')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    main(args.data, args.out)
