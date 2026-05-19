# Heart Disease Prediction

A machine learning system for predicting heart disease risk using classification models with feature engineering.

## Features

- **Data Preprocessing**: Handles missing values, feature scaling, and feature engineering
- **Feature Engineering**: Interaction terms (age×chol, age×trestbps), derived features (age binning), and ratios (thalach/oldpeak)
- **Model Comparison**: Tests 6 models: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes
- **Automated Best Model Selection**: Selects model with highest ROC AUC score
- **Evaluation Metrics**: Generates classification reports, ROC curves, and AUC scores
- **CLI Inference**: Simple command-line interface for predictions on new samples

## Quick Start

1. Place dataset CSV at `data/heart.csv` (with a `target` column).
2. Create virtual env and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. **Option A: Train baseline models**

```bash
python train.py --data data/heart.csv --out models/best_model.joblib
```

4. **Option B: Hyperparameter tuning** (Recommended)

```bash
python hyperparameter_tuning.py --data data/heart.csv --out models/tuned_model.joblib
```

5. Compare baseline vs tuned:

```bash
python compare_models.py --baseline models/best_model.joblib --tuned models/tuned_model.joblib
```

6. Evaluate:

```bash
python evaluate.py --data data/heart.csv --model models/best_model.joblib --out results
```

7. Inference examples:

```bash
# Zero-valued sample
python infer.py --model models/best_model.joblib --sample-quick

# From JSON file
python infer.py --model models/best_model.joblib --sample-file sample_patient.json

# From JSON stdin
echo '{"age":50, ...}' | python infer.py --model models/best_model.joblib --sample-stdin
```

## Dataset Format

Expected columns: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`

## Project Structure

```
heart_disease_project/
├── data/                       # Dataset (heart.csv)
├── models/                     # Trained models (best_model.joblib)
├── results/                    # Evaluation output (roc_curve.png, metrics)
├── train.py                    # Model training script
├── evaluate.py                 # Model evaluation script
├── infer.py                    # Inference CLI
├── preprocess.py               # Feature engineering and scaling
├── data_loader.py              # CSV data loader
├── utils.py                    # Helper functions (save/load artifacts)
├── generate_synthetic_heart.py # Synthetic dataset generator
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Model Performance (on synthetic dataset)

### Baseline Models (No Tuning)
| Model | ROC AUC | Accuracy |
|-------|---------|----------|
| SVM | 0.6271 | 0.9672 |
| Random Forest | 0.6059 | 0.9672 |
| Gradient Boosting | 0.5847 | 0.9508 |
| KNN | 0.4407 | 0.9672 |
| Logistic Regression | 0.4237 | 0.9672 |
| Naive Bayes | 0.2373 | 0.0656 |

### Tuned Models (GridSearchCV + RandomizedSearchCV)
| Model | ROC AUC | Hyperparameters |
|-------|---------|-----------------|
| **SVM** (Best) | **0.8305** ✓ | C=0.1, kernel='linear', gamma='scale' |
| Random Forest | 0.6059 | n_estimators=100, max_depth=10, min_samples_split=2 |
| Gradient Boosting | 0.1695 | n_estimators=100, learning_rate=0.05, max_depth=5 |
| Logistic Regression | 0.5000 | C=0.1, penalty='l1', solver='liblinear' |

### Performance Improvement
- **SVM Tuning Result: +32.4% ROC AUC improvement** (0.6271 → 0.8305)

## Feature Engineering

The preprocessing pipeline adds:
- **Interaction Features**: age×cholesterol, age×blood pressure
- **Derived Features**: max heart rate / ST depression ratio, age group binning
- **Scaling**: StandardScaler normalization for all features

## Hyperparameter Tuning

The project includes automated hyperparameter tuning using:
- **GridSearchCV** for SVM, Gradient Boosting, and Logistic Regression
- **RandomizedSearchCV** for Random Forest (faster for large search spaces)

Key tuning parameters:
- **SVM**: C (regularization), kernel (rbf/linear), gamma (kernel coefficient)
- **Random Forest**: n_estimators, max_depth, min_samples_split, min_samples_leaf
- **Gradient Boosting**: n_estimators, learning_rate, max_depth, subsample
- **Logistic Regression**: C (regularization), penalty (l1/l2), solver

Tuning results show SVM with linear kernel and C=0.1 achieves 0.8305 ROC AUC, a significant improvement over baseline models.
