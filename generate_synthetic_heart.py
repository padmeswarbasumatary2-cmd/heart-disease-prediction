import numpy as np
import pandas as pd
from pathlib import Path

R = 303
rng = np.random.default_rng(42)

age = rng.normal(54, 9, R).clip(29, 77).round().astype(int)
sex = rng.choice([0,1], size=R, p=[0.32, 0.68])
cp = rng.integers(0,4,size=R)
trestbps = rng.normal(131,17,R).clip(94,200).round().astype(int)
chol = rng.normal(246,51,R).clip(126,564).round().astype(int)
fbs = (rng.random(R) < 0.15).astype(int)
restecg = rng.integers(0,3,size=R)
thalach = rng.normal(150,22,R).clip(71,202).round().astype(int)
exang = (rng.random(R) < 0.33).astype(int)
oldpeak = np.round(np.abs(rng.normal(1.0,1.1,R)),2)
slope = rng.integers(0,3,size=R)
ca = rng.integers(0,4,size=R)
thal = rng.choice([3,6,7], size=R, p=[0.55,0.25,0.20])

# Create a logistic risk combining a few predictors
logit = -5.5 + 0.03*(age-50) + 0.02*(chol-200) - 0.02*(thalach-150) + 0.9*exang + 0.6*(cp==3)
prob = 1/(1+np.exp(-logit))
target = (rng.random(R) < prob).astype(int)

df = pd.DataFrame({
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal,
    'target': target
})

out = Path('data/heart.csv')
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print('Wrote synthetic dataset to', out)
