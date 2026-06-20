
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument('--reg_rate', type=float, default=0.01)
args = parser.parse_args()

# Generate synthetic data (so the workflow doesn't depend on data assets)
import numpy as np
np.random.seed(42)
n = 300
df = pd.DataFrame({
    'Pregnancies': np.random.randint(0, 15, n),
    'PlasmaGlucose': np.random.randint(70, 200, n),
    'BMI': np.round(np.random.uniform(18, 45, n), 2),
    'Age': np.random.randint(21, 70, n),
})
df['Diabetic'] = ((df['PlasmaGlucose'] > 140) | (df['BMI'] > 35)).astype(int)

X = df.drop('Diabetic', axis=1)
y = df['Diabetic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mlflow.autolog()
model = LogisticRegression(C=1/args.reg_rate, max_iter=1000)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
mlflow.log_metric('accuracy', acc)
print(f'Accuracy: {acc:.4f}')
