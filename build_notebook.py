import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""# Ensemble Learning Techniques

This notebook implements ensemble learning with a focus on Random Forest, compares it with other algorithms, and saves a trained model for deployment with Streamlit.
"""))

cells.append(nbf.v4.new_markdown_cell("""## 1. Introduction

Ensemble learning combines multiple models to create a stronger predictor. This notebook covers:
- Bagging and Boosting concepts
- Random Forest implementation
- Comparison with Decision Tree and SVM
- Model evaluation and serialization
- Streamlit front-end integration
"""))

cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Preparation

We generate a synthetic "Traffic Prediction" dataset containing various features like hour, temperature, and weather conditions.
"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

# Generate synthetic traffic data
np.random.seed(42)
n_samples = 1500

hours = np.random.randint(0, 24, n_samples)
temperatures = np.random.uniform(-10, 35, n_samples)
clouds = np.random.randint(0, 100, n_samples)
weather = np.random.choice([0, 1, 2, 3], n_samples) # 0: Clear, 1: Clouds, 2: Rain, 3: Snow

# Target: Traffic Volume (0: Low, 1: Medium, 2: High)
# Logic: Rush hours (7-9, 16-18) have higher traffic, Clear weather is slightly higher than snow
traffic = []
for h, t, c, w in zip(hours, temperatures, clouds, weather):
    score = 0
    if h in [7, 8, 9, 16, 17, 18]:
        score += 2
    elif 10 <= h <= 15:
        score += 1
    
    if w in [2, 3]: # Rain or Snow
        score -= 1
        
    if score >= 2:
        traffic.append(2) # High
    elif score == 1:
        traffic.append(1) # Medium
    else:
        traffic.append(0) # Low

df = pd.DataFrame({
    'Hour': hours,
    'Temperature_C': temperatures,
    'Clouds_Pct': clouds,
    'Weather_Condition': weather,
    'Traffic_Volume': traffic
})

df.to_csv('traffic_prediction.csv', index=False)
df.head()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Dataset description

The dataset contains features like Hour, Temperature, Clouds, and Weather Condition. The target is Traffic Volume (0: Low, 1: Medium, 2: High).
"""))

cells.append(nbf.v4.new_code_cell("""df.info()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 3. Preprocessing and Train/Test Split

We use feature scaling and split the dataset into training and testing sets.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv('traffic_prediction.csv')
X = df.drop('Traffic_Volume', axis=1)
y = df['Traffic_Volume']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler_pipeline = Pipeline([('scaler', StandardScaler())])
X_train_scaled = scaler_pipeline.fit_transform(X_train)
X_test_scaled = scaler_pipeline.transform(X_test)

X_train.shape, X_test.shape
"""))

cells.append(nbf.v4.new_markdown_cell("""## 4. Model Training

We implement a Random Forest and compare it with a single Decision Tree and SVM.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc})
    print(f'--- {name} ---')
    print(f'Accuracy: {acc:.4f}')
    if name == 'Random Forest':
        print('\\nClassification report:')
        print(classification_report(y_test, y_pred))
        print('Confusion matrix:')
        print(confusion_matrix(y_test, y_pred))
    print()

results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
"""))

cells.append(nbf.v4.new_markdown_cell("""## 5. Model Evaluation and Comparison

The table below summarizes the test accuracy for each algorithm. Random Forest aggregates many decision trees, typically improving performance over a single tree.
"""))

cells.append(nbf.v4.new_code_cell("""results_df
"""))

cells.append(nbf.v4.new_markdown_cell("""## 6. Save the Trained Model

We save the Random Forest model and the preprocessing scaler so the app can load them later.
"""))

cells.append(nbf.v4.new_code_cell("""import joblib

joblib.dump(models['Random Forest'], 'saved_models/random_forest_model.joblib')
joblib.dump(scaler_pipeline, 'saved_models/scaler_pipeline.joblib')
print('Saved saved_models/random_forest_model.joblib and saved_models/scaler_pipeline.joblib')
"""))

cells.append(nbf.v4.new_markdown_cell("""## 7. Sample Prediction from Saved Model

We load the saved model and make a test prediction to verify the pipeline.
"""))

cells.append(nbf.v4.new_code_cell("""loaded_model = joblib.load('saved_models/random_forest_model.joblib')
loaded_scaler = joblib.load('saved_models/scaler_pipeline.joblib')
sample = X_test.iloc[[0]]
sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled)[0]
print('Sample prediction:', prediction)
print('Actual label:', y_test.iloc[0])
"""))

nb.cells = cells
with open(r'c:\Users\user\OneDrive\Documents\Demo\Ensemble_Learning_Techniques.ipynb', 'w') as f:
    nbf.write(nb, f)
