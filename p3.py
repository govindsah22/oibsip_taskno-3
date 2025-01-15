import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
wine_data = pd.read_csv("C:/Users/niraj/Desktop/Internship/Wine Quality Prediction/WineQT.csv")

# Display basic information about the dataset
print(wine_data.info())
print(wine_data.describe())
print(wine_data.head())

# Check for missing values
print(wine_data.isnull().sum())

# Separate features and target variable
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Exploratory Data Analysis (EDA)
sns.countplot(x='quality', hue='quality', data=wine_data, palette='pastel', legend=False)
plt.title('Distribution of Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(wine_data.corr(), annot=False, cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.show()

# Initialize Models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SGDClassifier': SGDClassifier(random_state=42),
    'SVC': SVC(random_state=42)
}

# Hyperparameter Grids
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SGDClassifier': {
        'loss': ['hinge', 'log', 'modified_huber'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [1000, 2000, 3000]
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
}

# Train, Tune, and Evaluate Each Model
for model_name, model in models.items():
    print(f"Processing {model_name}...\n")
    
    # Grid Search for Hyperparameter Tuning
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}\n")
    
    # Evaluate Model
    y_pred = best_model.predict(X_test)
    print(f"{model_name} Classification Report (Tuned):\n")
    print(classification_report(y_test, y_pred))
    print(f"{model_name} Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    
    # Additional Analysis for Specific Models
    if model_name == 'Random Forest':
        # Feature Importances
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X.columns
        
        plt.figure(figsize=(10, 6))
        plt.title(f'{model_name} Feature Importances')
        plt.bar(range(X.shape[1]), importances[indices], align='center', color='skyblue')
        plt.xticks(range(X.shape[1]), features[indices], rotation=90)
        plt.tight_layout()
        plt.show()
        
    elif model_name == 'SGDClassifier':
        # Feature Coefficients
        coef = best_model.coef_.flatten()
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(coef)), coef, color='lightgreen')
        plt.xticks(range(len(coef)), X.columns, rotation=90)
        plt.title(f'{model_name} Feature Coefficients')
        plt.tight_layout()
        plt.show()
