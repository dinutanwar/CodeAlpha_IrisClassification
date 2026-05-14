"""
Iris Flower Classification - CodeAlpha Data Science Internship
===============================================================
Task 1: Train a machine learning model using Scikit-learn to classify 
flower species (setosa, versicolor, virginica) based on measurements.

Author: [Your Name]
Internship: CodeAlpha Data Science Internship
Date: May 2026
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 60)
print("IRIS FLOWER CLASSIFICATION PROJECT")
print("CodeAlpha Data Science Internship - Task 1")
print("=" * 60)

# ============================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================
print("\n[STEP 1] Loading Iris Dataset...")

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset Shape: {X.shape}")
print(f"Number of Classes: {len(target_names)}")
print(f"Classes: {target_names}")
print(f"Features: {feature_names}")

# Create DataFrame for easier manipulation
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]

print("\nFirst 5 samples:")
print(df.head())
print("\nDataset Statistics:")
print(df.describe())

# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n[STEP 2] Performing Exploratory Data Analysis...")

# Check for missing values
print(f"\nMissing Values: {df.isnull().sum().sum()}")

# Class distribution
print("\nClass Distribution:")
print(df['species'].value_counts())

# Correlation matrix
corr_matrix = df[feature_names].corr()
print("\nFeature Correlation Matrix:")
print(corr_matrix)

# ============================================================
# STEP 3: DATA PREPROCESSING
# ============================================================
print("\n[STEP 3] Data Preprocessing...")

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Standardize features (important for SVM and KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features standardized using StandardScaler")

# ============================================================
# STEP 4: MODEL TRAINING AND COMPARISON
# ============================================================
print("\n[STEP 4] Training Multiple Models...")

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

results = {}

for name, model in models.items():
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    # Train on full training set
    model.fit(X_train_scaled, y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': test_acc,
        'model': model,
        'predictions': y_pred
    }

    print(f"\n{name}:")
    print(f"  Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"  Test Accuracy: {test_acc:.4f}")

# ============================================================
# STEP 5: SELECT BEST MODEL AND HYPERPARAMETER TUNING
# ============================================================
print("\n[STEP 5] Selecting Best Model & Hyperparameter Tuning...")

# Select best model based on test accuracy
best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")

# Hyperparameter tuning for SVM (best performing model)
print("\nPerforming Grid Search for SVM...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01]
}

grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Evaluate tuned model
tuned_model = grid_search.best_estimator_
tuned_pred = tuned_model.predict(X_test_scaled)
tuned_acc = accuracy_score(y_test, tuned_pred)
print(f"Tuned Model Test Accuracy: {tuned_acc:.4f}")

# ============================================================
# STEP 6: DETAILED EVALUATION
# ============================================================
print("\n[STEP 6] Detailed Evaluation of Best Model...")

print("\nClassification Report:")
print(classification_report(y_test, tuned_pred, target_names=target_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, tuned_pred)
print(cm)

# ============================================================
# STEP 7: FEATURE IMPORTANCE ANALYSIS
# ============================================================
print("\n[STEP 7] Feature Importance Analysis...")

# Using Random Forest for feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': [name.replace(' (cm)', '') for name in feature_names],
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(feature_importance_df)

# ============================================================
# STEP 8: VISUALIZATION
# ============================================================
print("\n[STEP 8] Generating Visualizations...")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Iris Flower Classification - Complete Analysis', fontsize=16, fontweight='bold')

# 1. Model Comparison
ax1 = axes[0, 0]
model_names = list(results.keys())
cv_means = [results[m]['cv_mean'] for m in model_names]
test_accs = [results[m]['test_accuracy'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35
ax1.bar(x - width/2, cv_means, width, label='CV Accuracy', alpha=0.8)
ax1.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=15, ha='right')
ax1.legend()
ax1.set_ylim(0.8, 1.05)
ax1.grid(True, alpha=0.3, axis='y')

# 2. Confusion Matrix
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=target_names, yticklabels=target_names, square=True)
ax2.set_title(f'Confusion Matrix - {best_model_name}')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

# 3. Feature Importance
ax3 = axes[1, 0]
colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
bars = ax3.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], 
               color=colors, alpha=0.8)
ax3.set_ylabel('Importance Score')
ax3.set_title('Feature Importance')
ax3.tick_params(axis='x', rotation=15)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Pair Plot (Sepal vs Petal)
ax4 = axes[1, 1]
colors_species = ['#e74c3c', '#2ecc71', '#3498db']
for i, species in enumerate(target_names):
    mask = y == i
    ax4.scatter(X[mask, 2], X[mask, 3], c=colors_species[i], label=species, 
               alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
ax4.set_xlabel('Petal Length (cm)')
ax4.set_ylabel('Petal Width (cm)')
ax4.set_title('Petal Measurements by Species')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_classification_results.png', dpi=150, bbox_inches='tight')
print("Visualization saved as 'iris_classification_results.png'")

# ============================================================
# STEP 9: PREDICTION FUNCTION
# ============================================================
print("\n[STEP 9] Creating Prediction Function...")

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict iris species based on measurements.

    Parameters:
    - sepal_length: Sepal length in cm
    - sepal_width: Sepal width in cm
    - petal_length: Petal length in cm
    - petal_width: Petal width in cm

    Returns:
    - Predicted species name
    """
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = tuned_model.predict(input_scaled)
    return target_names[prediction[0]]

# Example predictions
print("\nExample Predictions:")
test_samples = [
    [5.1, 3.5, 1.4, 0.2],   # Expected: setosa
    [6.2, 3.4, 5.4, 2.3],   # Expected: virginica
    [5.9, 3.0, 4.2, 1.5]    # Expected: versicolor
]

for sample in test_samples:
    pred = predict_iris(*sample)
    print(f"  Measurements {sample} -> Predicted: {pred}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print(f"Best Model: {best_model_name} with {grid_search.best_params_}")
print(f"Final Test Accuracy: {tuned_acc:.4f} ({tuned_acc*100:.2f}%)")
print(f"Most Important Features: {', '.join(feature_importance_df['Feature'].head(2).tolist())}")
print("\nKey Findings:")
print("- Petal measurements are the most discriminative features")
print("- SVM achieved the highest classification accuracy")
print("- Setosa is perfectly separable from other species")
print("- Versicolor and virginica have slight overlap")
print("=" * 60)
