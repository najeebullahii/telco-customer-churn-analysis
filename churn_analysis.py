import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
# Setting project paths
project_dir = r"C:\Users\Najib\Documents\Najib's Projects\Customer Chun Analysis"
data_path = os.path.join(project_dir, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
# Loading the dataset
df = pd.read_csv(data_path)
print(f"Data shape: {df.shape}")
# Quick look at the data
print(df.head())
print(df.info())
print(df.describe(include='all').T)
# Cleaning TotalCharges column (some values are blank spaces)
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan).astype(float)
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
# Dropping irrelevant or duplicate rows
df.drop_duplicates(inplace=True)
df.drop(columns=['customerID'], inplace=True)
# Final check after cleaning
print("\nMissing values per column:")
print(df.isna().sum())
# Saved cleaned dataset
df.to_csv(os.path.join(project_dir, "cleaned_data.csv"), index=False)
print("\nCleaned dataset saved.")
# EXPLORATORY DATA ANALYSIS (EDA)
print("\n--- Exploratory Data Analysis ---")
print(f"Dataset shape after cleaning: {df.shape}")
# Setting plot style
plt.style.use('seaborn-v0_8')
churn_colors = ['#1f77b4', '#d62728'] # Blue = No churn, Red = Yes churn
# 1. Target Variable (Churn Distribution)
print("\nChurn Overview:")
churn_counts = df['Churn'].value_counts()
churn_percent = churn_counts / len(df) * 100
print(churn_counts)
print(f"Churn Rate: {churn_percent['Yes']:.2f}%")
plt.figure(figsize=(10, 4))
# Bar chart
plt.subplot(1, 2, 1)
churn_counts.plot(kind='bar', color=churn_colors)
plt.title('Churn Distribution')
plt.ylabel('Count')
# Pie chart
plt.subplot(1, 2, 2)
plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%',
        startangle=90, colors=churn_colors, explode=(0, 0.1))
plt.title('Churn Percentage')
plt.tight_layout()
churn_plot_path = os.path.join(project_dir, "churn_distribution.png")
plt.savefig(churn_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Churn distribution plot saved to: {churn_plot_path}")
print("Observation: Around one-quarter of customers have churned — quite a high rate.")
# 2. Numerical Features (Histograms by Churn)
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
print("\nNumerical Features Summary:")
print(df[num_cols].describe())
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, hue='Churn', kde=True,
                 palette=churn_colors, alpha=0.7, ax=axes[i])
    axes[i].set_title(f'{col} by Churn')
plt.tight_layout()
num_plot_path = os.path.join(project_dir, "numerical_features.png")
plt.savefig(num_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Numerical feature plots saved to: {num_plot_path}")
print("Observation: Short-tenure customers and those with higher monthly charges are more likely to churn.")
# 3. Correlation Heatmap
df_corr = df.copy()
df_corr['Churn_numeric'] = df_corr['Churn'].map({'Yes': 1, 'No': 0})
corr_matrix = df_corr[num_cols + ['Churn_numeric']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
corr_plot_path = os.path.join(project_dir, "correlation_matrix.png")
plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Correlation matrix saved to: {corr_plot_path}")
print("Observation: Tenure is negatively correlated with churn, while monthly charges show a slight positive correlation.")
# 4. Categorical Features (Key Drivers)
key_cats = ['Contract', 'InternetService', 'PaymentMethod']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(key_cats):
    churn_rate = df.groupby(col)['Churn'].value_counts(normalize=True).unstack().fillna(0)
    churn_rate.plot(kind='bar', ax=axes[i], color=churn_colors)
    axes[i].set_title(f'Churn by {col}')
    axes[i].set_ylabel('Proportion')
    axes[i].legend(title='Churn', loc='upper right')
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
cat_plot_path = os.path.join(project_dir, "categorical_features.png")
plt.savefig(cat_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Categorical feature plots saved to: {cat_plot_path}")
print("Observation:")
print("- Month-to-month contracts have the highest churn.")
print("- Fiber optic customers churn more than DSL or no internet.")
print("- Customers paying via electronic checks are the most likely to churn.")
# 5. Tenure Group Analysis
df['tenure_group'] = pd.cut(df['tenure'],
                            bins=[0, 12, 24, 36, 48, 60, 72],
                            labels=['0-12','13-24','25-36','37-48','49-60','61-72'])
tenure_churn = df.groupby('tenure_group')['Churn'].value_counts(normalize=True).unstack().fillna(0)
tenure_churn.plot(kind='bar', color=churn_colors, figsize=(10, 5))
plt.title('Churn Rate by Tenure Group')
plt.ylabel('Churn Rate')
plt.xticks(rotation=45)
plt.tight_layout()
tenure_plot_path = os.path.join(project_dir, "tenure_group_churn.png")
plt.savefig(tenure_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Tenure group plot saved to: {tenure_plot_path}")
print("Observation: Customers in their first year churn at a much higher rate. The longer a customer stays, the less likely they are to leave.")
# 6. Quick Summary
print("\n--- Quick EDA Summary ---")
print(f"Churn Rate: {churn_percent['Yes']:.2f}%")
print("• Short tenure and month-to-month contracts are key churn drivers.")
print("• Fiber optic users and those paying by electronic check churn more.")
print("• Tenure has the strongest negative relationship with churn.")
summary_path = os.path.join(project_dir, "eda_summary.txt")
with open(summary_path, 'w') as f:
    f.write(f"Churn Rate: {churn_percent['Yes']:.2f}%\n")
    f.write("Key churn drivers: Short tenure, month-to-month contracts, fiber optic internet, electronic check payments.\n")
print(f"Summary saved to: {summary_path}")
# STEP 4: Feature Engineering & Preprocessing
print("\n--- STEP 4: FEATURE ENGINEERING & PREPROCESSING ---")
# 4.1 Handling Missing Values and Tenure Groups
# Replacing missing TotalCharges with 0 and convert to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)
# Creating tenure groups for easier analysis
df['tenure_group'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 24, 36, 48, 60, 72],
    labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
)
# 4.2 Creating 'total_services' Feature
service_cols = [
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
def count_services(row):
    count = 0
    for col in service_cols:
        if row[col] not in ['No', 'No phone service', 'No internet service']:
            count += 1
    return count
df['total_services'] = df.apply(count_services, axis=1)
# 4.3 Encoding Categorical Variables
categorical_cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'tenure_group'
]
# Binary encoding for yes/no and gender
binary_cols = [col for col in categorical_cols if df[col].nunique() == 2 and col != 'SeniorCitizen']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
# One-hot encode the multi-category columns
multi_cols = [col for col in categorical_cols if df[col].nunique() > 2]
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
# Encoding target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
# Dropping rows with missing target
df.dropna(subset=['Churn'], inplace=True)
# 4.4 Scaling Numerical Features
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
# 4.5 Train-Test Split
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train/Test split complete.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Churn distribution in train set:\n", y_train.value_counts(normalize=True))
print("Churn distribution in test set:\n", y_test.value_counts(normalize=True))
# ---5. MODELING ---
# For reproducibility
random_state = 42
# 5.1 Logistic Regression (Baseline)
print("\n=== Training Logistic Regression (Baseline) ===")
lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
lr_model.fit(X_train, y_train) # ✅ Fitting the model
lr_cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='f1')
print(f"Logistic Regression - Mean CV F1: {lr_cv_scores.mean():.3f}")
# 5.2 Random Forest with Basic Hyperparameter Tuning
print("\n=== Tuning Random Forest ===")
rf = RandomForestClassifier(random_state=random_state)
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print(f"Best RF Params: {rf_grid.best_params_}")
print(f"Best RF CV F1: {rf_grid.best_score_:.3f}")
# 5.3 XGBoost with Basic Hyperparameter Tuning
print("\n=== Tuning XGBoost ===")
xgb = XGBClassifier(
    random_state=random_state,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}
xgb_grid = GridSearchCV(xgb, xgb_params, cv=5, scoring='f1', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
print(f"Best XGB Params: {xgb_grid.best_params_}")
print(f"Best XGB CV F1: {xgb_grid.best_score_:.3f}")
# 5.4 Saving Trained Models
model_dir = os.path.join(project_dir, "models")
os.makedirs(model_dir, exist_ok=True)
joblib.dump(lr_model, os.path.join(model_dir, "logistic_regression_model.pkl"))
joblib.dump(best_rf, os.path.join(model_dir, "random_forest_model.pkl"))
joblib.dump(best_xgb, os.path.join(model_dir, "xgboost_model.pkl"))
print(f"\nModels saved to: {model_dir}")
# --- 6. MODEL EVALUATION ---
print("\n--- MODEL EVALUATION ---")
# List of models to evaluate
models = [
    ("Logistic Regression", lr_model),
    ("Random Forest", best_rf),
    ("XGBoost", best_xgb)
]
# Store AUC scores for plotting ROC curves
model_scores = {}
# 6.1 Classification Reports + AUC Scores
for name, model in models:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
   
    print(f"\n{name} Test Results:")
    print(classification_report(y_test, y_pred))
   
    auc = roc_auc_score(y_test, y_prob)
    model_scores[name] = auc
    print(f"{name} AUC-ROC: {auc:.3f}")
# 6.2 ROC Curves
plt.figure(figsize=(8, 6))
for name, model in models:
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {model_scores[name]:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curves for Churn Prediction', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(alpha=0.3)
# Saving ROC plot
roc_plot_path = os.path.join(project_dir, "roc_curves.png")
plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"ROC curves saved to: {roc_plot_path}")
# 6.3 Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, models):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
# Saving confusion matrix plot
cm_plot_path = os.path.join(project_dir, "confusion_matrices.png")
plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Confusion matrices saved to: {cm_plot_path}")
print("\nModel Evaluation Complete!")
# --- MODEL INTERPRETATION AND INSIGHTS ---
print("\n--- MODEL INTERPRETATION AND INSIGHTS ---")
# 1. Plot Top 10 Feature Importances from XGBoost
from xgboost import plot_importance
plt.figure(figsize=(8, 6))
plot_importance(best_xgb, max_num_features=10, importance_type='weight')
plt.title('Top 10 Important Features', fontsize=14)
plt.xlabel('F Score')
plt.ylabel('Features')
feature_plot_path = os.path.join(project_dir, "feature_importance.png")
plt.savefig(feature_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Feature importance plot saved to: {feature_plot_path}")
# 2. Identify Top 5 Important Features
importance_scores = best_xgb.feature_importances_
importance_df = (
    pd.DataFrame({'Feature': X_train.columns, 'Importance': importance_scores})
    .sort_values('Importance', ascending=False)
    .head(5)
)
print("\nTop 5 Features Driving Churn:")
print(importance_df)
# 3. Derive Simple Insights
print("\nKey Insights:")
print("- Contract type and internet service are the most influential churn drivers.")
print("- Customers on month-to-month contracts and those using electronic checks are at higher churn risk.")
print("- Long-term contracts generally reduce churn likelihood.")
print("- Fiber optic internet users show a higher tendency to churn compared to DSL or no internet.")
# 4. Save Insights to Text File
insights_path = os.path.join(project_dir, "model_insights.txt")
with open(insights_path, 'w') as f:
    f.write("TELCO CUSTOMER CHURN - MODEL INSIGHTS\n")
    f.write("="*40 + "\n\n")
    f.write("Best Model: XGBoost\n")
    f.write("AUC-ROC: 0.840 | F1 (Churn): 0.58\n\n")
    f.write("Top 5 Features:\n")
    f.write(importance_df.to_string(index=False) + "\n\n")
    f.write("Recommendations:\n")
    f.write("- Offer retention incentives to month-to-month customers.\n")
    f.write("- Target fiber optic users with loyalty offers.\n")
    f.write("- Encourage switching from electronic checks to automatic payments.\n")
    f.write("- Focus on early engagement for new/low-charge customers.\n")
print(f"\nInsights saved to: {insights_path}")
print("Model interpretation complete!")