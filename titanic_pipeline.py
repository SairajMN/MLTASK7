#!/usr/bin/env python3
"""
Titanic Survival Prediction Pipeline
====================================
End-to-end binary classification pipeline using Logistic Regression
to predict passenger survival on the Titanic.

Author: Senior Data Scientist
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load Titanic dataset from seaborn or Kaggle source."""
    print("1. Loading Titanic Dataset...")
    
    try:
        # Try to load from seaborn first (fallback option)
        from seaborn import load_dataset
        df = load_dataset('titanic')
        print("Loaded dataset from seaborn")
    except:
        # Alternative: Load from Kaggle CSV if available
        try:
            df = pd.read_csv('train.csv')
            print("Loaded dataset from train.csv")
        except:
            print("Error: Could not load dataset from either seaborn or train.csv")
            print("Please ensure you have either seaborn installed or train.csv in the current directory")
            return None
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    return df

def explore_data(df):
    """Perform initial data exploration."""
    print("\n" + "="*50)
    print("2. DATA EXPLORATION")
    print("="*50)
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    print("\nTarget Variable Distribution (Survived):")
    print(df['survived'].value_counts())
    print(f"Survival Rate: {df['survived'].mean():.2%}")
    
    # Key features explanation
    print("\nKey Features Analysis:")
    print("- Age: Passenger age (numerical)")
    print("- Sex: Passenger gender (categorical: male/female)")
    print("- Fare: Ticket price paid (numerical)")
    print("- Pclass: Passenger class (1=1st, 2=2nd, 3=3rd)")
    print("- Embarked: Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)")

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    print("\n" + "="*50)
    print("3. MISSING VALUE HANDLING")
    print("="*50)
    
    # Check missing values
    missing_values = df.isnull().sum()
    print("Missing values before handling:")
    print(missing_values[missing_values > 0])
    
    # Handle Age missing values with median
    # Justification: Median is robust to outliers compared to mean
    age_median = df['age'].median()
    df['age'].fillna(age_median, inplace=True)
    print(f"Filled missing Age values with median: {age_median}")
    
    # Handle Embarked missing values with mode
    # Justification: Mode is appropriate for categorical data
    embarked_mode = df['embarked'].mode()[0]
    df['embarked'].fillna(embarked_mode, inplace=True)
    print(f"Filled missing Embarked values with mode: {embarked_mode}")
    
    # Cabin has too many missing values, will be dropped later
    if 'cabin' in df.columns:
        print(f"Note: Cabin column has {df['cabin'].isnull().sum()} missing values (will be dropped)")
    else:
        print("Note: Cabin column not found (already dropped)")
    
    # Verify no more critical missing values
    remaining_missing = df[['age', 'embarked', 'survived']].isnull().sum().sum()
    print(f"Remaining missing values in key columns: {remaining_missing}")
    
    return df

def feature_selection(df):
    """Select relevant features for modeling."""
    print("\n" + "="*50)
    print("4. FEATURE SELECTION")
    print("="*50)
    
    # Remove non-informative columns
    # Justification:
    # - PassengerId: Unique identifier, no predictive power
    # - Name: Text data, would require complex NLP processing
    # - Ticket: Complex format, limited predictive value
    # - Cabin: Too many missing values (77%)
    
    columns_to_drop = ['passenger_id', 'name', 'ticket', 'cabin']
    # Only drop columns that exist in the dataset
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_clean = df.drop(columns=columns_to_drop)
    
    print(f"Dropped columns: {columns_to_drop}")
    print(f"Remaining columns: {list(df_clean.columns)}")
    print(f"New dataset shape: {df_clean.shape}")
    
    return df_clean

def prepare_features(df):
    """Prepare features for machine learning."""
    print("\n" + "="*50)
    print("5. FEATURE PREPARATION")
    print("="*50)
    
    # Separate features and target
    X = df.drop('survived', axis=1)
    y = df['survived']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Identify numerical and categorical features
    numerical_features = ['age', 'fare']
    categorical_features = ['sex', 'embarked', 'pclass']
    
    print(f"\nNumerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")
    
    return X, y, numerical_features, categorical_features

def create_pipeline(numerical_features, categorical_features):
    """Create preprocessing pipeline."""
    print("\n" + "="*50)
    print("6. PREPROCESSING PIPELINE")
    print("="*50)
    
    # Numerical transformer: StandardScaler
    # Justification: Logistic Regression is sensitive to feature scale
    numerical_transformer = StandardScaler()
    
    # Categorical transformer: OneHotEncoder
    # Justification: Converts categorical variables to binary features
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    print("Created preprocessing pipeline:")
    print("  - Numerical: StandardScaler")
    print("  - Categorical: OneHotEncoder (drop first to avoid dummy variable trap)")
    print("  - Classifier: LogisticRegression")
    
    return pipeline

def train_model(pipeline, X_train, y_train, numerical_features, categorical_features):
    """Train the Logistic Regression model."""
    print("\n" + "="*50)
    print("7. MODEL TRAINING")
    print("="*50)
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Get feature names after preprocessing
    feature_names = (numerical_features + 
                    list(pipeline.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .get_feature_names_out(categorical_features)))
    
    print("Model trained successfully")
    print(f"Total features after encoding: {len(feature_names)}")
    
    # Show feature importance (coefficients)
    coefficients = pipeline.named_steps['classifier'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\nFeature Importance (by coefficient magnitude):")
    print(feature_importance.head(10).to_string(index=False))
    
    return pipeline, feature_importance

def evaluate_model(pipeline, X_test, y_test):
    """Evaluate model performance."""
    print("\n" + "="*50)
    print("8. MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print("Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (TN):  {cm[0,0]}")
    print(f"  False Positives (FP): {cm[0,1]}")
    print(f"  False Negatives (FN): {cm[1,0]}")
    print(f"  True Positives (TP):  {cm[1,1]}")
    
    return y_pred, y_pred_proba, accuracy, precision, recall, f1, auc, cm

def plot_results(y_test, y_pred, y_pred_proba, cm, auc):
    """Create and save visualization plots."""
    print("\n" + "="*50)
    print("9. VISUALIZATION & RESULTS")
    print("="*50)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Did Not Survive', 'Survived'],
                yticklabels=['Did Not Survive', 'Survived'])
    axes[0].set_title('Confusion Matrix\n(TN, FP, FN, TP)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    
    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('ROC Curve Analysis\n(Trade-off between Sensitivity and Specificity)', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('titanic_results.png', dpi=300, bbox_inches='tight')
    print("Saved visualization: titanic_results.png")
    plt.show()
    
    return fig

def explain_logistic_regression():
    """Explain Logistic Regression conceptually."""
    print("\n" + "="*50)
    print("10. LOGISTIC REGRESSION EXPLANATION")
    print("="*50)
    
    print("""
CONCEPTUAL UNDERSTANDING:

1. LOGISTIC REGRESSION BASICS:
   - Despite its name, it's a classification algorithm
   - Uses sigmoid function to map linear combinations to probabilities [0,1]
   - Formula: P(y=1|x) = 1 / (1 + e^(-z)) where z = β₀ + β₁x₁ + ... + βₙxₙ

2. DECISION BOUNDARY:
   - Threshold typically set at 0.5
   - P(y=1|x) > 0.5 → Predict Class 1 (Survived)
   - P(y=1|x) ≤ 0.5 → Predict Class 0 (Did Not Survive)

3. WHY STANDARDIZATION?
   - Logistic Regression uses gradient descent for optimization
   - Features on different scales converge at different rates
   - Standardization ensures equal contribution from all features
   - Prevents features with larger scales from dominating

4. WHY ROC-AUC OVER ACCURACY?
   - Accuracy can be misleading with imbalanced datasets
   - ROC-AUC evaluates performance across all possible thresholds
   - Provides insight into trade-off between True Positive Rate and False Positive Rate
   - More robust metric for binary classification evaluation

5. INTERPRETATION:
   - Positive coefficients: Increase probability of survival
   - Negative coefficients: Decrease probability of survival
   - Magnitude indicates strength of relationship
""")

def main():
    """Main pipeline execution."""
    print("TITANIC SURVIVAL PREDICTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load Data
    df = load_data()
    if df is None:
        return
    
    # Step 2: Explore Data
    explore_data(df)
    
    # Step 3: Handle Missing Values
    df = handle_missing_values(df)
    
    # Step 4: Feature Selection
    df = feature_selection(df)
    
    # Step 5: Prepare Features
    X, y, numerical_features, categorical_features = prepare_features(df)
    
    # Step 6: Create Pipeline
    pipeline = create_pipeline(numerical_features, categorical_features)
    
    # Step 7: Train-Test Split
    print("\n" + "="*50)
    print("6. TRAIN-TEST SPLIT")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Training set survival rate: {y_train.mean():.2%}")
    print(f"Test set survival rate: {y_test.mean():.2%}")
    print("Used stratified sampling to preserve class balance")
    
    # Step 8: Train Model
    pipeline, feature_importance = train_model(pipeline, X_train, y_train, numerical_features, categorical_features)
    
    # Step 9: Evaluate Model
    y_pred, y_pred_proba, accuracy, precision, recall, f1, auc, cm = evaluate_model(
        pipeline, X_test, y_test
    )
    
    # Step 10: Visualization
    fig = plot_results(y_test, y_pred, y_pred_proba, cm, auc)
    
    # Step 11: Explanation
    explain_logistic_regression()
    
    # Final Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f" Dataset loaded: {df.shape[0]} passengers, {df.shape[1]} features")
    print(f" Missing values handled: Age (median), Embarked (mode)")
    print(f" Features selected: {len(numerical_features + categorical_features)}")
    print(f" Model trained: Logistic Regression with preprocessing pipeline")
    print(f" Performance: AUC = {auc:.3f}, Accuracy = {accuracy:.3f}")
    print(f" Visualizations saved: titanic_results.png")
    print("\n Pipeline completed successfully!")
    print(" Ready for academic submission and viva explanation")

if __name__ == "__main__":
    main()