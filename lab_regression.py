"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, r2_score, recall_score)



def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset.

    Returns:
        DataFrame with all columns.
    """
    # TODO: Load the CSV and return the DataFrame
    df = pd.read_csv(filepath)
    return df


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # TODO: Separate features and target, then split with stratification
    X=df.drop(columns=[target_col])
    y=df[target_col]
    # إذا classification (binary)
    if y.nunique() <= 2:
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    # إذا regression
    else:
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )


def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """
    # TODO: Create and return a Pipeline with two steps
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    return pipe


def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """
    # TODO: Create and return a Pipeline for Ridge regression

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    return pipe



def build_lasso_pipeline():
    """Build a Pipeline with StandardScaler and Lasso regression."""
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1))
    ])
    
    return pipe    

def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics.

    Args:
        pipeline: sklearn Pipeline with a classifier.
        X_train, X_test: Feature arrays.
        y_train, y_test: Label arrays.

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    # TODO: Fit the pipeline on training data, predict on test, compute metrics
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }
  

def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics.

    Args:
        pipeline: sklearn Pipeline with a regressor.
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.

    Returns:
        Dictionary with keys: 'mae', 'r2'.
    """
    # TODO: Fit the pipeline, predict, and compute MAE and R²
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }
   


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """
    # TODO: Run cross_val_score with StratifiedKFold
    cv_splitter = StratifiedKFold(
    n_splits=cv,
    shuffle=True,
    random_state=42
    )
#Task 6    
#CV: 0.607 +/- 0.019
#0.607 → متوسط الدقة (accuracy) عبر 5 folds
#± 0.019 → التذبذب (variance) بين الفولدز

# Cross-validation results show that the model achieves an average accuracy of ~0.61
# with low variance (+/- 0.019), indicating consistent performance across folds.
# However, the accuracy is relatively moderate, which may be due to class imbalance
# in the churn dataset (~16% churn rate), making accuracy less reliable.

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring="accuracy"
    )

    return scores


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        # Select numeric features for classification
        numeric_features = ["tenure", "monthly_charges", "total_charges",
                           "num_support_calls", "senior_citizen",
                           "has_partner", "has_dependents"]

        # Classification: predict churn
        df_cls = df[numeric_features + ["churned"]].dropna()
        split = split_data(df_cls, "churned")
        if split:
            X_train, X_test, y_train, y_test = split
            pipe = build_logistic_pipeline()
            if pipe:
                metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
                print(f"Logistic Regression: {metrics}")

                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    print(f"CV: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Regression: predict monthly_charges
        df_reg = df[["tenure", "total_charges", "num_support_calls",
                     "senior_citizen", "has_partner", "has_dependents",
                     "monthly_charges"]].dropna()
        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg
            ridge_pipe = build_ridge_pipeline()
            if ridge_pipe:
                reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
                print(f"Ridge Regression: {reg_metrics}")
            lasso_pipe = build_lasso_pipeline()

            if lasso_pipe:
                lasso_pipe.fit(X_tr, y_tr)
                ridge_pipe.fit(X_tr, y_tr)

                ridge_coef = ridge_pipe.named_steps["model"].coef_
                lasso_coef = lasso_pipe.named_steps["model"].coef_

                feature_names = X_tr.columns

                print("\nFeature Coefficients Comparison:")
                for name, r_coef, l_coef in zip(feature_names, ridge_coef, lasso_coef):
                    print(f"{name:20} | Ridge: {r_coef:.4f} | Lasso: {l_coef:.4f}")    
# In this case, Lasso did not drive any feature coefficients to zero.
# This suggests that all features contribute to predicting monthly charges.
# It may also indicate that the regularization strength (alpha=0.1) is not strong enough
# to eliminate less important features.
# To see more sparsity, we could try increasing alpha or using a different dataset with more irrelevant features. 
    ##print(df.head())            



"""
Summary of Findings

1. Which features appear most important for predicting churn?
Based on the model coefficients, features such as total_charges and tenure appear to have the strongest influence on predicting churn. Other features like number of support calls and customer demographics also contribute, but to a lesser extent.

2. Model Performance:
The logistic regression model achieved moderate performance with an accuracy of around 0.61. However, due to class imbalance in the dataset, accuracy is not the most reliable metric. The recall score is particularly important, as it reflects the model's ability to correctly identify customers who are likely to churn.

3. Key Concern:
Recall is more critical than precision in this problem because failing to identify a customer who will churn (false negative) can result in lost revenue. Therefore, improving recall should be prioritized.

4. Recommendations for Improvement:
- Try different models such as Random Forest or Gradient Boosting
- Tune hyperparameters (e.g., regularization strength)
- Apply feature engineering to create more informative variables
- Use resampling techniques such as SMOTE to address class imbalance
- Evaluate using additional metrics such as ROC-AUC

Overall, the model provides a solid baseline but can be improved with more advanced techniques and better handling of imbalanced data.
"""