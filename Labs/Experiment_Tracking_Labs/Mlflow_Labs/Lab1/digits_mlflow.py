"""
Handwritten Digit Classification with MLflow Tracking

Features:
- Multiple model comparison (KNN vs SVM)
- Hyperparameter tuning with tracking
- Model registry integration
- Artifact logging (confusion matrix plots)
- Experiment organization
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Set experiment
mlflow.set_experiment("digits_classification")


def load_and_prepare_data():
    """Load, scale, and split Digits dataset"""
    logger.info("Loading Digits dataset...")

    digits = load_digits()
    X = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(64)])
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (critical for KNN and SVM)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
    return X_train_scaled, X_test_scaled, y_train, y_test, digits.target_names


def train_baseline_models(X_train, X_test, y_train, y_test):
    """Train baseline models and track with MLflow"""
    logger.info("Training baseline models...")

    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'SVM': SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
    }

    results = {}

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_baseline"):
            # Log tags
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("stage", "baseline")

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Log parameters
            if model_name == 'KNN':
                mlflow.log_param("n_neighbors", model.n_neighbors)
                mlflow.log_param("weights", model.weights)
            elif model_name == 'SVM':
                mlflow.log_param("kernel", model.kernel)
                mlflow.log_param("C", model.C)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)

            # Log confusion matrix as artifact
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=range(10), yticklabels=range(10))
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{model_name} Confusion Matrix')
            plt.tight_layout()
            mlflow.log_figure(fig, f"confusion_matrix_{model_name}.png")
            plt.close()

            # Log classification report as text
            report = classification_report(y_test, y_pred)
            mlflow.log_text(report, f"classification_report_{model_name}.txt")

            # Log model
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, "model", signature=signature)

            results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'run_id': mlflow.active_run().info.run_id
            }

            logger.info(f"{model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")

    return results


def hyperparameter_tuning(X_train, X_test, y_train, y_test):
    """Perform hyperparameter tuning on KNN with MLflow tracking"""
    logger.info("Starting hyperparameter tuning for KNN...")

    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)

    with mlflow.start_run(run_name="KNN_GridSearch"):
        mlflow.set_tag("model_type", "KNN")
        mlflow.set_tag("stage", "hyperparameter_tuning")

        # Fit grid search
        grid_search.fit(X_train, y_train)

        # Log all parameter combinations as nested runs
        cv_results = pd.DataFrame(grid_search.cv_results_)
        for idx, row in cv_results.iterrows():
            with mlflow.start_run(run_name=f"KNN_config_{idx}", nested=True):
                mlflow.log_params(row['params'])
                mlflow.log_metric("mean_cv_score", row['mean_test_score'])
                mlflow.log_metric("std_cv_score", row['std_test_score'])

        # Best model evaluation
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log best parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # Log model with registry
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            best_model,
            "model",
            signature=signature,
            registered_model_name="DigitsClassifier"
        )

        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best model: Accuracy={accuracy:.4f}, F1={f1:.4f}")

        return grid_search.best_estimator_, mlflow.active_run().info.run_id


def register_best_model(model_name, run_id):
    """Register model in MLflow Model Registry and transition to Production"""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    latest_versions = client.get_latest_versions(model_name, stages=["None"])

    if latest_versions:
        version = latest_versions[0].version

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )

        logger.info(f"Model {model_name} version {version} transitioned to Production")
        return version

    return None


def create_comparison_plot(results):
    """Create model comparison plot and log as artifact"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = list(results.keys())
    metrics = ['accuracy', 'f1_score']
    metric_labels = ['Accuracy', 'F1 Score (Weighted)']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[m][metric] for m in models]
        axes[idx].bar(models, values, color=['#1f77b4', '#ff7f0e'])
        axes[idx].set_ylabel(label)
        axes[idx].set_title(f'{label} Comparison')
        axes[idx].set_ylim([0.95, 1.0])

        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.002, f'{v:.4f}', ha='center')

    plt.tight_layout()
    return fig


def main():
    """Main pipeline with MLflow tracking"""
    logger.info("=" * 60)
    logger.info("DIGIT CLASSIFICATION WITH MLFLOW")
    logger.info("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test, target_names = load_and_prepare_data()

    # Train baseline models
    baseline_results = train_baseline_models(X_train, X_test, y_train, y_test)

    # Create comparison plot
    with mlflow.start_run(run_name="Model_Comparison"):
        mlflow.set_tag("type", "comparison")

        comparison_fig = create_comparison_plot(baseline_results)
        mlflow.log_figure(comparison_fig, "model_comparison.png")
        plt.close()

        comparison_text = "\n".join([
            f"{model}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}"
            for model, metrics in baseline_results.items()
        ])
        mlflow.log_text(comparison_text, "model_comparison.txt")

    # Hyperparameter tuning
    best_model, best_run_id = hyperparameter_tuning(X_train, X_test, y_train, y_test)

    # Register best model
    logger.info("Registering best model in Model Registry...")
    version = register_best_model("DigitsClassifier", best_run_id)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("View results: mlflow ui --port 5001")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()