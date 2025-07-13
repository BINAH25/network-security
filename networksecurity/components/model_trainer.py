import os
import sys
import mlflow
import dagshub

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
    write_yaml_file,
    upload_file_to_s3
)
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize Dagshub for MLflow tracking
dagshub.init(repo_owner='visteen192', repo_name='network-security', mlflow=True)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        """
        Initialize model trainer with config and transformed data artifact.
        """
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classificationmetric):
        """
        Logs F1, precision, recall scores to MLflow.
        """
        with mlflow.start_run():
            mlflow.log_metric("f1_score", classificationmetric.f1_score)
            mlflow.log_metric("precision", classificationmetric.precision_score)
            mlflow.log_metric("recall_score", classificationmetric.recall_score)
            mlflow.sklearn.log_model(best_model, "model")

    def train_model(self, X_train, y_train, x_test, y_test):
        """
        Compares multiple models, selects the best, saves and uploads it.
        """
        # Define models
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }

        # Define hyperparameters for tuning
        params = {
            "Decision Tree": {'criterion': ['gini', 'entropy', 'log_loss']},
            "Random Forest": {'n_estimators': [8, 16, 32, 128, 256]},
            "Gradient Boosting": {
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {},
            "AdaBoost": {
                'learning_rate': [.1, .01, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

        # Evaluate models and get report
        model_report = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            models=models,
            param=params
        )

        # 📄 Save the report locally
        report_save_path = os.path.join("report", "model_comparison_report.yaml")
        os.makedirs(os.path.dirname(report_save_path), exist_ok=True)
        write_yaml_file(file_path=report_save_path, content=model_report)
        logging.info(f"📄 Model comparison report saved to: {report_save_path}")

        # ✅ Upload report to S3
        upload_file_to_s3(report_save_path, "report/model_comparison_report.yaml", "REPORT")

        # Select best model
        best_model_score = max(model_report.values())
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name]

        # Train and evaluate on train data
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        self.track_mlflow(best_model, classification_train_metric)

        # Evaluate on test data
        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        self.track_mlflow(best_model, classification_test_metric)

        # Load preprocessor
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        # Save the final model (with preprocessor pipeline)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        final_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=final_model)
        save_object("final_model/model.pkl", best_model)

        # Upload model to S3
        upload_file_to_s3(self.model_trainer_config.trained_model_file_path, "model/final_model.pkl", "MODEL")

        # Return metadata/artifact
        return ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Kicks off model training pipeline and returns the artifact.
        """
        try:
            # Load transformed training/test arrays
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # Split features/labels
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # Train and return artifact
            return self.train_model(x_train, y_train, x_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
