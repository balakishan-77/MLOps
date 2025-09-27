
# Import necessary modules
from huggingface_hub import HfApi
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline  # Use imblearn pipeline for SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
import os
import numpy as np
import mlflow   

from sklearn.metrics import accuracy_score, classification_report, recall_score
import joblib




# Set up MLflow experiment tracking
mlflow.set_tracking_uri("http://localhost:5000")  # Set to localhost for local MLflow server
mlflow.set_experiment("Tourism_Package_Experiment")

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
REPO_PATH = "hf://datasets/balakishan77/Tourism_Package"

Xtrain = pd.read_csv(REPO_PATH + "/Xtrain.csv")
Xval = pd.read_csv(REPO_PATH + "/Xval.csv")
Xtest = pd.read_csv(REPO_PATH + "/Xtest.csv")

ytrain = pd.read_csv(REPO_PATH + "/ytrain.csv")
yval = pd.read_csv(REPO_PATH + "/yval.csv")
ytest = pd.read_csv(REPO_PATH + "/ytest.csv")

print("Splitted Dataset loaded successfully.")

# Variable Segregation
numeric_features  = ["Age","DurationOfPitch","MonthlyIncome"]
categorical_features  = ["TypeofContact", "CityTier","Occupation","Gender","NumberOfPersonVisiting","NumberOfFollowups","ProductPitched","PreferredPropertyStar", "MaritalStatus","NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar","NumberOfChildrenVisiting", "Designation"]

# Create preprocessor for handling different feature types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),  # Standardize numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-hot encode categorical features
    ])

# Create your pipeline using imblearn Pipeline to handle SMOTE
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),    # Preprocessing step
    ('sampler', SMOTE(sampling_strategy=0.8, k_neighbors=4, random_state=42)),  # SMOTE sampler step
    ('classifier', RandomForestClassifier(random_state=42))  # Your classifier
])

# Define hyperparameter grid
param_grid = {"classifier__n_estimators": np.arange(10, 70),
    "classifier__min_samples_leaf": np.arange(5, 50),
    "classifier__max_features": np.arange(0.3, 0.8, 0.1),
    "classifier__max_samples": np.arange(0.3, 0.7, 0.1),
    "classifier__class_weight" : ['balanced', 'balanced_subsample'],
    "classifier__max_depth":np.arange(3,7),
    "classifier__min_impurity_decrease":[0.001, 0.002, 0.003, 0.004, 0.005]
             }

# Create the f1_scorer
f1_scorer = make_scorer(f1_score)


# Hyperparameter tuning with RandomizedSearchCV
# rcv_obj = RandomizedSearchCV(pipeline, param_grid, n_iter=20, scoring=f1_scorer, cv=10, n_jobs=-1, verbose=2)
# rcv_obj = rcv_obj.fit(Xtrain, ytrain)
# print(rcv_obj.best_params_)

with mlflow.start_run():
    rcv_obj = RandomizedSearchCV(pipeline, param_grid, n_iter=20, scoring=f1_scorer, cv=10, n_jobs=-1, verbose=2)
    rcv_obj = rcv_obj.fit(Xtrain, ytrain)
    print(rcv_obj.best_params_)

    # Log all parameter combinations and their mean test scores
    results = rcv_obj.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)

            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)
            print(param_set, mean_score, std_score)

    # Log best parameters separately in main run
    mlflow.log_params(rcv_obj.best_params_)


    # Store and evaluate the best model
    best_model = rcv_obj.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_churn_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")
