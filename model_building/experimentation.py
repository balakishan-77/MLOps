# --------------------------------------------------------------------
# Import necessary modules
# --------------------------------------------------------------------


from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline  # Use imblearn pipeline for SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, make_scorer, recall_score
import numpy as np
import mlflow   

from sklearn.metrics import accuracy_score, classification_report, recall_score
import joblib

# --------------------------------------------------------------------
# Setup mlflow experiment configuration
# --------------------------------------------------------------------
mlflow.set_tracking_uri("http://localhost:5000")  # Set to localhost for local MLflow server
mlflow.set_experiment("Tourism_Package_Experiment")


# --------------------------------------------------------------------
# Create preprocessor, pipeline, and hyperparameter grid
# --------------------------------------------------------------------

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

# Create the recall_scorer
# recall_scorer = make_scorer(recall_score)

# --------------------------------------------------------------------
# Start experimentation with RandomizedSearchCV and log results to MLflow
# --------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Start an MLflow run
with mlflow.start_run():
    rcv_obj = RandomizedSearchCV(pipeline, param_grid, n_iter=200, scoring=f1_scorer, cv=skf, n_jobs=-1, verbose=2)
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


    # Evaluate on validation and testing data
    classification_threshold = 0.50

    y_pred_val_proba = best_model.predict_proba(Xval)[:, 1]
    y_pred_val = (y_pred_val_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    val_report = classification_report(yval, y_pred_val, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the validation and test metrics for the best model
    mlflow.log_metrics({
        "val_accuracy": val_report['accuracy'],
        "val_precision": val_report['1']['precision'],
        "val_recall": val_report['1']['recall'],
        "val_f1-score": val_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    print("Validation Classification Report:")
    print(val_report)
    print("Testing Classification Report:")
    print(test_report)


    # --------------------------------------------------------------------
    # Save the best model as a joblib file and upload to hugging face
    # --------------------------------------------------------------------

    # Save the model locally
    model_path = "best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "balakishan77/Tourism_Package"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_tourism_package_model_v1.joblib",
        path_in_repo="best_tourism_package_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
