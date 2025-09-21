
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
rcv_obj = RandomizedSearchCV(pipeline, param_grid, n_iter=2, scoring=f1_scorer, cv=10, n_jobs=-1, verbose=2)
rcv_obj = rcv_obj.fit(Xtrain, ytrain)
print(rcv_obj.best_params_)
