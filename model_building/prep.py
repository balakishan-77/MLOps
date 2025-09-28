# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi



# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/balakishan77/Tourism_Package/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ------------------------------------------------------------------------------------
# Data Exploration and Sanity Checks   
# ------------------------------------------------------------------------------------

# Display the first few rows of the dataset
print(df.head())
print(df.info())
print(df.describe().T)

# Check for missing values

# sanity check - null data
print('Checking for null data : ')
print(df.isnull().sum())
print()

# sanity check - duplicate data
print('Checking for duplicate data : ')
print(df.duplicated().sum())
print()


# ------------------------------------------------------------------------------------
# Variable Segregation
# ------------------------------------------------------------------------------------
# segregate continous / numeric variables
continousVariables  = ["Age","DurationOfPitch","MonthlyIncome"]
print("Continous Variables: ", len(continousVariables), continousVariables)

# segregate categorical variables
categoryVariables  = ["ProdTaken", "TypeofContact", "CityTier","Occupation","Gender","NumberOfPersonVisiting","NumberOfFollowups","ProductPitched","PreferredPropertyStar", "MaritalStatus","NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar","NumberOfChildrenVisiting", "Designation"]
print("Categorical Variables: ", len(categoryVariables), categoryVariables)

idVariables = ["CustomerID", "Unnamed: 0"]


# ------------------------------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------------------------------

# 1. Convert continuous variables to integers
# ----------------------------------------------------------------------------
print("=" * 50)
print(f"Number of continuous variables: {len(continousVariables)}")
print("Continuous Variables to convert:")
print(continousVariables)
print("=" * 50)

print("Converting continuous variables to integers...")

converted_count = 0

for var in continousVariables:
    if var in df.columns:
        original_dtype = df[var].dtype

        # # Check if variable has any NaN values
        # nan_count = df[var].isnull().sum()
        # if nan_count > 0:
        #     print(f"Warning: {var} has {nan_count} NaN values - filling with median")
        #     df[var] = df[var].fillna(df[var].median())

        # Check if variable has decimal values
        if df[var].dtype in ['float64', 'float32']:
            has_decimals = any(df[var].dropna() % 1 != 0)

            if not has_decimals:
                # Convert to integer if no decimal values
                df[var] = df[var].astype('int64')
                converted_count += 1
                print(f"✓ {var}: {original_dtype} → {df[var].dtype}")
            else:
                # Round to nearest integer if has decimals
                df[var] = df[var].round().astype('int64')
                converted_count += 1
                print(f"✓ {var}: {original_dtype} → {df[var].dtype} (rounded)")
        elif df[var].dtype in ['int32', 'int8', 'int16']:
            # Convert other integer types to int64 for consistency
            df[var] = df[var].astype('int64')
            converted_count += 1
            print(f"✓ {var}: {original_dtype} → {df[var].dtype}")
        else:
            print(f"→ {var}: {original_dtype} (already integer or no conversion needed)")

print(f"\nConversion completed! {converted_count} variables converted to int64.")

# ----------------------------------------------------------------------------
# 2. Convert categorical variables to 'category' data type
# ----------------------------------------------------------------------------
print("=" * 50)
print("Categorical Variables to convert:")
print(f"Number of categorical variables: {len(categoryVariables)}")
print("Variables:", categoryVariables)
print("=" * 50)

# Display current data types for these columns
print("\nCurrent data types for categorical variables:")
print("=" * 50)
for var in categoryVariables:
    if var in df.columns:
        print(f"{var}: {df[var].dtype}")
    else:
        print(f"{var}: Column not found in dataframe")

# Convert categorical variables to categorical data type
print("\nConverting categorical variables to 'category' data type:")
print("=" * 50)

converted_categorical = []
for var in categoryVariables:
    if var in df.columns:
        original_dtype = df[var].dtype
        df[var] = df[var].astype('category')
        converted_categorical.append(var)
        print(f"✓ Converted {var}: {original_dtype} → category")
    else:
        print(f"✗ Skipped {var}: Column not found in dataframe")

print(f"\nSuccessfully converted {len(converted_categorical)} columns to categorical data type.")

# Verify the conversion
print("\nVerification - Updated data types for categorical variables:")
print("=" * 50)
for var in categoryVariables:
    if var in df.columns:
        print(f"{var}: {df[var].dtype}")




# Show memory optimization benefit
print("\nMemory usage optimization:")
print("=" * 50)
memory_info = df.memory_usage(deep=True)
total_memory = memory_info.sum()
print(f"Current total memory usage: {total_memory:,} bytes ({total_memory / (1024**2):.2f} MB)")

# Store original shape before removing ID columns
original_shape = df.shape
print(f"Original dataframe shape: {original_shape}")
print(f"Columns before removal: {list(df.columns)}")

# ----------------------------------------------------------------------------
# 3. Remove ID variables from dataframe
# ----------------------------------------------------------------------------
print("=" * 50)
print(f"\nRemoving ID variables: {idVariables}")
print("=" * 50)

# Drop ID variables
df_cleaned = df.drop(columns=idVariables)

# Display new shape and remaining columns
new_shape = df_cleaned.shape
print(f"\nDataframe shape after removing ID columns: {new_shape}")
print(f"Columns removed: {original_shape[1] - new_shape[1]}")
print(f"Remaining columns: {new_shape[1]}")

# Verify ID columns are removed
print(f"\nVerification - ID columns removed:")
for var in idVariables:
    if var in df_cleaned.columns:
        print(f"  ❌ {var}: Still present")
    else:
        print(f"  ✅ {var}: Successfully removed")

# Update the main dataframe
df = df_cleaned.copy()
print(f"\nUpdated main dataframe shape: {df.shape}")
print("First few column names:", df.columns[:10].tolist())

# ----------------------------------------------------------------------------
# 4. Replace spaces with underscores in categorical column values
# ----------------------------------------------------------------------------
print("=" * 70)
print("Replacing spaces with underscores in categorical column values (Optimized)...")
print("=" * 70)

# Get categorical columns directly from dataframe
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Found {len(categorical_cols)} categorical columns: {categorical_cols}")

# Function to replace spaces with underscores
def replace_spaces_with_underscores(value):
    """Replace spaces with underscores in string values"""
    if isinstance(value, str):
        return value.replace(' ', '_')
    return value

# Track changes
total_changes = 0
processed_columns = []

# Process categorical columns using apply function
for col in categorical_cols:
    print(f"\nProcessing column: {col}")

    # Store original unique values
    original_values = df[col].unique()

    # Apply space replacement using vectorized operations
    df[col] = df[col].apply(replace_spaces_with_underscores)

    # Check what changed
    new_values = df[col].unique()

    # Find values that were modified
    changes_in_col = 0
    if not set(original_values) == set(new_values):
        changes_in_col = len([v for v in original_values if isinstance(v, str) and ' ' in v])

    if changes_in_col > 0:
        print(f"  ✅ Modified {changes_in_col} unique values")
        print(f"  Before: {original_values[:5].tolist()}")
        print(f"  After:  {new_values[:5].tolist()}")
        total_changes += changes_in_col
    else:
        print(f"  → No changes needed")

    processed_columns.append(col)

print(f"\n" + "=" * 70)
print(f"Optimization completed!")
print(f"Columns processed: {len(processed_columns)}")
print(f"Total unique values modified: {total_changes}")
print(f"Processed columns: {processed_columns}")

# Display final sample
if categorical_cols:
    print(f"\nFinal categorical column samples:")
    for col in categorical_cols[:3]:  # Show first 3 columns
        unique_vals = df[col].unique()
        print(f"  {col}: {unique_vals[:5].tolist()}")

print("=" * 70)

# ----------------------------------------------------------------------------
# 5. Fix specific Gender column value: Replace 'Fe_Male' with 'Female'
# ----------------------------------------------------------------------------
print("=" * 60)
print("Fixing specific Gender column value...")
print("=" * 60)

if 'Gender' in df.columns:
    # Check current unique values in Gender column
    gender_values_before = df['Gender'].unique()
    print(f"Gender values before fix: {gender_values_before.tolist()}")

    # Check if 'Fe_Male' exists
    if 'Fe_Male' in gender_values_before:
        # Count occurrences before replacement
        female_count = sum(df['Gender'] == 'Fe_Male')
        print(f"Found {female_count} occurrences of 'Fe_Male'")

        # Replace 'Fe_Male' with 'Female'
        df['Gender'] = df['Gender'].str.replace('Fe_Male', 'Female', regex=False)

        # Verify replacement
        gender_values_after = df['Gender'].unique()
        print(f"Gender values after fix: {gender_values_after.tolist()}")
        print(f"✅ Successfully replaced 'Fe_Male' with 'Female' ({female_count} records updated)")
    else:
        print("→ 'Fe_Male' not found in Gender column")
else:
    print("❌ Gender column not found in dataframe")

# ----------------------------------------------------------------------------
# Print summary for categorical variables
# ----------------------------------------------------------------------------

print('Checking for distribution of categorical variables after fixing sugar content label')
print()

for i in categoryVariables :
  print(i)
  print('--------------------------------')
  print(df[i].value_counts(normalize=True))
  print()

# ----------------------------------------------------------------------------
# Print summary for continous / numerical variables
# ----------------------------------------------------------------------------

# Get stats to another dataframe
df_stat_summary = df.describe().T

# Add median and mode
for i in df_stat_summary.index:
  df_stat_summary.loc[i, 'median'] = df[i].median()
  df_stat_summary.loc[i, 'mode'] = df[i].mode()[0]

# Make the stats readable
for col in df_stat_summary.columns:
  if col == 'count' :
    df_stat_summary[col] = df_stat_summary[col].astype(int)
  else :
    # df_stat_summary[col] = (df_stat_summary[col]).apply(lambda x : f"{x:.1f}")
    df_stat_summary[col] = (df_stat_summary[col]).apply(lambda x : round(x, 1))

# exclude order id, customer id from stats
df_stat_summary[:]  


# ----------------------------------------------------------------------------
# Create datasets
# ----------------------------------------------------------------------------
print(categoryVariables)

print(continousVariables)

target = 'ProdTaken'
if target in categoryVariables:
    categoryVariables.remove(target)

categorical_features = categoryVariables
numeric_features = continousVariables

# create X dataset having independent variables
X = df[numeric_features + categorical_features]

# create Y dataset having dependent / target variable
y = df[target]

#Split X, y datasets into train, val, test

Xtrain, Xtemp, ytrain, ytemp = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

Xval, Xtest, yval, ytest = train_test_split(
    Xtemp, ytemp,
    test_size=0.3,
    random_state=0,
    stratify=ytemp
)

print('Shape of splitted datasets : ')
print(Xtrain.shape, Xval.shape, Xtest.shape)

print('Target distribution in splitted datasets : ')
print(ytrain.value_counts(normalize=True))
print(yval.value_counts(normalize=True))
print(ytest.value_counts(normalize=True))


# Save datasets to CSV files
os.makedirs("data", exist_ok=True)
Xtrain.to_csv("data/Xtrain.csv", index=False)
Xval.to_csv("data/Xval.csv", index=False)
Xtest.to_csv("data/Xtest.csv", index=False)
ytrain.to_csv("data/ytrain.csv", index=False)
yval.to_csv("data/yval.csv", index=False)
ytest.to_csv("data/ytest.csv", index=False)


# ------------------------------------------------------------------------------------
# Upload train, test, val datasets to HF dataset space
# ------------------------------------------------------------------------------------

files = ["Xtrain.csv","Xval.csv","Xtest.csv","ytrain.csv","yval.csv","ytest.csv"]

repo_id = "balakishan77/Tourism_Package"
repo_type = "dataset"
print(f"Uploading splitted dataset to space '{repo_id}'...")

for file_path in files:
    print(f"Uploading file '{file_path}'...")
    api.upload_file(
        path_or_fileobj="data/"+file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=repo_id,
        repo_type=repo_type,
    )

print(f"Uploading of splitted dataset to space '{repo_id}' completed.")  


