from huggingface_hub import HfApi
import os

repo_id = "balakishan77/Tourism_Package"
repo_type = "space"
print(f"Uploading streamlit application to space '{repo_id}'...")

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="deployment",     # the local folder containing your files
    repo_id=repo_id,                    # the target repo
    repo_type=repo_type,                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)

print(f"Uploading of streamlit application to space '{repo_id}' completed.")


