import os
import gdown
import zipfile

# Dictionary of weights: {relative_path: google_drive_id}
weights = {
    "checkpoints/alias_final.pth": "18q4lS7cNt1_X8ewCgya1fq0dSk93jTL6",
    "checkpoints/gmm_final.pth": "1uDRPY8gh9sHb3UDonq6ZrINqDOd7pmTz",
    "checkpoints/seg_final.pth": "1d7lZNLh51Qt5Mi1lXqyi6Asb2ncLrEdC",
    "checkpoints/cloth_segm_u2net_latest.pth": "1ysEoAJNxou7RNuT9iKOxRhjVRNY5RLjx",
    "Self-Correction-Human-Parsing/checkpoints/final.pth": "1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH"
}

# The OpenPose models are hosted separately as a zip
OPENPOSE_MODELS_ZIP_ID = "1I4UWTXN5RjtmcmUy6oKUzdUAfD1haNYC"

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "Self-Correction-Human-Parsing", "checkpoints"), exist_ok=True)

    for relative_path, file_id in weights.items():
        output_path = os.path.join(base_dir, relative_path)
        if not os.path.exists(output_path):
            print(f"Downloading {relative_path}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, output_path, quiet=False)
            except Exception as e:
                print(f"Failed to download {relative_path}: {e}")
        else:
            print(f"Already exists: {relative_path}")

    # Handle OpenPose models
    openpose_models_dir = os.path.join(base_dir, "openpose", "models")
    body_25_model = os.path.join(openpose_models_dir, "pose", "body_25", "pose_iter_584000.caffemodel")
    
    if os.path.exists(openpose_models_dir) and not os.path.exists(body_25_model):
        print("Downloading OpenPose models from Google Drive...")
        zip_path = os.path.join(base_dir, "openpose_models.zip")
        url = f'https://drive.google.com/uc?id={OPENPOSE_MODELS_ZIP_ID}'
        try:
            gdown.download(url, zip_path, quiet=False)
            print("Extracting OpenPose models...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # The zip contains a 'models/' folder at its root, so extract to openpose directory directly
                zip_ref.extractall(os.path.join(base_dir, "openpose"))
            os.remove(zip_path)
            print("OpenPose models extracted successfully!")
        except Exception as e:
            print(f"Failed to download or extract OpenPose models: {e}")
    else:
        if os.path.exists(body_25_model):
            print("OpenPose models already exist.")

if __name__ == "__main__":
    main()
