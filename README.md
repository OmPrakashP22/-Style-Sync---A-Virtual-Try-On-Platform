# Virtual Try-On (Native Windows Local Setup)

This directory contains a complete native Windows port of the Virtual Try-On pipeline. It completely removes the need for Google Colab, Linux-only shell script commands, and ngrok dependencies. The client-side Flask frontend now communicates directly with the PyTorch AI pipeline seamlessly.

## Prerequisites

1. **Python 3.8+** (We highly advise using a virtual environment!)
2. **NVIDIA GPU** with CUDA installed (This project handles extremely heavy Deep Learning tasks).
   * CPU-only execution is possible but will be slow.
3. **OpenPose Windows Portable Release** (Required for Human Pose Estimation)

## Setup Instructions

### 1. Create a Python Virtual Environment
Open your command prompt or PowerShell inside the `virtual-try-on` directory:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Python Dependencies
Install the required packages.
```powershell
pip install -r requirements.txt
```
*Note: Make sure to install the PyTorch version that corresponds to your system's CUDA version via the official PyTorch website instructions if `requirements.txt` fails to detect your GPU.*

### 3. Download AI Model Weights
We have provided an automated script to download the huge PyTorch models directly to their correct folders.
```powershell
python download_weights.py
```
This script will download ~2GB of`.pth` files including the `ALIAS`, `GMM`, `Segmentation`, and `SCHP` networks.

### 4. Setup OpenPose
Because building OpenPose from source on Windows locally is prone to C++ build errors, we've designed this port to use the Portable Binary.
1. Download [OpenPose Portable for Windows](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases) (Look for `openpose-...-binaries-win64...zip`).
2. Extract the downloaded zip file.
3. Rename the extracted folder to exactly **`openpose`** and place it perfectly inside this `virtual-try-on` directory.
   - The path `virtual-try-on/openpose/bin/OpenPoseDemo.exe` MUST exist for the inference to work.
4. Run `openpose/models/getModels.bat` (if applicable) inside the OpenPose folder to fetch its internal body pose estimation models.

### 5. Setup SCHP (Self-Correction-Human-Parsing)
Clone the SCHP repository required for Human Parsing parsing inside this directory:
```powershell
git clone https://github.com/PeikeLi/Self-Correction-Human-Parsing.git
```
*(The weights downloaded in Step 3 automatically placed themselves inside the SCHP folder).*

## Run the Application!
Everything is configured. Simply launch the Flask server:
```powershell
python app.py
```
Go to `http://127.0.0.1:5000` in your web browser, upload a model and cloth image, and wait for the results!
