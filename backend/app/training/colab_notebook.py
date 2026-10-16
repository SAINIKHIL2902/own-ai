import json
from pathlib import Path
from typing import Any, Dict


def generate_colab_notebook_content(
    version_id: str,
    base_model: str = "llama3.2:1b",
    candidate_tag: str = "llama3.2:1b-behavior-v1",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
) -> Dict[str, Any]:
    """
    Generates a valid Jupyter / Colab Notebook (.ipynb) structure with 1-click execution cells
    for Google Drive integration, GPU training, and versioned model output.
    """
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# 🚀 own-ai Cloud GPU Fine-Tuning: Version `{version_id}`\n",
                f"**Base Model**: `{base_model}` | **Candidate Tag**: `{candidate_tag}`\n\n",
                "This notebook trains your personalized LoRA adapter on Google Colab (Free T4 GPU),\n",
                "reads your unified dataset from Google Drive, and automatically exports the versioned\n",
                "model back to Google Drive for 1-click import into own-ai.\n\n",
                "### Quick Start:\n",
                "1. Select **Runtime -> Change runtime type -> T4 GPU** (or A100).\n",
                "2. Click **Runtime -> Run all** (or run each cell sequentially below).\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 1: Verify NVIDIA GPU Availability\n",
                "!nvidia-smi\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 2: Mount Google Drive (Optional if uploading directly)\n",
                "try:\n",
                "    from google.colab import drive\n",
                "    drive.mount('/content/drive')\n",
                "    print('✅ Google Drive mounted successfully.')\n",
                "except Exception as e:\n",
                "    print('⚠️ Google Drive mount skipped (using direct upload/download).')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 3: Install Training Dependencies\n",
                "!pip install -q torch transformers datasets peft accelerate bitsandbytes\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# Step 4: Locate and Unpack Unified Dataset for Version: {version_id}\n",
                "import os\n",
                "import shutil\n",
                "import zipfile\n",
                "\n",
                f"VERSION_ID = '{version_id}'\n",
                "DRIVE_DATASET_DIR = f'/content/drive/MyDrive/own-ai/datasets/{VERSION_ID}'\n",
                "WORKSPACE_DIR = '/content/training_workspace'\n",
                "os.makedirs(WORKSPACE_DIR, exist_ok=True)\n",
                "\n",
                "# Check for directly uploaded unit zip in /content\n",
                "content_zips = [f for f in os.listdir('/content') if f.startswith('unit_') and f.endswith('.zip')]\n",
                "drive_zip = os.path.join(DRIVE_DATASET_DIR, f'unit_{VERSION_ID}.zip')\n",
                "\n",
                "if content_zips:\n",
                "    uploaded_zip = os.path.join('/content', content_zips[0])\n",
                "    print(f'✅ Found uploaded dataset in Colab: {uploaded_zip}')\n",
                "    with zipfile.ZipFile(uploaded_zip, 'r') as zf:\n",
                "        zf.extractall(WORKSPACE_DIR)\n",
                "elif os.path.exists(drive_zip):\n",
                "    print(f'✅ Extracting dataset from Google Drive: {drive_zip}')\n",
                "    with zipfile.ZipFile(drive_zip, 'r') as zf:\n",
                "        zf.extractall(WORKSPACE_DIR)\n",
                "elif os.path.exists(DRIVE_DATASET_DIR):\n",
                "    print(f'✅ Copying dataset from Google Drive: {DRIVE_DATASET_DIR}')\n",
                "    for f in os.listdir(DRIVE_DATASET_DIR):\n",
                "        src = os.path.join(DRIVE_DATASET_DIR, f)\n",
                "        if os.path.isfile(src):\n",
                "            shutil.copy2(src, os.path.join(WORKSPACE_DIR, f))\n",
                "else:\n",
                "    raise FileNotFoundError(\n",
                "        '⚠️ Dataset zip not found! Please drag and drop unit_v_....zip into the Files tab on the left in Colab.'\n",
                "    )\n",
                "\n",
                "%cd {WORKSPACE_DIR}\n",
                "!ls -la\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 5: Execute Cloud GPU PEFT LoRA Training\n",
                f"!python train_gpu.py --data_dir . --output_dir ./trained_adapter --epochs {epochs} --batch_size {batch_size} --learning_rate {learning_rate}\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 6: Export Trained Model to Google Drive & Create Version Metadata\n",
                "import json\n",
                "from datetime import datetime, timezone\n",
                "\n",
                "OUTPUT_DIR = './trained_adapter'\n",
                "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
                "DRIVE_MODEL_DIR = f'/content/drive/MyDrive/own-ai/models/{VERSION_ID}'\n",
                "os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)\n",
                "\n",
                "metadata = {\n",
                "    'version_id': VERSION_ID,\n",
                f"    'base_model': '{base_model}',\n",
                f"    'candidate_tag': '{candidate_tag}',\n",
                "    'completed_at': datetime.now(timezone.utc).isoformat(),\n",
                "    'status': 'TRAINING_COMPLETE',\n",
                f"    'epochs': {epochs},\n",
                f"    'learning_rate': {learning_rate},\n",
                "}\n",
                "\n",
                "with open(os.path.join(OUTPUT_DIR, 'training_complete.json'), 'w') as f:\n",
                "    json.dump(metadata, f, indent=2)\n",
                "\n",
                "# Copy to Google Drive\n",
                "for f in os.listdir(OUTPUT_DIR):\n",
                "    s = os.path.join(OUTPUT_DIR, f)\n",
                "    d = os.path.join(DRIVE_MODEL_DIR, f)\n",
                "    if os.path.isfile(s):\n",
                "        shutil.copy2(s, d)\n",
                "\n",
                "# Also create a zip archive for 1-click browser download\n",
                "zip_dest = f'/content/model_{VERSION_ID}.zip'\n",
                "shutil.make_archive(f'/content/model_{VERSION_ID}', 'zip', OUTPUT_DIR)\n",
                "shutil.copy2(zip_dest, os.path.join(DRIVE_MODEL_DIR, f'model_{VERSION_ID}.zip'))\n",
                "\n",
                "print(f'✅ Model version {VERSION_ID} successfully saved to Google Drive: {DRIVE_MODEL_DIR}')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 7: (Optional) Direct Browser Download\n",
                "from google.colab import files\n",
                "files.download(f'/content/model_{VERSION_ID}.zip')\n"
            ]
        }
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "toc_visible": True,
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
