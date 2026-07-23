# Vroom Cars — Car Damage Prediction Model

An end-to-end deep learning system for automotive damage image classification, built with PyTorch. The pipeline covers data preprocessing, transfer learning, systematic hyperparameter optimisation, and deployment via both a web GUI and a REST API for real-time predictions.

## Overview

Vroom Cars takes an image of a vehicle and predicts whether — and what type of — damage is present. It's designed as a production-style pipeline rather than a notebook demo: the model is tuned systematically, guarded against overfitting, and served through two different interfaces for different use cases (interactive testing vs. programmatic integration).

## Features

- CNN-based image classification using transfer learning
- Pretrained backbones: **ResNet** and **EfficientNet**
- Regularisation to prevent overfitting: **Dropout**, **Batch Normalisation**, **Early Stopping**
- Automated hyperparameter search with **Optuna** (learning rate, batch size, architecture selection)
- Two deployment surfaces:
  - **Streamlit GUI** for interactive, human-facing predictions
  - **FastAPI REST server** for real-time, programmatic predictions

## Tech Stack

| Component            | Technology              |
|-----------------------|--------------------------|
| Model framework       | PyTorch                  |
| Backbones              | ResNet, EfficientNet     |
| Hyperparameter tuning | Optuna                   |
| Web GUI                | Streamlit                |
| API server              | FastAPI                  |

## Project Structure

```
vroom-cars/
├── data/                  # Training/validation/test image data
├── notebooks/             # Exploration and experimentation notebooks
├── src/
│   ├── dataset.py         # Dataset loading and preprocessing
│   ├── model.py           # Model architecture (ResNet / EfficientNet heads)
│   ├── train.py           # Training loop with early stopping
│   ├── tune.py             # Optuna hyperparameter search
│   └── predict.py          # Inference utilities
├── app/
│   ├── streamlit_app.py    # Streamlit GUI
│   └── api.py               # FastAPI REST server
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <repo-url>
cd vroom-cars
pip install -r requirements.txt
```

## Usage

### Train the model
```bash
python src/train.py --config configs/train_config.yaml
```

### Run hyperparameter search
```bash
python src/tune.py --n-trials 50
```

### Launch the Streamlit GUI
```bash
streamlit run app/streamlit_app.py
```

### Run the FastAPI server
```bash
uvicorn app.api:app --reload
```

Example API request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample_car_image.jpg"
```

## Model Details

- **Approach:** Transfer learning on ResNet and EfficientNet backbones, fine-tuned for the damage classification task.
- **Overfitting controls:** Dropout layers, Batch Normalisation, and Early Stopping based on validation loss.
- **Tuning:** Optuna was used to search over learning rate, batch size, and backbone architecture to select the best-performing configuration.

## Future Improvements

- Add damage severity/localisation (bounding boxes or segmentation) instead of classification only
- Expand dataset coverage across more vehicle types and damage categories
- Add model explainability (e.g. Grad-CAM) to visualise which regions drove a prediction
- Containerise the FastAPI service for easier deployment
