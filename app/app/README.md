# Streamlit Inference Demo

This folder contains a Streamlit application for running inference using the
trained Hybrid CNN–Transformer intrusion detection model.

## Required Artifacts
Place the following files inside an `artifacts/` directory (not included in repo):

- hybrid_cnn_transformer_final.h5
- feature_order.txt
- scaler_mean.npy
- scaler_scale.npy
- label_classes.npy

## How to Run
```bash
pip install -r requirements.txt
streamlit run app/app.py
