# IoT Intrusion Detection using Hybrid CNN–Transformer

A deep learning–based Intrusion Detection System (IDS) for IoT networks,
performing a comparative study of CNN, LSTM, Transformer, and a proposed
Hybrid CNN–Transformer architecture using the CICIDS-2017 dataset.

## Motivation
Traditional rule-based intrusion detection systems fail to detect evolving
and zero-day attacks in IoT environments. This project explores deep learning
and hybrid architectures to improve detection accuracy, robustness, and
minority-class performance.

## Models Implemented
- Decision Tree (baseline)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Transformer Encoder
- **Hybrid CNN + Transformer (Proposed)**

## Dataset
- CICIDS-2017 (merged CSV format)
- Preprocessing steps:
  - Handling missing and infinite values
  - Feature normalization (mean–std scaling)
  - SMOTE-based class imbalance correction

> Dataset is not included due to size and licensing constraints.

## Results
| Model | Accuracy | F1 Score |
|------|----------|----------|
| CNN | 87.79% | 90.76% |
| LSTM | 83.97% | 88.06% |
| Transformer | 87.37% | 90.27% |
| **Hybrid CNN–Transformer** | **97.94%** | **97.98%** |

## Architecture Highlights
- CNN layers for spatial feature extraction
- Transformer encoder for contextual dependency modeling
- Balanced batch sampling and Focal Loss
- Streamlit-based real-time inference interface
- SHAP for model interpretability

## Technologies Used
Python, TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Streamlit, SHAP

## Project Report
Full academic report is available in the `docs/` folder.

## Author
Aishi Adhikari  
B.Tech Computer Science & Engineering, VIT Vellore
