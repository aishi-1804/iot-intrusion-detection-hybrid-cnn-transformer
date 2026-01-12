import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ------------------ USER CONFIG (keep artifact paths relative) ------------------
ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "hybrid_cnn_transformer_final.h5")
FEATURE_ORDER_PATH = os.path.join(ARTIFACT_DIR, "feature_order.txt")
SCALER_MEAN_PATH = os.path.join(ARTIFACT_DIR, "scaler_mean.npy")
SCALER_SCALE_PATH = os.path.join(ARTIFACT_DIR, "scaler_scale.npy")
LABEL_CLASSES_PATH = os.path.join(ARTIFACT_DIR, "label_classes.npy")
# ------------------------------------------------------------------------------

st.set_page_config(page_title="Hybrid CNN-Transformer - Inference", layout="wide")
st.title("Hybrid CNN + Transformer — Inference (Form-based)")

@st.cache_resource
def load_model_safe(path):
    try:
        model = tf.keras.models.load_model(path)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_feature_order(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        cols = [line.strip() for line in f if line.strip()]
    return cols

@st.cache_data
def load_scaler(mean_path, scale_path):
    if not os.path.exists(mean_path) or not os.path.exists(scale_path):
        return None, None
    mean = np.load(mean_path)
    scale = np.load(scale_path)
    return mean, scale

@st.cache_data
def load_label_classes(path):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)

def scale_X(X, mean, scale):
    return (X - mean) / (scale + 1e-12)

def predict_batch(model, X_scaled, threshold=0.5):
    preds = model.predict(X_scaled, verbose=0)
    if preds.ndim == 2 and preds.shape[1] > 1:
        prob = preds
        pred_labels = np.argmax(preds, axis=1)
    else:
        prob = preds.flatten()
        pred_labels = (prob >= threshold).astype(int)
    return prob, pred_labels

st.sidebar.header("Model & artifacts")
st.sidebar.write(ARTIFACT_DIR)

model, load_err = load_model_safe(MODEL_PATH)
if model is None:
    st.sidebar.error("Failed to load model with tf.keras.models.load_model().")
    st.sidebar.text("Error:")
    st.sidebar.text(load_err or "unknown error")
    st.warning("If you used custom layers (Lambda), load may fail. Use the notebook to rebuild the model and save in Keras native format (.keras).")
else:
    st.sidebar.success("Model loaded")

feature_order = load_feature_order(FEATURE_ORDER_PATH)
scaler_mean, scaler_scale = load_scaler(SCALER_MEAN_PATH, SCALER_SCALE_PATH)
label_classes = load_label_classes(LABEL_CLASSES_PATH)

if feature_order is None:
    st.error(f"Feature order file not found at {FEATURE_ORDER_PATH}. The app needs this to build the form.")
    st.stop()

if scaler_mean is None or scaler_scale is None:
    st.warning("Scaler mean/scale files not found. Predictions will be made without scaling.")
if label_classes is None:
    st.warning("Label classes file not found. Predictions will show numeric labels.")

threshold_slider = st.sidebar.slider("Decision threshold (for binary)", 0.0, 1.0, 0.5, 0.01)

st.subheader("Input mode")
mode = st.radio("Choose input method:", ("Single sample (form)", "Upload CSV (batch)"))

if mode == "Single sample (form)":
    st.info("Fill numeric values for each feature. For many features, use tab to move faster.")
    with st.form("single_form"):
        input_vals = []
        cols = st.columns(2)
        for i, fname in enumerate(feature_order):
            with cols[i % 2]:
                val = st.number_input(label=fname, key=f"f_{i}", value=0.0, format="%.6f", step=0.01)
                input_vals.append(val)
        submitted = st.form_submit_button("Predict single sample")
    if submitted:
        X = np.array(input_vals, dtype=np.float32).reshape(1, -1)
        if scaler_mean is not None:
            Xs = scale_X(X, scaler_mean, scaler_scale)
        else:
            Xs = X
        prob, pred_label = predict_batch(model, Xs, threshold_slider)
        if prob.ndim == 1:
            p = float(prob[0])
            pred = int(pred_label[0])
            label_name = label_classes[pred] if label_classes is not None else str(pred)
            st.write("### Prediction")
            st.write("Probability (attack class):", f"{p:.6f}")
            st.write("Predicted label:", label_name)
        else:
            st.write("Multiclass probabilities:")
            dfp = pd.DataFrame(prob, columns=(label_classes.tolist() if label_classes is not None else range(prob.shape[1])))
            st.dataframe(dfp.T)

else:
    st.info("Upload a CSV. Columns can be in any order; app will reorder using saved feature order. Missing columns will be filled with zeros.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error("Could not read CSV: " + str(e))
            st.stop()
        st.write("Uploaded rows:", df.shape[0], "columns:", df.shape[1])

        missing = [c for c in feature_order if c not in df.columns]
        if missing:
            st.warning(f"{len(missing)} features missing; filling with zeros. Missing example: {missing[:5]}")
            for c in missing:
                df[c] = 0.0

        X_df = df[feature_order].astype(float)
        X_np = X_df.values.astype(np.float32)
        if scaler_mean is not None:
            Xs = scale_X(X_np, scaler_mean, scaler_scale)
        else:
            Xs = X_np

        st.write("Running predictions...")
        batch_size = 1024
        preds = model.predict(Xs, batch_size=batch_size, verbose=0)
        if preds.ndim == 2 and preds.shape[1] > 1:
            prob = preds
            pred_labels = np.argmax(preds, axis=1)
        else:
            prob = preds.flatten()
            pred_labels = (prob >= threshold_slider).astype(int)

        out_df = df.copy()
        if prob.ndim == 1:
            out_df["attack_prob"] = prob
            out_df["pred_label"] = pred_labels
            if label_classes is not None:
                out_df["pred_label_name"] = [label_classes[i] for i in pred_labels]
        else:
            for i, name in enumerate(label_classes if label_classes is not None else range(preds.shape[1])):
                out_df[f"prob_{name}"] = preds[:, i]
            out_df["pred_label"] = pred_labels
            if label_classes is not None:
                out_df["pred_label_name"] = [label_classes[i] for i in pred_labels]

        st.write("Predictions complete. Example rows:")
        st.dataframe(out_df.head(20))

        to_download = out_df.copy()
        csv_bytes = to_download.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

        if "label" in df.columns or "y" in df.columns or "target" in df.columns:
            gt_col = "label" if "label" in df.columns else ("y" if "y" in df.columns else "target")
            st.write("Ground truth column found:", gt_col)
            try:
                from sklearn.metrics import classification_report, confusion_matrix
                y_true = df[gt_col].astype(int).values
                report = classification_report(y_true, pred_labels, output_dict=True)
                st.write(pd.DataFrame(report).transpose())
                cm = confusion_matrix(y_true, pred_labels)
                st.write("Confusion matrix (counts):")
                st.write(cm)
            except Exception as e:
                st.warning("Could not compute metrics: " + str(e))

st.markdown("---")
st.markdown("**Notes:**")
st.markdown("- This app expects the same feature set used during training. The `feature_order.txt` file defines the expected column names and order.")
st.markdown("- Scaler mean/scale are loaded from saved `.npy` files – if they are missing, the app will still run but predictions may be incorrect.")
st.markdown("- If the model fails to load due to custom layers, rebuild the model in notebook and save in native Keras format (`.keras`) or save weights + include a small loader function in this app.")
st.markdown("**Run:** `streamlit run app.py`")