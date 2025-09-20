import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# -------------------------------
# Generate Synthetic (Simulated) Data
# -------------------------------
def generate_simulation_data(n=1000):
    np.random.seed(42)
    X = np.random.rand(n, 2) * 10
    y = (X[:, 0] + X[:, 1] > 10).astype(int)
    return pd.DataFrame({"Feature1": X[:, 0], "Feature2": X[:, 1], "Label": y})

# -------------------------------
# Generate Real (Noisy) Data
# -------------------------------
def generate_real_data(n=300):
    np.random.seed(99)
    X = np.random.rand(n, 2) * 10
    y = (X[:, 0] + X[:, 1] + np.random.normal(0, 2, n) > 10).astype(int)
    return pd.DataFrame({"Feature1": X[:, 0], "Feature2": X[:, 1], "Label": y})

# -------------------------------
# Train Model Function
# -------------------------------
def train_model(df):
    X = df[["Feature1", "Feature2"]]
    y = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    return model, scaler, acc, y_test, y_pred

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Duality AI - Sim2Real", layout="wide")

st.title("🌐 Duality AI: Bridging Simulation and Real World")
st.markdown("### A Hackathon Prototype for Sim-to-Real Transfer Learning")

# Generate Data
sim_data = generate_simulation_data()
real_data = generate_real_data()

col1, col2 = st.columns(2)
with col1:
    st.subheader("🔵 Simulated Data Distribution")
    fig1 = px.scatter(sim_data, x="Feature1", y="Feature2", color=sim_data["Label"].astype(str),
                     title="Simulation Environment")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("🟠 Real-World Data Distribution")
    fig2 = px.scatter(real_data, x="Feature1", y="Feature2", color=real_data["Label"].astype(str),
                     title="Real Environment")
    st.plotly_chart(fig2, use_container_width=True)

# Train Models
sim_model, sim_scaler, sim_acc, y_test_sim, y_pred_sim = train_model(sim_data)
real_model, real_scaler, real_acc, y_test_real, y_pred_real = train_model(real_data)

st.markdown("---")
st.subheader("📊 Model Performance")

col3, col4 = st.columns(2)
with col3:
    st.metric("Simulation Accuracy", f"{sim_acc*100:.2f}%")
    cm_sim = confusion_matrix(y_test_sim, y_pred_sim)
    fig, ax = plt.subplots()
    sns.heatmap(cm_sim, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Simulation Confusion Matrix")
    st.pyplot(fig)

with col4:
    st.metric("Real-World Accuracy", f"{real_acc*100:.2f}%")
    cm_real = confusion_matrix(y_test_real, y_pred_real)
    fig, ax = plt.subplots()
    sns.heatmap(cm_real, annot=True, fmt="d", cmap="Oranges", ax=ax)
    ax.set_title("Real-World Confusion Matrix")
    st.pyplot(fig)

st.markdown("---")
st.subheader("⚡ Try a Prediction (Sim-to-Real)")

f1 = st.slider("Feature 1 Value", 0.0, 10.0, 5.0)
f2 = st.slider("Feature 2 Value", 0.0, 10.0, 5.0)

# Predict using both models
sim_pred = sim_model.predict(sim_scaler.transform([[f1, f2]]))[0]
real_pred = real_model.predict(real_scaler.transform([[f1, f2]]))[0]

col5, col6 = st.columns(2)
with col5:
    st.success(f"Simulation Model Prediction → {'High Risk' if sim_pred==1 else 'Low Risk'}")
with col6:
    st.warning(f"Real-World Model Prediction → {'High Risk' if real_pred==1 else 'Low Risk'}")

st.markdown("---")
st.info("✅ This prototype shows how models trained in a **simulation** can differ from **real-world environments**, and how Duality AI can help bridge the gap!")
