import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Generate Synthetic (Simulated) Data
# -------------------------------
def generate_simulation_data(n=800):
    np.random.seed(42)
    X = np.random.rand(n, 2) * 10
    y = (X[:, 0]**2 + X[:, 1] > 40).astype(int)
    return pd.DataFrame({"Feature1": X[:, 0], "Feature2": X[:, 1], "Label": y})

# -------------------------------
# Generate Real (Noisy) Data
# -------------------------------
def generate_real_data(n=300):
    np.random.seed(99)
    X = np.random.rand(n, 2) * 10
    y = (X[:, 0]**2 + X[:, 1] + np.random.normal(0, 3, n) > 40).astype(int)
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
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    return model, scaler, acc

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Duality AI Website", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    body { background: linear-gradient(to right, #141E30, #243B55); color: white; }
    h1,h2,h3,h4,h5,h6 { color: #FFD700; }
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Navigation Menu
# -------------------------------
menu = st.sidebar.radio("🌐 Navigate", ["Home", "Data", "Training", "Predictions", "About"])

# -------------------------------
# Home Page
# -------------------------------
if menu == "Home":
    st.markdown("""
    <div style='text-align:center;'>
        <h1>🌌 Duality AI</h1>
        <h3>Bridging Simulation & Reality with AI</h3>
        <p>Hackathon Project – Demonstrating Sim-to-Real Transfer Learning</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='glass-card'><h3>🚀 Simulation</h3><p>Train on synthetic data efficiently.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'><h3>🌍 Real World</h3><p>Adapt models to noisy real-world conditions.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='glass-card'><h3>⚡ Transfer</h3><p>Bridge the gap seamlessly between both domains.</p></div>", unsafe_allow_html=True)

# -------------------------------
# Data Page
# -------------------------------
elif menu == "Data":
    st.header("🔍 Data Visualization")
    sim_data = generate_simulation_data()
    real_data = generate_real_data()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Simulation Data")
        fig1 = px.scatter(sim_data, x="Feature1", y="Feature2", color=sim_data["Label"].astype(str))
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.subheader("Real-World Data")
        fig2 = px.scatter(real_data, x="Feature1", y="Feature2", color=real_data["Label"].astype(str))
        st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Training Page
# -------------------------------
elif menu == "Training":
    st.header("📊 Model Training Results")
    sim_data = generate_simulation_data()
    real_data = generate_real_data()
    sim_model, sim_scaler, sim_acc = train_model(sim_data)
    real_model, real_scaler, real_acc = train_model(real_data)

    df_compare = pd.DataFrame({
        "Domain": ["Simulation", "Real World"],
        "Accuracy": [sim_acc, real_acc]
    })

    st.metric("Simulation Accuracy", f"{sim_acc*100:.2f}%")
    st.metric("Real-World Accuracy", f"{real_acc*100:.2f}%")

    fig3 = px.bar(df_compare, x="Domain", y="Accuracy", color="Domain", text="Accuracy")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# Predictions Page
# -------------------------------
elif menu == "Predictions":
    st.header("⚡ Try Predictions")
    sim_data = generate_simulation_data()
    real_data = generate_real_data()
    sim_model, sim_scaler, sim_acc = train_model(sim_data)
    real_model, real_scaler, real_acc = train_model(real_data)

    f1 = st.slider("Feature 1", 0.0, 10.0, 5.0)
    f2 = st.slider("Feature 2", 0.0, 10.0, 5.0)

    sim_pred = sim_model.predict(sim_scaler.transform([[f1, f2]]))[0]
    real_pred = real_model.predict(real_scaler.transform([[f1, f2]]))[0]

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"Simulation Model → {'High Risk' if sim_pred==1 else 'Low Risk'}")
    with col2:
        st.warning(f"Real-World Model → {'High Risk' if real_pred==1 else 'Low Risk'}")

# -------------------------------
# About Page
# -------------------------------
elif menu == "About":
    st.header("ℹ️ About Duality AI")
    st.markdown("""
    **Duality AI** is a hackathon project showcasing how AI can transfer knowledge between **simulation** environments and **real-world** applications. 

    - 📊 Demonstrates sim-to-real transfer learning.
    - 🌍 Applies to robotics, healthcare, climate, and smart cities.
    - 🚀 Built with **Streamlit, Scikit-learn, Plotly**.

    <br>
    <b>Team Vision:</b> Making AI adaptive, robust, and trustworthy across dual domains.
    """, unsafe_allow_html=True)
