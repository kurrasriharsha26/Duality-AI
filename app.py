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
st.set_page_config(page_title="Duality AI Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .main { background: linear-gradient(to right, #0f2027, #203a43, #2c5364); color:white; }
    h1,h2,h3,h4,h5,h6 { color: #FFD700; }
    .stMetric { background: #1f2c3e; padding: 10px; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌌 Duality AI Dashboard")
st.caption("Sim-to-Real Transfer Learning Prototype")

# Sidebar Navigation
menu = st.sidebar.radio("📂 Navigate", ["Data", "Training & Accuracy", "Predictions"])

# Data
sim_data = generate_simulation_data()
real_data = generate_real_data()

if menu == "Data":
    st.subheader("🔍 Data Exploration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Simulation Data (Synthetic)**")
        fig1 = px.scatter_3d(sim_data, x="Feature1", y="Feature2", z="Label", color=sim_data["Label"].astype(str),
                             title="3D Simulation Data")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.markdown("**Real Data (Noisy)**")
        fig2 = px.scatter_3d(real_data, x="Feature1", y="Feature2", z="Label", color=real_data["Label"].astype(str),
                             title="3D Real Data")
        st.plotly_chart(fig2, use_container_width=True)

elif menu == "Training & Accuracy":
    st.subheader("📊 Model Training & Performance")
    sim_model, sim_scaler, sim_acc = train_model(sim_data)
    real_model, real_scaler, real_acc = train_model(real_data)

    col3, col4 = st.columns(2)
    with col3:
        st.metric("Simulation Accuracy", f"{sim_acc*100:.2f}%")
    with col4:
        st.metric("Real-World Accuracy", f"{real_acc*100:.2f}%")

    df_compare = pd.DataFrame({
        "Domain": ["Simulation", "Real World"],
        "Accuracy": [sim_acc, real_acc]
    })
    st.markdown("### Accuracy Comparison")
    fig3 = px.bar(df_compare, x="Domain", y="Accuracy", text="Accuracy", color="Domain",
                 color_discrete_map={"Simulation":"#1f77b4", "Real World":"#ff7f0e"})
    st.plotly_chart(fig3, use_container_width=True)

elif menu == "Predictions":
    st.subheader("⚡ Interactive Predictions")
    sim_model, sim_scaler, sim_acc = train_model(sim_data)
    real_model, real_scaler, real_acc = train_model(real_data)

    f1 = st.slider("Feature 1 Value", 0.0, 10.0, 5.0)
    f2 = st.slider("Feature 2 Value", 0.0, 10.0, 5.0)

    sim_pred = sim_model.predict(sim_scaler.transform([[f1, f2]]))[0]
    real_pred = real_model.predict(real_scaler.transform([[f1, f2]]))[0]

    col5, col6 = st.columns(2)
    with col5:
        st.success(f"Simulation Prediction → {'High Risk' if sim_pred==1 else 'Low Risk'}")
    with col6:
        st.warning(f"Real-World Prediction → {'High Risk' if real_pred==1 else 'Low Risk'}")

    st.info("This shows how predictions differ when moving from Simulation to Real-World data ✨")
