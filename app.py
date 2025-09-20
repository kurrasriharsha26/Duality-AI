import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# -------------------------------
# Custom Theme (Glassmorphism + Gradient)
# -------------------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
[data-testid="stSidebar"] {
    background: rgba(20, 20, 20, 0.8);
    backdrop-filter: blur(10px);
}
.card {
    background: rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("🌌 Duality AI")
menu = st.sidebar.radio("Navigation", ["Home", "Data", "Training", "Predictions", "Explainability", "About"])

# -------------------------------
# Home Page
# -------------------------------
if menu == "Home":
    st.title("🌌 Duality AI – Bridging Simulation & Reality")
    st.markdown("""
    Duality AI explores **Sim-to-Real Transfer Learning**: how models trained in simulation 
    can adapt to noisy, uncertain real-world data.
    """)
    st.markdown("<div class='card'>Hackathon Project | Professional Web UI | Transfer Learning</div>", unsafe_allow_html=True)

# -------------------------------
# Data Generation Page
# -------------------------------
elif menu == "Data":
    st.header("📊 Data Generation")
    samples = st.slider("Number of samples", 100, 2000, 500)
    noise = st.slider("Noise level (Real-world uncertainty)", 0.0, 2.0, 0.5, 0.1)

    X = np.linspace(-5, 5, samples)
    y_sim = np.sin(X) + np.random.normal(0, 0.1, samples)
    y_real = np.sin(X) + np.random.normal(0, noise, samples)

    df = pd.DataFrame({"X": X, "Simulation": y_sim, "Real": y_real})
    fig = px.line(df, x="X", y=["Simulation", "Real"], title="Simulation vs Real Data")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Training Page
# -------------------------------
elif menu == "Training":
    st.header("🤖 Train Models")
    X = np.linspace(-5, 5, 500).reshape(-1, 1)
    y_sim = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])
    y_real = np.sin(X).ravel() + np.random.normal(0, 0.5, X.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y_sim, test_size=0.2)
    sim_model = RandomForestRegressor().fit(X_train, y_train)
    sim_pred = sim_model.predict(X_test)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_real, test_size=0.2)
    real_model = RandomForestRegressor().fit(X_train_r, y_train_r)
    real_pred = real_model.predict(X_test_r)

    sim_acc = r2_score(y_test, sim_pred)
    real_acc = r2_score(y_test_r, real_pred)

    col1, col2 = st.columns(2)
    col1.metric("Simulation Model R²", f"{sim_acc:.2f}")
    col2.metric("Real Model R²", f"{real_acc:.2f}")

    # Error Distribution
    errors = pd.DataFrame({"Simulation Error": y_test - sim_pred, "Real Error": y_test_r - real_pred})
    fig_err = px.histogram(errors, barmode="overlay", title="Error Distribution")
    st.plotly_chart(fig_err, use_container_width=True)

# -------------------------------
# Predictions Page
# -------------------------------
elif menu == "Predictions":
    st.header("⚡ Dual Predictions")
    user_input = st.slider("Enter X value", -5.0, 5.0, 0.0, 0.1)

    sim_out = sim_model.predict([[user_input]])[0]
    real_out = real_model.predict([[user_input]])[0]

    col1, col2 = st.columns(2)
    col1.success(f"Simulation Model Prediction: {sim_out:.3f}")
    col2.info(f"Real Model Prediction: {real_out:.3f}")

# -------------------------------
# Explainability Page
# -------------------------------
elif menu == "Explainability":
    st.header("🔍 Feature Importance & Trust")
    importances_sim = sim_model.feature_importances_
    importances_real = real_model.feature_importances_

    df_imp = pd.DataFrame({
        "Feature": ["X"],
        "Simulation Importance": importances_sim,
        "Real Importance": importances_real
    })
    fig_imp = px.bar(df_imp, x="Feature", y=["Simulation Importance", "Real Importance"], barmode="group", title="Feature Importance Comparison")
    st.plotly_chart(fig_imp, use_container_width=True)

# -------------------------------
# About Page
# -------------------------------
elif menu == "About":
    st.header("ℹ️ About Duality AI")
    st.markdown("""
    Duality AI is built for hackathons to demonstrate:
    - **Sim-to-Real Transfer Learning**
    - **Model Robustness under Noise**
    - **Trust & Explainability**
    - **Interactive Predictions**
    
    Use Cases:
    - 🤖 Robotics
    - 🏥 Healthcare
    - 🌱 Climate Science
    - 🚗 Autonomous Vehicles
    """)
