import streamlit as st
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

# SHAP optional
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Solar XAI", page_icon="☀️", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#FF6B35'>
☀️ INTEGRASI XAI UNTUK SISTEM ENERGI SURYA
</h1>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# DATA GENERATOR
# ---------------------------------------------------------
@st.cache_data
def generate_sample_data(n=1000):
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="15min")
    hours = dates.hour + dates.minute/60
    irradiance = np.maximum(0, 800*np.sin((hours-6)*np.pi/12) + np.random.normal(0,50,n))
    ambient = 25 + 10*np.sin((hours-8)*np.pi/12) + np.random.normal(0,2,n)
    module = ambient + (irradiance/40) + np.random.normal(0,3,n)
    humidity = np.clip(60 - 20*np.sin((hours-8)*np.pi/12) + np.random.normal(0,10,n), 20, 95)
    wind = np.abs(5 + np.sin(hours*np.pi/6)*3 + np.random.normal(0,1.5,n))
    pressure = 1013 + np.random.normal(0,5,n)
    cloud = np.clip(50 + 20*np.sin(np.arange(n)/50) + np.random.normal(0,10,n), 0,100)
    eff = np.clip(18 - 0.04*(module-25) + np.random.normal(0,.5,n), 12,22)

    dc = irradiance * 1.6 * eff/100
    ac = np.maximum(0, dc*0.96 + np.random.normal(0,5,n))

    df = pd.DataFrame({
        "timestamp": dates,
        "irradiance": irradiance,
        "ambient_temperature": ambient,
        "module_temperature": module,
        "humidity": humidity,
        "wind_speed": wind,
        "atmospheric_pressure": pressure,
        "cloud_cover": cloud,
        "panel_efficiency": eff,
        "ac_power": ac
    })
    return df


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Konfigurasi")

    mode = st.radio("Data Source", ["Sample", "Upload CSV"])
    if mode == "Upload CSV":
        f = st.file_uploader("Upload CSV", type=["csv"])
        if f:
            df = pd.read_csv(f)
        else:
            df = generate_sample_data()
    else:
        size = st.slider("Jumlah data", 500, 3000, 1200, 100)
        df = generate_sample_data(size)

    st.divider()

    st.subheader("Model Settings")
    n_estimators = st.slider("Trees", 50, 300, 120, 10)
    max_depth = st.slider("Max Depth", 5, 50, 15, 5)
    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data", "🤖 Model Training", "🔍 XAI (SHAP)", "📈 Prediksi"
])


# ---------------------------------------------------------
# TAB 1 – DATA EXPLORATION
# ---------------------------------------------------------
with tab1:
    st.subheader("📊 Ringkasan Data")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Heatmap Korelasi")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# ---------------------------------------------------------
# TAB 2 – MODEL TRAINING
# ---------------------------------------------------------
with tab2:
    st.subheader("Latih Model Random Forest")

    if st.button("🚀 Train Model"):
        with st.spinner("Training..."):
            features = [
                "irradiance",
                "ambient_temperature",
                "module_temperature",
                "humidity",
                "wind_speed",
                "atmospheric_pressure",
                "cloud_cover",
                "panel_efficiency",
            ]

            X = df[features]
            y = df["ac_power"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=1
            )
            model.fit(X_train_s, y_train)

            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["features"] = features
            st.session_state["X_test_s"] = X_test_s
            st.session_state["y_test"] = y_test

            st.success("Model berhasil dilatih!")

            pred = model.predict(X_test_s)

            col1, col2 = st.columns(2)
            col1.metric("R²", f"{r2_score(y_test, pred):.3f}")
            col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, pred)):.2f}")


# ---------------------------------------------------------
# TAB 3 – SHAP XAI
# ---------------------------------------------------------
with tab3:
    st.subheader("Explainable AI (SHAP)")

    if "model" not in st.session_state:
        st.warning("Latih model dulu.")
    elif not SHAP_AVAILABLE:
        st.error("SHAP tidak tersedia di environment.")
    else:
        if st.button("🔬 Generate SHAP"):
            with st.spinner("Menghitung SHAP..."):
                model = st.session_state["model"]
                X_test_s = st.session_state["X_test_s"]
                features = st.session_state["features"]

                explainer = shap.TreeExplainer(model)
                values = explainer.shap_values(X_test_s)

                st.success("SHAP selesai!")

                # summary plot
                fig = plt.figure(figsize=(9,5))
                shap.summary_plot(values, X_test_s, feature_names=features, show=False)
                st.pyplot(fig)


# ---------------------------------------------------------
# TAB 4 – PREDIKSI MANUAL
# ---------------------------------------------------------
with tab4:
    st.subheader("Prediksi Manual")

    if "model" not in st.session_state:
        st.warning("Latih model dahulu.")
    else:
        cols = st.columns(4)

        irradiance = cols[0].number_input("Irradiance", 0.0, 1200.0, 500.0)
        ambient = cols[0].number_input("Ambient Temp", -10.0, 50.0, 25.0)
        module_t = cols[1].number_input("Module Temp", 0.0, 90.0, 45.0)
        humidity = cols[1].number_input("Humidity", 0.0, 100.0, 50.0)
        wind = cols[2].number_input("Wind Speed", 0.0, 30.0, 5.0)
        pressure = cols[2].number_input("Pressure", 900.0, 1100.0, 1013.0)
        cloud = cols[3].number_input("Cloud Cover", 0.0, 100.0, 30.0)
        eff = cols[3].number_input("Panel Efficiency", 10.0, 25.0, 18.0)

        if st.button("🔮 Predict"):
            model = st.session_state["model"]
            scaler = st.session_state["scaler"]
            features = st.session_state["features"]

            arr = np.array([[irradiance, ambient, module_t, humidity, wind, pressure, cloud, eff]])
            arr_s = scaler.transform(arr)

            pred = model.predict(arr_s)[0]

            st.success(f"Predicted AC Power: **{pred:.2f} W**")
