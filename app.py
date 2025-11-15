# app.py (final, fixed & hardened for Docker/headless)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for headless (Docker)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Import shap defensively (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Page config
st.set_page_config(page_title="Solar Energy XAI", page_icon="☀️", layout="wide")

# CSS
st.markdown(
    """
    <style>
    .main-header { font-size: 2.5rem; color: #FF6B35; text-align: center; margin-bottom: 2rem; }
    .sub-header { font-size: 1.5rem; color: #004E89; margin-top: 2rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #FF6B35; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-header">☀️ INTEGRASI EXPLAINABLE AI (XAI)<br>OPTIMALISASI PERFORMA ENERGI SURYA</p>',
    unsafe_allow_html=True,
)

# -----------------------------
# Data generation (sample)
# -----------------------------
@st.cache_data
def generate_sample_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="15min")
    hours = dates.hour + dates.minute / 60
    days = np.arange(n_samples) / 96  # 96 readings per day

    daily_pattern = np.sin((hours - 6) * np.pi / 12)
    seasonal_variation = 1 + 0.3 * np.sin(days * 2 * np.pi / 365)
    base_irradiance = 800 * daily_pattern * seasonal_variation
    irradiance = np.maximum(0, base_irradiance + np.random.normal(0, 50, n_samples))
    irradiance = np.where((hours < 6) | (hours > 18), 0, irradiance)

    daily_temp_pattern = 20 + 10 * np.sin((hours - 8) * np.pi / 12)
    seasonal_temp = 5 * np.sin(days * 2 * np.pi / 365)
    temperature = daily_temp_pattern + seasonal_temp + np.random.normal(0, 2, n_samples)

    temp_rise = 25 * (irradiance / 1000)
    module_temperature = temperature + temp_rise + np.random.normal(0, 3, n_samples)

    humidity_pattern = 60 - 20 * np.sin((hours - 8) * np.pi / 12)
    humidity = np.clip(humidity_pattern + np.random.normal(0, 10, n_samples), 20, 95)

    wind_speed = np.abs(5 + 3 * np.sin(hours * np.pi / 6) + np.random.normal(0, 1.5, n_samples))
    wind_speed = np.clip(wind_speed, 0, 20)

    pressure = 1013 + np.random.normal(0, 5, n_samples)

    base_clouds = 30 + 25 * np.sin(days * 2 * np.pi / 7)
    cloud_cover = np.clip(base_clouds + np.random.normal(0, 15, n_samples), 0, 100)

    base_efficiency = 18
    temp_coefficient = -0.004
    efficiency = base_efficiency + temp_coefficient * (module_temperature - 25)
    efficiency = np.clip(efficiency + np.random.normal(0, 0.5, n_samples), 12, 22)

    panel_area = 1.6
    dc_power = (irradiance * panel_area * efficiency / 100)
    dc_power *= (1 - cloud_cover / 200)
    dc_power = np.maximum(0, dc_power + np.random.normal(0, 5, n_samples))

    inverter_efficiency = 0.96
    ac_power = dc_power * inverter_efficiency
    ac_power = np.maximum(0, ac_power + np.random.normal(0, 3, n_samples))

    voltage = np.where(irradiance > 10, 30 + 5 * (irradiance / 1000) + np.random.normal(0, 1, n_samples), 0)
    current = np.where(voltage > 0, dc_power / voltage, 0)
    energy_yield = np.cumsum(ac_power / 1000 * 0.25)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "irradiance": irradiance,
            "ambient_temperature": temperature,
            "module_temperature": module_temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "atmospheric_pressure": pressure,
            "cloud_cover": cloud_cover,
            "panel_efficiency": efficiency,
            "dc_voltage": voltage,
            "dc_current": current,
            "dc_power": dc_power,
            "ac_power": ac_power,
            "daily_energy": energy_yield % 10,
        }
    )
    return df


# -----------------------------
# Sidebar: config & data
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=100)
    st.title("⚙️ Konfigurasi")

    data_source = st.radio("Sumber Data", ["Gunakan Data Sampel", "Upload Data CSV"])
    if data_source == "Upload Data CSV":
        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ Data berhasil diupload!")
            except Exception as e:
                st.error(f"Error membaca CSV: {e}")
                df = generate_sample_data()
                st.info("Menggunakan data sampel sementara")
        else:
            df = generate_sample_data()
            st.info("Menggunakan data sampel sementara")
    else:
        n_samples = st.slider("Jumlah Sampel Data", 500, 5000, 1000, 100)
        df = generate_sample_data(n_samples)

    st.markdown("---")
    st.subheader("🎯 Parameter Model RF")
    n_estimators = st.slider("Jumlah Trees", 50, 200, 100, 25)
    max_depth = st.slider("Max Depth", 5, 50, 10, 5)
    test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data & EDA", "🤖 Model Training", "🔍 Explainable AI", "📈 Prediksi"])

# -----------------------------
# TAB 1: Data & EDA
# -----------------------------
with tab1:
    st.markdown('<p class="sub-header">📊 Data Overview</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data Points", len(df))
    with col2:
        st.metric("Jumlah Fitur", len(df.columns) - 2)
    with col3:
        st.metric("Rata-rata AC Power", f"{df['ac_power'].mean():.2f} W")
    with col4:
        st.metric("Total Energy", f"{df['daily_energy'].iloc[-1]:.2f} kWh")

    st.subheader("🔍 Preview Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("📉 Visualisasi Data")
    # Histogram
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    df["ac_power"].hist(bins=50, edgecolor="black", ax=ax1)
    ax1.set_title("Distribusi AC Power Output")
    ax1.set_xlabel("AC Power Output (W)")
    ax1.set_ylabel("Frequency")
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)

    # Correlation heatmap
    features_to_plot = [
        "irradiance",
        "ambient_temperature",
        "module_temperature",
        "humidity",
        "wind_speed",
        "cloud_cover",
        "panel_efficiency",
        "ac_power",
    ]
    corr = df[features_to_plot].corr()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax2, fmt=".2f")
    ax2.set_title("Correlation Matrix")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("⏱️ Time Series - Power Output & Environmental Factors")
    sample_data = df[: 96 * 3]  # 3 days

    fig3, ax3 = plt.subplots(figsize=(14, 5))
    ax3.plot(sample_data["timestamp"], sample_data["dc_power"], linewidth=2, label="DC Power")
    ax3.plot(sample_data["timestamp"], sample_data["ac_power"], linewidth=2, label="AC Power")
    ax3.set_title("DC vs AC Power Output (3 Days)")
    ax3.set_xlabel("Timestamp")
    ax3.set_ylabel("Power (W)")
    ax3.legend()
    ax3.grid(alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")
    st.pyplot(fig3)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(14, 5))
    ax4.plot(sample_data["timestamp"], sample_data["irradiance"], linewidth=2, label="Irradiance")
    ax4.set_xlabel("Timestamp")
    ax4.set_ylabel("Irradiance (W/m²)")
    ax4.grid(alpha=0.3)
    ax4_t = ax4.twinx()
    ax4_t.plot(sample_data["timestamp"], sample_data["module_temperature"], linewidth=2, label="Module Temp", color="tab:red")
    ax4_t.set_ylabel("Module Temperature (°C)")
    ax4_t.set_title("Irradiance & Module Temperature (3 Days)")
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")
    st.pyplot(fig4)
    plt.close(fig4)

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.scatter(df["irradiance"], df["ac_power"], alpha=0.3, s=10)
    ax5.set_xlabel("Irradiance (W/m²)")
    ax5.set_ylabel("AC Power (W)")
    ax5.set_title("Power Output vs Irradiance")
    ax5.grid(alpha=0.3)
    st.pyplot(fig5)
    plt.close(fig5)

    fig6, ax6 = plt.subplots(figsize=(10, 6))
    ax6.scatter(df["module_temperature"], df["panel_efficiency"], alpha=0.3, s=10)
    ax6.set_xlabel("Module Temperature (°C)")
    ax6.set_ylabel("Panel Efficiency (%)")
    ax6.set_title("Efficiency vs Module Temperature")
    ax6.grid(alpha=0.3)
    st.pyplot(fig6)
    plt.close(fig6)

# -----------------------------
# TAB 2: Model Training
# -----------------------------
with tab2:
    st.markdown('<p class="sub-header">🤖 Training Model Random Forest</p>', unsafe_allow_html=True)

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
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
                X = df[features].copy()
                y = df["ac_power"].copy()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=1)
                rf_model.fit(X_train_scaled, y_train)

                y_pred_train = rf_model.predict(X_train_scaled)
                y_pred_test = rf_model.predict(X_test_scaled)

                train_mse = mean_squared_error(y_train, y_pred_train)
                test_mse = mean_squared_error(y_test, y_pred_test)
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)

                st.session_state["model"] = rf_model
                st.session_state["scaler"] = scaler
                st.session_state["X_test"] = X_test
                st.session_state["y_test"] = y_test
                st.session_state["y_pred"] = y_pred_test
                st.session_state["features"] = features
                st.session_state["X_test_scaled"] = X_test_scaled

                st.success("✅ Model berhasil dilatih!")

                st.subheader("📊 Performa Model")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Training Metrics**")
                    st.metric("R² Score", f"{train_r2:.4f}")
                    st.metric("RMSE", f"{np.sqrt(train_mse):.2f}")
                    st.metric("MAE", f"{train_mae:.2f}")
                with col2:
                    st.markdown("**Testing Metrics**")
                    st.metric("R² Score", f"{test_r2:.4f}")
                    st.metric("RMSE", f"{np.sqrt(test_mse):.2f}")
                    st.metric("MAE", f"{test_mae:.2f}")

                st.subheader("🎯 Actual vs Predicted")
                fig7, ax7 = plt.subplots(figsize=(8, 6))
                ax7.scatter(y_test, y_pred_test, alpha=0.5)
                ax7.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
                ax7.set_xlabel("Actual Power Output")
                ax7.set_ylabel("Predicted Power Output")
                st.pyplot(fig7)
                plt.close(fig7)

                fig8, ax8 = plt.subplots(figsize=(8, 6))
                residuals = y_test - y_pred_test
                ax8.scatter(y_pred_test, residuals, alpha=0.5)
                ax8.axhline(y=0, color="red", linestyle="--", lw=2)
                ax8.set_xlabel("Predicted Power Output")
                ax8.set_ylabel("Residuals")
                st.pyplot(fig8)
                plt.close(fig8)

                st.subheader("🎯 Feature Importance")
                feature_importance = pd.DataFrame({"Feature": features, "Importance": rf_model.feature_importances_}).sort_values(
                    "Importance", ascending=False
                )
                fig9, ax9 = plt.subplots(figsize=(10, 6))
                ax9.barh(feature_importance["Feature"], feature_importance["Importance"])
                ax9.set_xlabel("Importance")
                ax9.set_title("Feature Importance - Random Forest")
                ax9.grid(alpha=0.3, axis="x")
                st.pyplot(fig9)
                plt.close(fig9)

            except Exception as e:
                st.error(f"Error saat melatih model: {e}")
                st.exception(e)

# -----------------------------
# TAB 3: Explainable AI (SHAP)
# -----------------------------
with tab3:
    st.markdown('<p class="sub-header">🔍 Explainable AI dengan SHAP</p>', unsafe_allow_html=True)

    if "model" not in st.session_state:
        st.warning("⚠️ Silakan latih model terlebih dahulu di tab 'Model Training'")
    else:
        if not SHAP_AVAILABLE:
            st.warning("SHAP tidak tersedia di lingkungan ini. Install `shap` untuk menggunakan fitur XAI.")
        else:
            if st.button("🔬 Generate SHAP Analysis", type="primary"):
                with st.spinner("Menghitung SHAP values..."):
                    try:
                        model = st.session_state.get("model")
                        X_test_scaled = st.session_state.get("X_test_scaled")
                        features = st.session_state.get("features")
                        if model is None or X_test_scaled is None or features is None:
                            st.error("Tidak ada data uji — latih model terlebih dahulu.")
                        else:
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_test_scaled)

                            st.success("✅ SHAP analysis selesai!")

                            # Summary beeswarm
                            try:
                                fig_sh1 = plt.figure(figsize=(10, 6))
                                shap.summary_plot(shap_values, X_test_scaled, feature_names=features, show=False)
                                st.pyplot(fig_sh1)
                                plt.close(fig_sh1)
                            except Exception:
                                st.write("SHAP summary plot tidak tersedia untuk versi SHAP ini.")

                            # Bar plot
                            try:
                                fig_sh2 = plt.figure(figsize=(10, 6))
                                shap.summary_plot(shap_values, X_test_scaled, feature_names=features, plot_type="bar", show=False)
                                st.pyplot(fig_sh2)
                                plt.close(fig_sh2)
                            except Exception:
                                st.write("SHAP bar plot tidak tersedia untuk versi SHAP ini.")

                            # Individual waterfall
                            try:
                                st.subheader("🔍 Penjelasan Prediksi Individual")
                                max_idx = max(0, X_test_scaled.shape[0] - 1)
                                sample_idx = st.slider("Pilih Sample Index", 0, max_idx, 0)
                                if X_test_scaled.shape[0] > 0:
                                    try:
                                        expl_val = explainer.expected_value
                                        vals = shap_values[sample_idx]
                                        data_row = X_test_scaled[sample_idx]
                                        try:
                                            expl_obj = shap.Explanation(values=vals, base_values=expl_val, data=data_row, feature_names=features)
                                            fig_w = plt.figure(figsize=(10, 6))
                                            shap.waterfall_plot(expl_obj, show=False)
                                            st.pyplot(fig_w)
                                            plt.close(fig_w)
                                        except Exception:
                                            st.write("SHAP waterfall plot tidak tersedia pada versi SHAP ini.")
                                    except Exception as e:
                                        st.write("Error saat membuat SHAP individual:", e)
                            except Exception as e:
                                st.write("Error saat SHAP individual:", e)

# -----------------------------
# TAB 4: Prediction (manual input)
# -----------------------------
with tab4:
    st.markdown('<p class="sub-header">📈 Prediksi Power Output</p>', unsafe_allow_html=True)

    if "model" not in st.session_state:
        st.warning("⚠️ Silakan latih model terlebih dahulu di tab 'Model Training'")
    else:
        st.write("Masukkan parameter cuaca untuk memprediksi power output:")
        col1, col2, col3 = st.columns(3)
        with col1:
            irradiance_input = st.number_input("☀️ Irradiance (W/m²)", 0.0, 1200.0, 500.0, 10.0)
            ambient_temp_input = st.number_input("🌡️ Ambient Temp (°C)", -10.0, 50.0, 25.0, 0.5)
            module_temp_input = st.number_input("🔥 Module Temp (°C)", 0.0, 80.0, 45.0, 0.5)
        with col2:
            humidity_input = st.number_input("💧 Humidity (%)", 0.0, 100.0, 50.0, 1.0)
            wind_input = st.number_input("💨 Wind Speed (m/s)", 0.0, 20.0, 5.0, 0.5)
            pressure_input = st.number_input("🌀 Pressure (hPa)", 950.0, 1050.0, 1013.0, 1.0)
        with col3:
            cloud_input = st.number_input("☁️ Cloud Cover (%)", 0.0, 100.0, 30.0, 1.0)
            efficiency_input = st.number_input("⚡ Panel Efficiency (%)", 10.0, 25.0, 18.0, 0.1)

        if st.button("🔮 Predict Power Output", type="primary"):
            try:
                model = st.session_state.get("model")
                scaler = st.session_state.get("scaler")
                features = st.session_state.get("features")
                if model is None or scaler is None or features is None:
                    st.error("Model atau scaler tidak ditemukan. Latih model terlebih dahulu.")
                else:
                    input_data = np.array(
                        [
                            [
                                irradiance_input,
                                ambient_temp_input,
                                module_temp_input,
                                humidity_input,
                                wind_input,
                                pressure_input,
                                cloud_input,
                                efficiency_input,
                            ]
                        ]
                    )
                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)[0]

                    st.success("✅ Prediksi berhasil!")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Power Output", f"{prediction:.2f} W")
                    with col2:
                        eff = (prediction / irradiance_input * 100) if irradiance_input > 0 else 0
                        st.metric("Estimated Efficiency", f"{eff:.2f}%")
                    with col3:
                        daily_energy = prediction * 24 / 1000
                        st.metric("Est. Daily Energy", f"{daily_energy:.2f} kWh")

                    if SHAP_AVAILABLE:
                        try:
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(input_scaled)
                            try:
                                expl_obj = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=input_scaled[0], feature_names=features)
                                fig_pf = plt.figure(figsize=(10, 4))
                                shap.waterfall_plot(expl_obj, show=False)
                                st.pyplot(fig_pf)
                                plt.close(fig_pf)
                            except Exception:
                                st.write("SHAP waterfall not available for this SHAP version.")
                        except Exception as e:
                            st.write("SHAP explanation failed:", e)
            except Exception as e:
                st.error(f"Error saat prediksi: {e}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🌟 Aplikasi XAI untuk Optimalisasi Energi Surya | Powered by Streamlit & SHAP</p>
    </div>
    """,
    unsafe_allow_html=True,
)
