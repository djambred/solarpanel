# app.py — Streamlit app with RandomForest + LSTM + XAI support
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# optional: SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# optional: TensorFlow / Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Page config
st.set_page_config(page_title="Solar Energy XAI (RF + LSTM)", page_icon="☀️", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.0rem; color: #FF6B35; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.1rem; color: #004E89; margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">☀️ INTEGRASI EXPLAINABLE AI (XAI) — RANDOM FOREST & LSTM</p>', unsafe_allow_html=True)

# ---------------------------
# Utilities
# ---------------------------
@st.cache_data
def generate_sample_data(n_samples=1200, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="15min")
    hours = dates.hour + dates.minute/60
    days = np.arange(n_samples) / 96
    daily_pattern = np.sin((hours - 6) * np.pi / 12)
    seasonal_variation = 1 + 0.3 * np.sin(days * 2 * np.pi / 365)
    base_irradiance = 800 * daily_pattern * seasonal_variation
    irradiance = np.maximum(0, base_irradiance + np.random.normal(0, 50, n_samples))
    irradiance = np.where((hours < 6) | (hours > 18), 0, irradiance)
    ambient = 20 + 10 * np.sin((hours - 8) * np.pi / 12) + np.random.normal(0, 2, n_samples)
    module = ambient + 25 * (irradiance / 1000) + np.random.normal(0, 3, n_samples)
    humidity = np.clip(60 - 20 * np.sin((hours - 8) * np.pi / 12) + np.random.normal(0, 10, n_samples), 20, 95)
    wind = np.abs(5 + 3 * np.sin(hours * np.pi / 6) + np.random.normal(0, 1.5, n_samples))
    pressure = 1013 + np.random.normal(0, 5, n_samples)
    cloud = np.clip(30 + 25 * np.sin(days * 2 * np.pi / 7) + np.random.normal(0, 15, n_samples), 0, 100)
    efficiency = np.clip(18 - 0.004 * (module - 25) + np.random.normal(0, 0.5, n_samples), 12, 22)
    dc_power = (irradiance * 1.6 * efficiency / 100) * (1 - cloud / 200)
    dc_power = np.maximum(0, dc_power + np.random.normal(0, 5, n_samples))
    ac_power = np.maximum(0, dc_power * 0.96 + np.random.normal(0, 3, n_samples))
    voltage = np.where(irradiance > 10, 30 + 5 * (irradiance / 1000) + np.random.normal(0, 1, n_samples), 0)
    current = np.where(voltage > 0, dc_power / voltage, 0)
    energy_yield = np.cumsum(ac_power / 1000 * 0.25)
    df = pd.DataFrame({
        "timestamp": dates,
        "irradiance": irradiance,
        "ambient_temperature": ambient,
        "module_temperature": module,
        "humidity": humidity,
        "wind_speed": wind,
        "atmospheric_pressure": pressure,
        "cloud_cover": cloud,
        "panel_efficiency": efficiency,
        "dc_voltage": voltage,
        "dc_current": current,
        "dc_power": dc_power,
        "ac_power": ac_power,
        "daily_energy": energy_yield % 10
    })
    return df

def create_lstm_sequences(X, y, seq_len):
    """
    X: numpy array (n_samples, n_features)
    y: numpy array (n_samples,)
    Returns: X_seq (n_seq, seq_len, n_features), y_seq (n_seq,)
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, r2, mae

# ---------------------------
# Sidebar: data + model config
# ---------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=80)
    st.title("⚙️ Konfigurasi")

    data_source = st.radio("Sumber Data", ["Gunakan Data Sampel", "Upload CSV"])
    if data_source == "Upload Data CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success("✅ Data uploaded")
            except Exception as e:
                st.error(f"Error membaca CSV: {e}")
                df = generate_sample_data()
                st.info("Menggunakan data sampel")
        else:
            df = generate_sample_data()
            st.info("Menggunakan data sampel")
    else:
        n_samples = st.slider("Jumlah Sampel", 600, 5000, 1200, 100)
        df = generate_sample_data(n_samples)

    st.markdown("---")
    st.subheader("Pilih Model")
    model_choice = st.selectbox("Model", ["Random Forest", "LSTM"])

    # RF params
    st.markdown("**Parameter Random Forest**")
    rf_n_estimators = st.slider("n_estimators", 50, 500, 100, 50)
    rf_max_depth = st.slider("max_depth", 5, 50, 10, 5)

    # LSTM params
    st.markdown("**Parameter LSTM**")
    lstm_seq_len = st.slider("Sequence length (timesteps)", 4, 96, 24, 4)
    lstm_epochs = st.slider("Epochs", 1, 50, 10, 1)
    lstm_batch = st.selectbox("Batch size", [8, 16, 32, 64], index=2)

    st.markdown("---")
    st.subheader("Train/Test")
    test_size = st.slider("Test size (%)", 10, 40, 20, 5) / 100

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data & EDA", "🤖 Train", "🔍 XAI", "📈 Predict"])

# ---------------------------
# TAB 1: Data & EDA
# ---------------------------
with tab1:
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.write("Kolom:", df.columns.tolist())
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("AC Power distribution")
    fig, ax = plt.subplots(figsize=(8,4))
    df['ac_power'].hist(bins=50, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------
# TAB 2: Train
# ---------------------------
with tab2:
    st.subheader("Train Model")

    features = ['irradiance', 'ambient_temperature', 'module_temperature',
                'humidity', 'wind_speed', 'atmospheric_pressure',
                'cloud_cover', 'panel_efficiency']

    if model_choice == "Random Forest":
        st.write("Model: Random Forest")
        if st.button("🚀 Train Random Forest"):
            with st.spinner("Training Random Forest..."):
                try:
                    X = df[features].copy()
                    y = df['ac_power'].copy()
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42, n_jobs=1)
                    rf.fit(X_train_s, y_train)

                    y_pred = rf.predict(X_test_s)
                    rmse, r2, mae = evaluate_regression(y_test, y_pred)

                    st.session_state['model'] = rf
                    st.session_state['model_type'] = 'rf'
                    st.session_state['scaler'] = scaler
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    st.session_state['X_test_s'] = X_test_s

                    st.success("✅ Random Forest trained")
                    st.metric("RMSE", f"{rmse:.2f}")
                    st.metric("R2", f"{r2:.3f}")
                    st.metric("MAE", f"{mae:.2f}")

                except Exception as e:
                    st.error(f"Training error: {e}")

    else:  # LSTM
        st.write("Model: LSTM")
        if not TF_AVAILABLE:
            st.error("TensorFlow not available in this environment. Install `tensorflow` to use LSTM.")
        else:
            if st.button("🚀 Train LSTM"):
                with st.spinner("Preparing data & training LSTM..."):
                    try:
                        # prepare supervised sequences
                        X = df[features].copy().values
                        y = df['ac_power'].values
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)

                        X_seq, y_seq = create_lstm_sequences(X_scaled, y, lstm_seq_len)
                        # split sequences (train/test)
                        split_idx = int(len(X_seq) * (1 - test_size))
                        X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
                        y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]

                        # build simple LSTM model
                        n_features = X_seq.shape[2]
                        model = Sequential()
                        model.add(LSTM(64, input_shape=(lstm_seq_len, n_features), return_sequences=False))
                        model.add(Dropout(0.2))
                        model.add(Dense(32, activation='relu'))
                        model.add(Dense(1, activation='linear'))
                        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

                        # callbacks
                        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                        history = model.fit(
                            X_train_seq, y_train_seq,
                            validation_data=(X_test_seq, y_test_seq),
                            epochs=lstm_epochs,
                            batch_size=lstm_batch,
                            callbacks=[es],
                            verbose=0
                        )

                        # eval
                        y_pred = model.predict(X_test_seq).flatten()
                        rmse, r2, mae = evaluate_regression(y_test_seq, y_pred)

                        # store
                        st.session_state['model'] = model
                        st.session_state['model_type'] = 'lstm'
                        st.session_state['scaler'] = scaler
                        st.session_state['lstm_seq_len'] = lstm_seq_len
                        st.session_state['X_test_seq'] = X_test_seq
                        st.session_state['y_test_seq'] = y_test_seq
                        st.session_state['features'] = features

                        st.success("✅ LSTM trained")
                        st.metric("RMSE", f"{rmse:.2f}")
                        st.metric("R2", f"{r2:.3f}")
                        st.metric("MAE", f"{mae:.2f}")

                    except Exception as e:
                        st.error(f"LSTM training error: {e}")

# ---------------------------
# TAB 3: XAI (SHAP)
# ---------------------------
with tab3:
    st.subheader("Explainable AI (SHAP)")

    if 'model' not in st.session_state:
        st.warning("Latih model terlebih dahulu (tab Train).")
    else:
        mtype = st.session_state.get('model_type', 'rf')
        st.write(f"Using model: {mtype.upper()}")

        if not SHAP_AVAILABLE:
            st.error("SHAP not installed in environment. Install `shap` to enable XAI.")
        else:
            if mtype == 'rf':
                if st.button("🔬 Generate SHAP (RF)"):
                    with st.spinner("Computing SHAP for RF..."):
                        try:
                            rf = st.session_state['model']
                            X_test = st.session_state['X_test']
                            X_test_s = st.session_state['X_test_s']
                            explainer = shap.TreeExplainer(rf)
                            shap_values = explainer.shap_values(X_test_s)

                            # summary plot
                            fig = plt.figure(figsize=(10,5))
                            shap.summary_plot(shap_values, X_test_s, feature_names=st.session_state['X_test'].columns if hasattr(st.session_state['X_test'], 'columns') else None, show=False)
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"SHAP RF error: {e}")
            else:  # LSTM
                if not TF_AVAILABLE:
                    st.error("TensorFlow not available — cannot use SHAP for LSTM here.")
                else:
                    if st.button("🔬 Generate SHAP (LSTM)"):
                        with st.spinner("Computing SHAP for LSTM (DeepExplainer preferred)..."):
                            try:
                                model = st.session_state['model']
                                X_test_seq = st.session_state['X_test_seq']
                                # note: DeepExplainer expects a background dataset; use small sample from training or first N samples
                                background = X_test_seq[:min(50, len(X_test_seq))]
                                try:
                                    explainer = shap.DeepExplainer(model, background)
                                    shap_values = explainer.shap_values(X_test_seq[:min(200, len(X_test_seq))])
                                    # shap_values shape for DeepExplainer may be [ (n_examples, timesteps, features) ]
                                    # we'll summarize over timesteps for plotting: mean abs across timesteps
                                    vals = np.array(shap_values)
                                    # vals might be list; try convert robustly
                                    if isinstance(shap_values, list):
                                        arr = np.array(shap_values[0])
                                    else:
                                        arr = np.array(shap_values)
                                    # collapse timesteps
                                    arr_mean = np.mean(np.abs(arr), axis=1)  # (n_examples, n_features)
                                    mean_feat_imp = np.mean(arr_mean, axis=0)
                                    feat_names = st.session_state.get('features', [f"f{i}" for i in range(arr_mean.shape[1])])
                                    imp_df = pd.DataFrame({"feature": feat_names, "importance": mean_feat_imp})
                                    imp_df = imp_df.sort_values("importance", ascending=True)

                                    fig, ax = plt.subplots(figsize=(8,6))
                                    ax.barh(imp_df['feature'], imp_df['importance'])
                                    ax.set_title("Approx. SHAP feature importance (LSTM — mean over timesteps)")
                                    st.pyplot(fig)
                                    plt.close(fig)
                                except Exception as e:
                                    st.error(f"SHAP DeepExplainer failed: {e}. You can try KernelExplainer but it is slow.")
                            except Exception as e:
                                st.error(f"SHAP LSTM error: {e}")

# ---------------------------
# TAB 4: Predict
# ---------------------------
with tab4:
    st.subheader("Manual Prediction / Forecast")

    if 'model' not in st.session_state:
        st.warning("Latih model dahulu di tab Train.")
    else:
        mtype = st.session_state.get('model_type', 'rf')
        features = ['irradiance', 'ambient_temperature', 'module_temperature',
                    'humidity', 'wind_speed', 'atmospheric_pressure',
                    'cloud_cover', 'panel_efficiency']

        cols = st.columns(3)
        with cols[0]:
            irr = st.number_input("Irradiance (W/m²)", 0.0, 1200.0, 500.0, 10.0)
            amb = st.number_input("Ambient Temp (°C)", -10.0, 50.0, 25.0, 0.5)
            modt = st.number_input("Module Temp (°C)", 0.0, 80.0, 45.0, 0.5)
        with cols[1]:
            hum = st.number_input("Humidity (%)", 0.0, 100.0, 50.0, 1.0)
            wind = st.number_input("Wind Speed (m/s)", 0.0, 20.0, 5.0, 0.5)
            pres = st.number_input("Pressure (hPa)", 950.0, 1050.0, 1013.0, 1.0)
        with cols[2]:
            cloud = st.number_input("Cloud Cover (%)", 0.0, 100.0, 30.0, 1.0)
            eff = st.number_input("Panel Efficiency (%)", 10.0, 25.0, 18.0, 0.1)

        if mtype == 'rf':
            if st.button("Predict (RF)"):
                try:
                    model = st.session_state['model']
                    scaler = st.session_state['scaler']
                    X_input = np.array([[irr, amb, modt, hum, wind, pres, cloud, eff]])
                    Xs = scaler.transform(X_input)
                    pred = model.predict(Xs)[0]
                    st.success(f"Predicted AC Power (RF): {pred:.2f} W")
                except Exception as e:
                    st.error(f"Prediction error (RF): {e}")
        else:
            if st.button("Predict (LSTM)"):
                if not TF_AVAILABLE:
                    st.error("TensorFlow not available — cannot predict with LSTM.")
                else:
                    try:
                        model = st.session_state['model']
                        scaler = st.session_state['scaler']
                        seq_len = st.session_state.get('lstm_seq_len', lstm_seq_len)
                        # Build input sequence by repeating current input (best-effort) or using last available sequence if exists
                        X_input_base = np.array([[irr, amb, modt, hum, wind, pres, cloud, eff]])
                        X_scaled = scaler.transform(X_input_base)  # shape (1, n_feat)
                        # create seq by repeating last row
                        X_seq = np.repeat(X_scaled[np.newaxis, :, :], seq_len, axis=1)  # shape (1, seq_len, n_feat)
                        # but model expects (n_seq, seq_len, n_feat)
                        # create proper shape
                        X_seq = X_scaled.reshape(1,1,-1)
                        X_seq = np.tile(X_seq, (1, seq_len, 1))
                        pred = model.predict(X_seq).flatten()[0]
                        st.success(f"Predicted AC Power (LSTM): {pred:.2f} W")
                    except Exception as e:
                        st.error(f"Prediction error (LSTM): {e}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#666'>🌟 Solar Energy XAI — RandomForest & LSTM</div>", unsafe_allow_html=True)
