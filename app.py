# app.py — Streamlit app with RandomForest + LSTM + SHAP (RF & LSTM)
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

# Optional: SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Optional: TensorFlow / Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="Solar Energy XAI (RF + LSTM)", page_icon="☀️", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 1.8rem; color: #FF6B35; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.1rem; color: #004E89; margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">☀️ INTEGRASI EXPLAINABLE AI — RANDOM FOREST & LSTM</p>', unsafe_allow_html=True)

# -------------------------
# Utilities
# -------------------------
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
        "dc_power": dc_power,
        "ac_power": ac_power
    })
    return df

def create_lstm_sequences(X, y, seq_len):
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

# -------------------------
# Sidebar: configs
# -------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=80)
    st.title("⚙️ Konfigurasi")

    source = st.radio("Sumber Data", ["Sample", "Upload CSV"])
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success("✅ CSV loaded")
            except Exception as e:
                st.error(f"Failed loading CSV: {e}")
                df = generate_sample_data()
        else:
            df = generate_sample_data()
    else:
        n_samples = st.slider("Jumlah Sampel", 600, 5000, 1200, 100)
        df = generate_sample_data(n_samples)

    st.markdown("---")
    st.subheader("Pilih Model")
    model_choice = st.selectbox("Model", ["Random Forest", "LSTM"])

    st.markdown("**RF params**")
    rf_n_estimators = st.slider("n_estimators", 50, 500, 100, 50)
    rf_max_depth = st.slider("max_depth", 5, 50, 10, 5)

    st.markdown("**LSTM params**")
    lstm_seq_len = st.slider("Sequence length", 4, 96, 24, 4)
    lstm_epochs = st.slider("Epochs", 1, 50, 10, 1)
    lstm_batch = st.selectbox("Batch size", [8,16,32,64], index=2)

    st.markdown("---")
    st.subheader("Train/Test")
    test_size = st.slider("Test size (%)", 10, 40, 20, 5) / 100

# -------------------------
# Tabs UI
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🤖 Train", "🔍 XAI", "📈 Predict"])

# feature set
FEATURES = ['irradiance','ambient_temperature','module_temperature','humidity','wind_speed','atmospheric_pressure','cloud_cover','panel_efficiency']

# -------------------------
# Tab 1: Data
# -------------------------
with tab1:
    st.subheader("Preview Data")
    st.dataframe(df.head(), use_container_width=True)
    st.subheader("Descriptive Stats")
    st.dataframe(df.describe(), use_container_width=True)

# -------------------------
# Tab 2: Train
# -------------------------
with tab2:
    st.subheader("Train Model")

    if model_choice == "Random Forest":
        st.write("Model: Random Forest")
        if st.button("🚀 Train RF"):
            with st.spinner("Training RF..."):
                try:
                    X = df[FEATURES].copy()
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
                    st.session_state['X_test_s'] = X_test_s
                    st.session_state['y_test'] = y_test
                    st.session_state['features'] = FEATURES

                    st.success("✅ RF trained")
                    st.metric("RMSE", f"{rmse:.2f}")
                    st.metric("R2", f"{r2:.3f}")
                    st.metric("MAE", f"{mae:.2f}")

                except Exception as e:
                    st.error(f"RF training error: {e}")

    else:
        st.write("Model: LSTM")
        if not TF_AVAILABLE:
            st.error("TensorFlow not available. To use LSTM install `tensorflow` or `tensorflow-cpu`.")
        else:
            if st.button("🚀 Train LSTM"):
                with st.spinner("Preparing LSTM data & training..."):
                    try:
                        X = df[FEATURES].values
                        y = df['ac_power'].values
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)

                        X_seq, y_seq = create_lstm_sequences(X_scaled, y, lstm_seq_len)
                        if len(X_seq) < 10:
                            st.error("Not enough sequence samples for chosen sequence length.")
                        else:
                            split_idx = int(len(X_seq) * (1 - test_size))
                            X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
                            y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]

                            n_features = X_seq.shape[2]
                            model = Sequential()
                            model.add(LSTM(64, input_shape=(lstm_seq_len, n_features), return_sequences=False))
                            model.add(Dropout(0.2))
                            model.add(Dense(32, activation='relu'))
                            model.add(Dense(1, activation='linear'))
                            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

                            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                            history = model.fit(
                                X_train_seq, y_train_seq,
                                validation_data=(X_test_seq, y_test_seq),
                                epochs=lstm_epochs,
                                batch_size=lstm_batch,
                                callbacks=[es],
                                verbose=0
                            )

                            y_pred = model.predict(X_test_seq).flatten()
                            rmse, r2, mae = evaluate_regression(y_test_seq, y_pred)

                            st.session_state['model'] = model
                            st.session_state['model_type'] = 'lstm'
                            st.session_state['scaler'] = scaler
                            st.session_state['lstm_seq_len'] = lstm_seq_len
                            st.session_state['X_test_seq'] = X_test_seq
                            st.session_state['y_test_seq'] = y_test_seq
                            st.session_state['features'] = FEATURES

                            st.success("✅ LSTM trained")
                            st.metric("RMSE", f"{rmse:.2f}")
                            st.metric("R2", f"{r2:.3f}")
                            st.metric("MAE", f"{mae:.2f}")

                    except Exception as e:
                        st.error(f"LSTM training error: {e}")

# -------------------------
# Tab 3: XAI (SHAP)
# -------------------------
with tab3:
    st.subheader("Explainable AI (SHAP)")

    if 'model' not in st.session_state:
        st.warning("Train a model first in the Train tab.")
    else:
        mtype = st.session_state.get('model_type', 'rf')
        st.write(f"Explaining model: **{mtype.upper()}**")

        if not SHAP_AVAILABLE:
            st.error("SHAP not installed in environment. Install `shap` to use XAI features.")
        else:
            if mtype == 'rf':
                if st.button("🔬 SHAP for RF"):
                    with st.spinner("Computing SHAP values for RF..."):
                        try:
                            rf = st.session_state['model']
                            X_test_s = st.session_state['X_test_s']
                            feat_names = st.session_state.get('features', None)
                            explainer = shap.TreeExplainer(rf)
                            shap_values = explainer.shap_values(X_test_s)

                            # Summary plot (beeswarm)
                            fig = plt.figure(figsize=(10,6))
                            shap.summary_plot(shap_values, X_test_s, feature_names=feat_names, show=False)
                            st.pyplot(fig)
                            plt.close(fig)

                            # Feature importance bar (mean(|shap|))
                            avg_imp = np.mean(np.abs(shap_values), axis=0)
                            imp_df = pd.DataFrame({'feature': feat_names, 'importance': avg_imp}).sort_values('importance', ascending=True)
                            fig2, ax2 = plt.subplots(figsize=(8,6))
                            ax2.barh(imp_df['feature'], imp_df['importance'])
                            ax2.set_title("SHAP mean(|value|) - RF")
                            st.pyplot(fig2)
                            plt.close(fig2)

                        except Exception as e:
                            st.error(f"SHAP RF error: {e}")

            else:  # LSTM
                if not TF_AVAILABLE:
                    st.error("TensorFlow not available — cannot compute SHAP for LSTM.")
                else:
                    if st.button("🔬 SHAP for LSTM"):
                        with st.spinner("Computing SHAP values for LSTM (DeepExplainer)..."):
                            try:
                                model = st.session_state['model']
                                X_test_seq = st.session_state['X_test_seq']
                                feat_names = st.session_state.get('features', [f"f{i}" for i in range(X_test_seq.shape[2])])
                                if X_test_seq is None or len(X_test_seq) == 0:
                                    st.error("No LSTM test sequences found. Retrain with more data or shorter seq length.")
                                else:
                                    # choose small background (use first N train samples or random)
                                    background = X_test_seq[:min(50, len(X_test_seq))]
                                    try:
                                        explainer = shap.DeepExplainer(model, background)
                                        sample = X_test_seq[:min(200, len(X_test_seq))]
                                        shap_values = explainer.shap_values(sample)
                                        # shap_values might be list or array
                                        if isinstance(shap_values, list):
                                            arr = np.array(shap_values[0])
                                        else:
                                            arr = np.array(shap_values)
                                        # arr shape: (n_samples, seq_len, n_features)
                                        # compute mean absolute importance across samples and timesteps
                                        abs_arr = np.abs(arr)
                                        mean_over_samples = np.mean(abs_arr, axis=0)  # (seq_len, n_features)
                                        mean_feat_imp = np.mean(mean_over_samples, axis=0)  # (n_features,)

                                        # Bar plot: importance per feature (averaged over timesteps)
                                        imp_df = pd.DataFrame({'feature': feat_names, 'importance': mean_feat_imp}).sort_values('importance', ascending=True)
                                        fig1, ax1 = plt.subplots(figsize=(8,6))
                                        ax1.barh(imp_df['feature'], imp_df['importance'])
                                        ax1.set_title("Approx. SHAP importance (LSTM, averaged over timesteps)")
                                        st.pyplot(fig1)
                                        plt.close(fig1)

                                        # Heatmap: importance per timestep x feature (average over samples)
                                        fig2, ax2 = plt.subplots(figsize=(10,6))
                                        sns.heatmap(mean_over_samples.T, cmap='viridis', xticklabels=False, yticklabels=feat_names, ax=ax2)
                                        ax2.set_xlabel("Timestep (sequence index)")
                                        ax2.set_ylabel("Feature")
                                        ax2.set_title("SHAP abs mean (feature × timestep) — LSTM")
                                        st.pyplot(fig2)
                                        plt.close(fig2)

                                    except Exception as e:
                                        st.error(f"DeepExplainer failed: {e}. DeepExplainer requires TF model and compatible SHAP version. You may try KernelExplainer (very slow) as fallback.")
                            except Exception as e:
                                st.error(f"SHAP LSTM error: {e}")

# -------------------------
# Tab 4: Predict
# -------------------------
with tab4:
    st.subheader("Manual Prediction / Forecasting")

    if 'model' not in st.session_state:
        st.warning("Train a model first in the Train tab.")
    else:
        mtype = st.session_state.get('model_type', 'rf')
        cols = st.columns(3)
        with cols[0]:
            irr = st.number_input("Irradiance", 0.0, 1200.0, 500.0, 10.0)
            amb = st.number_input("Ambient Temp", -10.0, 50.0, 25.0, 0.5)
            modt = st.number_input("Module Temp", 0.0, 90.0, 45.0, 0.5)
        with cols[1]:
            hum = st.number_input("Humidity", 0.0, 100.0, 50.0, 1.0)
            wind = st.number_input("Wind Speed", 0.0, 30.0, 5.0, 0.5)
            pres = st.number_input("Pressure", 900.0, 1100.0, 1013.0, 1.0)
        with cols[2]:
            cloud = st.number_input("Cloud Cover", 0.0, 100.0, 30.0, 1.0)
            eff = st.number_input("Panel Efficiency", 10.0, 25.0, 18.0, 0.1)

        if mtype == 'rf':
            if st.button("Predict (RF)"):
                try:
                    model = st.session_state['model']
                    scaler = st.session_state['scaler']
                    X_in = np.array([[irr, amb, modt, hum, wind, pres, cloud, eff]])
                    Xs = scaler.transform(X_in)
                    pred = model.predict(Xs)[0]
                    st.success(f"Predicted AC Power (RF): {pred:.2f} W")
                except Exception as e:
                    st.error(f"RF predict error: {e}")
        else:
            if st.button("Predict (LSTM)"):
                if not TF_AVAILABLE:
                    st.error("TensorFlow not available — cannot predict with LSTM.")
                else:
                    try:
                        model = st.session_state['model']
                        scaler = st.session_state['scaler']
                        seq_len = st.session_state.get('lstm_seq_len', lstm_seq_len)
                        X_in = np.array([[irr, amb, modt, hum, wind, pres, cloud, eff]])
                        Xs = scaler.transform(X_in)  # (1, n_features)
                        # Build a sequence input: use last available test sequence if present, otherwise repeat current row
                        if 'X_test_seq' in st.session_state and len(st.session_state['X_test_seq'])>0:
                            last_seq = st.session_state['X_test_seq'][-1]  # shape (seq_len, n_features)
                            # replace last row in the sequence with current input (shift)
                            seq = np.copy(last_seq)
                            seq = np.vstack([seq[1:], Xs[0]])
                            seq = seq.reshape(1, seq.shape[0], seq.shape[1])
                        else:
                            # fallback: repeat current input
                            seq = np.tile(Xs.reshape(1,1,-1), (1, seq_len, 1))
                        pred = model.predict(seq).flatten()[0]
                        st.success(f"Predicted AC Power (LSTM): {pred:.2f} W")
                    except Exception as e:
                        st.error(f"LSTM predict error: {e}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#666'>🌟 Solar Energy XAI — Random Forest & LSTM</div>", unsafe_allow_html=True)
