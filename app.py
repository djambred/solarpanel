# app.py — Solar Energy XAI (RF + LSTM) with lazy SHAP import and robust fallbacks
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

# TensorFlow (optional for LSTM)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# NOTE: do NOT import shap at module load time. We'll import lazily when user requests SHAP.
SHAP_AVAILABLE = None  # None = unknown; True/False after dynamic import attempt

# --------------------
# Helpers
# --------------------
def make_arrow_safe(df_in):
    """Make DataFrame safe for pyarrow/Streamlit: convert datelike -> datetime, numeric-like -> numeric, else -> str"""
    if df_in is None:
        return df_in
    df = df_in.copy()
    for col in df.columns:
        try:
            series = df[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                df[col] = pd.to_datetime(series, errors="coerce")
                continue

            if series.dtype == object:
                sample = series.dropna().astype(str).head(20)
                if not sample.empty and all(any(ch in s for ch in [":", "-", "T"]) for s in sample):
                    parsed = pd.to_datetime(series, errors="coerce")
                    if parsed.notna().sum() / max(1, len(parsed)) > 0.5:
                        df[col] = parsed
                        continue

            if pd.api.types.is_numeric_dtype(series):
                df[col] = pd.to_numeric(series, errors="coerce")
                continue

            coerced_num = pd.to_numeric(series, errors="coerce")
            if coerced_num.notna().sum() > len(series) * 0.5:
                df[col] = coerced_num
                continue

            df[col] = series.astype(str)
        except Exception:
            df[col] = df[col].astype(str)
    return df

@st.cache_data
def generate_sample_data(n_samples=1200, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="15min")
    hours = dates.hour + dates.minute / 60
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
    df['timestamp'] = pd.to_datetime(df['timestamp'])
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
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    return rmse, r2, mae, mape

# --------------------
# Page config & UI
# --------------------
st.set_page_config(page_title="Solar Energy XAI (RF + LSTM)", page_icon="☀️", layout="wide")
st.markdown("""
    <style>
    .main-header { font-size: 1.8rem; color: #FF6B35; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.1rem; color: #004E89; margin-top: 1rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    </style>
""", unsafe_allow_html=True)
st.markdown('<p class="main-header">☀️ INTEGRASI EXPLAINABLE AI — RANDOM FOREST & LSTM</p>', unsafe_allow_html=True)

# --------------------
# Sidebar: data + hyperparams
# --------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=80)
    st.title("⚙️ Konfigurasi")

    source = st.radio("Sumber Data", ["Sample", "Upload CSV"])
    if source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                tmp = pd.read_csv(uploaded_file, nrows=0)
                parse_dates = ['timestamp'] if 'timestamp' in tmp.columns else None
                df = pd.read_csv(uploaded_file, parse_dates=parse_dates)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                st.success("✅ CSV loaded")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                df = generate_sample_data()
                st.info("Using sample data instead")
        else:
            df = generate_sample_data()
            st.info("Using sample data")
    else:
        n_samples = st.slider("Jumlah Sampel", 600, 5000, 1200, 100)
        df = generate_sample_data(n_samples)

    st.markdown("---")
    st.subheader("Hyperparameters")

    st.markdown("**Random Forest**")
    rf_n_estimators = st.slider("n_estimators", 50, 500, 100, 50)
    rf_max_depth = st.slider("max_depth", 5, 50, 10, 5)

    st.markdown("**LSTM**")
    lstm_seq_len = st.slider("Sequence length", 4, 96, 24, 4)
    lstm_epochs = st.slider("Epochs", 5, 100, 20, 5)
    lstm_batch = st.selectbox("Batch size", [8,16,32,64], index=2)

    st.markdown("---")
    st.subheader("Train/Test Split")
    test_size = st.slider("Test size (%)", 10, 40, 20, 5) / 100

# defensive timestamp normalization
if 'timestamp' in df.columns:
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    except Exception:
        pass

FEATURES = ['irradiance','ambient_temperature','module_temperature','humidity','wind_speed','atmospheric_pressure','cloud_cover','panel_efficiency']

# session state flags
if 'rf_trained' not in st.session_state:
    st.session_state['rf_trained'] = False
if 'lstm_trained' not in st.session_state:
    st.session_state['lstm_trained'] = False

# --------------------
# Tabs
# --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data & EDA", "🤖 Train Models", "🔍 XAI (SHAP)", "📈 Predictions", "⚖️ Model Comparison"])

# ---------- Tab 1 ----------
with tab1:
    st.subheader("📋 Data Preview")
    try:
        safe = make_arrow_safe(df.head(20))
        st.dataframe(safe, width='stretch')
    except Exception:
        st.write(df.head(20).astype(str))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Descriptive Statistics")
        try:
            stats = make_arrow_safe(df.describe())
            st.dataframe(stats, width='stretch')
        except Exception:
            st.write(df.describe().astype(str))
    with col2:
        st.subheader("🔢 Data Info")
        st.write(f"**Total Rows:** {len(df)}")
        st.write(f"**Features:** {len(FEATURES)}")
        st.write(f"**Target:** ac_power")
        st.write(f"**Missing Values (total):** {df.isnull().sum().sum()}")

    st.subheader("📈 Visualizations")
    v1, v2 = st.columns(2)
    with v1:
        fig1, ax1 = plt.subplots(figsize=(8,5))
        df['ac_power'].hist(bins=50, ax=ax1)
        ax1.set_xlabel("AC Power (W)")
        ax1.set_title("AC Power Distribution")
        st.pyplot(fig1)
        plt.close(fig1)
    with v2:
        fig2, ax2 = plt.subplots(figsize=(8,5))
        df['irradiance'].plot(ax=ax2, linewidth=0.8)
        ax2.set_xlabel("Index")
        ax2.set_title("Irradiance Over Time (index)")
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("🔥 Correlation Heatmap")
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[num_cols].corr()
        fig3, ax3 = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax3)
        ax3.set_title("Feature Correlation")
        st.pyplot(fig3)
        plt.close(fig3)
    except Exception:
        st.write("Correlation plot failed (maybe not enough numeric columns).")

# ---------- Tab 2: Train ----------
with tab2:
    st.subheader("🤖 Train Models")
    left, right = st.columns(2)

    # Random Forest
    with left:
        st.markdown("### 🌲 Random Forest")
        st.write(f"Estimators: {rf_n_estimators}, Max depth: {rf_max_depth}, Test size: {test_size*100:.0f}%")
        if st.button("🚀 Train Random Forest", use_container_width=True):
            with st.spinner("Training Random Forest..."):
                try:
                    X = df[FEATURES].copy()
                    y = df['ac_power'].copy()
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42, n_jobs=1)
                    rf.fit(X_train_s, y_train)

                    y_pred_train = rf.predict(X_train_s)
                    y_pred_test = rf.predict(X_test_s)
                    rmse_train, r2_train, mae_train, mape_train = evaluate_regression(y_train, y_pred_train)
                    rmse_test, r2_test, mae_test, mape_test = evaluate_regression(y_test, y_pred_test)

                    st.session_state['rf_model'] = rf
                    st.session_state['rf_scaler'] = scaler
                    st.session_state['rf_X_train'] = X_train
                    st.session_state['rf_X_test'] = X_test
                    st.session_state['rf_X_test_s'] = X_test_s
                    st.session_state['rf_y_train'] = y_train
                    st.session_state['rf_y_test'] = y_test
                    st.session_state['rf_y_pred'] = y_pred_test
                    st.session_state['rf_metrics'] = {
                        'rmse_train': rmse_train, 'r2_train': r2_train, 'mae_train': mae_train, 'mape_train': mape_train,
                        'rmse_test': rmse_test, 'r2_test': r2_test, 'mae_test': mae_test, 'mape_test': mape_test
                    }
                    st.session_state['rf_trained'] = True

                    st.success("✅ Random Forest trained")
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("RMSE (test)", f"{rmse_test:.2f}")
                    c2.metric("R² (test)", f"{r2_test:.3f}")
                    c3.metric("MAE (test)", f"{mae_test:.2f}")
                    c4.metric("MAPE (test)", f"{mape_test:.2f}%")
                except Exception as e:
                    st.error(f"RF training failed: {e}")

        if st.session_state.get('rf_trained', False):
            st.info("Random Forest is trained")

    # LSTM
    with right:
        st.markdown("### 🧠 LSTM")
        st.write(f"Seq len: {lstm_seq_len}, Epochs: {lstm_epochs}, Batch: {lstm_batch}")
        if not TF_AVAILABLE:
            st.error("TensorFlow not available. Install tensorflow to enable LSTM.")
        else:
            if st.button("🚀 Train LSTM", use_container_width=True):
                with st.spinner("Training LSTM..."):
                    try:
                        X = df[FEATURES].values
                        y = df['ac_power'].values
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        X_seq, y_seq = create_lstm_sequences(X_scaled, y, lstm_seq_len)

                        if len(X_seq) < 30:
                            st.error("Not enough sequence samples. Reduce seq length or increase data.")
                        else:
                            split_idx = int(len(X_seq) * (1 - test_size))
                            X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
                            y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]

                            n_features = X_seq.shape[2]
                            model = Sequential([
                                LSTM(128, input_shape=(lstm_seq_len, n_features), return_sequences=True),
                                Dropout(0.2),
                                LSTM(64, return_sequences=False),
                                Dropout(0.2),
                                Dense(32, activation='relu'),
                                Dense(16, activation='relu'),
                                Dense(1, activation='linear')
                            ])
                            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                            history = model.fit(
                                X_train_seq, y_train_seq,
                                validation_data=(X_test_seq, y_test_seq),
                                epochs=lstm_epochs,
                                batch_size=lstm_batch,
                                callbacks=[es],
                                verbose=0
                            )

                            y_pred_train = model.predict(X_train_seq, verbose=0).flatten()
                            y_pred_test = model.predict(X_test_seq, verbose=0).flatten()
                            rmse_train, r2_train, mae_train, mape_train = evaluate_regression(y_train_seq, y_pred_train)
                            rmse_test, r2_test, mae_test, mape_test = evaluate_regression(y_test_seq, y_pred_test)

                            st.session_state['lstm_model'] = model
                            st.session_state['lstm_scaler'] = scaler
                            st.session_state['lstm_seq_len'] = lstm_seq_len
                            st.session_state['lstm_X_test_seq'] = X_test_seq
                            st.session_state['lstm_y_train'] = y_train_seq
                            st.session_state['lstm_y_test'] = y_test_seq
                            st.session_state['lstm_y_pred'] = y_pred_test
                            st.session_state['lstm_history'] = history.history
                            st.session_state['lstm_metrics'] = {
                                'rmse_train': rmse_train, 'r2_train': r2_train, 'mae_train': mae_train, 'mape_train': mape_train,
                                'rmse_test': rmse_test, 'r2_test': r2_test, 'mae_test': mae_test, 'mape_test': mape_test
                            }
                            st.session_state['lstm_trained'] = True

                            st.success("✅ LSTM trained")
                            c1,c2,c3,c4 = st.columns(4)
                            c1.metric("RMSE (test)", f"{rmse_test:.2f}")
                            c2.metric("R² (test)", f"{r2_test:.3f}")
                            c3.metric("MAE (test)", f"{mae_test:.2f}")
                            c4.metric("MAPE (test)", f"{mape_test:.2f}%")

                    except Exception as e:
                        # note: if shap was imported earlier this error could appear; lazy import avoids that
                        st.error(f"LSTM training failed: {e}")

            if st.session_state.get('lstm_trained', False):
                st.info("LSTM is trained")
                if 'lstm_history' in st.session_state:
                    hist = st.session_state['lstm_history']
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.plot(hist.get('loss', []), label='train loss')
                    ax1.plot(hist.get('val_loss', []), label='val loss')
                    ax1.set_title("Loss")
                    ax1.legend()
                    ax2.plot(hist.get('mae', []), label='train mae')
                    ax2.plot(hist.get('val_mae', []), label='val mae')
                    ax2.set_title("MAE")
                    ax2.legend()
                    st.pyplot(fig)
                    plt.close(fig)

# ---------- Tab 3: XAI ----------
with tab3:
    st.subheader("🔍 Explainable AI (SHAP)")

    col_rf, col_lstm = st.columns(2)

    # ---------- RF SHAP (lazy import) ----------
    with col_rf:
        st.markdown("### 🌲 SHAP for Random Forest")
        if not st.session_state.get('rf_trained', False):
            st.warning("Train RF first")
        else:
            if st.button("🔬 Compute SHAP (RF)", use_container_width=True):
                with st.spinner("Computing SHAP for RF..."):
                    try:
                        # lazy import shap
                        global SHAP_AVAILABLE
                        if SHAP_AVAILABLE is None:
                            try:
                                import shap as _shap
                                SHAP_AVAILABLE = True
                                shap = _shap
                            except Exception as e:
                                SHAP_AVAILABLE = False
                                st.error(f"Could not import shap: {e}")
                                raise e
                        else:
                            # if previously imported, import into local name
                            if SHAP_AVAILABLE:
                                import importlib
                                shap = importlib.import_module("shap")

                        rf = st.session_state['rf_model']
                        X_test_s = st.session_state['rf_X_test_s']
                        expl = shap.TreeExplainer(rf)
                        shap_values = expl.shap_values(X_test_s)
                        st.session_state['rf_shap_values'] = shap_values
                        st.success("✅ SHAP computed for RF")

                        # Summary plot
                        fig = plt.figure(figsize=(10,6))
                        shap.summary_plot(shap_values, X_test_s, feature_names=FEATURES, show=False)
                        st.pyplot(fig)
                        plt.close(fig)

                        # Importance bar
                        avg_imp = np.mean(np.abs(shap_values), axis=0)
                        imp_df = pd.DataFrame({'feature': FEATURES, 'importance': avg_imp}).sort_values('importance', ascending=True)
                        fig2, ax2 = plt.subplots(figsize=(8,6))
                        ax2.barh(imp_df['feature'], imp_df['importance'])
                        ax2.set_title("RF SHAP mean(|value|)")
                        st.pyplot(fig2)
                        plt.close(fig2)

                    except Exception as e:
                        st.error(f"SHAP RF error: {e}")

    # ---------- LSTM SHAP (lazy import + robust) ----------
    with col_lstm:
        st.markdown("### 🧠 SHAP for LSTM (Deep -> Gradient -> Permutation fallback)")
        if not st.session_state.get('lstm_trained', False):
            st.warning("Train LSTM first")
        elif not TF_AVAILABLE:
            st.error("TensorFlow not available")
        else:
            if st.button("🔬 Compute SHAP (LSTM)", use_container_width=True):
                with st.spinner("Computing SHAP values for LSTM (attempting Deep->Gradient->Permutation)..."):
                    try:
                        # lazy import shap
                        global SHAP_AVAILABLE
                        shap = None
                        if SHAP_AVAILABLE is None:
                            try:
                                import shap as _shap
                                SHAP_AVAILABLE = True
                                shap = _shap
                            except Exception as e:
                                SHAP_AVAILABLE = False
                                shap = None
                                st.warning(f"Could not import shap: {e}. Will try permutation fallback.")
                        else:
                            if SHAP_AVAILABLE:
                                import importlib
                                shap = importlib.import_module("shap")

                        model = st.session_state['lstm_model']
                        X_test_seq = st.session_state.get('lstm_X_test_seq', None)
                        feat_names = FEATURES

                        if X_test_seq is None or len(X_test_seq) == 0:
                            st.error("No LSTM test sequences found. Retrain with more data or shorter seq length.")
                        else:
                            sample = X_test_seq[:min(200, len(X_test_seq))]
                            background = X_test_seq[:min(50, len(X_test_seq))]

                            arr = None
                            expl_used = None

                            # Attempt DeepExplainer
                            if shap is not None:
                                try:
                                    expl = shap.DeepExplainer(model, background)
                                    shap_values = expl.shap_values(sample)
                                    arr = np.array(shap_values[0]) if isinstance(shap_values, list) else np.array(shap_values)
                                    expl_used = "DeepExplainer"
                                except Exception as e_deep:
                                    st.warning(f"DeepExplainer failed: {e_deep!s}. Trying GradientExplainer...")

                            # Attempt GradientExplainer
                            if arr is None and shap is not None:
                                try:
                                    expl = shap.GradientExplainer(model, background)
                                    shap_values = expl.shap_values(sample)
                                    arr = np.array(shap_values[0]) if isinstance(shap_values, list) else np.array(shap_values)
                                    expl_used = "GradientExplainer"
                                except Exception as e_grad:
                                    st.warning(f"GradientExplainer failed: {e_grad!s}. Falling back to permutation importance.")

                            # Permutation fallback
                            if arr is None:
                                st.info("Permutation fallback: estimating importance by shuffling features (model-agnostic). This may be slow.")
                                try:
                                    preds_base = model.predict(sample, verbose=0).flatten()
                                    n_samples, seq_len, n_feat = sample.shape
                                    mean_imp = np.zeros(n_feat)
                                    for fi in range(n_feat):
                                        sample_perm = sample.copy()
                                        # shuffle feature fi across samples for each timestep
                                        for t in range(seq_len):
                                            np.random.shuffle(sample_perm[:, t, fi])
                                        preds_perm = model.predict(sample_perm, verbose=0).flatten()
                                        mean_imp[fi] = np.mean(np.abs(preds_base - preds_perm))
                                    arr = np.zeros((n_samples, seq_len, n_feat))
                                    for fi in range(n_feat):
                                        arr[:, :, fi] = mean_imp[fi]
                                    expl_used = "PermutationFallback"
                                except Exception as e_perm:
                                    st.error(f"Permutation fallback failed: {e_perm!s}")
                                    arr = None

                            # Visualize if arr available
                            if arr is not None:
                                st.success(f"SHAP-like importances computed (method={expl_used})")
                                abs_arr = np.abs(arr)
                                mean_over_samples = np.mean(abs_arr, axis=0)  # (seq_len, n_features)
                                mean_feat_imp = np.mean(mean_over_samples, axis=0)  # (n_features,)

                                imp_df = pd.DataFrame({'feature': feat_names, 'importance': mean_feat_imp}).sort_values('importance', ascending=True)
                                fig1, ax1 = plt.subplots(figsize=(8,6))
                                ax1.barh(imp_df['feature'], imp_df['importance'])
                                ax1.set_title(f"LSTM feature importance (method={expl_used})")
                                st.pyplot(fig1)
                                plt.close(fig1)

                                fig2, ax2 = plt.subplots(figsize=(10,6))
                                sns.heatmap(mean_over_samples.T, cmap='viridis', yticklabels=feat_names, xticklabels=False, ax=ax2)
                                ax2.set_xlabel("Timestep")
                                ax2.set_ylabel("Feature")
                                ax2.set_title("Importance per Timestep × Feature")
                                st.pyplot(fig2)
                                plt.close(fig2)
                            else:
                                st.error("Failed to compute SHAP-like importances for LSTM.")
                    except Exception as e_all:
                        st.error(f"Unexpected error during LSTM SHAP: {e_all!s}")

# ---------- Tab 4: Predictions ----------
with tab4:
    st.subheader("📈 Manual Predictions")
    st.markdown("### Input Features")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        irr = st.number_input("☀️ Irradiance (W/m²)", 0.0, 1200.0, 500.0, 10.0)
        amb = st.number_input("🌡️ Ambient Temp (°C)", -10.0, 50.0, 25.0, 0.5)
    with c2:
        modt = st.number_input("🔥 Module Temp (°C)", 0.0, 90.0, 45.0, 0.5)
        hum = st.number_input("💧 Humidity (%)", 0.0, 100.0, 50.0, 1.0)
    with c3:
        wind = st.number_input("💨 Wind Speed (m/s)", 0.0, 30.0, 5.0, 0.5)
        pres = st.number_input("📊 Pressure (hPa)", 900.0, 1100.0, 1013.0, 1.0)
    with c4:
        cloud = st.number_input("☁️ Cloud Cover (%)", 0.0, 100.0, 30.0, 1.0)
        eff = st.number_input("⚡ Panel Efficiency (%)", 10.0, 25.0, 18.0, 0.1)

    X_input = np.array([[irr, amb, modt, hum, wind, pres, cloud, eff]])

    col_pred1, col_pred2 = st.columns(2)
    # RF pred
    with col_pred1:
        st.markdown("### 🌲 Random Forest")
        if not st.session_state.get('rf_trained', False):
            st.warning("Train RF first")
        else:
            if st.button("🔮 Predict with RF", use_container_width=True):
                try:
                    rf = st.session_state['rf_model']
                    scaler = st.session_state['rf_scaler']
                    Xs = scaler.transform(X_input)
                    pred = rf.predict(Xs)[0]
                    st.success(f"🎯 Predicted AC Power (RF): {pred:.2f} W")
                    est_eff = (pred / max(irr, 1e-6)) * 100
                    est_daily = pred * 24 / 1000
                    st.write(f"Estimated efficiency: **{est_eff:.2f}%**")
                    st.write(f"Estimated daily energy: **{est_daily:.2f} kWh**")
                except Exception as e:
                    st.error(f"RF predict error: {e}")

    # LSTM pred
    with col_pred2:
        st.markdown("### 🧠 LSTM")
        if not st.session_state.get('lstm_trained', False):
            st.warning("Train LSTM first")
        elif not TF_AVAILABLE:
            st.error("TensorFlow not available")
        else:
            if st.button("🔮 Predict with LSTM", use_container_width=True):
                try:
                    model = st.session_state['lstm_model']
                    scaler = st.session_state['lstm_scaler']
                    seq_len = st.session_state.get('lstm_seq_len', lstm_seq_len)
                    Xs = scaler.transform(X_input)  # (1, n_features)

                    # try to build realistic sequence using last test sequence
                    if 'lstm_X_test_seq' in st.session_state and st.session_state['lstm_X_test_seq'] is not None and len(st.session_state['lstm_X_test_seq'])>0:
                        last_seq = st.session_state['lstm_X_test_seq'][-1].copy()
                        if last_seq.shape[0] >= seq_len:
                            seq = np.vstack([last_seq[1:], Xs[0]])
                        else:
                            seq = np.vstack([last_seq, np.tile(Xs[0], (seq_len - last_seq.shape[0], 1))])
                        seq = seq.reshape(1, seq.shape[0], seq.shape[1])
                    else:
                        seq = np.tile(Xs.reshape(1,1,-1), (1, seq_len, 1))

                    pred_lstm = model.predict(seq, verbose=0).flatten()[0]
                    st.success(f"🎯 Predicted AC Power (LSTM): {pred_lstm:.2f} W")
                    est_eff_l = (pred_lstm / max(irr, 1e-6)) * 100
                    est_daily_l = pred_lstm * 24 / 1000
                    st.write(f"Estimated efficiency: **{est_eff_l:.2f}%**")
                    st.write(f"Estimated daily energy: **{est_daily_l:.2f} kWh**")
                except Exception as e:
                    st.error(f"LSTM predict error: {e}")

# ---------- Tab 5: Model Comparison ----------
with tab5:
    st.subheader("⚖️ Model Comparison")
    rf_ready = st.session_state.get('rf_trained', False)
    lstm_ready = st.session_state.get('lstm_trained', False)

    if not rf_ready and not lstm_ready:
        st.info("Train at least one model to compare.")
    else:
        c1, c2 = st.columns(2)
        if rf_ready:
            with c1:
                st.markdown("### 🌲 Random Forest Metrics")
                m = st.session_state.get('rf_metrics', None)
                if m:
                    st.write(f"- RMSE (test): **{m['rmse_test']:.2f}**")
                    st.write(f"- R² (test): **{m['r2_test']:.3f}**")
                    st.write(f"- MAE (test): **{m['mae_test']:.2f}**")
                    st.write(f"- MAPE (test): **{m['mape_test']:.2f}%**")
                else:
                    st.write("No RF metrics stored.")
                rf_model = st.session_state.get('rf_model', None)
                if rf_model is not None:
                    try:
                        imp = rf_model.feature_importances_
                        fi_df = pd.DataFrame({'feature': FEATURES, 'importance': imp}).sort_values('importance', ascending=False)
                        st.markdown("Top RF features")
                        st.table(fi_df.head(8).set_index('feature'))
                    except Exception:
                        pass
        else:
            with c1:
                st.info("RF not trained")

        if lstm_ready:
            with c2:
                st.markdown("### 🧠 LSTM Metrics")
                lm = st.session_state.get('lstm_metrics', None)
                if lm:
                    st.write(f"- RMSE (test): **{lm['rmse_test']:.2f}**")
                    st.write(f"- R² (test): **{lm['r2_test']:.3f}**")
                    st.write(f"- MAE (test): **{lm['mae_test']:.2f}**")
                    st.write(f"- MAPE (test): **{lm['mape_test']:.2f}%**")
                else:
                    st.write("No LSTM metrics stored.")
                if 'lstm_y_test' in st.session_state and 'lstm_y_pred' in st.session_state:
                    try:
                        y_t = st.session_state['lstm_y_test']
                        y_p = st.session_state['lstm_y_pred']
                        fig, ax = plt.subplots(figsize=(8,4))
                        ax.plot(y_t[:200], label='actual')
                        ax.plot(y_p[:200], label='pred')
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception:
                        pass
        else:
            with c2:
                st.info("LSTM not trained")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#666'>🌟 Solar Energy XAI — Random Forest & LSTM</div>", unsafe_allow_html=True)
