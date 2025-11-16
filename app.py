import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

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
    tf.config.run_functions_eagerly(False)
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
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">☀️ INTEGRASI EXPLAINABLE AI — RANDOM FOREST & LSTM</p>', unsafe_allow_html=True)

# Important notice about workflow
st.info("""
    ⚠️ **Peringatan Alur Kerja**: Latih model **SEBELUM** menggunakan fitur SHAP/Explainability!
    - ✅ Urutan yang disarankan: Train → Predict → Use SHAP
    - Jika mengalami error setelah menggunakan SHAP pada LSTM, klik tombol "Reset TensorFlow" sebelum melatih ulang LSTM.
""")

# ------------ Utilities ------------

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

def reset_tensorflow_session():
    """Reset TensorFlow session to clear SHAP modifications"""
    if TF_AVAILABLE:
        try:
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            return True
        except Exception:
            return False
    return False

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
    # Returns a tuple of metrics
    return rmse, r2, mae, mape 

# Helper function to display metrics (used in combined results)
def display_metrics_card(metrics_tuple, model_name):
    rmse, r2, mae, mape = metrics_tuple
    st.markdown(f"### {model_name} Performance")
    st.metric("R² Score (Test)", f"{r2:.3f}")
    st.metric("RMSE (Test)", f"{rmse:.2f} W")
    st.metric("MAE (Test)", f"{mae:.2f} W")
    st.metric("MAPE (Test)", f"{mape:.2f}%")

# --- COMBINED TRAINING LOGIC ---
def train_all_models(df, test_size, rf_n_estimators, rf_max_depth, lstm_seq_len, lstm_epochs, lstm_batch):
    
    FEATURES = ['irradiance','ambient_temperature','module_temperature','humidity','wind_speed','atmospheric_pressure','cloud_cover','panel_efficiency']
    
    with st.status("Preparing data and initializing training...", expanded=True) as status:
        X = df[FEATURES].copy()
        y = df['ac_power'].copy()
        
        # Split for RF (tabular data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scaler for RF
        scaler_rf = StandardScaler()
        X_train_s = scaler_rf.fit_transform(X_train)
        X_test_s = scaler_rf.transform(X_test)
        
        # Scaler for LSTM (sequence data)
        scaler_lstm = StandardScaler()
        X_scaled = scaler_lstm.fit_transform(X.values)
        X_seq, y_seq = create_lstm_sequences(X_scaled, y.values, lstm_seq_len)
        
        if len(X_seq) < 50:
            status.update(label="❌ Training Failed: Not enough sequence samples for LSTM.", state="error")
            st.error("❌ Not enough sequence samples. Reduce sequence length or increase data size.")
            return

        split_idx = int(len(X_seq) * (1 - test_size))
        X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
        y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]
        
        status.update(label="Data Preparation Complete.", state="running")

        
    # ----------------------------------------------------
    # Step 2: Train Random Forest
    # ----------------------------------------------------
    with st.status("🌲 Training Random Forest...", expanded=True) as status_rf:
        try:
            rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, 
                                      random_state=42, n_jobs=-1)
            rf.fit(X_train_s, y_train)

            y_pred_train = rf.predict(X_train_s)
            y_pred_test = rf.predict(X_test_s)

            rmse_train, r2_train, mae_train, mape_train = evaluate_regression(y_train, y_pred_train)
            rmse_test, r2_test, mae_test, mape_test = evaluate_regression(y_test, y_pred_test)
            
            # Save results
            st.session_state['rf_model'] = rf
            st.session_state['rf_scaler'] = scaler_rf
            st.session_state['rf_X_test_s'] = X_test_s
            st.session_state['rf_y_test'] = y_test
            st.session_state['rf_y_pred'] = y_pred_test
            st.session_state['rf_metrics'] = (rmse_test, r2_test, mae_test, mape_test)
            st.session_state['rf_trained'] = True
            
            status_rf.update(label="✅ Random Forest Training Complete!", state="complete", expanded=False)
        except Exception as e:
            st.session_state['rf_trained'] = False
            status_rf.update(label=f"❌ Random Forest Training Failed: {e}", state="error")


    # ----------------------------------------------------
    # Step 3: Train LSTM
    # ----------------------------------------------------
    if TF_AVAILABLE:
        with st.status("🧠 Training LSTM Neural Network (may take a moment)...", expanded=True) as status_lstm:
            try:
                n_features = X_seq.shape[2]
                reset_tensorflow_session()
                
                model = Sequential([
                    LSTM(128, input_shape=(lstm_seq_len, n_features), return_sequences=True),
                    Dropout(0.2),
                    LSTM(64, return_sequences=False),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dense(16, activation='relu'),
                    Dense(1, activation='linear')
                ])
                
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
                es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                history = model.fit(X_train_seq, y_train_seq,
                          validation_data=(X_test_seq, y_test_seq),
                          epochs=lstm_epochs,
                          batch_size=lstm_batch,
                          callbacks=[es],
                          verbose=0) # Suppress Keras output
                
                y_pred_train = model.predict(X_train_seq, verbose=0).flatten()
                y_pred_test = model.predict(X_test_seq, verbose=0).flatten()

                rmse_test, r2_test, mae_test, mape_test = evaluate_regression(y_test_seq, y_pred_test)

                # Save results
                st.session_state['lstm_model'] = model
                st.session_state['lstm_scaler'] = scaler_lstm
                st.session_state['lstm_seq_len'] = lstm_seq_len
                st.session_state['lstm_X_test_seq'] = X_test_seq
                st.session_state['lstm_y_test'] = y_test_seq
                st.session_state['lstm_y_pred'] = y_pred_test
                st.session_state['lstm_metrics'] = (rmse_test, r2_test, mae_test, mape_test)
                st.session_state['lstm_trained'] = True
                st.session_state['lstm_history'] = history.history
                
                status_lstm.update(label="✅ LSTM Training Complete!", state="complete", expanded=False)
            
            except Exception as e:
                st.session_state['lstm_trained'] = False
                status_lstm.update(label=f"❌ LSTM Training Failed: {e}", state="error")
                st.error(f"LSTM Error: {e}")

# Display results after combined training
def display_combined_results():
    st.markdown("---")
    st.markdown("## 📊 Model Training Results")
    
    col_rf, col_lstm = st.columns(2)
    
    if st.session_state.get('rf_trained', False):
        with col_rf:
            display_metrics_card(st.session_state['rf_metrics'], "🌲 Random Forest")
            
    if st.session_state.get('lstm_trained', False):
        with col_lstm:
            display_metrics_card(st.session_state['lstm_metrics'], "🧠 LSTM")
            
    if st.session_state.get('rf_trained', False) and st.session_state.get('lstm_trained', False):
        st.info("Lanjut ke **Tab 5: Model Comparison** untuk detail perbandingan performa.")


# --- COMBINED PREDICTION LOGIC ---

def predict_all_models(X_input):
    results = {}
    
    # Random Forest Prediction
    if st.session_state.get('rf_trained', False):
        try:
            model = st.session_state['rf_model']
            scaler = st.session_state['rf_scaler']
            X_scaled = scaler.transform(X_input)
            pred = model.predict(X_scaled)[0]
            results['rf_pred'] = pred
            
            if 'rf_shap_explainer' in st.session_state:
                explainer = st.session_state['rf_shap_explainer']
                shap_values_single = explainer.shap_values(X_scaled)
                results['rf_shap'] = shap_values_single[0]
                results['rf_features'] = [
                    'irradiance','ambient_temperature','module_temperature','humidity','wind_speed','atmospheric_pressure','cloud_cover','panel_efficiency'
                ]
                
        except Exception:
            results['rf_pred'] = None

    # LSTM Prediction
    if st.session_state.get('lstm_trained', False):
        try:
            model = st.session_state['lstm_model']
            scaler = st.session_state['lstm_scaler']
            seq_len = st.session_state['lstm_seq_len']
            
            X_scaled = scaler.transform(X_input)
            
            if 'lstm_X_test_seq' in st.session_state and len(st.session_state['lstm_X_test_seq']) > 0:
                # Use the last sequence from test set + current input
                last_seq = st.session_state['lstm_X_test_seq'][-1].copy()
                new_seq = np.vstack([last_seq[1:], X_scaled[0]])
                seq = new_seq.reshape(1, seq_len, -1)
            else:
                # Default to sequence of repeated current input if test set is empty/missing
                seq = np.tile(X_scaled.reshape(1, 1, -1), (1, seq_len, 1))

            pred = model.predict(seq, verbose=0).flatten()[0]
            results['lstm_pred'] = pred
            
        except Exception:
            results['lstm_pred'] = None
            
    return results

def display_prediction_results(results, X_input):
    st.markdown("---")
    st.markdown("## 🎯 Hasil Prediksi Gabungan")
    
    col_rf_pred, col_lstm_pred = st.columns(2)
    
    if 'rf_pred' in results and results['rf_pred'] is not None:
        with col_rf_pred:
            st.success(f"### 🌲 RF Power: **{results['rf_pred']:.2f} W**")
            
            if 'rf_shap' in results:
                st.markdown("#### Feature Contributions (RF)")
                contrib_df = pd.DataFrame({
                    'Feature': results['rf_features'],
                    'Value': X_input[0],
                    'SHAP': results['rf_shap']
                })
                contrib_df['Abs_SHAP'] = np.abs(contrib_df['SHAP'])
                contrib_df = contrib_df.sort_values('Abs_SHAP', ascending=False)
                
                fig, ax = plt.subplots(figsize=(7, 5))
                colors = ['green' if x > 0 else 'red' for x in contrib_df['SHAP']]
                ax.barh(contrib_df['Feature'], contrib_df['SHAP'], color=colors, alpha=0.7)
                ax.set_xlabel('SHAP Value (Impact on Prediction)')
                ax.set_title('RF Feature Contribution')
                ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
        
    if 'lstm_pred' in results and results['lstm_pred'] is not None:
        with col_lstm_pred:
            st.success(f"### 🧠 LSTM Power: **{results['lstm_pred']:.2f} W**")
            st.info("💡 Prediksi LSTM menggunakan urutan data historis sebelumnya.")


# ------------ Sidebar config & data load ------------

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=80)
    st.title("⚙️ Konfigurasi")

    source = st.radio("Sumber Data", ["Sample", "Upload CSV"])
    # ... (omitted file upload logic for brevity, assuming standard implementation) ...
    # Placeholder for file upload logic:
    uploaded_file = None # Assume no file uploaded for this example
    if source == "Upload CSV" and uploaded_file is None:
        st.info("Using sample data (Upload disabled in this snippet)")
        df = generate_sample_data()
    else:
        n_samples = st.slider("Jumlah Sampel", 600, 5000, 1200, 100)
        df = generate_sample_data(n_samples)


    st.markdown("---")
    st.subheader("Hyperparameters")
    
    st.markdown("**Random Forest**")
    rf_n_estimators = st.slider("n_estimators", 50, 500, 100, 50)
    rf_max_depth = st.slider("max_depth", 5, 50, 10, 5)

    st.markdown("**LSTM**")
    lstm_seq_len = st.slider("Sequence length", 4, 96, 24, 4, key='lstm_seq_len_sidebar')
    lstm_epochs = st.slider("Epochs", 5, 100, 20, 5)
    lstm_batch = st.selectbox("Batch size", [8,16,32,64], index=2)

    st.markdown("---")
    st.subheader("Train/Test Split")
    test_size = st.slider("Test size (%)", 10, 40, 20, 5) / 100

FEATURES = ['irradiance','ambient_temperature','module_temperature','humidity','wind_speed','atmospheric_pressure','cloud_cover','panel_efficiency']

# Initialize session state
if 'rf_trained' not in st.session_state:
    st.session_state['rf_trained'] = False
if 'lstm_trained' not in st.session_state:
    st.session_state['lstm_trained'] = False

# ------------ Tabs UI ------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data & EDA", "🤖 Train Models", "🔍 XAI (SHAP)", "📈 Predictions", "⚖️ Model Comparison"])

# ---------- Tab 1: Data & EDA ----------
# ... (contents remain unchanged, omitted for brevity) ...

# ---------- Tab 2: Train Models (UPDATED) ----------
with tab2:
    st.subheader("🤖 Train Machine Learning Models")
    
    train_col1, train_col2 = st.columns(2)
    
    # Random Forest Summary
    with train_col1:
        st.markdown("### 🌲 Random Forest Regressor")
        st.write(f"**Estimators:** {rf_n_estimators}")
        st.write(f"**Max Depth:** {rf_max_depth}")
        
    # LSTM Summary
    with train_col2:
        st.markdown("### 🧠 LSTM Neural Network")
        st.write(f"**Sequence Length:** {lstm_seq_len}")
        st.write(f"**Epochs:** {lstm_epochs}")
        
    st.markdown("---")
    
    # Tombol GABUNGAN untuk Training
    if st.button("🚀 Train BOTH Models", key="train_all_btn", use_container_width=True):
        st.session_state['rf_trained'] = False
        st.session_state['lstm_trained'] = False
        
        train_all_models(df, test_size, rf_n_estimators, rf_max_depth, 
                         lstm_seq_len, lstm_epochs, lstm_batch)
        
    # Tampilkan tombol Reset TensorFlow di Tab Train, jika SHAP LSTM pernah dijalankan
    if st.session_state.get('lstm_trained', False) and ('lstm_shap_values' in st.session_state or 'lstm_perm_importance' in st.session_state) and TF_AVAILABLE:
        st.markdown("---")
        if st.button("🔄 Reset TensorFlow (Clear SHAP)", key="reset_tf_btn"):
            reset_tensorflow_session()
            for key in list(st.session_state.keys()):
                if 'shap' in key.lower() or 'perm' in key.lower():
                    del st.session_state[key]
            st.success("✅ TensorFlow session reset!")
            st.rerun()

    # Tampilkan Hasil Gabungan
    if st.session_state.get('rf_trained', False) or st.session_state.get('lstm_trained', False):
        display_combined_results()
        if st.session_state.get('lstm_trained', False) and 'lstm_history' in st.session_state:
             # Add LSTM History plot back here
             st.markdown("#### 📉 LSTM Training History")
             # ... (plotting code for LSTM history goes here, copied from original Tab 2) ...

# ---------- Tab 3: XAI (SHAP) ----------
# ... (contents remain largely unchanged, but ensure logic for RF SHAP is run before RF prediction display) ...

# ---------- Tab 4: Predictions (UPDATED) ----------
with tab4:
    st.subheader("📈 Manual Predictions")
    
    st.markdown("### Input Features")
    cols = st.columns(4)
    with cols[0]:
        irr = st.number_input("☀️ Irradiance (W/m²)", 0.0, 1200.0, 500.0, 10.0)
        amb = st.number_input("🌡️ Ambient Temp (°C)", -10.0, 50.0, 25.0, 0.5)
    with cols[1]:
        modt = st.number_input("🔥 Module Temp (°C)", 0.0, 90.0, 45.0, 0.5)
        hum = st.number_input("💧 Humidity (%)", 0.0, 100.0, 50.0, 1.0)
    with cols[2]:
        wind = st.number_input("💨 Wind Speed (m/s)", 0.0, 30.0, 5.0, 0.5)
        pres = st.number_input("📊 Pressure (hPa)", 900.0, 1100.0, 1013.0, 1.0)
    with cols[3]:
        cloud = st.number_input("☁️ Cloud Cover (%)", 0.0, 100.0, 30.0, 1.0)
        eff = st.number_input("⚡ Panel Efficiency (%)", 10.0, 25.0, 18.0, 0.1)
    
    X_input = np.array([[irr, amb, modt, hum, wind, pres, cloud, eff]])
    
    # Tombol GABUNGAN untuk Prediction
    if not st.session_state.get('rf_trained', False) and not st.session_state.get('lstm_trained', False):
         st.warning("⚠️ Latih model terlebih dahulu di Tab 2.")
    else:
        if st.button("🔮 Predict BOTH Models", key="predict_all_btn", use_container_width=True):
            with st.spinner("Calculating predictions..."):
                prediction_results = predict_all_models(X_input)
                st.session_state['last_prediction_results'] = prediction_results
                st.session_state['last_X_input'] = X_input
                
        # Tampilkan hasil prediksi
        if 'last_prediction_results' in st.session_state:
            display_prediction_results(st.session_state['last_prediction_results'], st.session_state['last_X_input'])

# ---------- Tab 5: Model Comparison ----------
# ... (contents remain unchanged, omitted for brevity) ...

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666; padding: 20px;'>
    <p style='font-size: 1.1em;'>🌟 <strong>Solar Energy XAI Dashboard</strong></p>
</div>
""", unsafe_allow_html=True)
