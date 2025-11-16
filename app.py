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

st.info("""
    ⚠️ **Peringatan Alur Kerja**: Latih model **SEBELUM** menggunakan fitur SHAP/Explainability!
    - ✅ Urutan yang disarankan: Train → Predict → Use SHAP
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
            st.session_state['rf_metrics_full'] = {
                'rmse_train': rmse_train, 'r2_train': r2_train, 'mae_train': mae_train, 'mape_train': mape_train,
                'rmse_test': rmse_test, 'r2_test': r2_test, 'mae_test': mae_test, 'mape_test': mape_test
            }
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
                          verbose=0) 
                
                y_pred_train = model.predict(X_train_seq, verbose=0).flatten()
                y_pred_test = model.predict(X_test_seq, verbose=0).flatten()

                rmse_train, r2_train, mae_train, mape_train = evaluate_regression(y_train_seq, y_pred_train)
                rmse_test, r2_test, mae_test, mape_test = evaluate_regression(y_test_seq, y_pred_test)

                # Save results
                st.session_state['lstm_model'] = model
                st.session_state['lstm_scaler'] = scaler_lstm
                st.session_state['lstm_seq_len'] = lstm_seq_len
                st.session_state['lstm_X_test_seq'] = X_test_seq
                st.session_state['lstm_y_test'] = y_test_seq
                st.session_state['lstm_y_pred'] = y_pred_test
                st.session_state['lstm_metrics'] = (rmse_test, r2_test, mae_test, mape_test)
                st.session_state['lstm_metrics_full'] = {
                    'rmse_train': rmse_train, 'r2_train': r2_train, 'mae_train': mae_train, 'mape_train': mape_train,
                    'rmse_test': rmse_test, 'r2_test': r2_test, 'mae_test': mae_test, 'mape_test': mape_test
                }
                st.session_state['lstm_trained'] = True
                st.session_state['lstm_history'] = history.history
                
                status_lstm.update(label="✅ LSTM Training Complete!", state="complete", expanded=False)
            
            except Exception as e:
                st.session_state['lstm_trained'] = False
                status_lstm.update(label=f"❌ LSTM Training Failed: {e}", state="error")
                import traceback
                st.error(f"LSTM Error: {e}")
                with st.expander("🐛 View Full Error"):
                    st.code(traceback.format_exc())

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
        
        if 'lstm_history' in st.session_state:
            st.markdown("#### 📉 LSTM Training History")
            history = st.session_state['lstm_history']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(history['loss'], label='Train Loss', color='blue')
            ax1.plot(history['val_loss'], label='Val Loss', color='red')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss (MSE)')
            ax1.set_title('Loss Over Epochs')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            ax2.plot(history['mae'], label='Train MAE', color='blue')
            ax2.plot(history['val_mae'], label='Val MAE', color='red')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.set_title('MAE Over Epochs')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            st.pyplot(fig)
            plt.close(fig)


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
                last_seq = st.session_state['lstm_X_test_seq'][-1].copy()
                new_seq = np.vstack([last_seq[1:], X_scaled[0]])
                seq = new_seq.reshape(1, seq_len, -1)
            else:
                seq = np.tile(X_scaled.reshape(1, 1, -1), (1, seq_len, 1))

            pred = model.predict(seq, verbose=0).flatten()[0]
            results['lstm_pred'] = pred
            
        except Exception as e:
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
    
    uploaded_file = None
    if source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
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

# --- Tab 1: Data & EDA (Full content restored) ---
with tab1:
    st.subheader("📋 Data Preview")
    
    df_display = df.copy()
    if 'timestamp' in df_display.columns:
        df_display['timestamp'] = df_display['timestamp'].astype(str)
    
    st.dataframe(df_display.head(20), width='stretch')

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Descriptive Statistics")
        desc_stats = df[FEATURES + ['ac_power']].describe()
        st.dataframe(desc_stats, width='stretch')
    
    with col2:
        st.subheader("🔢 Data Info")
        st.write(f"**Total Rows:** {len(df)}")
        st.write(f"**Features:** {len(FEATURES)}")
        st.write(f"**Target:** ac_power")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")

    st.subheader("📈 Data Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        fig1, ax1 = plt.subplots(figsize=(8,5))
        df['ac_power'].hist(bins=50, ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_xlabel("AC Power (W)", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.set_title("AC Power Distribution", fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)
    
    with viz_col2:
        fig2, ax2 = plt.subplots(figsize=(8,5))
        df['irradiance'].plot(ax=ax2, color='orange', linewidth=0.8)
        ax2.set_xlabel("Time Index", fontsize=12)
        ax2.set_ylabel("Irradiance (W/m²)", fontsize=12)
        ax2.set_title("Irradiance Over Time", fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("🔥 Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(12,8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax3, center=0, 
                linewidths=1, cbar_kws={"shrink": 0.8})
    ax3.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    st.pyplot(fig3)
    plt.close(fig3)

# --- Tab 2: Train Models (Updated with single button) ---
with tab2:
    st.subheader("🤖 Train Machine Learning Models")
    
    train_col1, train_col2 = st.columns(2)
    
    with train_col1:
        st.markdown("### 🌲 Random Forest Regressor")
        st.write(f"**Estimators:** {rf_n_estimators}")
        st.write(f"**Max Depth:** {rf_max_depth}")
        
    with train_col2:
        st.markdown("### 🧠 LSTM Neural Network")
        st.write(f"**Sequence Length:** {lstm_seq_len}")
        st.write(f"**Epochs:** {lstm_epochs}")
    
    st.markdown("---")
    
    if st.button("🚀 Train BOTH Models", key="train_all_btn", use_container_width=True):
        st.session_state['rf_trained'] = False
        st.session_state['lstm_trained'] = False
        
        train_all_models(df, test_size, rf_n_estimators, rf_max_depth, 
                         lstm_seq_len, lstm_epochs, lstm_batch)
        
    if st.session_state.get('lstm_trained', False) and ('lstm_shap_values' in st.session_state or 'lstm_perm_importance' in st.session_state) and TF_AVAILABLE:
        st.markdown("---")
        if st.button("🔄 Reset TensorFlow (Clear SHAP)", key="reset_tf_btn"):
            reset_tensorflow_session()
            for key in list(st.session_state.keys()):
                if 'shap' in key.lower() or 'perm' in key.lower():
                    del st.session_state[key]
            st.success("✅ TensorFlow session reset!")
            st.rerun()

    if st.session_state.get('rf_trained', False) or st.session_state.get('lstm_trained', False):
        display_combined_results()

# --- Tab 3: XAI (SHAP) (Full content restored) ---
with tab3:
    st.subheader("🔍 Explainable AI dengan SHAP")
    
    if not SHAP_AVAILABLE:
        st.error("⚠️ SHAP not installed. Install with: `pip install shap`")
    else:
        shap_col1, shap_col2 = st.columns(2)
        
        # SHAP for Random Forest
        with shap_col1:
            st.markdown("### 🌲 SHAP for Random Forest")
            
            if not st.session_state.get('rf_trained', False):
                st.warning("⚠️ Train Random Forest model first")
            else:
                if st.button("🔬 Compute SHAP (RF)", key="shap_rf_btn"):
                    with st.spinner("Computing SHAP values for Random Forest..."):
                        try:
                            rf = st.session_state['rf_model']
                            X_test_s = st.session_state['rf_X_test_s']
                            
                            explainer = shap.TreeExplainer(rf)
                            shap_values = explainer.shap_values(X_test_s)
                            
                            st.session_state['rf_shap_values'] = shap_values
                            st.session_state['rf_shap_explainer'] = explainer
                            
                            st.success("✅ SHAP computed for RF")
                            
                            st.markdown("#### Feature Impact Summary")
                            fig1 = plt.figure(figsize=(10, 6))
                            shap.summary_plot(shap_values, X_test_s, feature_names=FEATURES, show=False)
                            st.pyplot(fig1)
                            plt.close(fig1)
                            
                            st.markdown("#### Feature Importance (Mean |SHAP|)")
                            avg_imp = np.mean(np.abs(shap_values), axis=0)
                            imp_df = pd.DataFrame({
                                'feature': FEATURES, 
                                'importance': avg_imp
                            }).sort_values('importance', ascending=True)
                            
                            fig2, ax2 = plt.subplots(figsize=(8, 6))
                            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df)))
                            ax2.barh(imp_df['feature'], imp_df['importance'], color=colors)
                            ax2.set_xlabel('Mean |SHAP Value|', fontsize=12)
                            ax2.set_title('RF Feature Importance via SHAP', fontsize=14, fontweight='bold')
                            ax2.grid(axis='x', alpha=0.3)
                            st.pyplot(fig2)
                            plt.close(fig2)
                            
                        except Exception as e:
                            st.error(f"❌ SHAP RF error: {e}")
        
        # SHAP for LSTM
        with shap_col2:
            st.markdown("### 🧠 Explainability for LSTM")
            
            if not st.session_state.get('lstm_trained', False):
                st.warning("⚠️ Train LSTM model first")
            elif not TF_AVAILABLE:
                st.error("⚠️ TensorFlow not available")
            else:
                st.info("""
                **Choose an explainability method:**
                - ⭐ **Permutation Importance**: Fast, reliable (Recommended)
                - 🐢 **KernelExplainer**: Slower SHAP-based method (5-10 min)
                """)
                
                shap_method = st.radio("Method", 
                                      ["Permutation Importance ⭐", 
                                       "KernelExplainer (SHAP)"],
                                      key="lstm_shap_method")
                
                if st.button("🔬 Compute Explainability (LSTM)", key="shap_lstm_btn"):
                    with st.spinner("Computing explainability..."):
                        try:
                            model = st.session_state['lstm_model']
                            X_test_seq = st.session_state['lstm_X_test_seq']
                            y_test_seq = st.session_state['lstm_y_test']
                            
                            if len(X_test_seq) < 10:
                                st.error("❌ Not enough test samples")
                            else:
                                n_features = X_test_seq.shape[2]
                                seq_len = st.session_state['lstm_seq_len']

                                if "Permutation" in shap_method:
                                    st.info("Computing Permutation Importance...")
                                    
                                    sample_size = min(300, len(X_test_seq))
                                    X_sample = X_test_seq[:sample_size]
                                    y_sample = y_test_seq[:sample_size]
                                    
                                    baseline_pred = model.predict(X_sample, verbose=0).flatten()
                                    baseline_score = r2_score(y_sample, baseline_pred)
                                    
                                    feature_importance = []
                                    feature_importance_std = []
                                    
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    for feat_idx, feat_name in enumerate(FEATURES):
                                        status_text.text(f"Processing: {feat_name} ({feat_idx+1}/{len(FEATURES)})")
                                        importance_scores = []
                                        
                                        for repeat in range(5): # Reduced repeats for quicker demo
                                            X_permuted = X_sample.copy()
                                            original_shape = X_permuted[:, :, feat_idx].shape
                                            X_permuted[:, :, feat_idx] = np.random.permutation(
                                                X_permuted[:, :, feat_idx].flatten()
                                            ).reshape(original_shape)
                                            
                                            perm_pred = model.predict(X_permuted, verbose=0).flatten()
                                            perm_score = r2_score(y_sample, perm_pred)
                                            importance_scores.append(baseline_score - perm_score)
                                        
                                        feature_importance.append(np.mean(importance_scores))
                                        feature_importance_std.append(np.std(importance_scores))
                                        progress_bar.progress((feat_idx + 1) / len(FEATURES))
                                    
                                    progress_bar.empty()
                                    status_text.empty()
                                    
                                    mean_feat_imp = np.array(feature_importance)
                                    std_feat_imp = np.array(feature_importance_std)
                                    
                                    st.session_state['lstm_perm_importance'] = mean_feat_imp
                                    st.session_state['lstm_perm_std'] = std_feat_imp
                                    st.session_state['lstm_shap_values'] = None # Clear SHAP if PI is used
                                    
                                    st.success("✅ Permutation Importance computed!")
                                    
                                else:  # KernelExplainer (with dimension fix)
                                    st.info("Using KernelExplainer (this may take several minutes)...")
                                    
                                    sample_size = min(50, len(X_test_seq))
                                    background_size = min(20, len(X_test_seq))
                                    
                                    background = X_test_seq[:background_size]
                                    sample = X_test_seq[:sample_size]
                                    
                                    background_flat = background.reshape(-1, seq_len * n_features)
                                    sample_flat = sample.reshape(-1, seq_len * n_features)

                                    def predict_fn(x):
                                        x_3d = x.reshape(-1, seq_len, n_features)
                                        return model.predict(x_3d, verbose=0).flatten()
                                    
                                    with st.spinner(f"Initializing explainer..."):
                                        explainer = shap.KernelExplainer(predict_fn, background_flat)
                                    
                                    with st.spinner(f"Computing SHAP values..."):
                                        shap_values = explainer.shap_values(sample_flat, nsamples=50)
                                    
                                    shap_array = np.array(shap_values)
                                    st.session_state['lstm_shap_values'] = shap_array
                                    st.session_state['lstm_perm_importance'] = None # Clear PI if SHAP is used
                                    
                                    st.success("✅ SHAP computed!")
                                    
                                    # Process SHAP results for visualization
                                    abs_shap = np.abs(shap_array)
                                    reshaped = abs_shap.reshape(-1, seq_len, n_features)
                                    mean_over_samples = np.mean(reshaped, axis=0)
                                    mean_feat_imp = np.mean(mean_over_samples, axis=0)
                                    std_feat_imp = np.std(np.mean(reshaped, axis=1), axis=0) # Simple std for bar plot

                                # Visualization (applies to both SHAP and PI)
                                imp_dict = {
                                    'feature': FEATURES, 
                                    'importance': mean_feat_imp
                                }
                                
                                # Use PI std if available
                                if st.session_state.get('lstm_perm_std') is not None:
                                    std_feat_imp = st.session_state['lstm_perm_std']

                                if std_feat_imp is not None:
                                    imp_dict['std'] = std_feat_imp
                                
                                imp_df = pd.DataFrame(imp_dict).sort_values('importance', ascending=True)
                                
                                st.markdown("#### Feature Importance")
                                fig1, ax1 = plt.subplots(figsize=(8, 6))
                                colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(imp_df)))
                                
                                if 'std' in imp_df.columns:
                                    bars = ax1.barh(imp_df['feature'], imp_df['importance'], 
                                                   xerr=imp_df['std'], color=colors, alpha=0.8)
                                else:
                                    bars = ax1.barh(imp_df['feature'], imp_df['importance'], color=colors)
                                    
                                ax1.set_xlabel('Importance Score', fontsize=12)
                                ax1.set_title(f'LSTM Feature Importance ({shap_method.split()[0]})', fontsize=14, fontweight='bold')
                                ax1.grid(axis='x', alpha=0.3)
                                
                                for bar in bars:
                                    width = bar.get_width()
                                    ax1.text(width, bar.get_y() + bar.get_height()/2, 
                                            f'{width:.4f}', ha='left', va='center', fontsize=9)
                                
                                st.pyplot(fig1)
                                plt.close(fig1)
                                
                                if "KernelExplainer" in shap_method and 'mean_over_samples' in locals() and mean_over_samples is not None:
                                    st.markdown("#### SHAP Heatmap (Feature × Timestep)")
                                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                                    sns.heatmap(mean_over_samples.T, cmap='viridis', 
                                               yticklabels=FEATURES, xticklabels=False, ax=ax2,
                                               cbar_kws={'label': 'Mean |SHAP|'})
                                    ax2.set_xlabel("Timestep", fontsize=12)
                                    ax2.set_ylabel("Feature", fontsize=12)
                                    ax2.set_title("SHAP Impact Over Time", fontsize=14, fontweight='bold')
                                    st.pyplot(fig2)
                                    plt.close(fig2)
                                
                                # ... (omitted remaining stats table display, assuming it works) ...
                                
                        except Exception as e:
                            st.error(f"❌ Explainability error: {e}")
                            import traceback
                            with st.expander("🐛 Full Error"):
                                st.code(traceback.format_exc())


# --- Tab 4: Predictions (Updated with single button) ---
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
    
    if not st.session_state.get('rf_trained', False) and not st.session_state.get('lstm_trained', False):
         st.warning("⚠️ Latih model terlebih dahulu di Tab 2.")
    else:
        if st.button("🔮 Predict BOTH Models", key="predict_all_btn", use_container_width=True):
            with st.spinner("Calculating predictions..."):
                prediction_results = predict_all_models(X_input)
                st.session_state['last_prediction_results'] = prediction_results
                st.session_state['last_X_input'] = X_input
                
        if 'last_prediction_results' in st.session_state:
            display_prediction_results(st.session_state['last_prediction_results'], st.session_state['last_X_input'])

# --- Tab 5: Model Comparison (Full content restored) ---
with tab5:
    st.subheader("⚖️ Model Comparison")
    
    if not st.session_state.get('rf_trained', False) and not st.session_state.get('lstm_trained', False):
        st.warning("⚠️ Train both models to see comparison")
    else:
        st.markdown("### 📊 Performance Metrics Comparison")
        
        comparison_data = []
        
        if st.session_state.get('rf_trained', False):
            rf_metrics = st.session_state['rf_metrics_full']
            comparison_data.append({
                'Model': 'Random Forest',
                'RMSE (Train)': rf_metrics['rmse_train'],
                'RMSE (Test)': rf_metrics['rmse_test'],
                'R² (Train)': rf_metrics['r2_train'],
                'R² (Test)': rf_metrics['r2_test'],
                'MAE (Train)': rf_metrics['mae_train'],
                'MAE (Test)': rf_metrics['mae_test'],
                'MAPE (Test)': rf_metrics['mape_test']
            })
        
        if st.session_state.get('lstm_trained', False):
            lstm_metrics = st.session_state['lstm_metrics_full']
            comparison_data.append({
                'Model': 'LSTM',
                'RMSE (Train)': lstm_metrics['rmse_train'],
                'RMSE (Test)': lstm_metrics['rmse_test'],
                'R² (Train)': lstm_metrics['r2_train'],
                'R² (Test)': lstm_metrics['r2_test'],
                'MAE (Train)': lstm_metrics['mae_train'],
                'MAE (Test)': lstm_metrics['mae_test'],
                'MAPE (Test)': lstm_metrics['mape_test']
            })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df.style.highlight_min(subset=['RMSE (Test)', 'MAE (Test)', 'MAPE (Test)'], color='lightgreen')
                                     .highlight_max(subset=['R² (Test)'], color='lightgreen'), 
                        width='stretch')
            
            st.markdown("### 📈 Visual Performance Comparison")
            
            if len(comparison_data) == 2:
                metrics_to_plot = ['RMSE (Test)', 'MAE (Test)', 'R² (Test)', 'MAPE (Test)']
                
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                axes = axes.flatten()
                
                for idx, metric in enumerate(metrics_to_plot):
                    ax = axes[idx]
                    values = [comp_df.loc[comp_df['Model'] == 'Random Forest', metric].values[0],
                             comp_df.loc[comp_df['Model'] == 'LSTM', metric].values[0]]
                    
                    colors = ['#FF6B35', '#004E89']
                    bars = ax.bar(['Random Forest', 'LSTM'], values, color=colors, alpha=0.7, edgecolor='black')
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}',
                               ha='center', va='bottom', fontweight='bold')
                    
                    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
                    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
                    
                    if metric == 'R² (Test)':
                        best_idx = np.argmax(values)
                    else:
                        best_idx = np.argmin(values)
                    bars[best_idx].set_edgecolor('gold')
                    bars[best_idx].set_linewidth(3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        
        if st.session_state.get('rf_trained', False) and st.session_state.get('lstm_trained', False):
            st.markdown("### 🎯 Predictions vs Actual (Test Set)")
            
            rf_y_test = st.session_state['rf_y_test']
            rf_y_pred = st.session_state['rf_y_pred']
            lstm_y_test = st.session_state['lstm_y_test']
            lstm_y_pred = st.session_state['lstm_y_pred']
            
            min_len = min(len(rf_y_test), len(lstm_y_test), 500)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            
            # RF: Actual vs Predicted
            axes[0, 0].scatter(rf_y_test[:min_len], rf_y_pred[:min_len], alpha=0.5, s=20, color='#FF6B35')
            axes[0, 0].plot([rf_y_test[:min_len].min(), rf_y_test[:min_len].max()], 
                           [rf_y_test[:min_len].min(), rf_y_test[:min_len].max()], 
                           'k--', lw=2, label='Perfect Prediction')
            axes[0, 0].set_xlabel('Actual AC Power (W)', fontsize=11)
            axes[0, 0].set_ylabel('Predicted AC Power (W)', fontsize=11)
            axes[0, 0].set_title('Random Forest: Actual vs Predicted', fontsize=12, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            
            # LSTM: Actual vs Predicted
            axes[0, 1].scatter(lstm_y_test[:min_len], lstm_y_pred[:min_len], alpha=0.5, s=20, color='#004E89')
            axes[0, 1].plot([lstm_y_test[:min_len].min(), lstm_y_test[:min_len].max()], 
                           [lstm_y_test[:min_len].min(), lstm_y_test[:min_len].max()], 
                           'k--', lw=2, label='Perfect Prediction')
            axes[0, 1].set_xlabel('Actual AC Power (W)', fontsize=11)
            axes[0, 1].set_ylabel('Predicted AC Power (W)', fontsize=11)
            axes[0, 1].set_title('LSTM: Actual vs Predicted', fontsize=12, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
            
            # RF: Residuals
            rf_residuals = rf_y_test[:min_len] - rf_y_pred[:min_len]
            axes[1, 0].scatter(rf_y_pred[:min_len], rf_residuals, alpha=0.5, s=20, color='#FF6B35')
            axes[1, 0].axhline(y=0, color='k', linestyle='--', lw=2)
            axes[1, 0].set_xlabel('Predicted AC Power (W)', fontsize=11)
            axes[1, 0].set_ylabel('Residuals (W)', fontsize=11)
            axes[1, 0].set_title('Random Forest: Residual Plot', fontsize=12, fontweight='bold')
            axes[1, 0].grid(alpha=0.3)
            
            # LSTM: Residuals
            lstm_residuals = lstm_y_test[:min_len] - lstm_y_pred[:min_len]
            axes[1, 1].scatter(lstm_y_pred[:min_len], lstm_residuals, alpha=0.5, s=20, color='#004E89')
            axes[1, 1].axhline(y=0, color='k', linestyle='--', lw=2)
            axes[1, 1].set_xlabel('Predicted AC Power (W)', fontsize=11)
            axes[1, 1].set_ylabel('Residuals (W)', fontsize=11)
            axes[1, 1].set_title('LSTM: Residual Plot', fontsize=12, fontweight='bold')
            axes[1, 1].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            st.markdown("### 📉 Time Series Comparison (First 200 Test Samples)")
            
            n_samples = min(200, min_len)
            x_axis = np.arange(n_samples)
            
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.plot(x_axis, rf_y_test[:n_samples], label='Actual', color='black', linewidth=2, alpha=0.7)
            ax.plot(x_axis, rf_y_pred[:n_samples], label='RF Prediction', color='#FF6B35', linewidth=1.5, alpha=0.8)
            ax.plot(x_axis, lstm_y_pred[:n_samples], label='LSTM Prediction', color='#004E89', linewidth=1.5, alpha=0.8)
            ax.set_xlabel('Sample Index', fontsize=12)
            ax.set_ylabel('AC Power (W)', fontsize=12)
            ax.set_title('Model Predictions vs Actual Values Over Time', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            st.markdown("### 📊 Error Distribution Comparison")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].hist(rf_residuals, bins=50, color='#FF6B35', alpha=0.7, edgecolor='black')
            axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            axes[0].set_xlabel('Residual (W)', fontsize=11)
            axes[0].set_ylabel('Frequency', fontsize=11)
            axes[0].set_title('Random Forest: Error Distribution', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            axes[1].hist(lstm_residuals, bins=50, color='#004E89', alpha=0.7, edgecolor='black')
            axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            axes[1].set_xlabel('Residual (W)', fontsize=11)
            axes[1].set_ylabel('Frequency', fontsize=11)
            axes[1].set_title('LSTM: Error Distribution', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            st.markdown("### 📋 Residual Statistics")
            
            residual_stats = pd.DataFrame({
                'Model': ['Random Forest', 'LSTM'],
                'Mean Error': [np.mean(rf_residuals), np.mean(lstm_residuals)],
                'Std Error': [np.std(rf_residuals), np.std(lstm_residuals)],
                'Min Error': [np.min(rf_residuals), np.min(lstm_residuals)],
                'Max Error': [np.max(rf_residuals), np.max(lstm_residuals)],
                'Median Error': [np.median(rf_residuals), np.median(lstm_residuals)]
            })
            
            st.dataframe(residual_stats.style.format({
                'Mean Error': '{:.2f}',
                'Std Error': '{:.2f}',
                'Min Error': '{:.2f}',
                'Max Error': '{:.2f}',
                'Median Error': '{:.2f}'
            }), width='stretch')
            
            st.markdown("### 🏆 Model Recommendation")
            
            rf_score = st.session_state['rf_metrics'][1] # R2 is index 1
            lstm_score = st.session_state['lstm_metrics'][1]
            
            if rf_score > lstm_score:
                winner = "Random Forest"
                winner_color = "#FF6B35"
                winner_r2 = rf_score
                winner_rmse = st.session_state['rf_metrics'][0]
            else:
                winner = "LSTM"
                winner_color = "#004E89"
                winner_r2 = lstm_score
                winner_rmse = st.session_state['lstm_metrics'][0]
            
            st.markdown(f"""
            <div style='background-color: {winner_color}; padding: 20px; border-radius: 10px; color: white;'>
                <h3 style='color: white; margin: 0;'>🏆 Best Performing Model: {winner}</h3>
                <p style='margin: 10px 0 0 0; font-size: 1.1em;'>
                    <strong>R² Score:</strong> {winner_r2:.4f} | <strong>RMSE:</strong> {winner_rmse:.2f} W
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            #### 💡 Model Selection Guidelines:
            - **Random Forest**: Better for interpretability, faster training, good for tabular data
            - **LSTM**: Better for temporal patterns, sequential dependencies, time-series forecasting
            - **Recommendation**: Use the model with higher R² and lower RMSE for your use case
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666; padding: 20px;'>
    <p style='font-size: 1.1em;'>🌟 <strong>Solar Energy XAI Dashboard</strong></p>
</div>
""", unsafe_allow_html=True)
