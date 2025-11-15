import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import shap
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Solar Energy XAI",
    page_icon="☀️",
    layout="wide"
)

# CSS Custom
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #004E89;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">☀️ INTEGRASI EXPLAINABLE AI (XAI)<br>OPTIMALISASI PERFORMA ENERGI SURYA</p>', unsafe_allow_html=True)

# Fungsi untuk generate data contoh berdasarkan Solar PV System Dataset
@st.cache_data
def generate_sample_data(n_samples=1000):
    """Generate data simulasi energi surya berdasarkan karakteristik PV System"""
    np.random.seed(42)
    
    # Generate timestamp dengan interval 15 menit (realistis untuk monitoring PV)
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='15min')
    
    # Ekstrak informasi waktu
    hours = dates.hour + dates.minute/60
    days = np.arange(n_samples) / 96  # 96 readings per day (15 min interval)
    
    # 1. IRRADIANCE (W/m²) - Radiasi matahari
    # Pola harian dengan variasi musiman
    daily_pattern = np.sin((hours - 6) * np.pi / 12)  # Peak di tengah hari
    seasonal_variation = 1 + 0.3 * np.sin(days * 2 * np.pi / 365)  # Variasi tahunan
    base_irradiance = 800 * daily_pattern * seasonal_variation
    irradiance = np.maximum(0, base_irradiance + np.random.normal(0, 50, n_samples))
    irradiance = np.where((hours < 6) | (hours > 18), 0, irradiance)  # Tidak ada radiasi malam
    
    # 2. AMBIENT TEMPERATURE (°C)
    daily_temp_pattern = 20 + 10 * np.sin((hours - 8) * np.pi / 12)
    seasonal_temp = 5 * np.sin(days * 2 * np.pi / 365)
    temperature = daily_temp_pattern + seasonal_temp + np.random.normal(0, 2, n_samples)
    
    # 3. MODULE TEMPERATURE (°C) - Selalu lebih tinggi dari ambient
    # Cell temp biasanya 20-30°C lebih tinggi dari ambient saat ada irradiance
    temp_rise = 25 * (irradiance / 1000)  # Temperature rise factor
    module_temperature = temperature + temp_rise + np.random.normal(0, 3, n_samples)
    
    # 4. HUMIDITY (%)
    humidity_pattern = 60 - 20 * np.sin((hours - 8) * np.pi / 12)  # Rendah siang, tinggi malam
    humidity = np.clip(humidity_pattern + np.random.normal(0, 10, n_samples), 20, 95)
    
    # 5. WIND SPEED (m/s)
    wind_speed = np.abs(5 + 3 * np.sin(hours * np.pi / 6) + np.random.normal(0, 1.5, n_samples))
    wind_speed = np.clip(wind_speed, 0, 20)
    
    # 6. ATMOSPHERIC PRESSURE (hPa/mbar)
    pressure = 1013 + np.random.normal(0, 5, n_samples)
    
    # 7. CLOUD COVER (%)
    base_clouds = 30 + 25 * np.sin(days * 2 * np.pi / 7)  # Weekly pattern
    cloud_cover = np.clip(base_clouds + np.random.normal(0, 15, n_samples), 0, 100)
    
    # 8. PANEL EFFICIENCY (%) - Menurun dengan suhu
    base_efficiency = 18  # 18% baseline efficiency
    temp_coefficient = -0.004  # -0.4%/°C typical for silicon
    efficiency = base_efficiency + temp_coefficient * (module_temperature - 25)
    efficiency = np.clip(efficiency + np.random.normal(0, 0.5, n_samples), 12, 22)
    
    # 9. DC POWER OUTPUT (W)
    # Power = Irradiance × Area × Efficiency
    panel_area = 1.6  # m² (typical panel size)
    dc_power = (irradiance * panel_area * efficiency / 100)
    # Losses due to cloud cover
    dc_power *= (1 - cloud_cover / 200)
    dc_power = np.maximum(0, dc_power + np.random.normal(0, 5, n_samples))
    
    # 10. AC POWER OUTPUT (W) - After inverter losses
    inverter_efficiency = 0.96  # 96% inverter efficiency
    ac_power = dc_power * inverter_efficiency
    ac_power = np.maximum(0, ac_power + np.random.normal(0, 3, n_samples))
    
    # 11. VOLTAGE (V) - DC voltage
    voltage = np.where(irradiance > 10, 
                       30 + 5 * (irradiance / 1000) + np.random.normal(0, 1, n_samples),
                       0)
    
    # 12. CURRENT (A) - DC current
    current = np.where(voltage > 0, dc_power / voltage, 0)
    
    # 13. ENERGY YIELD (kWh) - Cumulative
    energy_yield = np.cumsum(ac_power / 1000 * 0.25)  # 0.25 hour intervals
    
    df = pd.DataFrame({
        'timestamp': dates,
        'irradiance': irradiance,
        'ambient_temperature': temperature,
        'module_temperature': module_temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'atmospheric_pressure': pressure,
        'cloud_cover': cloud_cover,
        'panel_efficiency': efficiency,
        'dc_voltage': voltage,
        'dc_current': current,
        'dc_power': dc_power,
        'ac_power': ac_power,
        'daily_energy': energy_yield % 10  # Reset daily
    })
    
    return df

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=100)
    st.title("⚙️ Konfigurasi")
    
    # Upload data atau gunakan sample
    data_source = st.radio(
        "Sumber Data",
        ["Gunakan Data Sampel", "Upload Data CSV"]
    )
    
    if data_source == "Upload Data CSV":
        uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data berhasil diupload!")
        else:
            df = generate_sample_data()
            st.info("Menggunakan data sampel sementara")
    else:
        n_samples = st.slider("Jumlah Sampel Data", 500, 5000, 1000, 100)
        df = generate_sample_data(n_samples)
    
    st.markdown("---")
    
    # Parameter Model
    st.subheader("🎯 Parameter Model RF")
    n_estimators = st.slider("Jumlah Trees", 50, 500, 100, 50)
    max_depth = st.slider("Max Depth", 5, 50, 10, 5)
    test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100

# Tab Navigation
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data & EDA", "🤖 Model Training", "🔍 Explainable AI", "📈 Prediksi"])

# TAB 1: Data & EDA
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
    
    # Preview data
    st.subheader("🔍 Preview Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistik deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Visualisasi
    st.subheader("📉 Visualisasi Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        df['ac_power'].hist(bins=50, edgecolor='black', ax=ax, color='#FF6B35')
        ax.set_title('Distribusi AC Power Output', fontsize=14, fontweight='bold')
        ax.set_xlabel('AC Power Output (W)')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        features_to_plot = ['irradiance', 'ambient_temperature', 'module_temperature', 
                           'humidity', 'wind_speed', 'cloud_cover', 'panel_efficiency', 'ac_power']
        correlation = df[features_to_plot].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f', 
                   cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
    
    # Time series plot
    st.subheader("⏱️ Time Series - Power Output & Environmental Factors")
    
    # Plot 1: Power outputs
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # AC vs DC Power
    sample_data = df[:96*3]  # 3 days data
    axes[0].plot(sample_data['timestamp'], sample_data['dc_power'], 
                linewidth=2, color='#FF6B35', label='DC Power', alpha=0.7)
    axes[0].plot(sample_data['timestamp'], sample_data['ac_power'], 
                linewidth=2, color='#004E89', label='AC Power', alpha=0.7)
    axes[0].set_title('DC vs AC Power Output (3 Days)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Timestamp')
    axes[0].set_ylabel('Power (W)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Irradiance and Temperature
    ax2 = axes[1]
    ax3 = ax2.twinx()
    
    line1 = ax2.plot(sample_data['timestamp'], sample_data['irradiance'], 
                     linewidth=2, color='#FFB400', label='Irradiance')
    line2 = ax3.plot(sample_data['timestamp'], sample_data['module_temperature'], 
                     linewidth=2, color='#E63946', label='Module Temp')
    
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Irradiance (W/m²)', color='#FFB400')
    ax3.set_ylabel('Module Temperature (°C)', color='#E63946')
    ax2.tick_params(axis='y', labelcolor='#FFB400')
    ax3.tick_params(axis='y', labelcolor='#E63946')
    ax2.set_title('Irradiance & Module Temperature (3 Days)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Additional plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['irradiance'], df['ac_power'], alpha=0.3, color='#FF6B35', s=10)
        ax.set_xlabel('Irradiance (W/m²)')
        ax.set_ylabel('AC Power (W)')
        ax.set_title('Power Output vs Irradiance', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['module_temperature'], df['panel_efficiency'], 
                  alpha=0.3, color='#004E89', s=10)
        ax.set_xlabel('Module Temperature (°C)')
        ax.set_ylabel('Panel Efficiency (%)')
        ax.set_title('Efficiency vs Module Temperature', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        st.pyplot(fig)

# TAB 2: Model Training
with tab2:
    st.markdown('<p class="sub-header">🤖 Training Model Random Forest</p>', unsafe_allow_html=True)
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Prepare data
            features = ['irradiance', 'ambient_temperature', 'module_temperature', 
                       'humidity', 'wind_speed', 'atmospheric_pressure', 'cloud_cover', 
                       'panel_efficiency']
            X = df[features]
            y = df['ac_power']  # Target: AC Power Output
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train = rf_model.predict(X_train_scaled)
            y_pred_test = rf_model.predict(X_test_scaled)
            
            # Metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Save to session state
            st.session_state['model'] = rf_model
            st.session_state['scaler'] = scaler
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred_test
            st.session_state['features'] = features
            
            st.success("✅ Model berhasil dilatih!")
            
            # Display metrics
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
            
            # Visualisasi prediksi
            st.subheader("🎯 Actual vs Predicted")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, y_pred_test, alpha=0.5, color='#FF6B35')
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                ax.set_xlabel('Actual Power Output')
                ax.set_ylabel('Predicted Power Output')
                ax.set_title('Actual vs Predicted Values')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                residuals = y_test - y_pred_test
                ax.scatter(y_pred_test, residuals, alpha=0.5, color='#004E89')
                ax.axhline(y=0, color='red', linestyle='--', lw=2)
                ax.set_xlabel('Predicted Power Output')
                ax.set_ylabel('Residuals')
                ax.set_title('Residual Plot')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
            # Feature importance
            st.subheader("🎯 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='#FF6B35')
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance - Random Forest')
            ax.grid(alpha=0.3, axis='x')
            st.pyplot(fig)

# TAB 3: Explainable AI (XAI)
with tab3:
    st.markdown('<p class="sub-header">🔍 Explainable AI dengan SHAP</p>', unsafe_allow_html=True)
    
    if 'model' in st.session_state:
        if st.button("🔬 Generate SHAP Analysis", type="primary"):
            with st.spinner("Menghitung SHAP values..."):
                model = st.session_state['model']
                X_test = st.session_state['X_test']
                scaler = st.session_state['scaler']
                features = st.session_state['features']
                
                X_test_scaled = scaler.transform(X_test)
                
                # SHAP Explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_scaled)
                
                st.success("✅ SHAP analysis selesai!")
                
                # SHAP Summary Plot
                st.subheader("📊 SHAP Summary Plot")
                st.write("Visualisasi kontribusi setiap fitur terhadap prediksi model")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_scaled, feature_names=features, show=False)
                st.pyplot(fig)
                plt.close()
                
                # SHAP Bar Plot
                st.subheader("📊 SHAP Feature Importance")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_scaled, feature_names=features, plot_type="bar", show=False)
                st.pyplot(fig)
                plt.close()
                
                # Individual prediction explanation
                st.subheader("🔍 Penjelasan Prediksi Individual")
                sample_idx = st.slider("Pilih Sample Index", 0, len(X_test)-1, 0)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[sample_idx],
                        base_values=explainer.expected_value,
                        data=X_test_scaled[sample_idx],
                        feature_names=features
                    ),
                    show=False
                )
                st.pyplot(fig)
                plt.close()
                
                # Display sample details
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Input Features:**")
                    for i, feat in enumerate(features):
                        st.write(f"- {feat}: {X_test.iloc[sample_idx][feat]:.2f}")
                
                with col2:
                    st.markdown("**Prediction:**")
                    pred_value = model.predict(X_test_scaled[sample_idx].reshape(1, -1))[0]
                    actual_value = st.session_state['y_test'].iloc[sample_idx]
                    st.metric("Predicted Power", f"{pred_value:.2f} W")
                    st.metric("Actual Power", f"{actual_value:.2f} W")
                    st.metric("Error", f"{abs(pred_value - actual_value):.2f} W")
    else:
        st.warning("⚠️ Silakan latih model terlebih dahulu di tab 'Model Training'")

# TAB 4: Prediksi
with tab4:
    st.markdown('<p class="sub-header">📈 Prediksi Power Output</p>', unsafe_allow_html=True)
    
    if 'model' in st.session_state:
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
            model = st.session_state['model']
            scaler = st.session_state['scaler']
            
            # Prepare input
            input_data = np.array([[irradiance_input, ambient_temp_input, module_temp_input, 
                                   humidity_input, wind_input, pressure_input, 
                                   cloud_input, efficiency_input]])
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            
            st.success("✅ Prediksi berhasil!")
            
            # Display result
            st.markdown("### 🎯 Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Power Output", f"{prediction:.2f} W", 
                         delta=None)
            with col2:
                efficiency = (prediction / irradiance_input * 100) if irradiance_input > 0 else 0
                st.metric("Estimated Efficiency", f"{efficiency:.2f}%")
            with col3:
                daily_energy = prediction * 24 / 1000  # kWh
                st.metric("Est. Daily Energy", f"{daily_energy:.2f} kWh")
            
            # SHAP explanation for this prediction
            st.markdown("### 🔍 Penjelasan Prediksi")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=input_scaled[0],
                    feature_names=st.session_state['features']
                ),
                show=False
            )
            st.pyplot(fig)
            plt.close()
    else:
        st.warning("⚠️ Silakan latih model terlebih dahulu di tab 'Model Training'")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌟 Aplikasi XAI untuk Optimalisasi Energi Surya | Powered by Streamlit & SHAP</p>
    </div>
""", unsafe_allow_html=True)
