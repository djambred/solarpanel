"""
Solar Panel PV System Dataset Generator
Generates realistic solar panel data based on Kaggle PV System Dataset structure
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def generate_solar_pv_dataset(n_days=30, interval_minutes=15, save_csv=True):
    """
    Generate realistic solar PV system dataset
    
    Parameters:
    -----------
    n_days : int
        Number of days to generate data for
    interval_minutes : int
        Time interval between measurements (default: 15 minutes)
    save_csv : bool
        Whether to save the dataset as CSV
    
    Returns:
    --------
    DataFrame with solar PV system data
    """
    
    # Calculate total samples
    samples_per_day = (24 * 60) // interval_minutes
    n_samples = n_days * samples_per_day
    
    print(f"Generating {n_samples} samples ({n_days} days, {interval_minutes} min intervals)...")
    
    np.random.seed(42)
    
    # Generate timestamps
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(minutes=interval_minutes * i) for i in range(n_samples)]
    
    # Extract time features
    hours = np.array([d.hour + d.minute/60 for d in dates])
    days_of_year = np.array([(d - start_date).days for d in dates])
    
    print("Generating environmental parameters...")
    
    # 1. SOLAR IRRADIANCE (W/m²)
    # Realistic pattern: 0 at night, peak ~1000 W/m² at noon
    daily_pattern = np.sin((hours - 6) * np.pi / 12)
    daily_pattern = np.maximum(0, daily_pattern)  # No negative values
    
    # Seasonal variation (higher in summer)
    seasonal_factor = 1 + 0.3 * np.sin((days_of_year - 80) * 2 * np.pi / 365)
    
    # Base irradiance with weather variations
    base_irradiance = 1000 * daily_pattern * seasonal_factor
    
    # Add weather effects (clouds, etc.)
    weather_noise = np.random.normal(0, 100, n_samples)
    irradiance = np.maximum(0, base_irradiance + weather_noise)
    
    # Set to 0 during night hours
    irradiance = np.where((hours < 5) | (hours > 19), 0, irradiance)
    
    # 2. AMBIENT TEMPERATURE (°C)
    daily_temp_pattern = 15 + 12 * np.sin((hours - 8) * np.pi / 12)
    seasonal_temp = 10 * np.sin((days_of_year - 80) * 2 * np.pi / 365)
    ambient_temperature = daily_temp_pattern + seasonal_temp + np.random.normal(0, 2, n_samples)
    
    # 3. MODULE/CELL TEMPERATURE (°C)
    # Module temperature is higher than ambient when irradiance is present
    # Rule: T_module = T_ambient + (NOCT - 20) * Irradiance / 800
    # NOCT (Nominal Operating Cell Temperature) typically 45°C
    noct = 45
    temp_coefficient = (noct - 20) / 800
    temp_rise = temp_coefficient * irradiance
    module_temperature = ambient_temperature + temp_rise + np.random.normal(0, 2, n_samples)
    
    # 4. HUMIDITY (%)
    # Higher at night/morning, lower during day
    humidity_pattern = 70 - 25 * np.sin((hours - 8) * np.pi / 12)
    humidity = np.clip(humidity_pattern + np.random.normal(0, 10, n_samples), 20, 95)
    
    # 5. WIND SPEED (m/s)
    # Varies throughout day with some randomness
    wind_base = 3 + 4 * np.sin(hours * np.pi / 12)
    wind_speed = np.abs(wind_base + np.random.normal(0, 1.5, n_samples))
    wind_speed = np.clip(wind_speed, 0, 15)
    
    # 6. ATMOSPHERIC PRESSURE (hPa)
    pressure = 1013 + 10 * np.sin(days_of_year * 2 * np.pi / 365) + np.random.normal(0, 5, n_samples)
    
    # 7. CLOUD COVER (%)
    # Affects irradiance - random weather patterns
    cloud_base = 20 + 30 * np.sin(days_of_year * 2 * np.pi / 7)  # Weekly patterns
    cloud_cover = np.clip(cloud_base + np.random.normal(0, 20, n_samples), 0, 100)
    
    print("Calculating PV system parameters...")
    
    # 8. PANEL EFFICIENCY (%)
    # Decreases with temperature
    # Typical silicon panel: 18% @ 25°C, -0.4%/°C
    base_efficiency = 18.0
    temp_coefficient_eff = -0.004  # -0.4% per °C
    panel_efficiency = base_efficiency + temp_coefficient_eff * (module_temperature - 25)
    panel_efficiency = np.clip(panel_efficiency + np.random.normal(0, 0.3, n_samples), 12, 22)
    
    # 9. DC VOLTAGE (V)
    # Typical range: 30-40V for a panel
    # Voltage decreases slightly with temperature
    nominal_voltage = 36
    voltage_temp_coeff = -0.08  # V/°C
    dc_voltage = np.where(
        irradiance > 50,
        nominal_voltage + voltage_temp_coeff * (module_temperature - 25) + np.random.normal(0, 0.5, n_samples),
        0
    )
    dc_voltage = np.maximum(0, dc_voltage)
    
    # 10. DC POWER (W)
    # Power = Irradiance × Area × Efficiency
    panel_area = 1.7  # m² (typical 300W panel)
    dc_power = (irradiance * panel_area * panel_efficiency / 100)
    
    # Apply cloud cover losses
    cloud_loss_factor = 1 - (cloud_cover / 150)
    dc_power *= cloud_loss_factor
    
    # Apply soiling/degradation (1-2% loss)
    soiling_factor = 0.98
    dc_power *= soiling_factor
    
    dc_power = np.maximum(0, dc_power + np.random.normal(0, 5, n_samples))
    
    # 11. DC CURRENT (A)
    # Current = Power / Voltage
    dc_current = np.where(dc_voltage > 0, dc_power / dc_voltage, 0)
    dc_current = np.maximum(0, dc_current)
    
    # 12. AC POWER (W)
    # After inverter conversion (typically 95-97% efficiency)
    inverter_efficiency = 0.96
    ac_power = dc_power * inverter_efficiency
    ac_power = np.maximum(0, ac_power + np.random.normal(0, 3, n_samples))
    
    # 13. DAILY ENERGY YIELD (kWh)
    # Cumulative energy for the day, resets at midnight
    hours_per_interval = interval_minutes / 60
    energy_increments = ac_power * hours_per_interval / 1000  # Convert to kWh
    
    # Reset cumulative sum at midnight
    day_numbers = np.array([d.day for d in dates])
    daily_energy = []
    cumsum = 0
    prev_day = day_numbers[0]
    
    for i, day in enumerate(day_numbers):
        if day != prev_day:
            cumsum = 0
        cumsum += energy_increments[i]
        daily_energy.append(cumsum)
        prev_day = day
    
    daily_energy = np.array(daily_energy)
    
    print("Creating DataFrame...")
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'irradiance': np.round(irradiance, 2),
        'ambient_temperature': np.round(ambient_temperature, 2),
        'module_temperature': np.round(module_temperature, 2),
        'humidity': np.round(humidity, 2),
        'wind_speed': np.round(wind_speed, 2),
        'atmospheric_pressure': np.round(pressure, 2),
        'cloud_cover': np.round(cloud_cover, 2),
        'panel_efficiency': np.round(panel_efficiency, 2),
        'dc_voltage': np.round(dc_voltage, 2),
        'dc_current': np.round(dc_current, 2),
        'dc_power': np.round(dc_power, 2),
        'ac_power': np.round(ac_power, 2),
        'daily_energy': np.round(daily_energy, 3)
    })
    
    # Add additional derived features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    if save_csv:
        filename = f'solar_pv_dataset_{n_days}days.csv'
        df.to_csv(filename, index=False)
        print(f"\n✅ Dataset saved as: {filename}")
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nKey Metrics:")
    print(f"  Average AC Power: {df['ac_power'].mean():.2f} W")
    print(f"  Peak AC Power: {df['ac_power'].max():.2f} W")
    print(f"  Average Daily Energy: {df.groupby(df['timestamp'].dt.date)['daily_energy'].max().mean():.2f} kWh")
    print(f"  Total Energy Generated: {df['daily_energy'].max() * n_days:.2f} kWh")
    print(f"  Average Panel Efficiency: {df['panel_efficiency'].mean():.2f}%")
    print(f"  Average Module Temperature: {df['module_temperature'].mean():.2f}°C")
    
    return df


def plot_dataset_overview(df, save_plots=True):
    """
    Create overview plots of the generated dataset
    """
    print("\nGenerating visualization plots...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Power Output Time Series (3 days)
    ax1 = plt.subplot(3, 3, 1)
    sample_data = df.iloc[:96*3]  # 3 days
    ax1.plot(sample_data['timestamp'], sample_data['dc_power'], 
             label='DC Power', linewidth=1.5, alpha=0.8)
    ax1.plot(sample_data['timestamp'], sample_data['ac_power'], 
             label='AC Power', linewidth=1.5, alpha=0.8)
    ax1.set_title('Power Output (3 Days Sample)', fontweight='bold')
    ax1.set_ylabel('Power (W)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Irradiance vs Power
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(df['irradiance'], df['ac_power'], alpha=0.3, s=5)
    ax2.set_title('Power vs Irradiance', fontweight='bold')
    ax2.set_xlabel('Irradiance (W/m²)')
    ax2.set_ylabel('AC Power (W)')
    ax2.grid(alpha=0.3)
    
    # Plot 3: Temperature vs Efficiency
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(df['module_temperature'], df['panel_efficiency'], alpha=0.3, s=5, c='red')
    ax3.set_title('Efficiency vs Module Temperature', fontweight='bold')
    ax3.set_xlabel('Module Temperature (°C)')
    ax3.set_ylabel('Panel Efficiency (%)')
    ax3.grid(alpha=0.3)
    
    # Plot 4: Daily Energy Profile
    ax4 = plt.subplot(3, 3, 4)
    daily_totals = df.groupby(df['timestamp'].dt.date)['daily_energy'].max()
    ax4.bar(range(len(daily_totals)), daily_totals.values, color='orange', alpha=0.7)
    ax4.set_title('Daily Energy Generation', fontweight='bold')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Energy (kWh)')
    ax4.grid(alpha=0.3, axis='y')
    
    # Plot 5: Hourly Average Power
    ax5 = plt.subplot(3, 3, 5)
    hourly_avg = df.groupby('hour')['ac_power'].mean()
    ax5.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=8)
    ax5.set_title('Average Power by Hour', fontweight='bold')
    ax5.set_xlabel('Hour of Day')
    ax5.set_ylabel('Average AC Power (W)')
    ax5.grid(alpha=0.3)
    ax5.set_xticks(range(0, 24, 2))
    
    # Plot 6: Environmental Conditions
    ax6 = plt.subplot(3, 3, 6)
    sample_env = df.iloc[:96*2]  # 2 days
    ax6_twin = ax6.twinx()
    ax6.plot(sample_env['timestamp'], sample_env['ambient_temperature'], 
             label='Ambient Temp', color='blue', linewidth=1.5)
    ax6_twin.plot(sample_env['timestamp'], sample_env['humidity'], 
                  label='Humidity', color='green', linewidth=1.5)
    ax6.set_title('Environmental Conditions', fontweight='bold')
    ax6.set_ylabel('Temperature (°C)', color='blue')
    ax6_twin.set_ylabel('Humidity (%)', color='green')
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 7: Correlation Heatmap
    ax7 = plt.subplot(3, 3, 7)
    corr_features = ['irradiance', 'ambient_temperature', 'module_temperature', 
                     'humidity', 'panel_efficiency', 'ac_power']
    corr_matrix = df[corr_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax7, cbar_kws={'shrink': 0.8})
    ax7.set_title('Feature Correlation Matrix', fontweight='bold')
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax7.yaxis.get_majorticklabels(), rotation=0)
    
    # Plot 8: Power Distribution
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(df['ac_power'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax8.set_title('AC Power Distribution', fontweight='bold')
    ax8.set_xlabel('AC Power (W)')
    ax8.set_ylabel('Frequency')
    ax8.grid(alpha=0.3, axis='y')
    
    # Plot 9: Efficiency Distribution
    ax9 = plt.subplot(3, 3, 9)
    ax9.hist(df['panel_efficiency'], bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax9.set_title('Panel Efficiency Distribution', fontweight='bold')
    ax9.set_xlabel('Efficiency (%)')
    ax9.set_ylabel('Frequency')
    ax9.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('solar_pv_dataset_overview.png', dpi=300, bbox_inches='tight')
        print("✅ Plots saved as: solar_pv_dataset_overview.png")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    print("="*60)
    print("SOLAR PANEL PV SYSTEM DATASET GENERATOR")
    print("="*60)
    print()
    
    # Generate dataset
    # Options: 7 days, 30 days, 90 days, 365 days
    df = generate_solar_pv_dataset(
        n_days=30,           # Number of days
        interval_minutes=15,  # Measurement interval
        save_csv=True        # Save to CSV
    )
    
    # Display first few rows
    print("\n" + "="*60)
    print("SAMPLE DATA (First 10 rows)")
    print("="*60)
    print(df.head(10).to_string())
    
    # Generate plots
    plot_dataset_overview(df, save_plots=True)
    
    print("\n" + "="*60)
    print("✅ DATASET GENERATION COMPLETE!")
    print("="*60)
