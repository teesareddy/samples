import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)  # For reproducibility

num_records = 1000
date_rng = pd.date_range(start='1/1/2022', end='12/31/2022', freq='H')
sample_dates = np.random.choice(date_rng, num_records, replace=True)
temperature = np.random.uniform(low=0, high=35, size=num_records)  # Temperatures in Celsius
humidity = np.random.uniform(low=10, high=100, size=num_records)   # Humidity in percentage
wind_speed = np.random.uniform(low=0, high=20, size=num_records)   # Wind speed in km/h

data = {
    'date': sample_dates,
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed
}

df = pd.DataFrame(data)
df = df.sort_values(by='date')  # Sort by date for better readability

# Save the dataset to an Excel file
file_path = 'sample_weather_data.xlsx'
df.to_excel(file_path, index=False)

print(f'Sample dataset saved to {file_path}')
