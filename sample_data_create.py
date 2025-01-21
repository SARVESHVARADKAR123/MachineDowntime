
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

# Seed for reproducibility
np.random.seed(42)

# Generate 100 rows of data
data = {
    "Machine_ID": range(1, 101),
    "Temperature": np.random.randint(65, 95, size=100),
    "Run_Time": np.random.randint(80, 160, size=100),
}

# Apply downtime rule: Downtime_Flag is 1 if Temperature > 85 or Run_Time > 130
data["Downtime_Flag"] = [
    1 if temp > 85 or runtime > 130 else 0
    for temp, runtime in zip(data["Temperature"], data["Run_Time"])
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Introduce outliers for Temperature and Run_Time
outliers = pd.DataFrame({
    "Machine_ID": range(101, 111),
    "Temperature": [150, 10, 180, 5, 200, 300, -50, 400, 0, 250],  # Extreme temperatures
    "Run_Time": [5, 200, 300, 10, 400, -20, 500, 600, 700, 800],   # Extreme runtimes
    "Downtime_Flag": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Assume all outliers lead to downtime
})

# Append outliers to the dataset
df = pd.concat([df, outliers], ignore_index=True)

# Save to CSV
df.to_csv("enhanced_sample_data.csv", index=False)

print("Dataset with rules and outliers saved to 'enhanced_sample_data.csv'.")
