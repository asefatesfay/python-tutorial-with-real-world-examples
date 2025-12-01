import numpy as np
import pandas as pd

# --- 1. Create the Sample Dataset (using NumPy arrays this time for easier math) ---
# Data is structured as: [ [Age1, Income1], [Age2, Income2], ... ]
data = np.array([
    [30, 60000],
    [45, 90000],
    [25, 40000],
    [50, 120000],
    [35, 75000]
])

print("Original Data (NumPy Array):")
print(data)
print("-" * 40)

# --- 2. Calculate the Mean (mu) and Standard Deviation (sigma) for each feature ---

# NumPy can calculate the mean and std deviation along an axis.
# axis=0 means "calculate across the rows" for each column independently.

# Calculate the mean for Age (column 0) and Income (column 1)
means = np.mean(data, axis=0)

# Calculate the standard deviation for Age and Income
# NOTE: np.std calculates population std dev by default.
# Scikit-learn's StandardScaler uses sample std dev (ddof=1) for consistency in stats.
# We'll use ddof=0 for simplicity here unless specified otherwise.
std_devs = np.std(data, axis=0)

print(f"Mean of features (Age, Income): {means}")
print(f"Std Dev of features (Age, Income): {std_devs}")
print("-" * 40)


# --- 3. Apply the Standardization Formula Manually ---

# We define a function to apply the Z-score formula
def manual_standardize(data_array, means_array, std_devs_array):
    # The formula can be applied directly to the entire NumPy array because
    # NumPy supports element-wise operations (broadcasting).
    scaled_data = (data_array - means_array) / std_devs_array
    return scaled_data

# Run the function
data_scaled_manual = manual_standardize(data, means, std_devs)


# --- 4. Display the Results in a nice Pandas DataFrame ---
scaled_df = pd.DataFrame(data_scaled_manual, columns=['Age Scaled (Z)', 'Income Scaled (Z)'])

print("Manually Scaled Data (Standardized):")
print(scaled_df)
print("-" * 40)

# --- 5. Verification (Check mean is approx 0 and std dev is approx 1) ---
print("Verification Statistics:")
print(f"Mean of Scaled Age: {scaled_df['Age Scaled (Z)'].mean():.4f}")
print(f"Std Dev of Scaled Age: {scaled_df['Age Scaled (Z)'].std():.4f}")
print(f"Mean of Scaled Income: {scaled_df['Income Scaled (Z)'].mean():.4f}")
print(f"Std Dev of Scaled Income: {scaled_df['Income Scaled (Z)'].std():.4f}")
