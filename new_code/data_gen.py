import numpy as np
import pandas as pd

def generate_data(num_samples, num_users, num_antennas, num_irs_elements):
    data = []
    for _ in range(num_samples):
        # Generate random channel matrices for users and IRS
        H_users = np.random.randn(num_users, num_antennas)  # Shape: (num_users, num_antennas)
        H_irs = np.random.randn(num_irs_elements, num_antennas)  # Shape: (num_irs_elements, num_antennas)
        
        # Concatenate users and IRS data along the rows
        X = np.concatenate((H_users, H_irs), axis=0)  # Shape: (num_users + num_irs_elements, num_antennas)
        
        # Add Gaussian noise to create noisy data
        noise = np.random.randn(*X.shape) * 0.1  # Noise with small variance
        Y = X + noise  # Noisy data (target is noise-free X)

        data.append((X, Y))

    return data

def save_data_to_excel(data, filename='generated_data.xlsx'):
    users_data, irs_data = [], []
    for X, Y in data:
        # Ensure X has at least 4 rows for users and the rest for IRS
        if X.shape[0] < 4:
            raise ValueError("X must have at least 4 rows for users.")
        
        users_data.append(X[:4, :].tolist())  # First 4 rows are Users
        irs_data.append(X[4:, :].tolist())    # Remaining rows are IRS

    # Create a DataFrame and save to Excel
    df = pd.DataFrame({'Users': users_data, 'IRS': irs_data})
    df.to_excel(filename, index=False)

# Parameters
num_samples = 1000
num_users = 4
num_antennas = 8
num_irs_elements = 16

# Generate data
data = generate_data(num_samples, num_users, num_antennas, num_irs_elements)

# Save data to Excel
save_data_to_excel(data)

print("Data generation completed. 'generated_data.xlsx' created.")