import numpy as np
import pandas as pd

def generate_data(num_samples=10000, num_antennas=4, num_users=2, num_irs_elements=8, noise_std=0.1):
    np.random.seed(42)
    # Rayleigh fading channel
    H = (np.random.randn(num_samples, num_antennas, num_users, num_irs_elements) +
         1j * np.random.randn(num_samples, num_antennas, num_users, num_irs_elements)) / np.sqrt(2)
    H = np.abs(H)  # Take magnitude for simplicity
    noise = noise_std * np.random.randn(num_samples, num_antennas, num_users, num_irs_elements)
    H_noisy = H + noise  # Adding noise
    return H_noisy, H

def save_data_to_excel(data, filename='generated_data.xlsx'):
    users_data, irs_data = [], []
    for X, Y in data:
        users_data.append(X[:4, :].tolist())  # First 4 rows are Users
        irs_data.append(X[4:, :].tolist())    # Last rows are IRS

    df = pd.DataFrame({'Users': users_data, 'IRS': irs_data})
    df.to_excel(filename, index=False)

# Parameters
num_samples = 10000
num_users = 4
num_antennas = 8
num_irs_elements = 16

data = generate_data(num_samples, num_users, num_antennas, num_irs_elements)
save_data_to_excel(data)

print("Data generation completed. 'generated_data.xlsx' created.")