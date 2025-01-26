import numpy as np
import pandas as pd

def generate_data(num_samples, num_users, num_antennas, num_irs_elements):
    data = []
    for _ in range(num_samples):
        H_users = np.random.randn(num_users, num_antennas)
        H_irs = np.random.randn(num_irs_elements, num_antennas)
        
        X = np.concatenate((H_users, H_irs), axis=0)
        noise = np.random.randn(*X.shape) * 0.1  # Noise with small variance
        Y = X + noise  # Noisy data (target is noise-free X)

        data.append((X, Y))

    return data

def save_data_to_excel(data, filename='generated_data.xlsx'):
    users_data, irs_data = [], []
    for X, Y in data:
        users_data.append(X[:4, :].tolist())  # First 4 rows are Users
        irs_data.append(X[4:, :].tolist())    # Last rows are IRS

    df = pd.DataFrame({'Users': users_data, 'IRS': irs_data})
    df.to_excel(filename, index=False)

# Parameters
num_samples = 1000
num_users = 4
num_antennas = 8
num_irs_elements = 16

data = generate_data(num_samples, num_users, num_antennas, num_irs_elements)
save_data_to_excel(data)

print("Data generation completed. 'generated_data.xlsx' created.")
