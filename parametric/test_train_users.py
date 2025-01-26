import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from resunet import resunet_model  # Importing the ResUNet model from resunet.py

# Data Generation Function (with parametric variation for number of users)
def generate_data(num_samples=1000, num_antennas=4, num_users=2, num_irs_elements=8, noise_std=0.1):
    np.random.seed(42)
    H = np.random.randn(num_samples, num_antennas, num_users, num_irs_elements)  # Channel matrix
    noise = noise_std * np.random.randn(num_samples, num_antennas, num_users, num_irs_elements)
    H_noisy = H + noise  # Adding noise
    return H_noisy, H

# Model Compilation and Training
def train_model(model, x_train, y_train, x_val, y_val, epochs=2, batch_size=32):
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history

# Evaluation Function
def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)

    print(f"Shape of predictions: {predictions.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    if predictions.shape[-1] != y_test.shape[-1]:
        # Reshape ground truth to match predicted output channels if needed
        y_test = y_test[..., :predictions.shape[-1]]

    predictions_flat = predictions.reshape(predictions.shape[0], -1)
    y_test_flat = y_test.reshape(y_test.shape[0], -1)

    nmse = np.mean(np.linalg.norm(y_test_flat - predictions_flat, axis=1) ** 2 /
                   np.linalg.norm(y_test_flat, axis=1) ** 2)
    return nmse

# Main Function
if __name__ == "__main__":
    # Number of users configurations for parametric variation
    user_configurations = [2,4,6]
    
    # Initialize results dictionary
    nmse_values_dict = {num_users: [] for num_users in user_configurations}
    epoch_values_dict = {num_users: [] for num_users in user_configurations}
    
    # Generate data (using 8 antennas and 8 IRS elements as constants)
    X_noisy, X_clean = generate_data(num_samples=2000, num_antennas=8, num_irs_elements=8)
    
    # Loop over number of users configurations
    for num_users in user_configurations:
        # Update data based on the current number of users
        X_noisy, X_clean = generate_data(num_samples=2000, num_antennas=8, num_users=num_users, num_irs_elements=8)
        
        # Split data into training, validation, and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X_noisy, X_clean, test_size=0.2, random_state=42)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

        # Get input shape for model
        input_shape = X_train.shape[1:]

        # Load ResUNet model
        model = resunet_model(input_shape)

        # Train the model
        history = train_model(model, X_train, Y_train, X_val, Y_val, epochs=50, batch_size=16)

        # Track NMSE over epochs
        nmse_per_epoch = []
        for epoch in range(50):  # Number of epochs trained
            nmse = evaluate_model(model, X_test, Y_test)
            nmse_per_epoch.append(nmse)
        
        # Store the results for this number of users configuration
        nmse_values_dict[num_users] = nmse_per_epoch
        epoch_values_dict[num_users] = list(range(1, 51))  # Epoch values

    # Save results to Excel
    results_df = pd.DataFrame()

    for num_users in user_configurations:
        results_df[f'NMSE_{num_users}_users'] = nmse_values_dict[num_users]
        results_df['Epoch'] = epoch_values_dict[num_users]
    
    results_df.to_excel("parametric_results_users.xlsx", index=False)
    print("Results saved successfully!")

    # Plot NMSE vs. Epoch for all user configurations
    plt.figure(figsize=(10, 6))
    for num_users in user_configurations:
        plt.plot(epoch_values_dict[num_users], nmse_values_dict[num_users], label=f'{num_users} users')

    plt.xlabel('Epoch')
    plt.ylabel('NMSE')
    plt.title('NMSE vs. Epoch for Different Number of Users Configurations')
    plt.legend()
    plt.grid()
    plt.show()
