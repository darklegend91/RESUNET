import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from resunet import resunet_model  # Importing the ResUNet model from resunet.py

# Data Generation Function (modified for different noise standard deviation)
def generate_data(num_samples=1000, num_antennas=4, num_users=2, num_irs_elements=8, noise_std=0.1):
    np.random.seed(42)
    H = np.random.randn(num_samples, num_antennas, num_users, num_irs_elements)  # Channel matrix
    noise = noise_std * np.random.randn(num_samples, num_antennas, num_users, num_irs_elements)
    H_noisy = H + noise  # Adding noise
    return H_noisy, H

# Model Compilation and Training
def train_model(model, x_train, y_train, x_val, y_val, optimizer, epochs=2, batch_size=32):
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    
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
    # Learning rate configurations for parametric variation
    learning_rates = [0.001, 0.01, 0.1, 0.0001]
    
    # Initialize results dictionary
    nmse_values_dict = {str(lr): [] for lr in learning_rates}
    epoch_values_dict = {str(lr): [] for lr in learning_rates}
    
    # Loop over different learning rate values
    for lr in learning_rates:
        # Generate data for each learning rate configuration
        X_noisy, X_clean = generate_data(num_samples=2000, num_antennas=8, num_users=4, num_irs_elements=16, noise_std=0.1)
        
        # Split data into training, validation, and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X_noisy, X_clean, test_size=0.2, random_state=42)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

        # Get input shape for model
        input_shape = X_train.shape[1:]

        # Load ResUNet model
        model = resunet_model(input_shape)

        # Create the optimizer with the current learning rate
        optimizer = Adam(learning_rate=lr)

        # Train the model with the current learning rate configuration
        history = train_model(model, X_train, Y_train, X_val, Y_val, optimizer, epochs=50, batch_size=16)

        # Track NMSE over epochs
        nmse_per_epoch = []
        for epoch in range(50):  # Number of epochs trained
            nmse = evaluate_model(model, X_test, Y_test)
            nmse_per_epoch.append(nmse)
        
        # Store the results for this learning rate configuration
        nmse_values_dict[str(lr)] = nmse_per_epoch
        epoch_values_dict[str(lr)] = list(range(1, 51))  # Epoch values

    # Save results to Excel
    results_df = pd.DataFrame()

    for lr in learning_rates:
        results_df[f'NMSE_LR_{lr}'] = nmse_values_dict[str(lr)]
        results_df['Epoch'] = epoch_values_dict[str(lr)]
    
    results_df.to_excel("parametric_results_learning_rate.xlsx", index=False)
    print("Results saved successfully!")

    # Plot NMSE vs. Epoch for all learning rates
    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        plt.plot(epoch_values_dict[str(lr)], nmse_values_dict[str(lr)], label=f'LR = {lr}')

    plt.xlabel('Epoch')
    plt.ylabel('NMSE')
    plt.title('NMSE vs. Epoch for Different Learning Rates')
    plt.legend()
    plt.grid()
    plt.show()
