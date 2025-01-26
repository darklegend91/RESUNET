import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from resunet import resunet_model  # Importing the ResUNet model from resunet.py

# Data Generation Function
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
    # Generate data
    X_noisy, X_clean = generate_data(num_samples=2000)

    # Split data into training, validation, and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_noisy, X_clean, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

    # Get input shape for model
    input_shape = X_train.shape[1:]

    # Load ResUNet model
    model = resunet_model(input_shape)

    # Train the model
    history = train_model(model, X_train, Y_train, X_val, Y_val, epochs=2, batch_size=16)

    # Evaluate the model on test data
    nmse_values = []
    snr_values = np.linspace(0, 30, 10)  # Example SNR values from 0 to 30 dB

    for snr in snr_values:
        noise_std = 10 ** (-snr / 20)  # Convert SNR to noise standard deviation
        X_test_noisy = X_test + noise_std * np.random.randn(*X_test.shape)
        nmse = evaluate_model(model, X_test_noisy, Y_test)
        nmse_values.append(nmse)
        print(f"SNR: {snr:.1f} dB, NMSE: {nmse:.6f}")

    # Ensure consistent lengths before creating DataFrame
    min_length = min(len(nmse_values), len(snr_values))
    nmse_values = nmse_values[:min_length]
    snr_values = snr_values[:min_length]
    epoch_values = list(range(1, min_length + 1))

    # Save results to Excel
    results_df = pd.DataFrame({
        'SNR': snr_values,
        'NMSE': nmse_values,
        'Epoch': epoch_values
    })

    results_df.to_excel("results.xlsx", index=False)
    print("Results saved successfully!")

    # Plot NMSE vs SNR
    plt.figure()
    plt.plot(snr_values, nmse_values, marker='o', label='NMSE vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE')
    plt.title('NMSE vs SNR')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot validation loss vs. test loss over epochs
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()
