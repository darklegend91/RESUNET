import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from resunet import resunet_model  # Importing the ResUNet model from resunet.py

# Data Generation Function
def generate_data(num_samples=20000, num_antennas=4, num_users=2, num_irs_elements=8, noise_std=0.1):
    np.random.seed(42)
    # Rayleigh fading channel
    H = (np.random.randn(num_samples, num_antennas, num_users, num_irs_elements) +
         1j * np.random.randn(num_samples, num_antennas, num_users, num_irs_elements)) / np.sqrt(2)
    H = np.abs(H)  # Take magnitude for simplicity
    noise = noise_std * np.random.randn(num_samples, num_antennas, num_users, num_irs_elements)
    H_noisy = H + noise  # Adding noise
    return H_noisy, H

# Model Compilation and Training
def train_model(model, x_train, y_train, x_val, y_val, epochs=50, batch_size=64):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )
    optimizer = Adam(learning_rate=lr_schedule)
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
    predictions_flat = predictions.reshape(predictions.shape[0], -1)
    y_test_flat = y_test.reshape(y_test.shape[0], -1)

    nmse = np.mean(np.linalg.norm(y_test_flat - predictions_flat, axis=1) ** 2 /
                   np.linalg.norm(y_test_flat, axis=1) ** 2)
    nmse_db = 10 * np.log10(nmse)  # Convert to dB scale
    return nmse_db

# Main Function
if __name__ == "__main__":
    # Generate data
    X_noisy, X_clean = generate_data(num_samples=20000)

    # Split data into training, validation, and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_noisy, X_clean, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

    # Get input shape for model
    input_shape = X_train.shape[1:]

    # Load ResUNet model
    model = resunet_model(input_shape)

    # Train the model
    history = train_model(model, X_train, Y_train, X_val, Y_val, epochs=50, batch_size=64)

    # Evaluate the model on test data
    nmse_values = []
    snr_values = np.linspace(0, 30, 10)  # Wider SNR range
    for snr in snr_values:
        noise_std = 10 ** (-snr / 20)
        X_test_noisy = X_test + noise_std * np.random.randn(*X_test.shape)
        nmse_db = evaluate_model(model, X_test_noisy, Y_test)
        nmse_values.append(nmse_db)
        print(f"SNR: {snr:.1f} dB, NMSE: {nmse_db:.2f} dB")

    # Save and plot results
    results_df = pd.DataFrame({'SNR': snr_values, 'NMSE': nmse_values})
    results_df.to_excel("results.xlsx", index=False)
    print("Results saved successfully!")

    # Plot NMSE vs SNR
    plt.figure()
    plt.plot(snr_values, nmse_values, marker='o', label='NMSE vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE (dB)')
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