import tensorflow as tf
from tensorflow.keras import layers, models

# Residual block definition
def residual_block(x, filters):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(x)

    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)

    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

# ResUNet Model Definition
def resunet_model(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder (Downsampling path)
    c1 = residual_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2), padding='same')(c1)

    c2 = residual_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2), padding='same')(c2)

    c3 = residual_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2), padding='same')(c3)

    c4 = residual_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2), padding='same')(c4)

    # Bottleneck
    c5 = residual_block(p4, 1024)

    # Decoder (Upsampling path)
    u4 = layers.UpSampling2D((2, 2))(c5)
    u4 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(u4)
    u4 = layers.Resizing(height=c4.shape[1], width=c4.shape[2])(u4)  # Resize to match encoder layer
    u4 = layers.Concatenate()([u4, c4])
    c6 = residual_block(u4, 512)

    u3 = layers.UpSampling2D((2, 2))(c6)
    u3 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(u3)
    u3 = layers.Resizing(height=c3.shape[1], width=c3.shape[2])(u3)  # Resize to match encoder layer
    u3 = layers.Concatenate()([u3, c3])
    c7 = residual_block(u3, 256)

    u2 = layers.UpSampling2D((2, 2))(c7)
    u2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(u2)
    u2 = layers.Resizing(height=c2.shape[1], width=c2.shape[2])(u2)  # Resize to match encoder layer
    u2 = layers.Concatenate()([u2, c2])
    c8 = residual_block(u2, 128)

    u1 = layers.UpSampling2D((2, 2))(c8)
    u1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(u1)
    u1 = layers.Resizing(height=c1.shape[1], width=c1.shape[2])(u1)  # Resize to match encoder layer
    u1 = layers.Concatenate()([u1, c1])
    c9 = residual_block(u1, 64)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='linear', padding='same')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Define input shape based on the generated data
input_shape = (20, 8, 1)  # 20 rows (4 users + 16 IRS elements), 8 antennas, 1 channel

# Create the model
model = resunet_model(input_shape)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()
