import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout
)

# Build architecture manually

model = Sequential([

    Conv1D(
        filters=64,
        kernel_size=5,
        activation='relu',
        input_shape=(250,21)
    ),

    BatchNormalization(),

    MaxPooling1D(pool_size=2),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation='relu'
    ),

    BatchNormalization(),

    MaxPooling1D(pool_size=2),

    Conv1D(
        filters=256,
        kernel_size=3,
        activation='relu'
    ),

    BatchNormalization(),

    MaxPooling1D(pool_size=2),

    Flatten(),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.5),

    Dense(
        1,
        activation='sigmoid'
    )

])

# Build model

model.build((None,250,21))

# Load weights

model.load_weights("model.weights.h5")

print("SUCCESS")

model.summary()