# ============================================================
# CNN IMPORTS
# ============================================================

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (

    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout

)

from tensorflow.keras.optimizers import Adam

print("CNN imports loaded.")




# ============================================================
# BUILD CNN MODEL
# ============================================================

model = Sequential([

    # ========================================================
    # BLOCK 1
    # ========================================================

    Conv1D(

        filters=64,

        kernel_size=5,

        activation='relu',

        input_shape=(250, 21)

    ),

    BatchNormalization(),

    MaxPooling1D(pool_size=2),

    # ========================================================
    # BLOCK 2
    # ========================================================

    Conv1D(

        filters=128,

        kernel_size=3,

        activation='relu'

    ),

    BatchNormalization(),

    MaxPooling1D(pool_size=2),

    # ========================================================
    # BLOCK 3
    # ========================================================

    Conv1D(

        filters=256,

        kernel_size=3,

        activation='relu'

    ),

    BatchNormalization(),

    MaxPooling1D(pool_size=2),

    # ========================================================
    # CLASSIFIER
    # ========================================================

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

# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss='binary_crossentropy',

    metrics=[

        'accuracy',

        tf.keras.metrics.Precision(),

        tf.keras.metrics.Recall(),

        tf.keras.metrics.AUC()

    ]

)

# ============================================================
# SUMMARY
# ============================================================

model.summary()