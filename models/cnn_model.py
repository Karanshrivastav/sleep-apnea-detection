# models/cnn_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_cnn_model(input_shape, num_classes, hp=None):
    if hp:
        # Use hyperparameters from Keras Tuner (still supported optionally)
        filters = hp.Int('filters', min_value=16, max_value=128, step=16)
        kernel_size = hp.Choice('kernel_size', [3, 5, 7])
        dense_units = hp.Int('dense_units', 32, 256, step=32)
        dropout_rate = hp.Float('dropout', 0.0, 0.5, step=0.1)
        learning_rate = hp.Choice('lr', [1e-2, 1e-3, 1e-4])
    else:
        # 🔧 Best hyperparameters after tuning
        filters = 96               # Best filter count
        kernel_size = 3            # Best kernel size
        dense_units = 128          # Best dense layer size
        dropout_rate = 0.0         # No dropout performed best
        learning_rate = 1e-3       # Adam optimizer learning rate

    model = Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Flatten(),
        Dense(units=dense_units, activation='relu'),
        Dropout(rate=dropout_rate),
        Dense(units=num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
