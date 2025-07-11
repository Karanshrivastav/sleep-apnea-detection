# models/cnn_model.py
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, InputLayer

# def build_cnn_model(input_shape, num_classes):
#     model = Sequential([
#         InputLayer(input_shape=input_shape),
#         Conv1D(64, kernel_size=5, activation='relu', padding='same'),
#         BatchNormalization(),
#         MaxPooling1D(pool_size=2),
#         Dropout(0.3),

#         Conv1D(128, kernel_size=5, activation='relu', padding='same'),
#         BatchNormalization(),
#         MaxPooling1D(pool_size=2),
#         Dropout(0.3),

#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.3),
#         Dense(num_classes, activation='softmax')
#     ])

#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# models/cnn_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_cnn_model(input_shape, num_classes, hp=None):
    if hp:
        # Use hyperparameters from Keras Tuner
        filters = hp.Int('filters', min_value=16, max_value=128, step=16)
        kernel_size = hp.Choice('kernel_size', [3, 5, 7])
        dense_units = hp.Int('dense_units', 32, 256, step=32)
        dropout_rate = hp.Float('dropout', 0.0, 0.5, step=0.1)
        learning_rate = hp.Choice('lr', [1e-2, 1e-3, 1e-4])
    else:
        # Default config
        filters = 64
        kernel_size = 3
        dense_units = 128
        dropout_rate = 0.3
        learning_rate = 1e-3

    model = Sequential([
        Conv1D(filters, kernel_size, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        BatchNormalization(),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
