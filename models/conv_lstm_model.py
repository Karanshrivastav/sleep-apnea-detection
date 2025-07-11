# models/conv_lstm_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense, Dropout, InputLayer

def build_convlstm_model(input_shape, num_classes):
    model = Sequential([
        ConvLSTM2D(filters=32, kernel_size=(3, 1), activation='relu',
                   input_shape=input_shape, return_sequences=False),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model