import pandas as pd
import numpy as np
from kerastuner.tuners import RandomSearch
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from models.cnn_model import build_cnn_model

def preprocess(data_path):
    df = pd.read_csv(data_path)
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    X = df[[col for col in df.columns if col.startswith("feature_")]].values.astype(np.float32)
    y = df['label_encoded'].values.astype(int)

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    X = X[..., np.newaxis]  # reshape for Conv1D

    y_cat = to_categorical(y, num_classes=len(np.unique(y)))
    return train_test_split(X, y_cat, test_size=0.2, random_state=42)

def model_builder(hp):
    return build_cnn_model(input_shape=(X_train.shape[1], 1), num_classes=y_train.shape[1], hp=hp)

# Load data
data_path = "dataset/breathing_dataset.csv"
X_train, X_val, y_train, y_val = preprocess(data_path)

# Tuner
tuner = RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='tuner_logs',
    project_name='cnn_tuning'
)

tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=1)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

