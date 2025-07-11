# scripts/train_model.py
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

from models.cnn_model import build_cnn_model
from models.conv_lstm_model import build_convlstm_model

from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


def preprocess_data(df):
    label_enc = LabelEncoder()
    df['label_encoded'] = label_enc.fit_transform(df['label'])
    class_names = label_enc.classes_

    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values.astype(np.float32)
    y = df['label_encoded'].values.astype(int)
    
    # Normalize input
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    return X, y, df['participant_id'].values, class_names


def evaluate_model(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm


def print_report(metrics, cm, class_names):
    print("\nClassification Report:")
    print(pd.DataFrame(metrics).transpose())
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))


def leave_one_participant_out(X, y, participant_ids, model_type, class_names):
    unique_participants = np.unique(participant_ids)
    fold_metrics = []

    for participant in unique_participants:
        print(f"\n📊 Fold: Leaving out {participant}")

        train_idx = participant_ids != participant
        test_idx = participant_ids == participant

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = {i: w for i, w in enumerate(class_weights)}

        num_classes = len(class_names)
        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_test_cat = to_categorical(y_test, num_classes=num_classes)

        input_shape = X_train.shape[1:]

        if model_type == 'cnn':
            model = build_cnn_model(input_shape=(input_shape[0], 1), num_classes=num_classes)
            X_train = X_train[..., np.newaxis]
            X_test = X_test[..., np.newaxis]

        elif model_type == 'convlstm':
            # Choose appropriate time_steps and spatial dims
            time_steps = 5
            step_length = input_shape[0] // time_steps  # should be divisible
            X_train = X_train.reshape((-1, time_steps, step_length, 1, 1))  # 5D
            X_test = X_test.reshape((-1, time_steps, step_length, 1, 1))
            model = build_convlstm_model(input_shape=(time_steps, step_length, 1, 1), num_classes=num_classes)

        else:
            raise ValueError("Unsupported model type")

        model.fit(
            X_train, y_train_cat,
            epochs=30,
            batch_size=32,
            validation_split=0.1,
            class_weight=class_weights,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )

        y_pred = np.argmax(model.predict(X_test), axis=1)
        report, cm = evaluate_model(y_test, y_pred, class_names)

        print_report(report, cm, class_names)
        fold_metrics.append(report)

    # Aggregate
    print("\n🔎 Average Metrics Across Folds:")
    avg_metrics = defaultdict(list)
    for fold in fold_metrics:
        for k, v in fold.items():
            if isinstance(v, dict):
                avg_metrics[k].append(v['f1-score'])

    for label in class_names:
        scores = avg_metrics[label]
        print(f"{label}: Mean F1 = {np.mean(scores):.3f}, Std = {np.std(scores):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['cnn', 'convlstm'], required=True)
    parser.add_argument('--data_path', type=str, default='dataset/breathing_dataset.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    X, y, participant_ids, class_names = preprocess_data(df)
    leave_one_participant_out(X, y, participant_ids, args.model, class_names)


if __name__ == '__main__':
    main()
