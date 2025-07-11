
# 🩺 Sleep Apnea Detection using Deep Learning

This project implements a deep learning pipeline to classify different types of sleep apnea events from respiratory signal features. It leverages participant-specific data and uses **leave-one-participant-out cross-validation** to ensure robust generalization across unseen individuals.

---

## 📌 Problem Statement

Sleep apnea is a sleep disorder characterized by repeated interruptions in breathing. The objective of this project is to build a model that classifies **respiratory signal segments** into the following three categories:

- `Normal`
- `Hypopnea` (partial obstruction of airflow)
- `Obstructive Apnea` (complete blockage)

We aim to automate detection using deep learning techniques and evaluate how well models generalize across different individuals.

---

## 📦 Dataset

- **Source**: [breathing_dataset.csv](dataset/breathing_dataset.csv)
- **Shape**: Each row contains:
  - Participant ID (`participant_id`)
  - Label (`label`: Normal, Hypopnea, or Obstructive Apnea)
  - Extracted features (`feature_1`, `feature_2`, ..., `feature_N`)


Each participant's data includes time-series signals recorded during overnight sleep:

- **Nasal Airflow** (sampled at 32 Hz)
- **Thoracic/Abdominal Movement** (32 Hz)
- **SpO₂** (4 Hz)
- **Event Annotations** including start and end times for apnea and hypopnea episodes

Participants: `AP01` to `AP05`

- **Preprocessing**:
  - Label encoding
  - Feature normalization
  - Leave-one-subject-out (LOSO) splitting

The dataset is preprocessed into features saved in `breathing_dataset.csv` containing:
- `feature_0` to `feature_n`
- `label`: target class
- `participant_id`: ID of the person


---

## 📉 Signal Visualization

To aid clinical interpretation, we provide visual plots of 8-hour recordings with event overlays using `scripts/vis.py`.

### Features:
- 📈 Plots for:
  - Nasal Airflow
  - Thoracic Movement
  - SpO₂
- 🚩 Annotated overlays for `Apnea`, `Hypopnea`, etc.
- 🧾 Output: Exported as **multipage PDFs** (5-minute windows per page)
- 📂 Example output saved to:

### Run Command:
```bash
python -m scripts.vis -name data/AP01
```
---

## 🧠 Models Used

### ✅ 1. CNN (Convolutional Neural Network)
- 1D Conv Layers
- Batch Normalization
- Dropout
- Dense Softmax Output
- Tuned for better class balance performance

### ✅ 2. ConvLSTM (Exploratory)
- Time-distributed convolution
- LSTM temporal modeling
- Required 5D reshaping of input
- Did not significantly outperform CNN

---

## 🔁 Evaluation Strategy

We use **Leave-One-Participant-Out Cross-Validation** (LOPO-CV), which ensures that models are trained on all but one subject and evaluated on the held-out subject. This simulates real-world performance on unseen users.

---
## 🛠️ Hyperparameter Tuning (`tune_cnn.py`)

To optimize model performance, we implemented a dedicated script `tune_cnn.py` for hyperparameter tuning of the CNN model. This script uses a **grid search strategy** combined with **Leave-One-Participant-Out (LOPO) cross-validation** to identify the best combination of:

- Number of convolutional filters
- Kernel sizes
- Dropout rates
- Dense layer units
- Learning rate

Each combination is evaluated based on the **macro F1-score**, which ensures fair weighting across imbalanced classes. The best configuration is then passed to the `cnn_model.py` architecture and used for training in `train_model.py`.

### ✅ Best Outcome:
The tuned CNN significantly improved performance:
- Stronger F1-score for the dominant `Normal` class.
- Moderate gains in `Hypopnea` detection.
- Maintains generalizability across unseen participants.

### ▶️ How to Run Tuning:

```bash
python -m scripts.tune_cnn
```

## 📈 Results (Tuned CNN Model)

The best performance was obtained using the **tuned CNN** architecture. Below are aggregated results across all folds:

| Class               | Mean F1 Score | Std Dev |
|--------------------|---------------|---------|
| **Normal**          | 0.864         | 0.094   |
| **Hypopnea**        | 0.088         | 0.089   |
| **Obstructive Apnea** | 0.022       | 0.044   |

- **Best Accuracy**: 94.8% (Fold: AP01)
- **Challenges**: The model struggles with **Hypopnea** and **Obstructive Apnea** due to severe class imbalance.

---

## 🧪 Example Confusion Matrix (Fold: AP05)

```
                   Pred_Hypopnea  Pred_Normal  Pred_Obstructive
True_Hypopnea            50          119             13
True_Normal             218         1004             34
True_Obstructive         37           95             11
```

---

## 🔧 Future Improvements

- 📊 Use **focal loss** or **class-weighted loss** to tackle imbalance
- 🔁 Try **oversampling (SMOTE or ADASYN)**
- 🧠 Implement **transformer-based models**

---

## 🚀 How to Run

```bash
# Train with CNN
python -m scripts.train_model --model cnn

# Train with ConvLSTM
python -m scripts.train_model --model convlstm
```

---

## 📁 Project Structure

```
.
│   README.md
│   requirements.txt
│
├───data
│   ├───AP01
│   │       Flow - 30-05-2024.txt
│   │       Flow Events - 30-05-2024.txt
│   │       Sleep profile - 30-05-2024.txt
│   │       SPO2 - 30-05-2024.txt
│   │       Thorac - 30-05-2024.txt
│   │
│   ├───AP02
│   │       Flow  - 30.05.2024.txt
│   │       Flow Events  - 30.05.2024.txt
│   │       Sleep profile  - 30.05.2024.txt
│   │       SPO2  - 30.05.2024.txt
│   │       Thorac  - 30.05.2024.txt
│   │
│   ├───AP03
│   │       Flow - 29_05_2024.txt
│   │       Flow Events - 29_05_2024.txt
│   │       Sleep profile - 29_05_2024.txt
│   │       SPO2 - 29_05_2024.txt
│   │       Thorac - 29_05_2024.txt
│   │
│   ├───AP04
│   │       Flow Events - 29.05.2024.txt
│   │       Flow Signal - 29.05.2024.txt
│   │       Sleep profile - 29.05.2024.txt
│   │       SPO2 Signal - 29.05.2024.txt
│   │       Thorac Signal - 29.05.2024.txt
│   │
│   └───AP05
│           Flow Events - 28.05.2024.txt
│           Flow Nasal - 28.05.2024.txt
│           Sleep profile - 28.05.2024.txt
│           SPO2 - 28.05.2024.txt
│           Thorac Movement - 28.05.2024.txt
│
├───dataset
│       breathing_dataset.csv
│       sleep_stage_dataset.csv
│
├───models
│   │   cnn_model.py
│   │   conv_lstm_model.py
│   │   __init__.py
│   │
│   └───__pycache__
│           cnn_model.cpython-311.pyc
│           conv_lstm_model.cpython-311.pyc
│           __init__.cpython-311.pyc
│
├───scripts
│   │   cnn_model.py
│   │   create_dataset.py
│   │   train_model.py
│   │   tune_cnn.py
│   │   vis.py
│   │
│   └───__pycache__
│           train_model.cpython-311.pyc
│           tune_cnn.cpython-311.pyc
│           vis.cpython-311.pyc
│
├───tuner_logs
│   └───cnn_tuning
│
└───visualizations
        AP01_visualization.pdf
        AP02_visualization.pdf
        AP03_visualization.pdf
        AP04_visualization.pdf
```

---
## ⚙️ Environment Setup

To replicate this project, follow these steps:

### 1. Create and Activate the Virtual Environment

Create a virtual environment named `health_sensing` and activate it:

```bash
# Create environment
python -m venv health_sensing

# Activate environment (Windows)
.\health_sensing\Scripts\activate

# OR (Linux/macOS)
source health_sensing/bin/activate

```
### 2. Install Dependencies
Make sure you’re in the activated environment, then install all required packages using:

```bash
pip install -r requirements.txt
```
## 👨‍💻 Author

Karan Shrivastava

---

## 📜 License

This project is for academic/research purposes only.
