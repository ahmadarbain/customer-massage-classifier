
# Customer Message Classification using Machine Learning

## Project Overview

This project classifies customer messages into three categories:

- **Information**: General inquiries about products, status, billing, promos, etc.
- **Request**: Service requests such as installation, relocation, reset password, invoice, etc.
- **Problem**: Complaints such as slow connection, no internet, or quota not resetting.

The project uses NLP techniques with TF-IDF and traditional machine learning classifiers.

---

## Methods Used

### 1. Preprocessing
- Lowercasing
- Stopword removal using **Sastrawi**
- TF-IDF vectorization (with unigram & bigram)

### 2. Labeling (Rule-Based)
- Label generation using keyword matching (regex)

### 3. Models Trained
- Logistic Regression
- Random Forest Classifier
- Linear Support Vector Machine (SVM)

### 4. Handling Imbalanced Data
- SMOTE (Synthetic Minority Over-sampling Technique)

### 5. Model Selection
- Model with the **highest macro F1-score** is selected and saved.

---

## Code Flow

```
main.py
├── --train (default): Train the model
├── --predict --csv [file.csv]: Predict labels from a CSV file
```

### Structure:
- `scripts/train_model.py`: Training pipeline, model selection, save model
- `app/predictor.py`: Loads model, predicts labels from CSV
- `data/`: Dataset (raw, train, and test)
- `models/`: Trained model & vectorizer
- `logs/`: Training and prediction logs

---

## Installation

```bash
git clone https://github.com/username/repo-name.git
cd repo-name
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

---

## Training the Model

```bash
python main.py
```

Model will be saved to `/models` as `final_model.pkl`.

---

## Predicting from CSV

```bash
python main.py --predict --csv data/question_list.csv --out hasil_prediksi.csv
```

The prediction will be saved as `hasil_prediksi.csv`.

---

## Project Files

- `main.py`
- `scripts/train_model.py`
- `app/predictor.py`
- `data/question_list.csv`
- `data/train.csv`
- `data/test.csv`
- `models/final_model.pkl`
- `models/final_vectorizer.pkl`
- `models/final_label_encoder.pkl`
- `logs/train.log`
- `logs/predict.log`
- `requirements.txt`
- `README.md`
