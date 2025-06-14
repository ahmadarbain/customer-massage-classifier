import pandas as pd
import re
import joblib
from pathlib import Path
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from loguru import logger

# Setup log folder and logger
Path("logs").mkdir(exist_ok=True)
logger.add("logs/train.log", rotation="1 MB", retention="10 days", level="INFO")


class _TrainModel:
    def __init__(self, data_path="data/question_list.csv"):
        self.data_path = data_path
        self.model = LinearSVC()
        self.vectorizer = TfidfVectorizer(
            stop_words=StopWordRemoverFactory().get_stop_words(),
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.label_encoder = LabelEncoder()

    def label_message(self, msg: str) -> str:
        msg = msg.lower()
        if re.search(r"tidak dapat|modem|ruter|perangkat|kenapa|mengapa|mati|putus|tidak bisa|error|lambat|gangguan|kuota", msg):
            return "Problem"
        elif re.search(r"terminasi|kedatangan teknisi|berhenti|tolong|penggantian paket|invoice|add on|internet|tv cable|reset password|instalasi baru|coverage area|relokasi|ganti|aktifkan|reset|order|kirimkan", msg):
            return "Request"
        elif re.search(r"berapa|info|informasi|status registrasi|pembayaran|promo|billing", msg):
            return "Information"
        else:
            return "Information"

    def run(self, **kwargs) -> Any:
        logger.info("Loading and labeling data...")
        df = pd.read_csv(self.data_path)
        df = df.iloc[1:].copy()
        df.columns = ['message']
        df['message'] = df['message'].str.replace('"', '').str.strip()
        df['label'] = df['message'].apply(self.label_message)

        logger.info("Splitting dataset...")
        X = df['message']
        y = self.label_encoder.fit_transform(df['label'])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # Jika y_train dan y_test berupa angka
        y_train_labels = self.label_encoder.inverse_transform(y_train)
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        
        # Gabungkan kembali X dan y jadi DataFrame
        train_df = pd.DataFrame({'message': X_train, 'label': y_train_labels})
        test_df = pd.DataFrame({'message': X_test, 'label': y_test_labels})

        # Simpan ke file CSV
        train_df.to_csv("data/train.csv", index=False)
        test_df.to_csv("data/test.csv", index=False)

        logger.info("Vectorizing...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        logger.info("Balancing with SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_vec, y_train)

        logger.info("Training and evaluating models...")
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Linear SVM": LinearSVC()
        }

        best_model = None
        best_score = 0
        best_model_name = ""

        for name, model in models.items():
            logger.info(f"🔍 Model: {name}")
            model.fit(X_train_sm, y_train_sm)
            y_pred = model.predict(X_test_vec)
            
            report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
            macro_f1 = f1_score(y_test, y_pred, average='macro')

            logger.info(f"Macro F1-score: {macro_f1:.4f}")
            print(report)

            if macro_f1 > best_score:
                best_score = macro_f1
                best_model = model
                best_model_name = name

        logger.success(f"🏆 Best Model: {best_model_name} with Macro F1-score = {best_score:.4f}")

        # Save only best model
        logger.info("Saving best model artifacts...")
        Path("models").mkdir(exist_ok=True)
        joblib.dump(best_model, "models/final_model.pkl")
        joblib.dump(self.vectorizer, "models/final_vectorizer.pkl")
        joblib.dump(self.label_encoder, "models/final_label_encoder.pkl")

        logger.success("Best model and components saved to /models.")
        return self

def Train_Pipeline(**kwargs) -> _TrainModel:
    return _TrainModel().run(**kwargs)
