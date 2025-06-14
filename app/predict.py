import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
from typing import Optional

# Setup logger
Path("logs").mkdir(exist_ok=True)
logger.add("logs/predict.log", rotation="500 KB", retention="7 days", level="INFO")


class _Predictor:
    def __init__(self):
        try:
            self.model = joblib.load("models/final_model_svm.pkl")
            self.vectorizer = joblib.load("models/final_vectorizer.pkl")
            self.label_encoder = joblib.load("models/final_label_encoder.pkl")
            logger.info("Model components loaded.")
        except Exception as e:
            logger.error(f"Failed to load model components: {e}")
            raise

    def run(self, csv_path: str, save_path: str = "predicted_results.csv") -> Optional[str]:
        try:
            df = pd.read_csv(csv_path)

            if 'message' not in df.columns:
                logger.error("File CSV tidak memiliki kolom 'message'.")
                return None

            logger.info(f"Memuat {len(df)} pesan dari: {csv_path}")
            vec = self.vectorizer.transform(df['message'].astype(str))
            predictions = self.model.predict(vec)
            predicted_labels = self.label_encoder.inverse_transform(predictions)
            df['predicted_label'] = predicted_labels

            df.to_csv(save_path, index=False)
            logger.success(f"Prediksi selesai. Hasil disimpan ke: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Gagal memproses prediksi: {e}")
            return None


# Public interface
def Predictor(csv_path: str, save_path: str = "predicted_results.csv") -> Optional[str]:
    return _Predictor().run(csv_path, save_path)
