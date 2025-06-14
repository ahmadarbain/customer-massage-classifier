import argparse
from loguru import logger
from app.train import Train_Pipeline
from app.predict import Predictor


def main():
    parser = argparse.ArgumentParser(description="Train or Predict customer message classification")
    parser.add_argument("--predict", action="store_true", help="Jalankan prediksi dari CSV")
    parser.add_argument("--csv", type=str, default="data/question_list.csv", help="Path ke file CSV input")
    parser.add_argument("--out", type=str, default="predicted_results.csv", help="Path file hasil prediksi")
    args = parser.parse_args()

    if args.predict:
        logger.info("Mode prediksi dari CSV")
        Predictor(csv_path=args.csv, save_path=args.out)
    else:
        Train_Pipeline()

if __name__ == "__main__":
    main()
