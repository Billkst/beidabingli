import os
import json
import joblib
import pandas as pd

MODEL_DIR = '/home/UserData/ljx/beidabingli/models'

SHORT_MODEL = os.path.join(MODEL_DIR, 'short_term_xgb.joblib')
LONG_MODEL = os.path.join(MODEL_DIR, 'long_term_xgb.joblib')
SHORT_META = os.path.join(MODEL_DIR, 'short_term_xgb_meta.json')
LONG_META = os.path.join(MODEL_DIR, 'long_term_xgb_meta.json')


def load_meta(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def predict(input_csv: str, output_csv: str):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(input_csv)

    short_meta = load_meta(SHORT_META)
    long_meta = load_meta(LONG_META)

    df = pd.read_csv(input_csv)

    # Ensure required columns exist
    for col in short_meta['features']:
        if col not in df.columns:
            df[col] = pd.NA
    for col in long_meta['features']:
        if col not in df.columns:
            df[col] = pd.NA

    short_model = joblib.load(SHORT_MODEL)
    long_model = joblib.load(LONG_MODEL)

    short_probs = short_model.predict_proba(df[short_meta['features']])[:, 1]
    long_probs = long_model.predict_proba(df[long_meta['features']])[:, 1]

    out = df.copy()
    out['pred_label1_prob'] = short_probs
    out['pred_label2_prob'] = long_probs

    out.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict short/long-term response probabilities')
    parser.add_argument('--input', required=True, help='CSV with clinical features')
    parser.add_argument('--output', required=True, help='CSV to save predictions')
    args = parser.parse_args()

    predict(args.input, args.output)
