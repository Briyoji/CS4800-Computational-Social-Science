import pandas as pd

# ==== CONFIG ====
CSV_PATH = "data/processed/processed_sentiment_analysis.csv"          # change this
VADER_COL = "vader_label"       # column name in your CSV
BERT_COL  = "bert_label"        # column name in your CSV
# =================

def main():
    # Read CSV
    df = pd.read_csv(CSV_PATH)

    # Basic checks
    if VADER_COL not in df.columns:
        raise ValueError(f"Column '{VADER_COL}' not found in CSV.")

    if BERT_COL not in df.columns:
        raise ValueError(f"Column '{BERT_COL}' not found in CSV.")

    # Print counts
    print("\n=== VADER SENTIMENT COUNTS ===")
    print(df[VADER_COL].value_counts(dropna=False))

    print("\n=== BERT SENTIMENT COUNTS ===")
    print(df[BERT_COL].value_counts(dropna=False))

if __name__ == "__main__":
    main()
