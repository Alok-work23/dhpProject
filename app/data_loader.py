import os
import pandas as pd
from tqdm import tqdm
from app.sentiment import FinBERTSentiment

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def preprocess_and_save(batch_size=32):
    sentiment_analyzer = FinBERTSentiment()
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    for filename in tqdm(csv_files, desc="Overall Progress", unit="file"):
        file_path = os.path.join(DATA_DIR, filename)

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        if 'message' not in df.columns:
            print(f"Skipped {filename}: No 'message' column.")
            continue

        print(f"\nProcessing: {filename}")
        df['Post'] = df['message'].astype(str)

        sentiments = []
        sentiment_scores = []
        posts = df['Post'].tolist()

        for i in tqdm(range(0, len(posts), batch_size), desc=f"Sentiment ({filename})", unit="batch", leave=False):
            batch = posts[i:i + batch_size]
            try:
                results = sentiment_analyzer.pipeline(batch)
                sentiments.extend([r.get("label", "neutral") for r in results])
                sentiment_scores.extend([r.get("score", 0.0) for r in results])
            except Exception as e:
                print(f"Batch failed at index {i}: {e}")
                sentiments.extend(["neutral"] * len(batch))
                sentiment_scores.extend([0.0] * len(batch))

        df['sentiment'] = sentiments
        df['sentiment_score'] = sentiment_scores

        df.to_csv(file_path, index=False)
        print(f"Processed and saved: {filename}")
