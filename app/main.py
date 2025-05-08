from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pathlib import Path
import os
import pandas as pd
import logging
import re
from google_news_scraper import fetch_links_from_google_news
from sentiment import FinBERTSentiment

# Configuration and Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
logger.info(f"Data directory is {DATA_DIR}")

# Securely load Gemini API Key
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

COMPANY_MAPPING = {
    "hdfc": "HDFC",
    "reliance": "Reliance",
    "tcs": "TCS",
    "infosys": "Infosys",
    "icici": "ICICI",
    "adani": "Adani Enterprises",
    "lnt": "Larsen & Toubro",
    "kotak": "Kotak Mahindra",
    "hul": "Hindustan Unilever",
    "itc": "ITC"
}

def get_company_filename(company_name: str) -> str:
    company_name = company_name.lower()
    return COMPANY_MAPPING.get(company_name, company_name)

def load_company_data(company: str) -> pd.DataFrame:
    file_path = DATA_DIR / f"{company}_Forum_data.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Company not found")
    return pd.read_csv(file_path)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/sentiment-summary/{company_name}")
def sentiment_summary(company_name: str):
    company = get_company_filename(company_name)
    df = load_company_data(company)

    sentiment_counts = df["sentiment"].value_counts().to_dict()
    return sentiment_counts

@app.get("/sentiment-trend/{company_name}")
def sentiment_trend(company_name: str, year: int = None, month: int = None):
    company = get_company_filename(company_name)
    df = load_company_data(company)

    df["date"] = pd.to_datetime(df["date"])
    if year:
        df = df[df["date"].dt.year == year]
    if month:
        df = df[df["date"].dt.month == month]

    trend = df.groupby(df["date"].dt.date)["sentiment"].value_counts().unstack().fillna(0).astype(int)
    return trend.reset_index().to_dict(orient="records")

@app.get("/price-vs-sentiment/{company_name}")
def price_vs_sentiment(company_name: str):
    company = get_company_filename(company_name)
    df = load_company_data(company)

    if "NSE_price_change" not in df.columns:
        raise HTTPException(status_code=404, detail="Price change data not available")

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["NSE_price_change", "sentiment_score"])
    df_grouped = df.groupby(df["date"].dt.date).agg({"sentiment_score": "mean", "NSE_price_change": "mean"}).reset_index()
    return df_grouped.to_dict(orient="records")

@app.get("/volume-insights/{company_name}")
def volume_insights(company_name: str):
    company = get_company_filename(company_name)
    df = load_company_data(company)

    df["date"] = pd.to_datetime(df["date"])
    volume = df.groupby(df["date"].dt.date).size().reset_index(name="message_volume")
    return volume.to_dict(orient="records")

@app.get("/top-keywords/{company_name}")
def top_keywords(company_name: str):
    company = get_company_filename(company_name)
    df = load_company_data(company)

    all_text = " ".join(df["text"].astype(str).tolist()).lower()
    words = re.findall(r"\b\w{5,}\b", all_text)
    word_freq = pd.Series(words).value_counts().head(20).to_dict()
    return word_freq

@app.get("/compare-companies")
def compare_companies():
    result = []
    for short_name, company in COMPANY_MAPPING.items():
        try:
            df = load_company_data(company)
            sentiment_counts = df["sentiment"].value_counts().to_dict()
            sentiment_counts["company"] = company
            result.append(sentiment_counts)
        except Exception as e:
            continue
    return result

@app.get("/scrape-news/{company_name}")
def scrape_news(company_name: str):
    try:
        links = fetch_links_from_google_news(company_name)
        return {"company": company_name, "links": links}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze-sentiment")
def analyze_sentiment(text: str):
    try:
        sentiment_analyzer = FinBERTSentiment()
        result = sentiment_analyzer.predict(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
