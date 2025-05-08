from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import List
import os
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
import re
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import json
import logging
import numpy as np
from pathlib import Path

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Paths ---
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
logger.info(f"Data directory: {DATA_DIR}")

# --- FastAPI Initialization ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Lazy Load Functions ---
def get_sentiment_model():
    from app.sentiment import FinBERTSentiment
    return FinBERTSentiment()

def get_gemini_model():
    genai.configure(api_key="AIzaSyDSc67L2N1KCVH7MNFL6qONMQpNqE760A8")
    return genai.GenerativeModel('gemini-2.0-flash')

# --- Utility Functions ---
def get_news_links(keyword: str, date: str, num_results: int = 3):
    try:
        date_str = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
        query = f"{keyword} stock {date_str}"
        url = f"https://news.google.com/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []
        for article in soup.find_all('article')[:num_results]:
            a_tag = article.find('a', href=True)
            if a_tag:
                href = a_tag['href']
                if href.startswith('./'):
                    href = f"https://news.google.com{href[1:]}"
                links.append({"title": a_tag.get_text().strip(), "url": href})
        return links
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []

@app.get("/")
def home():
    return {"message": "Stock Vibe API is running"}

@app.get("/get-news/")
async def get_news(company: str, date: str):
    return {"date": date, "news": get_news_links(company, date)}

@app.get("/overall-sentiment/{company}")
def overall_sentiment(company: str):
    path = DATA_DIR / f"{company}_Forum_data.csv"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "Company not found"})
    try:
        df = pd.read_csv(path, usecols=['sentiment', 'sentiment_score'])
        df['net'] = df.apply(lambda r: r['sentiment_score'] if r['sentiment'] == 'positive' else -r['sentiment_score'] if r['sentiment'] == 'negative' else 0, axis=1)
        return {
            "net_sentiment_score": round(df['net'].mean(), 3),
            "sentiment_counts": df['sentiment'].value_counts().to_dict()
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/sentiment-trend/{company}")
def sentiment_trend(company: str, year: int = None, month: int = None, start_date: str = None, end_date: str = None):
    path = DATA_DIR / f"{company}_Forum_data.csv"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "Company not found"})
    try:
        df = pd.read_csv(path, usecols=['Date', 'sentiment'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        if year:
            df = df[df['Date'].dt.year == year]
            if month:
                df = df[df['Date'].dt.month == month]
        if start_date and end_date:
            df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
        trend = df.groupby([df['Date'].dt.date, 'sentiment']).size().unstack(fill_value=0).reset_index()
        trend.columns.name = None
        return trend.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/price-vs-sentiment/{company}")
def price_vs_sentiment(company: str):
    path = DATA_DIR / f"{company}_Forum_data.csv"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "Company not found"})
    try:
        df = pd.read_csv(path, usecols=['Date', 'sentiment', 'sentiment_score', 'NSE_current_price'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['net'] = df.apply(lambda r: r['sentiment_score'] if r['sentiment'] == 'positive' else -r['sentiment_score'] if r['sentiment'] == 'negative' else 0, axis=1)
        summary = df.groupby(df['Date'].dt.date).agg(avg_net_sentiment=('net', 'mean'), avg_current_price=('NSE_current_price', 'mean')).reset_index()
        summary['date'] = summary['Date'].astype(str)
        return summary[['date', 'avg_net_sentiment', 'avg_current_price']].to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/compare-companies")
def compare_companies(companies: List[str] = Query(...)):
    results = []
    for company in companies:
        path = DATA_DIR / f"{company}_Forum_data.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=['sentiment', 'sentiment_score'])
            df['net'] = df.apply(lambda r: r['sentiment_score'] if r['sentiment'] == 'positive' else -r['sentiment_score'] if r['sentiment'] == 'negative' else 0, axis=1)
            results.append({"company": company, "average_net_sentiment": round(df['net'].mean(), 3)})
        except:
            continue
    if not results:
        raise HTTPException(status_code=404, detail="No valid company data")
    return results

@app.get("/message-volume/{company}")
def message_volume(company: str, start_date: str = None, end_date: str = None):
    path = DATA_DIR / f"{company}_Forum_data.csv"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "Company not found"})
    try:
        df = pd.read_csv(path, usecols=['Date', 'message'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        if start_date and end_date:
            df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
        result = df.groupby(df['Date'].dt.date).agg(message_count=('Date', 'count'), messages=('message', list)).reset_index()
        result['date'] = result['Date'].astype(str)
        return result[['date', 'message_count', 'messages']].to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/company-files")
def get_company_files():
    return [f.name for f in DATA_DIR.glob("*_Forum_data.csv")]
