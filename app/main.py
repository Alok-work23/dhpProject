from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import math
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
import re
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import json
from app.sentiment import FinBERTSentiment
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get absolute path to data directory
BASE_DIR = Path(__file__).parent.resolve()  # .resolve() gets absolute path
DATA_DIR = BASE_DIR / "data"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)
logger.info(f"Data directory location: {DATA_DIR}")


app = FastAPI()

# Configure Gemini (replace with your API key)
genai.configure(api_key="AIzaSyDSc67L2N1KCVH7MNFL6qONMQpNqE760A8")
model = genai.GenerativeModel('gemini-2.0-flash')

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize FinBERT
sentiment_model = FinBERTSentiment()

def get_news_links(keyword: str, date: str, num_results: int = 3):
    """Scrape news links related to keyword and date"""
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        date_str = date_obj.strftime("%Y-%m-%d")
        query = f"{keyword} stock {date_str}"
        
        # Search Google News
        url = f"https://news.google.com/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = []
        for article in soup.find_all('article')[:num_results]:
            a_tag = article.find('a', href=True)
            if a_tag:
                href = a_tag['href']
                if href.startswith('./'):
                    href = f"https://news.google.com{href[1:]}"
                links.append({
                    "title": a_tag.get_text().strip(),
                    "url": href
                })
        
        return links
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []
    
@app.get("/get-news/")
async def get_news(company: str, date: str):
    """
    Fetch AI-integrated news links for a given company and date.
    """
    news_links = get_news_links(company, date)
    return {"date": date, "news": news_links}

@app.get("/")
def home():
    return {"message": "Stock Vibe API is running"}

@app.get("/sentiment/{company}")
def get_sentiment(company: str):
    file_path = os.path.join(os.path.dirname(__file__), "data", f"{company}_Forum_data.csv")
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Company not found"})

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to read CSV: {e}"})

    if 'sentiment' not in df.columns:
        return JSONResponse(status_code=500, content={"error": "Sentiment column not found in preprocessed data"})

    return df.head().to_dict(orient="records")

@app.get("/sentiment-summary/{company}")
def sentiment_summary(company: str, year: int = None, month: int = None):
    file_path = os.path.join(os.path.dirname(__file__), "data", f"{company}_Forum_data.csv")
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Company not found"})

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to read CSV: {e}"})

    if 'sentiment' not in df.columns:
        return JSONResponse(status_code=500, content={"error": "Sentiment column not found in preprocessed data"})

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    if year:
        df = df[df['Date'].dt.year == year]
        if month:
            df = df[df['Date'].dt.month == month]

    counts = df['sentiment'].value_counts().to_dict()
    return counts

@app.get("/timeline/{company}")
def sentiment_timeline(company: str):
    file_path = os.path.join(os.path.dirname(__file__), "data", f"{company}_Forum_data.csv")
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Company not found"})

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to read CSV: {e}"})

    if 'Date' not in df.columns:
        return JSONResponse(status_code=400, content={"error": "No 'Date' column in CSV"})

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    if 'sentiment' not in df.columns:
        return JSONResponse(status_code=500, content={"error": "Sentiment column not found in preprocessed data"})

    timeline = df.groupby([df['Date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
    timeline.index = timeline.index.astype(str)
    timeline = timeline.rename_axis('date').reset_index()
    return timeline.to_dict(orient="records")

@app.get("/top-keywords/{company}")
def top_keywords(company: str, top_n: int = 10):
    file_path = os.path.join(os.path.dirname(__file__), "data", f"{company}_Forum_data.csv")
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Company not found"})

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to read CSV: {e}"})

    if 'message' not in df.columns:
        return JSONResponse(status_code=400, content={"error": "No 'message' column in data"})

    messages = df['message'].dropna().astype(str)
    if messages.empty:
        return JSONResponse(status_code=400, content={"error": "No valid messages found"})

    text = " ".join(messages.tolist()).lower()
    text = re.sub(r"[^a-z\s'-]", "", text)
    words = [word.strip() for word in text.split() if len(word.strip()) > 2]

    if not words:
        return JSONResponse(status_code=400, content={"error": "No valid keywords found after processing"})

    common_words = Counter(words).most_common(top_n)
    return [{"word": word, "count": count} for word, count in common_words]

@app.get("/overall-sentiment/{company}")
def overall_sentiment(company: str):
    file_path = os.path.join(os.path.dirname(__file__), "data", f"{company}_Forum_data.csv")
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Company not found"})

    df = pd.read_csv(file_path)
    if 'sentiment' not in df.columns or 'sentiment_score' not in df.columns:
        return JSONResponse(status_code=500, content={"error": "Required sentiment columns not found"})

    df['net_sentiment'] = df.apply(
        lambda row: row['sentiment_score'] if row['sentiment'] == 'positive'
        else (-row['sentiment_score'] if row['sentiment'] == 'negative' else 0), axis=1)

    net_sentiment_score = df['net_sentiment'].mean()
    sentiment_counts = df['sentiment'].value_counts().to_dict()

    return {
        "net_sentiment_score": round(net_sentiment_score, 3),
        "sentiment_counts": sentiment_counts
    }

@app.get("/sentiment-trend/{company}")
def sentiment_trend(company: str, year: int = None, month: int = None, start_date: str = None, end_date: str = None):
    file_path = os.path.join(os.path.dirname(__file__), "data", f"{company}_Forum_data.csv")
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Company not found"})

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to read CSV: {e}"})

    if 'Date' not in df.columns or 'sentiment' not in df.columns:
        return JSONResponse(status_code=500, content={"error": "Required columns missing"})

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    if year:
        df = df[df['Date'].dt.year == year]
        if month:
            df = df[df['Date'].dt.month == month]

    if start_date and end_date:
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "Invalid date format"})

    trend = df.groupby([df['Date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
    trend.index = trend.index.astype(str)
    trend = trend.rename_axis('date').reset_index()

    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment not in trend.columns:
            trend[sentiment] = 0

    trend = trend[['date', 'positive', 'neutral', 'negative']]
    return trend.to_dict(orient="records")

@app.get("/price-vs-sentiment/{company}")
def price_vs_sentiment(company: str, year: int = None, month: int = None, start_date: str = None, end_date: str = None):
    file_path = os.path.join(os.path.dirname(__file__), "data", f"{company}_Forum_data.csv")
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Company not found"})

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to read CSV: {e}"})

    required_columns = ['Date', 'sentiment_score', 'NSE_current_price', 'sentiment']
    if not all(col in df.columns for col in required_columns):
        return JSONResponse(status_code=500, content={"error": "Required columns missing"})

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    if year:
        df = df[df['Date'].dt.year == year]
        if month:
            df = df[df['Date'].dt.month == month]

    if start_date and end_date:
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "Invalid date format"})

    df['net_sentiment'] = df.apply(
        lambda row: row['sentiment_score'] if row['sentiment'] == 'positive'
        else (-row['sentiment_score'] if row['sentiment'] == 'negative' else 0), axis=1)

    summary = df.groupby(df['Date'].dt.date).agg({
        'net_sentiment': 'mean',
        'NSE_current_price': 'mean'
    })
    summary.index = summary.index.astype(str)
    summary = summary.rename_axis('date').reset_index()
    summary.columns = ['date', 'avg_net_sentiment', 'avg_current_price']

    return summary.to_dict(orient="records")

@app.get("/compare-companies")
def compare_companies(companies: List[str] = Query(...)):
    base_path = os.path.join(os.path.dirname(__file__), "data")
    comparison_data = []
    missing_companies = []

    if not companies:
        raise HTTPException(status_code=422, detail="At least one company must be provided")

    for company in companies:
        file_path = os.path.join(base_path, f"{company}_Forum_data.csv")
        if not os.path.exists(file_path):
            missing_companies.append(company)
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read CSV for {company}: {str(e)}")

        if 'sentiment_score' not in df.columns or 'sentiment' not in df.columns:
            raise HTTPException(status_code=500, detail=f"Required sentiment columns missing for {company}")

        df['net_sentiment'] = df.apply(
            lambda row: row['sentiment_score'] if row['sentiment'] == 'positive'
            else (-row['sentiment_score'] if row['sentiment'] == 'negative' else 0), axis=1)

        avg_sentiment = df['net_sentiment'].mean()
        comparison_data.append({
            "company": company,
            "average_net_sentiment": round(avg_sentiment, 3)
        })

    if not comparison_data:
        raise HTTPException(status_code=404, detail=f"No valid data found for companies: {', '.join(companies)}")

    return comparison_data

@app.get("/message-volume/{company}")
def message_volume(company: str, start_date: str = None, end_date: str = None):
    try:
        file_path = os.path.join(os.path.dirname(__file__), "data", f"{company}_Forum_data.csv")
        
        if not os.path.exists(file_path):
            return JSONResponse(status_code=404, content={"error": "Company not found"})

        df = pd.read_csv(file_path)

        # Check required columns
        if 'Date' not in df.columns or 'message' not in df.columns:
            return JSONResponse(status_code=400, content={"error": "CSV missing 'Date' or 'message' column"})

        # Convert 'Date' column
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

        # Filter by date range if provided
        if start_date and end_date:
            try:
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            except ValueError:
                return JSONResponse(status_code=400, content={"error": "Invalid date format"})

        # Drop rows with NaN in 'message'
        df = df.dropna(subset=['message'])

        # Group by date
        result = df.groupby(df['Date'].dt.date).agg(
            message_count=('Date', 'count'),
            messages=('message', list)
        ).reset_index()

        result = result.rename(columns={'Date': 'date'})
        result['date'] = result['date'].astype(str)

        # Ensure no NaNs in the output
        data = result.where(pd.notnull(result), None)

        return JSONResponse(content=data.to_dict(orient="records"))

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "details": str(e)}
        )
    







COMPANY_MAPPING = {
    'hdfc': 'HDFC',
    'reliance': 'Reliance',
    'tcs': 'TCS',
    'infosys': 'Infosys',
    'icici': 'ICICI',
    'adani_enterprises': 'Adani_Enterprises',
    'larsen_toubro': 'Larsen_Toubro',
    'kotak_mahi': 'Kotak_Mahi',
    'hindustan_unilever': 'Hindustan_Unilever',
    'itc': 'ITC'
}

def get_company_filename(company_name: str) -> str:
    standardized_name = COMPANY_MAPPING.get(company_name.lower())
    if not standardized_name:
        available = list(COMPANY_MAPPING.keys())
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid company name",
                "available_companies": available,
                "suggestion": f"Try one of these: {', '.join(available)}"
            }
        )
    return f"{standardized_name}_Forum_data.csv"

@app.get("/volume-insights/{company_name}")
async def get_volume_insights(
    company_name: str,
    days: int = Query(None, gt=0, le=365),
    year: int = Query(None, gt=2000, le=datetime.now().year),
    month: int = Query(None, gt=0, le=12)
):
    try:
        filename = get_company_filename(company_name)
        file_path = DATA_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Company data not found")

        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Apply date filters
        now = datetime.now()
        if year:
            df = df[df['Date'].dt.year == year]
            if month:
                df = df[df['Date'].dt.month == month]
        elif days:
            start_date = now - timedelta(days=days)
            df = df[df['Date'] >= start_date]
        else:
            # Default to last 30 days if no filters provided
            df = df[df['Date'] >= (now - timedelta(days=30))]
        
        # Calculate message volume
        volume_data = df.groupby(df['Date'].dt.date).size().reset_index(name='count')
        volume_data['date'] = volume_data['Date'].astype(str)
        volume_data['change'] = volume_data['count'].pct_change().replace([np.inf, -np.inf], np.nan) * 100
        volume_data['change'] = volume_data['change'].round(2)
        
        # Identify significant spikes (>50% increase)
        spikes = volume_data[volume_data['change'] > 50].sort_values('change', ascending=False)
        
        # Generate AI insights
        insights = []
        if not spikes.empty:
            try:
                # Prepare context for Gemini
                context = f"""
                Analyze message volume spikes for {company_name} stock discussions.
                Time period: {year if year else f'last {days} days' if days else 'last 30 days'}
                Significant spikes found: {len(spikes)}
                """
                
                # Add news context for top 3 spikes
                for i, spike in spikes.head(3).iterrows():
                    news = get_news_links(company_name, spike['date'])
                    if news:
                        context += f"\nOn {spike['date']} ({spike['change']}% increase):\n"
                        context += "\n".join([f"- {item['title']}" for item in news])
                
                prompt = f"""
                {context}
                
                Provide a comprehensive analysis including:
                1. Overview of message volume trends
                2. Potential reasons for major spikes
                3. Market sentiment implications
                4. Any notable patterns or anomalies
                
                Format as bullet points with clear explanations.
                """
                
                response = model.generate_content(prompt)
                insights = [line.strip() for line in response.text.split('\n') if line.strip()]
            except Exception as e:
                logger.error(f"Gemini AI error: {e}")
                insights = ["Could not generate insights at this time."]

        return {
            "company": company_name,
            "period": {
                "days": days,
                "year": year,
                "month": month
            },
            "volume_data": jsonable_encoder(volume_data.replace({np.nan: None}).to_dict(orient='records')),
            "spikes": jsonable_encoder(spikes.replace({np.nan: None}).to_dict(orient='records')),
            "insights": insights
        }
        
    except Exception as e:
        logger.error(f"Error in volume insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check-data-directory")
async def check_data_directory():
    """Endpoint to verify data directory setup"""
    return {
        "base_directory": str(BASE_DIR),
        "data_directory": str(DATA_DIR),
        "exists": DATA_DIR.exists(),
        "files": [f.name for f in DATA_DIR.glob("*") if f.is_file()]
    }

@app.get("/sentiment-snapshot/{company}")
def sentiment_snapshot(company: str):
    file_path = os.path.join(os.path.dirname(__file__), "data", f"{company}_Forum_data.csv")
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Company not found"})

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to read CSV: {e}"})

    required_columns = ['message', 'sentiment', 'sentiment_score']
    if not all(col in df.columns for col in required_columns):
        return JSONResponse(status_code=500, content={"error": "Required columns missing"})

    snapshot = []
    
    # Get one positive message (highest sentiment_score)
    positive_df = df[df['sentiment'] == 'positive']
    if not positive_df.empty:
        positive_message = positive_df.loc[positive_df['sentiment_score'].idxmax()]
        snapshot.append({
            "sentiment": "positive",
            "message": positive_message['message'],
            "sentiment_score": round(float(positive_message['sentiment_score']), 3)
        })

    # Get one negative message (lowest sentiment_score, i.e., highest absolute negative score)
    negative_df = df[df['sentiment'] == 'negative']
    if not negative_df.empty:
        negative_message = negative_df.loc[negative_df['sentiment_score'].idxmax()]
        snapshot.append({
            "sentiment": "negative",
            "message": negative_message['message'],
            "sentiment_score": round(float(negative_message['sentiment_score']), 3)
        })

    # Get one neutral message (sentiment_score closest to 0)
    neutral_df = df[df['sentiment'] == 'neutral']
    if not neutral_df.empty:
        neutral_message = neutral_df.loc[neutral_df['sentiment_score'].abs().idxmin()]
        snapshot.append({
            "sentiment": "neutral",
            "message": neutral_message['message'],
            "sentiment_score": round(float(neutral_message['sentiment_score']), 3)
        })

    if not snapshot:
        return JSONResponse(status_code=400, content={"error": "No valid messages found for snapshot"})

    return snapshot


