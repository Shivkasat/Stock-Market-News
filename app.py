from flask import Flask, jsonify, render_template, request, Response
import pandas as pd
import feedparser
from transformers import pipeline
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import concurrent.futures
import logging
import torch
import trafilatura
from newspaper import Article
import requests
import re
import json
from collections import defaultdict, Counter
import time
import threading
import queue
import sys
from io import StringIO

app = Flask(__name__)

# Load company data
try:
    company_df = pd.read_csv('company.csv')
    company_df.columns = company_df.columns.str.strip()
    company_df = company_df.dropna(subset=['COMPANY_NAME', 'SECTOR'])
    print(f"✅ Loaded {len(company_df)} companies from CSV")
except Exception as e:
    print(f"❌ Error loading company data: {e}")
    # Create fallback data
    company_df = pd.DataFrame({
        'COMPANY_NAME': ['Reliance Industries', 'TCS', 'HDFC Bank'],
        'SYMBOL': ['RELIANCE', 'TCS', 'HDFCBANK'],
        'SECTOR': ['Oil & Gas', 'IT', 'Banking']
    })

# Initialize AI models
try:
    sentiment_pipeline = pipeline("sentiment-analysis")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    print("✅ AI models loaded successfully")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    sentiment_pipeline = None
    summarizer = None

# **ENHANCED SECTOR CONFIGURATION**
ENHANCED_SECTOR_KEYWORDS = {
    "Banking": {
        "companies": ['hdfc bank', 'icici bank', 'sbi', 'state bank', 'axis bank', 'kotak mahindra',
                     'yes bank', 'indusind bank', 'federal bank', 'rbl bank', 'idfc first',
                     'hdfc', 'icici', 'kotak', 'axis', 'canara bank', 'pnb', 'punjab national'],
        "keywords": ['banking', 'bank', 'finance', 'loans', 'deposits', 'npa', 'credit', 
                    'financial services', 'nbfc', 'interest rates', 'rbi policy', 'repo rate'],
        "symbols": ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK']
    },
    
    "IT": {
        "companies": ['tcs', 'tata consultancy', 'infosys', 'wipro', 'hcl tech', 'hcltech',
                     'tech mahindra', 'mindtree', 'mphasis', 'ltts', 'cognizant', 'accenture'],
        "keywords": ['software', 'technology', 'information technology', 'digital', 
                    'tech services', 'it services', 'programming', 'cloud computing', 'ai'],
        "symbols": ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM']
    },
    
    "Oil & Gas": {
        "companies": ['reliance industries', 'reliance', 'ongc', 'oil and natural gas corporation', 
                     'bpcl', 'bharat petroleum', 'ioc', 'indian oil corporation', 'indian oil',
                     'gail', 'oil india', 'mrpl', 'hpcl', 'hindustan petroleum', 'gspl',
                     'igl', 'indraprastha gas', 'petronet lng', 'mangalore refinery'],
        "keywords": ['oil', 'gas', 'petroleum', 'refinery', 'crude oil', 'energy', 'petrol',
                    'diesel', 'lng', 'cng', 'natural gas', 'petrochemical', 'fuel', 'opec',
                    'drilling', 'exploration'],
        "symbols": ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL', 'OIL', 'MRPL', 'HPCL']
    }
}

# Enhanced RSS feeds
ENHANCED_RSS_FEEDS = {
    "economic_times_market": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "economic_times_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
    "financial_express": "https://www.financialexpress.com/market/feed/",
    "livemint": "https://www.livemint.com/rss/markets",
    "ndtv_business": "https://feeds.feedburner.com/ndtvprofit-latest",
    "zeebiz": "https://www.zeebiz.com/rss/markets.xml",
    "google_india_business": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp0Y0RvU0FtVnVHZ0pKVGtnQVAB?hl=en-IN&gl=IN&ceid=IN:en",
    "google_india_stocks": "https://news.google.com/rss/search?q=indian%20stocks&hl=en-IN&gl=IN&ceid=IN:en"
}

# Global variables for real-time CLI logs
processing_logs = []
log_lock = threading.Lock()

def add_log(message):
    """Add log message with timestamp"""
    with log_lock:
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}"
            processing_logs.append(log_message)
            print(log_message)  # Also print to console
            
            # Keep only last 50 logs
            if len(processing_logs) > 50:
                processing_logs.pop(0)
        except Exception as e:
            print(f"Logging error: {e}")

def get_logs():
    """Get current logs"""
    with log_lock:
        return processing_logs.copy()

# Comprehensive sentiment keywords
POSITIVE_WORDS = [
    'profit', 'growth', 'up', 'rise', 'gain', 'surge', 'bullish', 'positive', 'beat', 'strong',
    'earnings', 'revenue', 'high', 'record', 'boost', 'rally', 'jump', 'soar', 'climb',
    'acquisition', 'merger', 'expansion', 'launch', 'breakthrough', 'milestone', 'upgrade',
    'outperform', 'exceed', 'surpass', 'boom', 'optimistic', 'favorable', 'opportunity'
]

NEGATIVE_WORDS = [
    'loss', 'down', 'fall', 'decline', 'crash', 'bearish', 'negative', 'miss', 'weak', 'drop',
    'slump', 'plunge', 'tumble', 'collapse', 'worry', 'fear', 'concern', 'risk', 'warning',
    'lawsuit', 'investigation', 'fraud', 'penalty', 'fine', 'delay', 'cancellation',
    'underperform', 'disappoint', 'shortfall', 'struggle', 'challenge', 'pressure'
]

def clean_stock_mentions(text):
    """Extract ONLY valid stock symbols and remove garbage terms"""
    if not text:
        return []
    
    stock_patterns = [
        r'\b[A-Z]{3,8}\b',  # Stock symbols (3-8 chars)
        r'\bNifty\s*\d*\b', 
        r'\bSensex\b', 
        r'\bBSE\b', 
        r'\bNSE\b'
    ]
    
    mentions = []
    for pattern in stock_patterns:
        matches = re.findall(pattern, text)
        mentions.extend(matches)
    
    # Enhanced filtering to remove garbage terms
    invalid_terms = [
        'june', 'july', 'april', 'may', 'march', 'january', 'february', 'august',
        'september', 'october', 'november', 'december', 'monday', 'tuesday', 'wednesday',
        'thursday', 'friday', 'saturday', 'sunday', 'money', 'were', 'was', 'will',
        'the', 'and', 'for', 'with', 'this', 'that', 'from', 'they', 'can', 'all',
        'new', 'one', 'two', 'get', 'now', 'way', 'also', 'time', 'year', 'day',
        'week', 'month', 'today', 'yesterday', 'tomorrow', 'here', 'there', 'when',
        'where', 'what', 'who', 'how', 'why', 'which', 'than', 'more', 'most',
        'some', 'any', 'each', 'every', 'many', 'much', 'few', 'several'
    ]
    
    valid_mentions = []
    for mention in mentions:
        mention_clean = mention.strip()
        # Only include if:
        # 1. Not in invalid terms
        # 2. Length between 2-8 characters
        # 3. Contains at least one uppercase letter
        # 4. Not purely numeric
        # 5. Contains only alphabetic characters or numbers
        if (mention_clean.lower() not in invalid_terms and 
            2 <= len(mention_clean) <= 8 and
            any(c.isupper() for c in mention_clean) and
            not mention_clean.isdigit() and
            mention_clean.isalnum()):
            valid_mentions.append(mention_clean)
    
    return list(set(valid_mentions))

def enhanced_sector_classification(title, description):
    """Enhanced sector classification with better detection"""
    article_text = f"{title} {description}".lower()
    
    add_log(f"🔍 Classifying: {title[:50]}...")
    
    # Check for specific sector classification
    sector_scores = {}
    best_matches = {}
    
    for sector, data in ENHANCED_SECTOR_KEYWORDS.items():
        score = 0
        matched_companies = []
        matched_keywords = []
        
        # Company name matches (highest priority)
        for company in data['companies']:
            if company in article_text:
                score += 10  # High weight for companies
                matched_companies.append(company)
        
        # Keyword matches (medium priority)  
        for keyword in data['keywords']:
            if keyword in article_text:
                score += 3  # Medium weight for keywords
                matched_keywords.append(keyword)
        
        if score > 0:
            sector_scores[sector] = score
            best_matches[sector] = {
                'companies': matched_companies,
                'keywords': matched_keywords,
                'total_score': score
            }
    
    # Determine final classification
    if sector_scores:
        best_sector = max(sector_scores.items(), key=lambda x: x[1])
        sector_name, score = best_sector
        
        # Adjusted thresholds for better detection
        threshold = 3 if sector_name == "Oil & Gas" else 5
        
        if score >= threshold:
            matches = best_matches[sector_name]
            add_log(f"✅ Classified as {sector_name} (score: {score})")
            return sector_name, matches
    
    # Check for general Indian market indicators
    market_indicators = ['sensex', 'nifty', 'bse', 'nse', 'indian markets', 'stock market', 'equity']
    if any(indicator in article_text for indicator in market_indicators):
        add_log(f"📊 Classified as Indian Markets (general market news)")
        return "Indian Markets", {}
    
    add_log(f"❌ No classification found - excluding article")
    return None, {}

def enhanced_sentiment_analysis(text, title=""):
    """Enhanced sentiment analysis with comprehensive keywords"""
    try:
        combined_text = f"{title} {text}".lower() if text else ""
        
        positive_score = sum(1 for word in POSITIVE_WORDS if word in combined_text)
        negative_score = sum(1 for word in NEGATIVE_WORDS if word in combined_text)
        
        # Use AI if available
        if sentiment_pipeline and text:
            try:
                ai_result = sentiment_pipeline(combined_text[:512])[0]
                ai_label = ai_result['label'].upper()
                ai_score = ai_result['score']
                
                # Combine keyword and AI analysis
                if positive_score > negative_score:
                    final_label = "Positive"
                    confidence = min(0.7 + (positive_score * 0.05) + (ai_score * 0.2), 0.95)
                elif negative_score > positive_score:
                    final_label = "Negative"
                    confidence = min(0.7 + (negative_score * 0.05) + (ai_score * 0.2), 0.95)
                else:
                    final_label = "Neutral" 
                    confidence = 0.5
                
                return final_label, confidence
                
            except Exception as e:
                add_log(f"⚠️ AI sentiment failed, using fallback: {e}")
        
        # Fallback keyword analysis
        if positive_score > negative_score:
            return "Positive", min(0.6 + (positive_score * 0.1), 0.9)
        elif negative_score > positive_score:
            return "Negative", min(0.6 + (negative_score * 0.1), 0.9)
        else:
            return "Neutral", 0.5
            
    except Exception as e:
        add_log(f"❌ Sentiment analysis error: {e}")
        return "Neutral", 0.5

def predict_stock_movement(sector, articles):
    """Predict likely stock movement based on sentiment analysis"""
    if not articles:
        return "Uncertain", "No data", []
    
    positive_articles = [a for a in articles if a['sentiment_label'] == 'Positive']
    negative_articles = [a for a in articles if a['sentiment_label'] == 'Negative']
    
    # Calculate weighted sentiment scores
    total_positive_score = sum(a['sentiment'] for a in positive_articles)
    total_negative_score = sum(a['sentiment'] for a in negative_articles)
    
    # Get most mentioned stocks in positive/negative news
    positive_stocks = []
    negative_stocks = []
    
    for article in positive_articles:
        positive_stocks.extend(article.get('stock_mentions', []))
    
    for article in negative_articles:
        negative_stocks.extend(article.get('stock_mentions', []))
    
    # Prediction logic
    if len(positive_articles) > len(negative_articles) and total_positive_score > total_negative_score:
        prediction = "Likely Up 📈"
        confidence = min((len(positive_articles) / len(articles)) * 100, 95)
        key_stocks = list(set(positive_stocks))[:3]
    elif len(negative_articles) > len(positive_articles) and total_negative_score > total_positive_score:
        prediction = "Likely Down 📉"
        confidence = min((len(negative_articles) / len(articles)) * 100, 95)
        key_stocks = list(set(negative_stocks))[:3]
    else:
        prediction = "Sideways 📊"
        confidence = 50
        key_stocks = []
    
    return prediction, f"{confidence:.0f}%", key_stocks

def is_fresh_news(published_date_str, hours_threshold=24):
    """Check if news is fresh (within specified hours)"""
    try:
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %Z',
            '%a, %d %b %Y %H:%M:%S %z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        for fmt in date_formats:
            try:
                published_date = datetime.strptime(published_date_str, fmt)
                if published_date.tzinfo is None:
                    published_date = published_date.replace(tzinfo=datetime.now().astimezone().tzinfo)
                
                threshold = datetime.now().astimezone() - timedelta(hours=hours_threshold)
                is_fresh = published_date > threshold
                
                if is_fresh:
                    add_log(f"🆕 Fresh news from {published_date.strftime('%H:%M')}")
                
                return is_fresh
            except ValueError:
                continue
        return True  # Default to fresh if can't parse
    except Exception as e:
        add_log(f"Date parsing error: {e}")
        return True

def resolve_final_url(url):
    """Enhanced URL resolution for different news sources"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        # Handle Google News redirects
        if 'news.google.com' in url and '/articles/' in url:
            response = requests.get(url, headers=headers, allow_redirects=True, timeout=15)
            final_url = response.url
            add_log(f"🔗 Google redirect resolved: {final_url[:50]}...")
            return final_url
        
        # For other URLs, check if they're accessible
        response = requests.head(url, headers=headers, allow_redirects=True, timeout=10)
        if response.status_code == 200:
            return response.url
        else:
            add_log(f"⚠️ URL returned status {response.status_code}")
            return url
            
    except Exception as e:
        add_log(f"⚠️ URL resolution failed: {e}")
        return url

def process_rss_feed_enhanced(feed_name, feed_url, results_queue, max_articles=20):
    """Enhanced RSS processing with detailed logging"""
    try:
        add_log(f"🔄 Connecting to {feed_name}...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        feed = feedparser.parse(feed_url)
        
        if not hasattr(feed, 'entries') or len(feed.entries) == 0:
            add_log(f"⚠️ No articles found in {feed_name}")
            results_queue.put((feed_name, {}))
            return
        
        add_log(f"📰 Found {len(feed.entries)} articles in {feed_name}")
        
        sector_articles = defaultdict(list)
        processed = 0
        
        for entry in feed.entries[:max_articles]:
            title = entry.get('title', '')
            link = entry.get('link', '')
            description = BeautifulSoup(entry.get('summary', ''), 'html.parser').get_text()
            published = entry.get('published', '')
            
            if not title or not link:
                continue
            
            # Check freshness
            if not is_fresh_news(published, 24):
                continue
            
            # Enhanced sector classification
            sector, matches = enhanced_sector_classification(title, description)
            
            if sector:
                sentiment_label, sentiment_score = enhanced_sentiment_analysis(description, title)
                stock_mentions = clean_stock_mentions(f"{title} {description}")
                
                article_data = {
                    'title': title,
                    'description': description,
                    'url': link,
                    'sentiment': sentiment_score,
                    'sentiment_label': sentiment_label,
                    'publishedAt': published,
                    'source': feed_name.replace('_', ' ').title(),
                    'stock_mentions': stock_mentions,
                    'matched_terms': matches,
                    'summary': None
                }
                
                sector_articles[sector].append(article_data)
                processed += 1
                
                add_log(f"✅ Added to {sector}: {title[:40]}... (Sentiment: {sentiment_label})")
        
        add_log(f"📊 {feed_name} completed: {processed} articles processed")
        results_queue.put((feed_name, dict(sector_articles)))
        
    except Exception as e:
        add_log(f"❌ Error in {feed_name}: {str(e)}")
        results_queue.put((feed_name, {}))

def fetch_enhanced_news():
    """Multi-threaded news fetching with enhanced logging"""
    add_log("🚀 Starting enhanced news collection...")
    
    results_queue = queue.Queue()
    threads = []
    
    # Create threads
    for feed_name, feed_url in ENHANCED_RSS_FEEDS.items():
        thread = threading.Thread(
            target=process_rss_feed_enhanced,
            args=(feed_name, feed_url, results_queue),
            daemon=True
        )
        threads.append(thread)
        thread.start()
        time.sleep(0.2)
    
    # Wait for completion
    for thread in threads:
        thread.join(timeout=30)
    
    # Collect results
    final_articles = defaultdict(list)
    
    while not results_queue.empty():
        try:
            feed_name, sector_articles = results_queue.get_nowait()
            for sector, articles in sector_articles.items():
                final_articles[sector].extend(articles)
        except queue.Empty:
            break
    
    # Remove duplicates
    for sector in final_articles:
        seen = set()
        unique = []
        for article in final_articles[sector]:
            key = f"{article['title'][:30]}"
            if key not in seen:
                seen.add(key)
                unique.append(article)
        final_articles[sector] = unique
    
    # Log final results
    add_log(f"🎉 Collection completed!")
    for sector, articles in final_articles.items():
        add_log(f"📈 {sector}: {len(articles)} articles")
    
    return dict(final_articles)

def get_top_gainers():
    data = pd.read_csv("nifty500.csv")
    data.columns = data.columns.str.strip()
    data = data.dropna(subset=["LTP", "%CHNG"])
    data["%CHNG"] = pd.to_numeric(data["%CHNG"], errors="coerce")
    gainers = data.sort_values(by="%CHNG", ascending=False).head(5)
    return [{
        "name": row.get("SYMBOL") or row.get("COMPANY_NAME", "Unknown"),
        "price": row["LTP"],
        "change": row["CHNG"],
        "percent": f"{row['%CHNG']:.2f}%"
    } for _, row in gainers.iterrows()]

def get_top_losers():
    data = pd.read_csv("nifty500.csv")
    data.columns = data.columns.str.strip()
    data = data.dropna(subset=["LTP", "%CHNG"])
    data["%CHNG"] = pd.to_numeric(data["%CHNG"], errors="coerce")
    losers = data.sort_values(by="%CHNG", ascending=True).head(5)
    return [{
        "name": row.get("SYMBOL") or row.get("COMPANY_NAME", "Unknown"),
        "price": row["LTP"],
        "change": row["CHNG"],
        "percent": f"{row['%CHNG']:.2f}%"
    } for _, row in losers.iterrows()]

def generate_enhanced_insights(sector_articles):
    """Generate enhanced insights with stock predictions"""
    insights = {}
    
    for sector, articles in sector_articles.items():
        if not articles:
            continue
        
        # Sentiment counts
        positive = [a for a in articles if a['sentiment_label'] == 'Positive']
        negative = [a for a in articles if a['sentiment_label'] == 'Negative']
        neutral = [a for a in articles if a['sentiment_label'] == 'Neutral']
        
        # Stock movement prediction
        prediction, confidence, key_stocks = predict_stock_movement(sector, articles)
        
        # Clean trending stocks
        all_mentions = []
        for article in articles:
            all_mentions.extend(article.get('stock_mentions', []))
        
        valid_mentions = [m for m in all_mentions if len(m) >= 2 and m.isalpha()]
        trending_stocks = [item for item, count in Counter(valid_mentions).most_common(5)]
        
        insights[sector] = {
            "total_articles": len(articles),
            "positive_count": len(positive),
            "negative_count": len(negative),
            "neutral_count": len(neutral),
            "prediction": prediction,
            "prediction_confidence": confidence,
            "key_stocks": key_stocks,
            "trending_stocks": trending_stocks,
            "latest_articles": sorted(articles, key=lambda x: x.get('publishedAt', ''), reverse=True)[:5]
        }
    
    return insights

# Flask Routes
@app.route("/")
def dashboard():
    """Enhanced dashboard with all features"""
    try:
        add_log("🏠 Loading enhanced dashboard...")
        
        sector_articles = fetch_enhanced_news()
        sector_insights = generate_enhanced_insights(sector_articles)
        
        total_articles = sum(len(articles) for articles in sector_articles.values())
        
        add_log(f"✅ Dashboard loaded with {total_articles} articles")
        
        return render_template(
            "complete_dashboard.html",
            gainers=get_top_gainers(),
            losers=get_top_losers(),
            sector_articles=sector_articles,
            sector_insights=sector_insights,
            total_articles=total_articles,
            logs=get_logs()[-10:],  # Last 10 logs
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as e:
        add_log(f"❌ Dashboard error: {str(e)}")
        return f"<h1>Dashboard Error: {e}</h1><p>Check console logs for details.</p>"

@app.route("/api/logs")
def api_logs():
    """API endpoint for real-time logs"""
    return jsonify({"logs": get_logs()})

@app.route("/summarize")
def summarize_url():
    """Enhanced AI-powered article summarization that works for ALL sectors"""
    url = request.args.get("url")
    if not url:
        return jsonify({"summary": "❌ No URL provided."})
    
    try:
        add_log(f"🤖 AI summarizing: {url[:50]}...")
        
        # Enhanced URL resolution for different news sources
        resolved_url = resolve_final_url(url)
        
        # Try multiple extraction methods
        article_content = None
        extraction_method = ""
        
        # Method 1: Newspaper3k
        try:
            article = Article(resolved_url)
            article.download()
            article.parse()
            
            if article.text and len(article.text.split()) >= 30:
                article_content = article.text
                extraction_method = "Newspaper3k"
                add_log("✅ Content extracted via Newspaper3k")
        except Exception as e:
            add_log(f"⚠️ Newspaper3k failed: {e}")
        
        # Method 2: Trafilatura (fallback)
        if not article_content:
            try:
                downloaded = trafilatura.fetch_url(resolved_url)
                if downloaded:
                    extracted = trafilatura.extract(downloaded)
                    if extracted and len(extracted.split()) >= 30:
                        article_content = extracted
                        extraction_method = "Trafilatura"
                        add_log("✅ Content extracted via Trafilatura")
            except Exception as e:
                add_log(f"⚠️ Trafilatura failed: {e}")
        
        # Method 3: Direct requests + BeautifulSoup (last resort)
        if not article_content:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(resolved_url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                if len(text.split()) >= 30:
                    article_content = text
                    extraction_method = "BeautifulSoup"
                    add_log("✅ Content extracted via BeautifulSoup")
            except Exception as e:
                add_log(f"⚠️ BeautifulSoup failed: {e}")
        
        if not article_content:
            add_log("❌ All extraction methods failed")
            return jsonify({
                "summary": "⚠️ Could not extract article content. The article may be behind a paywall or have restricted access.",
                "stock_mentions": [],
                "sentiment": "Neutral",
                "sentiment_score": "0.50",
                "extraction_method": "Failed",
                "analysis_success": False
            })
        
        # AI Summarization
        if summarizer and len(article_content.split()) >= 50:
            try:
                # Prepare text for summarization (limit to 1024 tokens)
                max_length = 1000
                if len(article_content) > max_length:
                    # Take beginning and middle parts
                    words = article_content.split()
                    if len(words) > 200:
                        summary_input = ' '.join(words[:100] + words[len(words)//2:len(words)//2+100])
                    else:
                        summary_input = article_content[:max_length]
                else:
                    summary_input = article_content
                
                # Generate AI summary
                summary_result = summarizer(
                    summary_input,
                    max_length=130,
                    min_length=40,
                    do_sample=False,
                    truncation=True
                )[0]['summary_text']
                
                add_log("✅ AI summary generated successfully")
                
            except Exception as e:
                add_log(f"⚠️ AI summarization failed: {e}")
                # Fallback to extractive summary
                sentences = article_content.split('. ')
                summary_result = '. '.join(sentences[:3]) + '.' if len(sentences) >= 3 else article_content[:300] + '...'
                
        else:
            # Extractive summarization fallback
            sentences = article_content.split('. ')
            summary_result = '. '.join(sentences[:3]) + '.' if len(sentences) >= 3 else article_content[:300] + '...'
            add_log("⚠️ Using extractive summarization (AI not available)")
        
        # Enhanced analysis
        stock_mentions = clean_stock_mentions(summary_result + " " + article_content[:500])
        sentiment_label, sentiment_score = enhanced_sentiment_analysis(summary_result, "")
        
        # Detect sector from content
        detected_sector = "Unknown"
        content_lower = article_content.lower()
        for sector, data in ENHANCED_SECTOR_KEYWORDS.items():
            for company in data['companies']:
                if company in content_lower:
                    detected_sector = sector
                    break
            if detected_sector != "Unknown":
                break
        
        add_log(f"✅ Summary completed for {detected_sector} article")
        
        return jsonify({
            "summary": summary_result,
            "stock_mentions": stock_mentions[:6],  # Limit to 6 mentions
            "sentiment": sentiment_label,
            "sentiment_score": f"{sentiment_score:.2f}",
            "word_count": len(article_content.split()),
            "summary_length": len(summary_result.split()),
            "compression_ratio": f"{(len(summary_result.split()) / len(article_content.split())) * 100:.1f}%",
            "extraction_method": extraction_method,
            "detected_sector": detected_sector,
            "analysis_success": True
        })
        
    except Exception as e:
        add_log(f"❌ Summarization error: {str(e)}")
        return jsonify({
            "summary": f"❌ Error processing article: {str(e)}",
            "stock_mentions": [],
            "sentiment": "Neutral",
            "sentiment_score": "0.50",
            "analysis_success": False
        })

if __name__ == "__main__":
    add_log("🚀 Starting Enhanced Indian Stock Market Dashboard")
    add_log("🤖 AI summarization enabled")
    add_log("📊 Real-time CLI logs active")
    add_log("📈 Stock movement predictions active")
    add_log("⚡ Oil & Gas detection enhanced")
    add_log("🔧 Multi-method content extraction ready")
    
    print("\n" + "="*60)
    print("🇮🇳 ENHANCED INDIAN STOCK MARKET DASHBOARD")
    print("="*60)
    print("🌐 Dashboard: http://localhost:5000")
    print("📊 Features: AI Summaries, Real-time Logs, Stock Predictions")
    print("⚡ Auto-refresh: Every 10 minutes")
    print("🔧 Multi-threaded RSS processing")
    print("="*60)
    
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000, threaded=True)
