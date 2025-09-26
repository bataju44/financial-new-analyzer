# news_fetcher.py
import os
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

NEWS_API_URL = "https://newsapi.org/v2/top-headlines"
DEFAULT_COUNTRY = "us"
DEFAULT_CATEGORY = "business"

def _create_session(max_retries=3, backoff_factor=0.5, timeout=10):
    """Create a requests session with retries and timeouts."""
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.request_timeout = timeout
    return session

def fetch_financial_news(country=DEFAULT_COUNTRY, category=DEFAULT_CATEGORY, page_size=50):
    """
    Fetches top business headlines from News API.
    Returns a list of articles with title, description, and URL.
    """
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        raise ValueError("NEWS_API_KEY not found in environment variables. Please check your .env file.")
    
    params = {
        "category": category,
        "country": country,
        "pageSize": page_size,
        "apiKey": api_key
    }

    session = _create_session()
    try:
        r = session.get(NEWS_API_URL, params=params, timeout=session.request_timeout)
        r.raise_for_status()
        data = r.json()
        articles = []
        for article in data.get('articles', []):
            title = article.get('title', '')
            description = article.get('description', '')
            url = article.get('url')
            published_at = article.get('publishedAt')
            if not url or not title:
                continue
            content = f"{title}. {description or ''}".strip()
            articles.append({
                'title': title,
                'content': content,
                'url': url,
                'published_at': published_at
            })
        logger.info("Fetched %d articles", len(articles))
        return articles
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching news: %s", e)
        return []

if __name__ == "__main__":
    news = fetch_financial_news()
    for article in news[:2]:
        print(article['title'])
        print("---")
