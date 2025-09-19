# news_fetcher.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables


def fetch_financial_news():
    """
    Fetches top business headlines from News API.
    Returns a list of articles with title, description, and URL.
    """
    api_key = os.getenv('NEWS_API_KEY')

    # Ensure the API key is available
    if not api_key:
        raise ValueError("NEWS_API_KEY not found in environment variables. Please check your .env file.")
    url = f'https://newsapi.org/v2/top-headlines?category=business&country=us&apiKey={api_key}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an error for bad status codes
        data = response.json()
        
        articles = []
        for article in data.get('articles', []):
            # Combine title and description for LLM analysis
            content = f"{article['title']}. {article.get('description', '')}"
            if content.strip():  # Only add if there's actual text
                articles.append({
                    'title': article['title'],
                    'content': content,
                    'url': article['url'],
                    'published_at': article['publishedAt']
                })
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

# Test the function
if __name__ == "__main__":
    news = fetch_financial_news()
    for article in news[:2]:  # Print first 2 articles
        print(article['title'])
        print("---")