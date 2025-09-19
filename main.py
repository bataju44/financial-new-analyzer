# main.py
from news_fetcher import fetch_financial_news
from llm_analyzer_hf import analyze_news_article
from database import save_analysis_to_db
import time

def run_full_pipeline():
    print("Starting news analysis pipeline...")
    
    # 1. Fetch News
    print("Fetching latest news...")
    articles = fetch_financial_news()
    print(f"Fetched {len(articles)} articles.")
    
    # 2. Process each article
    for article in articles:
        print(f"Analyzing: {article['title'][:50]}...")
        
        # 3. Analyze with LLM
        analysis = analyze_news_article(article['content'])
        
        # 4. Save to Database
        save_analysis_to_db(article, analysis)
        
        # Be nice to the API - add a short delay to avoid rate limits
        time.sleep(1) 
    
    print("Pipeline run completed!")

if __name__ == "__main__":
    run_full_pipeline()