from news_fetcher import fetch_financial_news
from database import save_analysis_to_db
from model_comparison_analyzer import BatchModelComparison
import time

def run_full_pipeline():
    print("Starting news analysis pipeline...")

    # 1. Fetch news
    articles = fetch_financial_news()
    print(f"Fetched {len(articles)} articles.")

    # 2. Initialize batch analyzer once
    batch_analyzer = BatchModelComparison(
        sentiment_models=['twitter-roberta', 'distilbert-sst2'],
        ner_models=['bert-base-ner']
    )

    # 3. Collect all article texts
    texts = [a['content'] for a in articles]

    # 4. Run batch analysis
    print("Running batch analysis...")
    batch_results = batch_analyzer.analyze_batch(texts)

    # 5. Save to database
    for article, result in zip(articles, batch_results):
        save_analysis_to_db(article, result)
        time.sleep(0.5)

    print("Pipeline run completed!")

if __name__ == "__main__":
    run_full_pipeline()
