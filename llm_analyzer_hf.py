# llm_analyzer_hf.py
from transformers import pipeline
import time  # To add a small delay between model loads

print("Loading specialized sentiment and entity recognition models...")

# Load the sentiment analysis model first
# This model is specifically fine-tuned for sentiment on tweets (short text), great for headlines.
print("1. Loading sentiment model...")
sentiment_analyzer = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=-1,  # Use CPU
)

# Load the Named Entity Recognition (NER) model second
# This model is great at identifying organizations (companies), people, locations, etc.
print("2. Loading company recognition model...")
ner_analyzer = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    device=-1,
    aggregation_strategy="simple" # Groups related word pieces into a single entity
)

print("All models loaded successfully!")

def analyze_news_article(article_text):
    """
    Analyzes article text using two specialized local models:
    1. A sentiment analysis model to score the text from -1 to 1.
    2. A named entity recognition (NER) model to extract company names.
    """
    # Use the first 512 characters to avoid overloading the models
    analysis_text = article_text[:512]
    results = {"sentiment": 0.0, "companies": []}

    try:
        # Analyze Sentiment 
        sentiment_result = sentiment_analyzer(analysis_text)
        # The model returns a list of dicts: [{'label': 'negative', 'score': 0.96}]
        top_sentiment = sentiment_result[0]
        
        # Map the text label to a numeric score and weight it by the model's confidence
        label_map = {'negative': 0.0, 'neutral': 1.0, 'positive': 2.0}
        sentiment_label = top_sentiment['label']
        results["sentiment"] = label_map.get(sentiment_label, 0.0) * top_sentiment['score']
        print(f"DEBUG - Sentiment: {sentiment_label} (confidence: {top_sentiment['score']:.2f}) -> score: {results['sentiment']:.2f}")

        # Extract Company Names
        ner_results = ner_analyzer(analysis_text)
        companies = set()  # Use a set to avoid duplicates
        print(f"DEBUG - NER: {ner_results}")
        
        for entity in ner_results:
            # The NER model labels organizations as 'ORG'. This is our primary target.
            if entity['entity_group'] == 'ORG':
                # Clean the company name (the model might split words, e.g., 'Apple' vs '##pple')
                # The 'word' might have '##' in front of it if it's part of a word split by the tokenizer.
                company_name = entity['word'].replace('#', '').strip()
                # Add some basic filters: must be longer than 2 characters and not a common false positive
                if len(company_name) > 2 and company_name.lower() not in ['inc', 'ltd', 'corp', 'company']:
                    companies.add(company_name)
        
        results["companies"] = list(companies) # Convert the set to a list
        print(f"DEBUG - Companies found: {results['companies']}")

    except Exception as e:
        print(f"Error during analysis: {e}")
    
    return results

# Test the function with a variety of headlines
if __name__ == "__main__":
    test_headlines = [
        "Apple stock soared today after announcing record earnings.",
        "Microsoft faces antitrust investigation in the European Union.",
        "Tesla shares drop after CEO makes controversial public statement.",
        "The Federal Reserve raised interest rates by 0.5% today."
    ]
    
    for headline in test_headlines:
        print(f"\n--- Analyzing: '{headline}' ---")
        result = analyze_news_article(headline)
        print("Result:", result)
        time.sleep(1) # Be nice to the system