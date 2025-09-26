from transformers import pipeline
from typing import Dict, List, Any
import time
from abc import ABC, abstractmethod

# ==================== BASE ABSTRACT CLASSES ====================
class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers."""
    
    @abstractmethod
    def analyze(self, text: str) -> Any:
        """Analyze text and return results."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Return information about the model."""
        pass

# ==================== SENTIMENT ANALYZERS ====================
class SentimentAnalyzer(BaseAnalyzer):
    """Base class for sentiment analysis models."""
    
    def get_model_info(self) -> Dict:
        return {"type": "sentiment", "name": self.model_name}

class TransformerSentimentAnalyzer(SentimentAnalyzer):
    """Handles transformer-based sentiment models."""
    
    def __init__(self, model_name: str, model_key: str):
        self.model_name = model_name
        self.model_key = model_key
        print(f"Loading sentiment model: {model_name}")
        self.pipeline = pipeline(
            "text-classification",
            model=model_key,
            device=-1
        )
    
    def analyze(self, text: str) -> Dict:
        analysis_text = text[:512]
        try:
            result = self.pipeline(analysis_text)[0]
            return {
                'label': result['label'],
                'score': result['score'],
                'success': True
            }
        except Exception as e:
            return {'error': str(e), 'success': False}

class PromptBasedSentimentAnalyzer(SentimentAnalyzer):
    """Handles sentiment models that require custom prompting."""
    
    def __init__(self, model_name: str, model_key: str, prompt_template: str):
        self.model_name = model_name
        self.model_key = model_key
        self.prompt_template = prompt_template
        print(f"Loading prompt-based sentiment model: {model_name}")
        self.pipeline = pipeline(
            "text-generation",
            model=model_key,
            device=-1
        )
    
    def analyze(self, text: str) -> Dict:
        analysis_text = text[:512]
        try:
            prompt = self.prompt_template.format(text=analysis_text)
            result = self.pipeline(
                prompt,
                max_new_tokens=50,
                return_full_text=False
            )
            generated_text = result[0]['generated_text']
            return self._parse_sentiment_response(generated_text)
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    def _parse_sentiment_response(self, response: str) -> Dict:
        """Parse the model's response into structured sentiment data."""
        response_lower = response.lower()
        if 'positive' in response_lower:
            return {'label': 'positive', 'score': 0.9, 'success': True}
        elif 'negative' in response_lower:
            return {'label': 'negative', 'score': 0.9, 'success': True}
        else:
            return {'label': 'neutral', 'score': 0.5, 'success': True}

# ==================== NER ANALYZERS ====================
class NERAnalyzer(BaseAnalyzer):
    """Base class for NER analysis models."""
    
    def __init__(self):
        self.NEWS_SOURCE_BLOCKLIST = {
            'reuters', 'bloomberg', 'cnbc', 'wsj', 'financial', 'times', 'ap',
            'com', 'cnn', 'bbc', 'nytimes', 'theguardian', 'forbes', 'fortune'
        }
    
    def get_model_info(self) -> Dict:
        return {"type": "ner", "name": self.model_name}
    
    def _filter_companies(self, companies: List[str]) -> List[str]:
        """Filter out news sources and clean company names."""
        filtered = set()
        for company in companies:
            company_clean = company.replace('#', '').strip()
            if (len(company_clean) > 2 and 
                company_clean.lower() not in ['inc', 'ltd', 'corp', 'company', 'co'] and
                not any(src in company_clean.lower() for src in self.NEWS_SOURCE_BLOCKLIST)):
                filtered.add(company_clean)
        return list(filtered)

class TransformerNERAnalyzer(NERAnalyzer):
    """Handles transformer-based NER models."""
    
    def __init__(self, model_name: str, model_key: str):
        super().__init__()
        self.model_name = model_name
        self.model_key = model_key
        print(f"Loading NER model: {model_name}")
        self.pipeline = pipeline(
            "token-classification",
            model=model_key,
            device=-1,
            aggregation_strategy="simple"
        )
    
    def analyze(self, text: str) -> Dict:
        analysis_text = text[:512]
        try:
            ner_results = self.pipeline(analysis_text)
            companies = set()
            for entity in ner_results:
                if entity['entity_group'] == 'ORG':
                    companies.add(entity['word'])
            filtered_companies = self._filter_companies(companies)
            return {
                'companies': filtered_companies,
                'count': len(filtered_companies),
                'success': True
            }
        except Exception as e:
            return {'error': str(e), 'success': False}


class BatchModelComparison:
    SENTIMENT_MODELS = {
        'twitter-roberta': "cardiffnlp/twitter-roberta-base-sentiment-latest",
        'distilbert-sst2': "distilbert-base-uncased-finetuned-sst-2-english",
        'finbert': "yiyanghkust/finbert-tone"
    }

    NER_MODELS = {
        'bert-base-ner': "dslim/bert-base-NER",
        'bert-large-ner': "dslim/bert-large-NER"
    }

    def __init__(self, sentiment_models=None, ner_models=None):
        sentiment_models = sentiment_models or ['twitter-roberta']
        ner_models = ner_models or ['bert-base-ner']

        self.sentiment_analyzers = {
            name: pipeline("text-classification", model=self.SENTIMENT_MODELS[name], device=-1)
            for name in sentiment_models
        }
        self.ner_analyzers = {
            name: pipeline("token-classification", model=self.NER_MODELS[name],
                           device=-1, aggregation_strategy="simple")
            for name in ner_models
        }

    def analyze_batch(self, texts):
        """Run all models in batch on texts, return structured results per article and per model."""
        texts = [t[:512] for t in texts]  # limit to first 512 chars

        # Run sentiment models in batch
        sentiment_results_all = {
            model_name: analyzer(texts)
            for model_name, analyzer in self.sentiment_analyzers.items()
        }

        # Run NER models in batch
        ner_results_all = {
            model_name: analyzer(texts)
            for model_name, analyzer in self.ner_analyzers.items()
        }

        # Build results per article
        combined_results = []
        for i in range(len(texts)):
            per_article_sentiments = {}
            per_article_ner = {}

            # Per sentiment model
            for model_name, results in sentiment_results_all.items():
                per_article_sentiments[model_name] = self._parse_sentiment(results[i])

            # Per NER model
            for model_name, results in ner_results_all.items():
                per_article_ner[model_name] = self._parse_ner(results[i])

            # Combine into structured dict
            combined_results.append({
                'all_sentiments': per_article_sentiments,  # per model
                'all_ner': per_article_ner                 # per model
            })

        return combined_results

    def _parse_sentiment(self, result):
        """Map label to numeric score: positive -> +score, negative -> -score, neutral -> 0"""
        label = result['label'].lower()
        score = result['score']
        if 'positive' in label:
            value = score
        elif 'negative' in label:
            value = -score
        else:
            value = 0.0
        return {'label': label, 'score': value}

    def _parse_ner(self, ner_results):
        """Extract company names from NER results."""
        companies = set()
        for entity in ner_results:
            if entity['entity_group'] == 'ORG':
                name = entity['word'].replace('#', '').strip()
                if len(name) > 2 and name.lower() not in ['inc', 'ltd', 'corp', 'company']:
                    companies.add(name)
        return {'companies': list(companies)}

# ==================== MODEL COMPARISON MANAGER ====================

class ModelComparisonManager:
    """Manages multiple analyzers and coordinates comparisons."""
    
    SENTIMENT_MODELS = {
        'twitter-roberta': ("cardiffnlp/twitter-roberta-base-sentiment-latest", "transformer"),
        'distilbert-sst2': ("distilbert-base-uncased-finetuned-sst-2-english", "transformer"),
        'finbert': ("yiyanghkust/finbert-tone", "transformer"),
        'llama-sentiment': ("meta-llama/Llama-2-7b-chat-hf", "prompt")
    }
    
    NER_MODELS = {
        'bert-base-ner': "dslim/bert-base-NER",
        'bert-large-ner': "dslim/bert-large-NER"
    }
    
    def __init__(self, sentiment_models: List[str] = None, ner_models: List[str] = None):
        self.sentiment_analyzers = []
        self.ner_analyzers = []
        self._load_sentiment_models(sentiment_models or ['twitter-roberta'])
        self._load_ner_models(ner_models or ['bert-base-ner'])
    
    def _load_sentiment_models(self, model_keys: List[str]):
        for model_key in model_keys:
            if model_key in self.SENTIMENT_MODELS:
                model_id, model_type = self.SENTIMENT_MODELS[model_key]
                if model_type == "transformer":
                    analyzer = TransformerSentimentAnalyzer(model_key, model_id)
                elif model_type == "prompt":
                    prompt_template = "Analyze the sentiment of this financial news: {text}\nSentiment:"
                    analyzer = PromptBasedSentimentAnalyzer(model_key, model_id, prompt_template)
                else:
                    continue
                self.sentiment_analyzers.append(analyzer)
    
    def _load_ner_models(self, model_keys: List[str]):
        for model_key in model_keys:
            if model_key in self.NER_MODELS:
                analyzer = TransformerNERAnalyzer(model_key, self.NER_MODELS[model_key])
                self.ner_analyzers.append(analyzer)
    
    def compare_sentiment_models(self, text: str) -> Dict[str, Any]:
        results = {}
        for analyzer in self.sentiment_analyzers:
            result = analyzer.analyze(text)
            results[analyzer.model_name] = result
            time.sleep(0.1)
        return results
    
    def compare_ner_models(self, text: str) -> Dict[str, Any]:
        results = {}
        for analyzer in self.ner_analyzers:
            result = analyzer.analyze(text)
            results[analyzer.model_name] = result
            time.sleep(0.1)
        return results
    
    def comprehensive_comparison(self, text: str, title: str = None) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE COMPARISON: {title or 'Untitled'}")
        print(f"{'='*60}")
        results = {
            'text_sample': text[:100] + '...' if len(text) > 100 else text,
            'timestamp': time.time(),
            'sentiment_results': self.compare_sentiment_models(text),
            'ner_results': self.compare_ner_models(text)
        }
        return results

# ==================== USAGE EXAMPLE ====================
def main():
    """Demonstrate the refactored model comparison."""
    print("Loading models with separate responsibilities...")
    manager = ModelComparisonManager(
        sentiment_models=['twitter-roberta', 'distilbert-sst2'],
        ner_models=['bert-base-ner', 'bert-large-ner']
    )
    test_text = "Apple reported strong earnings despite market challenges. Microsoft also performed well."
    results = manager.comprehensive_comparison(test_text, "Tech Earnings Report")
    print("\nSENTIMENT RESULTS:")
    for model_name, result in results['sentiment_results'].items():
        if result.get('success'):
            print(f"  {model_name}: {result['label']} ({result['score']:.3f})")
        else:
            print(f"  {model_name}: ERROR - {result.get('error', 'Unknown error')}")
    print("\nNER RESULTS:")
    for model_name, result in results['ner_results'].items():
        if result.get('success'):
            print(f"  {model_name}: {result['companies']} ({result['count']} companies)")
        else:
            print(f"  {model_name}: ERROR - {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
