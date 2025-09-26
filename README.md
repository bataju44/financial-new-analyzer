# Financial News Analyzer

A Python-based NLP pipeline to fetch, process, and analyze financial news using multiple sentiment analysis and Named Entity Recognition (NER) models. Built for systematic evaluation of NLP models and efficient storage of per-model results for downstream analytics.

---

## Features

- **News Fetching:** Collects financial news articles from multiple online sources.  
- **Batch NLP Analysis:** Processes multiple articles at once using Hugging Face transformers for:
  - Sentiment Analysis (multi-model comparison)
  - Named Entity Recognition (NER) for companies and organizations
- **Per-Model Database Storage:** Saves sentiment scores and named entities per model in a relational database using SQLAlchemy ORM.  
- **Optimized Performance:** Batch processing reduces inference time by ~60% compared to sequential processing.  
- **Extensible Architecture:** Easily add new sentiment or NER models with minimal code changes.

---

## Tech Stack

- **Language:** Python 3.10+  
- **NLP Libraries:** Hugging Face Transformers, Tokenizers, PyTorch  
- **Database:** SQLite / PostgreSQL via SQLAlchemy ORM  
- **Task Scheduling:** Python scripts with optional time delays for API rate limits  
- **Other Tools:** Pandas, NumPy

---

## Installation

```bash
# Clone the repository
git clone https://github.com/bataju44/financial-news-analyzer.git
cd financial-news-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

The project is designed to fetch financial news, analyze it using multiple NLP models, and store the results in a database. You can run it in two simple steps:

1. **Run the main analysis pipeline**  
   This fetches financial news and runs sentiment and NER models:
   ```bash
   python main.py
   ```
2. **Run the dashboard**  
  This shows a bashboard for the articles in the database and shows a bargraph of the sentiment scores.
  ```bash
   python dashboard.py
   ```
