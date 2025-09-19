# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
Base = declarative_base()

# Define the SQL Table
class ProcessedArticle(Base):
    __tablename__ = 'processed_articles'
    
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    original_content = Column(String, nullable=False)
    url = Column(String, nullable=False)
    published_at = Column(DateTime)
    sentiment = Column(Float)  # -1 to 1
    companies = Column(JSON)   # Stores list of companies as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)  # Creates the table if it doesn't exist
Session = sessionmaker(bind=engine)

def save_analysis_to_db(article_data, analysis_result):
    """Saves the original article and its analysis to the database."""
    session = Session()
    try:
        # Convert published_at string to datetime object if it exists
        pub_date = datetime.fromisoformat(article_data['published_at'].replace('Z', '+00:00')) if article_data.get('published_at') else None
        
        new_article = ProcessedArticle(
            title=article_data['title'],
            original_content=article_data['content'],
            url=article_data['url'],
            published_at=pub_date,
            sentiment=analysis_result.get('sentiment', 0),
            companies=analysis_result.get('companies', [])
        )
        session.add(new_article)
        session.commit()
        print(f"Saved article: {article_data['title'][:50]}...")
    except Exception as e:
        print(f"Error saving to database: {e}")
        session.rollback()
    finally:
        session.close()