# database.py
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

# ------------------- DATABASE SETUP -------------------
Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"
    id = Column(Integer, primary_key=True)
    title = Column(String(512))
    content = Column(Text)
    published_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship to analysis results
    analyses = relationship("AnalysisResult", back_populates="article")

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey("articles.id"))
    model_name = Column(String(128))          # store per model
    sentiment = Column(Float, nullable=True)  # numeric sentiment score
    ner_entities = Column(Text, nullable=True) # comma-separated company names
    
    article = relationship("Article", back_populates="analyses")

# Create engine and session
engine = create_engine("sqlite:///financial_news.db")  # or your DB URI
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

# ------------------- DATABASE FUNCTIONS -------------------
def save_analysis_to_db(article_data: dict, analysis_results: list):
    """
    Save article and model analysis results to database.

    article_data: {'title': str, 'content': str}
    analysis_results: output of BatchModelComparison.analyze_batch()
    """
    session = Session()
    try:
        # 1️⃣ Save the article
        article = Article(
            title=article_data.get('title', '')[:512],
            content=article_data.get('content', '')
        )
        session.add(article)
        session.commit()  # commit to get article.id

        # 2️⃣ Save all model analyses
        for res in analysis_results:
            # res has: 'all_sentiments', 'all_ner'
            for model_name, sentiment_data in res['all_sentiments'].items():
                ner_data = res['all_ner'].get(model_name, {'companies': []})
                analysis = AnalysisResult(
                    article_id=article.id,
                    model_name=model_name,
                    sentiment=sentiment_data['score'],  # float
                    ner_entities=",".join(ner_data.get('companies', []))
                )
                session.add(analysis)

        session.commit()
        print(f"Saved article '{article.title[:50]}...' with {len(analysis_results[0]['all_sentiments'])} models analyzed.")

    except Exception as e:
        session.rollback()
        print(f"Error saving to database: {e}")
    finally:
        session.close()
