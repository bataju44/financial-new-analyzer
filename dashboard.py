# dashboard.py
import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))

st.title("📈 Financial News Sentiment Dashboard")

# Write a SQL query to get the latest analyzed articles
query = text("""
    SELECT title, sentiment, companies, published_at, url
    FROM processed_articles
    ORDER BY created_at DESC
    LIMIT 20;
""")

with engine.connect() as conn:
    df = pd.read_sql(query, conn)

# Display results
st.write("### Latest Analyzed News")
for _, row in df.iterrows():
    with st.expander(f"{row['title']} (Sentiment: {row['sentiment']:.2f})"):
        st.write(f"**Companies:** {', '.join(row['companies'])}")
        st.write(f"**Published:** {row['published_at']}")
        st.write(f"**Read more:** [Link]({row['url']})")

# Show a simple chart
st.write("### Sentiment Distribution")
st.bar_chart(df['sentiment'])