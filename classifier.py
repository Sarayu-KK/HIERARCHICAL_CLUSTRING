import pandas as pd
import joblib
from scraper import scrape_karkidi_jobs

def classify_new_jobs():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    df = scrape_karkidi_jobs()
    X_new = vectorizer.transform(df['Skills'].str.lower())
    df['category'] = model.predict(X_new)
    df.to_csv("new_classified_jobs.csv", index=False)
    return df
if __name__ == "__main__":
    classify_new_jobs()
