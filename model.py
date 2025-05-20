import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib

def train_model():
    df = pd.read_csv("data/jobs_data.csv")

    # Clean skills column
    df['Skills'] = df['Skills'].str.lower()

    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Skills'])

    # KMeans clustering
    model = KMeans(n_clusters=5, random_state=42)
    df['category'] = model.fit_predict(X)

    # Save model and vectorizer
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    df.to_csv('clustered_jobs.csv', index=False)
    print("Model trained and saved.")
if __name__ == "__main__":
    train_model()
