import streamlit as st
import pandas as pd
import joblib
from classifier import classify_new_jobs
from scraper import scrape_karkidi_jobs

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

# Notify users based on skills
def notify_users(df, user_interests):
    alerts = {}
    for user, skills in user_interests.items():
        matching_jobs = df[df['Skills'].str.lower().apply(
            lambda x: any(skill in x for skill in skills)
        )]
        alerts[user] = matching_jobs
    return alerts

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Job Alert App", layout="wide")
    st.title("🤖 AI-Powered Job Alert & Clustering App")
    
    st.markdown("Scrape and classify jobs from **Karkidi**. Get alerts based on your skills!")

    with st.sidebar:
        st.header("🔧 Job Scraper Settings")
        keyword = st.text_input("Search keyword", "data science")
        pages = st.slider("Number of pages to scrape", 1, 5, 1)

        if st.button("🚀 Scrape & Classify Jobs"):
            with st.spinner("Scraping and classifying jobs..."):
                model, vectorizer = load_model()
                df = scrape_karkidi_jobs(keyword, pages)
                X_new = vectorizer.transform(df['Skills'].str.lower())
                df['category'] = model.predict(X_new)
                df.to_csv("new_classified_jobs.csv", index=False)
                st.success(f"{len(df)} jobs scraped and classified!")
                st.session_state['df'] = df

    # If jobs are already scraped
    if 'df' in st.session_state:
        df = st.session_state['df']

        st.subheader("📂 Clustered Jobs")
        st.dataframe(df[['Title', 'Company', 'Location', 'Skills', 'category']], use_container_width=True)

        st.markdown("---")
        st.subheader("📣 User-Specific Job Alerts")

        num_users = st.slider("Number of users to simulate", 1, 5, 2)
        user_interests = {}
        for i in range(num_users):
            user = st.text_input(f"User {i+1} name", f"User{i+1}")
            skills = st.text_input(f"Skills for {user} (comma-separated)", "python, sql")
            user_interests[user] = [s.strip().lower() for s in skills.split(",")]

        if st.button("🔔 Check Alerts"):
            alerts = notify_users(df, user_interests)
            for user, jobs in alerts.items():
                if not jobs.empty:
                    st.success(f"📢 {user} — {len(jobs)} matching job(s) found!")
                    st.dataframe(jobs[['Title', 'Company', 'Skills']], use_container_width=True)
                else:
                    st.info(f"✅ No matching jobs for {user}.")

    else:
        st.info("👈 Start by scraping jobs from the sidebar.")

if __name__ == "__main__":
    main()
