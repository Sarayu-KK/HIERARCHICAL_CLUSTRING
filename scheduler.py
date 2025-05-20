import schedule
import time
from classifier import classify_new_jobs

# Step 1: Define user interests (you can later load this from a file if needed)
user_interests = {
    "Alice": ["python", "machine learning", "sql"],
    "Bob": ["java", "cloud", "aws"]
}

# Step 2: Add alert logic
def notify_users(df):
    for user, skills in user_interests.items():
        matching_jobs = df[df['Skills'].str.lower().apply(
            lambda x: any(skill in x for skill in skills)
        )]

        if not matching_jobs.empty:
            print(f"\n📢 ALERT for {user} — Matching Jobs Found:")
            print(matching_jobs[['Title', 'Company', 'Skills']].to_string(index=False))
        else:
            print(f"\n✅ No matching jobs for {user} today.")

# Step 3: Integrate with job() function
def job():
    print("\nRunning scheduled job check...")
    df = classify_new_jobs()
    notify_users(df)

# Step 4: Schedule and run
schedule.every().day.at("01:36").do(job)

if __name__ == "__main__":
    print("Scheduler started. Waiting for 01:19 AM job to run...")
    while True:
        schedule.run_pending()
        time.sleep(60)
