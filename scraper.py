import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

def scrape_karkidi_jobs(keyword="data science", pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        print(f"Scraping page: {page} - {url}")
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        job_blocks = soup.find_all("div", class_="ads-details")
        for job in job_blocks:
            try:
                title = (job.find("h4") or job.find("h2")).get_text(strip=True) if job.find("h4") or job.find("h2") else ""
                company = ""
                company_tag = job.find("a", href=lambda x: x and "Employer-Profile" in x)
                if not company_tag:
                    company_tag = job.find("span", class_="company-name")
                if company_tag:
                    company = company_tag.get_text(strip=True)

                location = job.find("p").get_text(strip=True) if job.find("p") else ""
                experience = job.find("p", class_="emp-exp").get_text(strip=True) if job.find("p", class_="emp-exp") else ""
                summary = ""
                skills = ""

                key_skills_tag = job.find("span", string="Key Skills")
                if key_skills_tag:
                    skills = key_skills_tag.find_next("p").get_text(strip=True)

                summary_tag = job.find("span", string="Summary")
                if summary_tag:
                    summary = summary_tag.find_next("p").get_text(strip=True)

                # Alternative fallback for skills block
                if not skills:
                    skills_block = job.find("div", class_="job-skills")
                    skills = skills_block.get_text(strip=True) if skills_block else ""

                jobs_list.append({
                    "Title": title,
                    "Company": company,
                    "Location": location,
                    "Experience": experience,
                    "Summary": summary,
                    "Skills": skills
                })
            except Exception as e:
                print(f"Error parsing job block: {e}")
                continue

        time.sleep(1)  # Be nice to the server

    df = pd.DataFrame(jobs_list)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/jobs_data.csv", index=False)
    print(f" Saved {len(df)} jobs to data/jobs_data.csv")
    return df

if __name__ == "__main__":
    df_jobs = scrape_karkidi_jobs(keyword="data science", pages=2)
    print(df_jobs.head())