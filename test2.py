import streamlit as st
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
# import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load NLP model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# Function to extract skills from text
def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE"]]
    return list(set(skills))

# Scrape job postings from Indeed (Example)
def fetch_jobs_from_indeed(query, location=""):
    url = f"https://www.indeed.com/jobs?q={query.replace(' ', '+')}&l={location}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    jobs = []
    for job_card in soup.select(".job_seen_beacon"):
        title = job_card.select_one(".jobTitle span").text.strip()
        company = job_card.select_one(".companyName").text.strip()
        job_link = "https://www.indeed.com" + job_card.select_one("a")["href"]
        jobs.append({"title": title, "company": company, "link": job_link})
    
    return jobs[:10]  # Return top 10 jobs

# Function to rank jobs using FAISS
def rank_jobs(resume_text, job_descriptions):
    # resume_vector = model.encode([resume_text])
    # job_vectors = model.encode(job_descriptions)

    # job_vectors = np.array(job_vectors).astype('float32')
    # index = faiss.IndexFlatL2(job_vectors.shape[1])
    # index.add(job_vectors)

    # _, indices = index.search(resume_vector, 5)  # Top 5 matches
    # return [job_descriptions[i] for i in indices[0]]
    job_texts = [resume_text] + job_descriptions  # Combine resume with job descriptions
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(job_texts)

    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]
    ranked_indices = similarity_scores.argsort()[::-1]  # Sort by similarity score (descending)
    
    return [job_descriptions[i] for i in ranked_indices[:5]]  # Return top 5 matched jobs



# Streamlit UI
# st.title("AI Resume Matcher")

# uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
# if uploaded_file:
#     resume_text = extract_text_from_pdf(uploaded_file)
#     st.text_area("Extracted Resume Text", resume_text, height=200)

#     skills = extract_skills(resume_text)
#     st.write("Extracted Skills:", ", ".join(skills))

#     job_query = st.text_input("Enter Job Role to Search")
#     if st.button("Find Jobs"):
#         job_listings = fetch_jobs_from_indeed(job_query)
#         job_descriptions = [job["title"] + " at " + job["company"] for job in job_listings]

#         ranked_jobs = rank_jobs(resume_text, job_descriptions)

#         st.subheader("Top Matched Jobs")
#         for job in ranked_jobs:
#             job_data = next(j for j in job_listings if j["title"] in job)
#             st.markdown(f"[{job_data['title']} at {job_data['company']}]({job_data['link']})")

if __name__ == '__main__':
    print(fetch_jobs_from_indeed(query='Data Engineer', location='Sydney'))