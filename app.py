import streamlit as st
from resume import Resume
from summarizer import Summarizer
from text_embedding import Embedder
import json
import faiss
import pandas as pd

if __name__ == '__main__':
    # Streamlit app UI
    st.title("AI-Powered Resume - Job Matching App")

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

    with open('api_keys.json', 'r') as f:
        keys = json.load(f)

    with open('resume_prompt.txt', 'r') as f:
        resume_prompt = f.read()

    with open('job_prompt.txt', 'r') as f:
        job_prompt = f.read()

    api_key = keys['UTS_OPENAI_KEY']

    if uploaded_file:
        with st.spinner('Extracting and summarizing your resume...'):
            resume = Resume(uploaded_file)
            summarizer = Summarizer(api_key=api_key)

            resume_text = resume.text
            # Replace resume content in the prompt
            resume_summary = summarizer.summarize_info(prompt=resume_prompt, query=resume_text)

            st.subheader("Your Resume Summary")
            st.text(resume_summary)

        with st.spinner('Matching your resume with job descriptions...'):
            embedder = Embedder(api_key=api_key)

            # read embeddings from .faiss file
            job_embeds = faiss.read_index('job_embeds.faiss')

            # Find top 5 matching jobs
            sim_scores, indices = embedder.get_topk_jobs(resume_summary, job_embeds, k=5)

            st.subheader("Top Matching Jobs")

            with open('id_mapping.json', 'r') as f:
                id_mapping = json.load(f)
            
            for score, index in zip(sim_scores[0], indices[0]):
                job_id = id_mapping[str(index)]
                # Load job descriptions from CSV
                df = pd.read_csv('datasets/job_descs_sum_exp.csv')
                
                job_description = df[df['job_id'] == job_id]['description'].item()
                job_summary = df[df['job_id'] == job_id]['job_summary'].item()
                
                st.write(f"**Job Index:** {index}")
                st.write(f"**Similarity Score:** {score}")
                st.write(f"**Job Description:** {job_description}")
                st.write(f"**Job Summary:** {job_summary}")
                st.write("---")
