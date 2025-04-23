import streamlit as st
from resume import Resume
from summarizer import Summarizer
from text_embedding import Embedder
import json
import base64
import os.path as osp
import pandas as pd
import faiss

def display_job_details(index, score, job_id, df):
    # job_description = df[df['job_id'] == job_id]['description'].item()
    # job_summary = df[df['job_id'] == job_id]['job_summary'].item()
    job = df[df['job_id'] == job_id]

    st.write(f"### Rank {index + 1}")

    st.write(f"#### {job['title'].item()} at {job['name'].item()}")
    st.write(f"**Similarity Score:** {score:.2f}")
    st.write(f"**Location:** {job['location'].item()}")
    st.write(f"**Industry:** {job['industry'].item()}")

    st.write("**Job Summary:** ")
    st.write(job['job_summary'].item())
    with st.expander("Detail Description"):
        st.write(job['description_x'].item())
    # st.write(f"**Company Size:** {row['company_size']}")
    st.write(f"**Company Location:** {job['city'].item()}, {job['state'].item()}, {job['country'].item()}")
    st.write(f"[Apply Here]({job['job_posting_url'].item()})")
    st.write("---")

def display_resume_section(api_key, resume_prompt, pdf_file):
    st.subheader("Resume Preview")
    # Convert PDF to base64
    base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')

    # Embed PDF into HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'

    # Display PDF in Streamlit app
    st.markdown(pdf_display, unsafe_allow_html=True)

    st.subheader("Resume Summary")
    with st.spinner('Extracting and summarizing your resume...'):
        resume = Resume(pdf_file)
        summarizer = Summarizer(api_key=api_key)

        resume_text = resume.text
        resume_summary = summarizer.summarize_info(prompt=resume_prompt, query=resume_text)
    
    st.write(resume_summary)

    return resume_summary

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    with open(osp.join(config['PROMPT_DIR'], 'resume_prompt.txt'), 'r') as f:
        resume_prompt = f.read()

    # job dataset
    df = pd.read_csv(osp.join(config['DATA_DIR'], 'job_merged.csv'))

    # job embeddings
    job_embeds = faiss.read_index(osp.join(config['EMBEDDING_DIR'], 'job_embeds.faiss'))

    # id mapping
    with open(osp.join(config['EMBEDDING_DIR'], 'id_mapping.json'), 'r') as f:
        id_mapping = json.load(f)

    api_key = config['UTS_OPENAI_KEY']

    st.header("Resume Upload")
    st.markdown("Upload your resume to find the best matching jobs based on your skills and experience.")

    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if pdf_file:
        resume_summary = display_resume_section(api_key, resume_prompt, pdf_file)

        with st.spinner('Matching your resume with job descriptions...'):
            embedder = Embedder(api_key=api_key)

            sim_scores, indices = embedder.get_topk_jobs(resume_summary, job_embeds, k=5)

            st.subheader("Top Matching Jobs")

            for rank, (score, index) in enumerate(zip(sim_scores[0][::-1], indices[0][::-1])):
                job_id = id_mapping[str(index)]
                display_job_details(rank, score, job_id, df)

    return

if __name__ == '__main__':
    main()
