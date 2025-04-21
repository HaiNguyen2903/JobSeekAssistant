import streamlit as st
from resume import Resume
from summarizer import Summarizer
from text_embedding import Embedder
import json
import faiss
import pandas as pd
import base64

def display_job_details(index, score, job_id, df):
    job_description = df[df['job_id'] == job_id]['description'].item()
    job_summary = df[df['job_id'] == job_id]['job_summary'].item()

    st.write(f"### Rank {index + 1}")
    st.write(f"**Similarity Score:** {score:.2f}")
    st.write("**Job Description:**")
    st.text_area("", job_description, height=300, disabled=True)
    st.write(f"**Job Summary:**")
    st.write(job_summary)
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
    with open('api_keys.json', 'r') as f:
        keys = json.load(f)

    with open('resume_prompt.txt', 'r') as f:
        resume_prompt = f.read()

    with open('job_prompt.txt', 'r') as f:
        job_prompt = f.read()

    api_key = keys['UTS_OPENAI_KEY']

    st.title("AI-Powered Resume - Job Matching App")

    st.header("Resume Upload")
    st.markdown("Upload your resume to find the best matching jobs based on your skills and experience.")

    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if pdf_file:
        resume_summary = display_resume_section(api_key, resume_prompt, pdf_file)

        with st.spinner('Matching your resume with job descriptions...'):
            embedder = Embedder(api_key=api_key)

            job_embeds = faiss.read_index('job_embeds.faiss')

            sim_scores, indices = embedder.get_topk_jobs(resume_summary, job_embeds, k=5)

            st.subheader("Top Matching Jobs")

            with open('id_mapping.json', 'r') as f:
                id_mapping = json.load(f)

            df = pd.read_csv('datasets/job_descs_sum_exp.csv')

            for rank, (score, index) in enumerate(zip(sim_scores[0][::-1], indices[0][::-1])):
                job_id = id_mapping[str(index)]
                display_job_details(rank, score, job_id, df)

    return

if __name__ == '__main__':
    main()
    
    # st.markdown("Upload your resume to find the best matching jobs based on your skills and experience.")

    # uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

    # with open('api_keys.json', 'r') as f:
    #     keys = json.load(f)

    # with open('resume_prompt.txt', 'r') as f:
    #     resume_prompt = f.read()

    # with open('job_prompt.txt', 'r') as f:
    #     job_prompt = f.read()

    # api_key = keys['UTS_OPENAI_KEY']

    # if uploaded_file:
    #     col1, col2 = st.columns(2)

    #     with col1:
    #         st.subheader("Uploaded Resume")
    #         st.markdown("You can preview your uploaded resume below:")
    #         st.download_button("Download Resume", uploaded_file, file_name=uploaded_file.name)
    #         st.markdown("---")
    #         st.markdown("### Resume Preview")
    #         st.markdown("Below is the visualized PDF of your resume:")
    #         with st.expander("View PDF"):
    #             st.file_uploader("", type="pdf", disabled=True, key="pdf_preview")

    #     with st.spinner('Extracting and summarizing your resume...'):
    #         resume = Resume(uploaded_file)
    #         summarizer = Summarizer(api_key=api_key)

    #         resume_text = resume.text
    #         resume_summary = summarizer.summarize_info(prompt=resume_prompt, query=resume_text)

    #     with col2:
    #         st.subheader("Your Resume Summary")
    #         st.markdown("---")
    #         st.markdown("### Highlighted Summary Area")
    #         st.text_area("", resume_summary, height=200, disabled=True, key="highlighted_summary")

    #     with st.spinner('Matching your resume with job descriptions...'):
    #         embedder = Embedder(api_key=api_key)

    #         job_embeds = faiss.read_index('job_embeds.faiss')

    #         sim_scores, indices = embedder.get_topk_jobs(resume_summary, job_embeds, k=5)

    #         st.subheader("Top Matching Jobs")

    #         with open('id_mapping.json', 'r') as f:
    #             id_mapping = json.load(f)

    #         df = pd.read_csv('datasets/job_descs_sum_exp.csv')

    #         for rank, (score, index) in enumerate(zip(sim_scores[0], indices[0])):
    #             job_id = id_mapping[str(index)]
    #             display_job_details(rank, score, job_id, df)
