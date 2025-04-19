import streamlit as st
from resume import Resume
from summarizer import Summarizer
import json

if __name__ == '__main__':
    # Streamlit app UI
    st.title("AI-Powered Resume - Job Matching App")

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

    with open('api_keys.json', 'r') as f:
        keys = json.load(f)

    with open('resume_summarize_prompt.txt', 'r') as f:
        prompt = f.read()
    

    api_key = keys['UTS_OPENAI_KEY']

    # if uploaded_file is not None:
    with st.spinner('Extracting and summarizing your resume...'):
        if uploaded_file:
            resume = Resume(uploaded_file)
            summarizer = Summarizer(api_key=api_key)

            resume_text = resume.text
            # replace resume content in the prompt
            prompt = prompt.replace('resume_content', resume_text)
            resume_summary = summarizer.summarize_info(prompt)
        # resume_embedding = get_embedding(resume_summary)
    
            st.subheader("Your Resume Summary")
            st.text(resume_summary)

        # num_results = st.slider("Select number of job matches", 1, len(jobs), 3)

        # # Match jobs by similarity
        # similarities = [
        #     cosine_similarity([resume_embedding], [job['embedding']])[0][0]
        #     for job in jobs
        # ]

        # # Sort jobs by similarity
        # matched_jobs = sorted(zip(jobs, similarities), key=lambda x: x[1], reverse=True)[:num_results]

        # st.subheader("Matched Jobs")
        # for job, similarity in matched_jobs:
        #     st.markdown(f"**{job['title']}** — Similarity: {similarity:.2f}")
        #     st.text(job['summary'])

        #     # Highlight matching parts
        #     st.markdown("**Matched Sections:**")
        #     job_sections = job['summary'].split('\n')
        #     resume_sections = resume_summary.split('\n')

        #     for j_sec in job_sections:
        #         if any(j_sec.split(':')[0] in r_sec for r_sec in resume_sections):
        #             st.markdown(f":green[{j_sec}]")
        #         else:
        #             st.markdown(j_sec)
