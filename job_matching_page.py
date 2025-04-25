import streamlit as st
from resume import Resume
from summarizer import Summarizer
from text_embedding import Embedder
import json
import base64
import os
import os.path as osp
import pandas as pd
import faiss



def display_job_details(index, score, job_id, df, api_key, resume_summary, summarizer):
    job = df[df['job_id'] == job_id].iloc[0]

    st.write(f"### Rank {index + 1}")
    st.write(f"#### {job['title']} at {job['name']}")
    st.write(f"**Similarity Score:** {score:.2f}")
    st.write(f"**Location:** {job['location']}")
    st.write(f"**Industry:** {job['industry']}")

    st.write("**Job Summary:**")
    st.write(job['job_summary'])


    with st.expander("🔍 Skill Gap Recommendation"):
        try:
            gap_feedback = summarizer.get_skill_gap(resume_summary, job['job_summary'])
            st.markdown(gap_feedback)
        except Exception as e:
            st.error(f"Error generating skill gap: {e}")

    with st.expander("Detail Description"):
        st.write(job['description_x'])

    with st.expander("🛠 Tailor My CV for this Job"):
        if st.button(f"Tailor CV - Rank {index + 1}", key=f"tailor_{index}"):
            tailored = summarizer.tailor_cv(resume_summary, job['job_summary'])
            st.write(tailored)

    with st.expander("✉️ Generate Cover Letter"):
        if st.button(f"Generate Cover Letter - Rank {index + 1}", key=f"cover_{index}"):
            letter = summarizer.generate_cover_letter(resume_summary, job['job_summary'])
            st.write(letter)


    st.write(f"**Company Location:** {job['city']}, {job['state']}, {job['country']}")
    st.write(f"[Apply Here]({job['job_posting_url']})")
    st.write("---")

def display_resume_section(api_key, resume_prompt, pdf_file):
    st.subheader("Resume Preview")

    base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

    st.subheader("Resume Summary")
    with st.spinner('Extracting and summarizing your resume...'):
        resume = Resume(pdf_file)
        summarizer = Summarizer(api_key=api_key)
        resume_text = resume.text
        resume_summary = summarizer.summarize_info(prompt=resume_prompt, query=resume_text)

        st.markdown("### 📝 Generated Resume Summary")
        st.write(resume_summary)

    return resume_summary, summarizer

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    with open(osp.join(config['PROMPT_DIR'], 'resume_prompt.txt'), 'r') as f:
        resume_prompt = f.read()

    df = pd.read_csv(osp.join(config['DATA_DIR'], 'job_merged.csv'))
    job_embeds = faiss.read_index(osp.join(config['EMBEDDING_DIR'], 'job_embeds.faiss'))

    with open(osp.join(config['EMBEDDING_DIR'], 'id_mapping.json'), 'r') as f:
        id_mapping = json.load(f)

    api_key = config['UTS_OPENAI_KEY']
    os.environ["OPENAI_API_KEY"] = api_key

    st.header("Resume Matching")
    st.markdown("Upload your resume and find the best matching jobs based on your skills and experience.")

    pdf_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

    if pdf_file:
        resume_summary, summarizer = display_resume_section(api_key, resume_prompt, pdf_file)

        with st.spinner('Matching your resume with job descriptions...'):
            embedder = Embedder(api_key=api_key)
            sim_scores, indices = embedder.get_topk_jobs(resume_summary, job_embeds, k=5)

            st.subheader("Top Matching Jobs")

            for rank, (score, index) in enumerate(zip(sim_scores[0][::-1], indices[0][::-1])):
                job_id = id_mapping[str(index)]
                display_job_details(rank, score, job_id, df, api_key, resume_summary, summarizer)

if __name__ == '__main__':
    main()
