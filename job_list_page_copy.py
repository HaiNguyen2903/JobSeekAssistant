import streamlit as st
import pandas as pd
import json
import os.path as osp
import faiss
from resume import Resume
from summarizer import Summarizer
from text_embedding import Embedder
import numpy as np

def filter_data(df, filters):
    filtered_df = df
    for key, value in filters.items():
        if value:
            filtered_df = filtered_df[filtered_df[key].str.contains(value, case=False, na=False)]
    return filtered_df

def display_job_details(filtered_df, filter_by_resume=False, 
                        sim_scores=None, embed_matched_ids=None,
                        id_mapping=None):
    if filter_by_resume:
        for rank, (score, index) in enumerate(zip(sim_scores[0][::-1], embed_matched_ids[0][::-1])):
            job_id = id_mapping[str(index)]
            job = filtered_df[filtered_df['job_id'] == job_id]
            st.write(f"#### {job['title'].item()} at {job['name'].item()}")
            st.write(f"## Rank {rank + 1}")
            st.write(f"**Similarity Score:** {score:.2f}")
            st.write(f"**Location:** {job['location'].item()}")
            st.write(f"**Industry:** {job['industry'].item()}")

            st.write("**Job Summary:** ")
            st.write(job['job_summary'].item())
            with st.expander("Detail Description"):
                st.write(job['description_x'].item())

            st.write(f"**Company Location:** {job['city'].item()}, {job['state'].item()}, {job['country'].item()}")
            st.write(f"[Apply Here]({job['job_posting_url'].item()})")
            st.write("---")
        return
    
    for _, row in filtered_df.iterrows():
        st.write(f"#### {row['title']} at {row['name']}")
        st.write(f"**Location:** {row['location']}")
        st.write(f"**Industry:** {row['industry']}")

        st.write("**Job Summary:** ")
        st.write(row['job_summary'])
        with st.expander("Detail Description"):
            st.write(row['description_x'])
        # st.write(f"**Company Size:** {row['company_size']}")
        st.write(f"**Company Location:** {row['city']}, {row['state']}, {row['country']}")
        st.write(f"[Apply Here]({row['job_posting_url']})")
        st.write("---")

    return

def get_job_embed_ids(job_ids, id_mapping):
    # Get indexes of those values
    value_index_map = {v: i for i, v in enumerate(id_mapping.values())}
    indexes = [value_index_map.get(v, None) for v in job_ids]
    return indexes


def main():
    if 'filter_by_resume' not in st.session_state:
        st.session_state.filter_by_resume = False

    with open('config.json', 'r') as f:
        config = json.load(f)

    with open(osp.join(config['PROMPT_DIR'], 'resume_prompt.txt'), 'r') as f:
        resume_prompt = f.read()

    # job dataset
    df = pd.read_csv(osp.join(config['DATA_DIR'], 'job_merged.csv'))

    # job embeddings
    job_embeds = faiss.read_index(osp.join(config['EMBEDDING_DIR'], 'job_embeds.faiss'))

    print(type(job_embeds))
    return
    # id mapping
    with open(osp.join(config['EMBEDDING_DIR'], 'id_mapping.json'), 'r') as f:
        id_mapping = json.load(f)

    api_key = config['UTS_OPENAI_KEY']

    job_merged_path = osp.join(config['DATA_DIR'], 'job_merged.csv')

    df = pd.read_csv(job_merged_path)

    st.sidebar.header("Filter Options")
    company_name_filter = st.sidebar.text_input("Company Name")
    job_title_filter = st.sidebar.text_input("Job Title")
    location_filter = st.sidebar.text_input("Location")
    industry_filter = st.sidebar.text_input("Industry")

    filters = {
        'name': company_name_filter,
        'title': job_title_filter,
        'location': location_filter,
        'industry': industry_filter
    }

    # Resume Upload Section
    st.sidebar.header("Resume Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file is not None:
        st.sidebar.success("Resume uploaded successfully!")

    # filter by features
    filtered_df = filter_data(df, filters)
    filtered_df = filtered_df.head(50)
    
    # filter by resume macthing
    if st.sidebar.button("Filter by Resume"):
        # update st session state
        st.session_state.filter_by_resume = True

        filtered_df = filtered_df.head(1)

        # summary and display resume summary
        with st.spinner('Extracting and summarizing your resume...'):
            resume = Resume(uploaded_file)
            summarizer = Summarizer(api_key=api_key)

            resume_text = resume.text
            resume_summary = summarizer.summarize_info(prompt=resume_prompt, query=resume_text)
        
        st.sidebar.header("Resume Summary")
        st.sidebar.write(resume_summary)

        # summary and display job matched
        embedder = Embedder(api_key=api_key)

        # get only embeddings on filtered jobs
        filtered_job_ids = filtered_df['job_id'].tolist()
        filtered_embed_ids = get_job_embed_ids(filtered_job_ids, id_mapping)
        filtered_job_embeds = np.array([job_embeds.reconstruct(int(id)) for id in filtered_embed_ids])

        # calculate similarity score for filtered jobs
        sim_scores, embed_matched_ids = embedder.get_topk_jobs(resume_summary, filtered_job_embeds, k=5)

        job_matched_ids = [id_mapping[str(index)] for index in embed_matched_ids[0]]
        filtered_df = filtered_df[filtered_df['job_id'].isin(job_matched_ids)]

        # for rank, (score, index) in enumerate(zip(sim_scores[0][::-1], indices[0][::-1])):
        #     job_id = id_mapping[str(index)]
        #     display_job_details(rank, score, job_id, df)


    st.write(f"### Showing {len(filtered_df)} Job Postings")

    if st.session_state.filter_by_resume:
        display_job_details(filtered_df=filtered_df,
                            filter_by_resume=True, 
                            sim_scores=sim_scores,
                            embed_matched_ids=embed_matched_ids,
                            id_mapping=id_mapping)
    else:
        display_job_details(filtered_df=filtered_df, filter_by_resume=False)

if __name__ == '__main__':
    main()