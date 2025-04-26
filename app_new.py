import streamlit as st
import pandas as pd
import json
import os.path as osp
import faiss
from resume import Resume
from text_embedding import LLM_Embedder
import numpy as np
from assistant import LLM_Assistant
from streamlit import session_state as ss

def filter_data(df, filters):
    filtered_df = df
    for key, value in filters.items():
        if value:
            filtered_df = filtered_df[filtered_df[key].str.contains(value, case=False, na=False)]
    return filtered_df

def display_ai_options(index, job_summary=None):
    st.write("**Better highlighting yourself with AI ✨**")

    col1, col2, col3 = st.columns(3)

    ai_option = f'ai_option_{index}'
    fit_check = f'fit_check_{index}'
    skill_gap_check = f'skill_gap_check_{index}'
    cover_letter_gen = f'cover_letter_gen_{index}'

    input = f"""
        Resume Summary:
        {ss['resume_summary']}

        Job Summary:
        {job_summary}
        """

    if ai_option not in ss:
        ss[ai_option] = None

    if fit_check not in ss:
        ss[fit_check] = False

    if skill_gap_check not in ss:
        ss[skill_gap_check] = False

    if cover_letter_gen not in ss:
        ss[cover_letter_gen] = False

    with col1:
        if st.button("✨ How am I fit?", key=f"explain_{index}"):
            ss[ai_option] = "fit_check"
            ss[fit_check] = not ss[fit_check]

    with col2:
        if st.button("✨ What am I missing?", key=f"missing_{index}"):
            ss[ai_option] = "skill_gap_check"
            ss[skill_gap_check] = not ss[skill_gap_check]

    with col3:
        if st.button("✨ Help me with cover letter", key=f"cover_{index}"):
            ss[ai_option] = "cover_letter_gen"
            ss[cover_letter_gen] = not ss[cover_letter_gen]

    if ss[ai_option] == "fit_check":
        if ss[fit_check]:
            if ss['resume_summary'] is None:
                st.write("Please upload your resume to check if this job is a good fit for you.")
            else:
                with st.spinner("Comparing your resume with the job description..."):
                    st.write("**How do you match this job**")
                    response = ss.llm_assistant.\
                        get_llm_response(instruction=ss.llm_assistant.fit_explain_prompt, 
                                         user_input=input)
                    st.write(f"{response}")

    elif ss[ai_option] == "skill_gap_check":
        if ss[skill_gap_check]:
            if ss['resume_summary'] is None:
                st.write("Please upload your resume to check if this job is a good fit for you.")
            else:
                with st.spinner("Comparing your resume with the job description..."):
                    st.write("**What you are missing**")
                    response = ss.llm_assistant.\
                        get_llm_response(instruction=ss.llm_assistant.skill_gap_prompt, 
                                        user_input=input)
                    st.write(f"{response}")

    elif ss[ai_option] == "cover_letter_gen":
        if ss[cover_letter_gen]:
            if ss['resume_summary'] is None:
                st.write("Please upload your resume to check if this job is a good fit for you.")
            else:
                with st.spinner("Comparing your resume with the job description..."):
                    st.write("**Draft Cover Letter**")
                    response = ss.llm_assistant.\
                        get_llm_response(instruction=ss.llm_assistant.cover_letter_prompt, 
                                        user_input=input)
                    st.write(f"{response}")

def display_job_post(filtered_df):
    used_feats = ['title', 'name', 'company_name', 'industry', 'location', 
                  'description_y', 'job_summary', 'description_x', 'job_posting_url']
    
    # drop na values
    filtered_df.dropna(subset=used_feats, inplace=True)

    if 'rank' in filtered_df:
        filtered_df = filtered_df.sort_values(by='rank', ascending=True)

    for index, row in filtered_df.iterrows():
        st.write(f"#### [{row['title']}]({row['job_posting_url']})")
        
        # # if filter by resume
        # if filter_by_resume:
        #     st.write(f"**Rank {row['rank']}**")
        #     st.write(f"**Similarity Score:** {row['score']:.2f}")

        st.write(f"**Company:** {row['company_name']}")
        st.write(f"**Industry:** {row['industry']}")
        st.write(f"**Location:** {row['location']}")

        st.write("**Job Summary:** ")
        st.write(row['job_summary'])

        with st.expander("**Read more**"):
            st.write("**About Company**")
            st.write(row['description_y'])
            st.write("**Job Description**")
            st.write(row['description_x'])

        # st.write(f"**Company Size:** {row['company_size']}")
        # st.write(f"**Company Location:** {row['city']}, {row['state']}, {row['country']}")

        display_ai_options(index=index, job_summary=row['job_summary'])

        st.write("---")

    return

def get_job_embed_ids(job_ids, id_mapping):
    # Get indexes of those values
    value_index_map = {v: i for i, v in enumerate(id_mapping.values())}
    indexes = [value_index_map.get(v, None) for v in job_ids]
    return indexes

def filter_topk_jobs(filtered_df, num_matched_jobs):
    # get only embeddings on filtered jobs
    filtered_job_ids = filtered_df['job_id'].tolist()
    
    # get job embedding ids
    filtered_embed_ids = get_job_embed_ids(filtered_job_ids,
                                        ss.llm_embedder.id_mapping)
    
    # get filtered job embeddings
    filtered_job_embeds = np.array([ss.llm_embedder.job_embeds.reconstruct(int(id)) \
                                    for id in filtered_embed_ids])

    # generate new filtered faiss index
    faiss_index = faiss.IndexFlatIP(filtered_job_embeds.shape[1])
    faiss_index.add(filtered_job_embeds)

    # mapping new embed ids to old embed ids
    embed_id_dict ={new_id: old_id for new_id, old_id in enumerate(filtered_embed_ids)}

    # calculate similarity score for filtered jobs
    sim_scores, embed_matched_ids = ss['llm_embedder'].\
        get_sim_scores(ss['resume_summary'], 
        faiss_index, k=num_matched_jobs)

    # ignore -1 value
    job_sim_scores = {}

    for rank, (score, index) in enumerate(zip(sim_scores[0], embed_matched_ids[0])):
        if index < 0:
            break
        
        # mapping new embed id -> old embed id -> job id
        job_id = ss.llm_embedder.id_mapping[str(embed_id_dict[index])]

        job_sim_scores[job_id] = {'rank': rank + 1, 'score': score}

    # filterd only matched jobs
    filtered_df = filtered_df[filtered_df['job_id'].isin(job_sim_scores.keys())]

    # add job rank and sim scores to df
    filtered_df['rank'] = filtered_df['job_id'].map(lambda x: job_sim_scores.get(x, {}).get('rank', None))
    filtered_df['score'] = filtered_df['job_id'].map(lambda x: job_sim_scores.get(x, {}).get('score', None))

    return filtered_df

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    job_merged_path = osp.join(config['DATA_DIR'], 'job_merged.csv')

    st.sidebar.header("Filter Options")

    company_name_filter = st.sidebar.text_input("Company Name")
    job_title_filter = st.sidebar.text_input("Job Title")
    location_filter = st.sidebar.text_input("Location")
    industry_filter = st.sidebar.text_input("Industry")

    filters = {
        'company_name': company_name_filter,
        'title': job_title_filter,
        'location': location_filter,
        'industry': industry_filter
    }

    df = pd.read_csv(job_merged_path)

    if 'job_list' not in ss:
        ss['job_list'] = df

    # init resume_summary variable to use across session
    if 'resume_summary' not in ss:
        ss['resume_summary'] = None

    if 'uploaded_file' not in ss:
        ss['uploaded_file'] = None

    if 'llm_assistant' not in ss:
        ss.llm_assistant = LLM_Assistant(config=config)

    if 'llm_embedder' not in ss:
        ss.llm_embedder = LLM_Embedder(config=config)

    # Resume Upload Section
    st.sidebar.header("Resume Upload")
    
    uploaded_file = st.sidebar.file_uploader("Upload your resume PDF file here", type=["pdf"])

    if uploaded_file is not None:
        num_matched_jobs = st.sidebar.number_input("Maximum jobs you want to return", 
                                                   min_value=1, max_value=50, value=5, step=1)
        # only summary once across reruns
        if ss['resume_summary'] is None:
            
            with st.spinner("Scanning Resume..."):
                resume = Resume(uploaded_file)
                resume_summary = ss.llm_assistant.\
                    get_llm_response(instruction=ss.llm_assistant.resume_prompt,
                                    user_input=resume.text)
                ss['resume_summary'] = resume_summary
        
        # display resume summary
        if ss['resume_summary'] == 'Resume Invalid':
            st.error("The document is not a resume. Please upload a valid resume.")
        else:
            st.sidebar.header("Resume Summary")
            st.sidebar.write(ss['resume_summary'])

    else:
        # if file is removed or replace -> no summary
        ss.resume_summary = None


    if st.sidebar.button("Search Jobs"):
        # Apply filters
        ss.job_list = filter_data(df, filters)

        # If resume is uploaded, apply resume matching
        if ss.resume_summary is not None:
            ss.job_list = filter_topk_jobs(ss.job_list, num_matched_jobs=num_matched_jobs)

    # display filtered job postings (only first 20 jobs)
    st.write(f"### Showing {len(ss.job_list)} Job Postings")
    display_job_post(filtered_df=ss.job_list[:20])

if __name__ == '__main__':
    main()