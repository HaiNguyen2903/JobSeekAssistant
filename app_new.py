import streamlit as st
import pandas as pd
import json
import os.path as osp
import faiss
from resume import Resume
from summarizer import Summarizer
from text_embedding import Embedder
import numpy as np
from assistant import LLM_Assistant

def filter_data(df, filters):
    filtered_df = df
    for key, value in filters.items():
        if value:
            filtered_df = filtered_df[filtered_df[key].str.contains(value, case=False, na=False)]
    return filtered_df

# def display_popup_message(key, title, content):
#     modal = Modal(
#         title=title, 
#         key=key,
        
#         # Optional
#         padding=20,    # default value
#         max_width=744,  # default value
#     )

#     modal.open()
#     with modal.container():
#         st.write(content)

def display_ai_options(index, job_summary=None):
    st.write("**Better highlighting yourself with AI ✨**")

    col1, col2, col3 = st.columns(3)

    ai_option = f'ai_option_{index}'
    fit_check = f'fit_check_{index}'
    skill_gap_check = f'skill_gap_check_{index}'
    cover_letter_gen = f'cover_letter_gen_{index}'

    if ai_option not in st.session_state:
        st.session_state[ai_option] = None

    if fit_check not in st.session_state:
        st.session_state[fit_check] = False

    if skill_gap_check not in st.session_state:
        st.session_state[skill_gap_check] = False

    if cover_letter_gen not in st.session_state:
        st.session_state[cover_letter_gen] = False

    with col1:
        if st.button("✨ How am I fit?", key=f"explain_{index}"):
            st.session_state[ai_option] = "fit_check"
            st.session_state[fit_check] = not st.session_state[fit_check]

    with col2:
        if st.button("✨ What am I missing?", key=f"missing_{index}"):
            st.session_state[ai_option] = "skill_gap_check"
            st.session_state[skill_gap_check] = not st.session_state[skill_gap_check]

    with col3:
        if st.button("✨ Help me with cover letter", key=f"cover_{index}"):
            st.session_state[ai_option] = "cover_letter_gen"
            st.session_state[cover_letter_gen] = not st.session_state[cover_letter_gen]

    if st.session_state[ai_option] == "fit_check":
        if st.session_state[fit_check]:
            if st.session_state['resume_summary'] is None:
                st.write("Please upload your resume to check if this job is a good fit for you.")
            else:
                with st.spinner("Comparing your resume with the job description..."):

                    input = f"""
                    Resume Summary:
                    {st.session_state['resume_summary']}

                    Job Summary:
                    {job_summary}
                    """
                    response = st.session_state.llm_assistant.get_llm_response(instruction=st.session_state.llm_assistant.fit_explain_prompt, 
                                                                               user_input=input)
                    
                    st.write(f"{response}")

    elif st.session_state[ai_option] == "skill_gap_check":
        if st.session_state[skill_gap_check]:
            st.write("You are missing some skills for this job.")

    elif st.session_state[ai_option] == "cover_letter_gen":
        if st.session_state[cover_letter_gen]:
            st.write("Generating a cover letter for this job.") 

def display_job_post(filtered_df):
    filter_by_resume = False

    used_feats = ['title', 'name', 'company_name', 'industry', 'location', 
                  'description_y', 'job_summary', 'description_x', 'job_posting_url']
    
    filtered_df.dropna(subset=used_feats, inplace=True)
    filtered_df = filtered_df[1:]

    if 'rank' in filtered_df:
        filtered_df = filtered_df.sort_values(by='rank', ascending=True)
        filter_by_resume = True

    for index, row in filtered_df.iterrows():
        st.write(f"#### [{row['title']}]({row['job_posting_url']})")
        
        # if filter by resume
        if filter_by_resume:
            st.write(f"**Rank {row['rank']}**")
            st.write(f"**Similarity Score:** {row['score']:.2f}")

        st.write(f"**Company::** {row['company_name']}")
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
    
    # init job_list variable to use across session
    if 'job_list' not in st.session_state:
        st.session_state['job_list'] = df.head(50)

    # init resume_summary variable to use across session
    if 'resume_summary' not in st.session_state:
        st.session_state['resume_summary'] = None

    if 'sim_scores' not in st.session_state:
        st.session_state.sim_scores = None

    if 'embed_matched_ids' not in st.session_state:
        st.session_state.embed_matched_ids = None

    if 'llm_assistant' not in st.session_state:
        st.session_state.llm_assistant = LLM_Assistant(config=config)

    # Resume Upload Section
    st.sidebar.header("Resume Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your resume PDF file here", type=["pdf"])

    if uploaded_file is not None:
        # st.sidebar.success("Resume uploaded successfully!")
        num_matched_jobs = st.sidebar.number_input("Maximum jobs you want to return", min_value=1, max_value=50, value=5, step=1)

        # only summary once across reruns
        if st.session_state['resume_summary'] is None:
            with st.spinner("Scanning Resume..."):
                resume = Resume(uploaded_file)
                summarizer = Summarizer(api_key=api_key)

                resume_text = resume.text
                resume_summary = summarizer.summarize_info(prompt=resume_prompt, query=resume_text)

                if resume_summary == 'Resume Invalid':
                    st.error("The document is not a resume. Please upload a valid resume.")
                else:
                    # save varibales to session state
                    st.session_state['resume_summary'] = resume_summary
                    st.sidebar.header("Resume Summary")
                    st.sidebar.write(st.session_state['resume_summary'])


    if st.sidebar.button("Search Jobs"):
        # Apply filters
        filtered_df = filter_data(df, filters)

        # If resume is uploaded, apply resume matching
        if st.session_state['resume_summary'] is not None:

            # resume = Resume(uploaded_file)
            # summarizer = Summarizer(api_key=api_key)

            # resume_text = resume.text
            # resume_summary = summarizer.summarize_info(prompt=resume_prompt, query=resume_text)

            # if resume_summary == 'Resume Invalid':
            #     st.error("The document is not a resume. Please upload a valid resume.")

            # summary and display job matched
            embedder = Embedder(api_key=api_key)

            # get only embeddings on filtered jobs
            filtered_job_ids = filtered_df['job_id'].tolist()
            filtered_embed_ids = get_job_embed_ids(filtered_job_ids, id_mapping)
            filtered_job_embeds = np.array([job_embeds.reconstruct(int(id)) for id in filtered_embed_ids])

            # convert to faiss index IP type
            faiss_index = faiss.IndexFlatIP(filtered_job_embeds.shape[1])
            faiss_index.add(filtered_job_embeds)

            # calculate similarity score for filtered jobs
            sim_scores, embed_matched_ids = embedder.get_topk_jobs(resume_summary, faiss_index, k=num_matched_jobs)

            # ignore -1 value
            job_sim_scores = {}

            for rank, (score, index) in enumerate(zip(sim_scores[0], embed_matched_ids[0])):
                if index < 0:
                    break

                job_id = id_mapping[str(index)]
                job_sim_scores[job_id] = {'rank': rank + 1, 'score': score}

            # filterd only matched jobs
            filtered_df = filtered_df[filtered_df['job_id'].isin(job_sim_scores.keys())]

            # add job rank and sim scores to df
            filtered_df['rank'] = filtered_df['job_id'].map(lambda x: job_sim_scores.get(x, {}).get('rank', None))
            filtered_df['score'] = filtered_df['job_id'].map(lambda x: job_sim_scores.get(x, {}).get('score', None))

        
        st.session_state['job_list'] = filtered_df


    # Display job listings from session state
    if st.session_state['job_list'] is not None:
        st.write(f"### Job Postings")
        display_job_post(filtered_df=st.session_state['job_list'])

if __name__ == '__main__':
    main()