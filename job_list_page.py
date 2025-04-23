import streamlit as st
import pandas as pd
import json
import os.path as osp

def filter_data(df, filters):
    filtered_df = df
    for key, value in filters.items():
        if value:
            filtered_df = filtered_df[filtered_df[key].str.contains(value, case=False, na=False)]
    return filtered_df

def display_job_details(row):
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

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

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

    filtered_df = filter_data(df, filters)
    filtered_df = filtered_df.head(50)

    st.write(f"### Showing {len(filtered_df)} Job Postings")

    for _, row in filtered_df.iterrows():
        display_job_details(row)

if __name__ == '__main__':
    main()