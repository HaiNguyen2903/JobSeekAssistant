import pandas as pd
from assistant import LLM_Assistant
import json
import os.path as osp
import os
import argparse

def mkdir_if_missing(path):
    if not osp.exists(path):
        os.makedirs(path)
    return

def extract_job_data(df_path, llm_assistant, save_file='', max_jobs=500):
    df = pd.read_csv(df_path)

    title_keywords = ['Data ', 'AI ', 'ML ', 'Machine Learning ']

    # filter jobs
    df = df[df['title'].str.contains('|'.join(title_keywords), case=False, na=False)]

    # extract subset of jobs
    df = df.head(max_jobs)

    df['job_summary'] = df['description'].apply(llm_assistant.get_llm_response, args=(llm_assistant.job_prompt,))

    if save_file != '':
        df.to_csv(save_file, index=False)

    return df

def merge_job_data(company_df, job_post_df, company_industry_df, save_file=''):
    merged_df = pd.merge(job_post_df, company_df,
                         on='company_id', how='inner')
    
    final = pd.merge(merged_df, company_industry_df,
                     on='company_id', how='inner')
    
    if save_file != '':
        final.to_csv(save_file, index=False)

    return final


def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(description="Generate job dataset")
    parser.add_argument('--save_path', type=str, help='Path to save the generated job dataset',
                        default=osp.join(config['DATA_DIR'], 'job_merged.csv'))
    parser.add_argument('--max_jobs', type=int, help='Maximum number of jobs to extract',
                        default=500)
    
    args = parser.parse_args()

    job_post_path = osp.join(config['DATA_DIR'], 'linkedin-jobs-2023-2024/postings.csv')
    company_path = osp.join(config['DATA_DIR'], 'linkedin-jobs-2023-2024/companies/companies.csv')
    company_industry_path = osp.join(config['DATA_DIR'], 'linkedin-jobs-2023-2024/companies/company_industries.csv')

    llm_assistant = LLM_Assistant(config=config)

    job_post_df = extract_job_data(job_post_path, llm_assistant, max_jobs=args.max_jobs)
    company_df = pd.read_csv(company_path)
    company_industry_df = pd.read_csv(company_industry_path)

    merge_job_data(company_df=company_df, 
                job_post_df=job_post_df,
                company_industry_df=company_industry_df,
                save_file=args.save_path)

    return

if __name__ == '__main__':
    main()