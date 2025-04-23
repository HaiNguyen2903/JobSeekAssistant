import pandas as pd
from summarizer import Summarizer
import json
import os.path as osp
import os

def mkdir_if_missing(path):
    if not osp.exists(path):
        os.makedirs(path)
    return

def extract_job_data(df_path, summarizer, prompt, save_file=''):
    df = pd.read_csv(df_path)

    title_keywords = ['Data ', 'AI ', 'ML ', 'Machine Learning ']

    # filter jobs
    df = df[df['title'].str.contains('|'.join(title_keywords), case=False, na=False)]

    df = df.head(10)
    df['job_summary'] = df['description'].apply(summarizer.summarize_info, args=(prompt,))

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

    job_post_path = osp.join(config['DATA_DIR'], 'linkedin-jobs-2023-2024/postings.csv')
    company_path = osp.join(config['DATA_DIR'], 'linkedin-jobs-2023-2024/companies/companies.csv')
    company_industry_path = osp.join(config['DATA_DIR'], 'linkedin-jobs-2023-2024/companies/company_industries.csv')

    with open(osp.join(config['PROMPT_DIR'], 'job_prompt.txt'), 'r') as f:
        job_prompt = f.read()

    summarizer = Summarizer(api_key=config['UTS_OPENAI_KEY'])

    job_post_df = extract_job_data(job_post_path, summarizer, job_prompt)
    company_df = pd.read_csv(company_path)
    company_industry_df = pd.read_csv(company_industry_path)

    merge_job_data(company_df=company_df, 
                job_post_df=job_post_df,
                company_industry_df=company_industry_df,
                save_file=osp.join(config['DATA_DIR'], 'job_merged.csv'))

    return

if __name__ == '__main__':
    main()