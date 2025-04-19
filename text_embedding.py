from openai import OpenAI
import json
from resume import Resume
import pandas as pd
import openai
import numpy as np
from summarizer import Summarizer
import faiss

class Embedder:
    def __init__(self, api_key, embed_dims=1536):
        self.client = OpenAI(api_key=api_key)
        # embedding dimensions
        self.embed_dims = embed_dims
        return
    
    def _get_embedding(self, text):
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )

        return response.data[0].embedding

    def _cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get_similarity_score(self, resume, job):
        resume_embed = self._get_embedding(resume)
        job_embed = self._get_embedding(job)

        score = self._cosine_similarity(resume_embed, job_embed)
        return score
    
    def store_job_embeddings(self, embed_output, id_output, job_df):
        index = faiss.IndexFlatL2(self.embed_dims)

        id_mapping = {i: job_id for i, job_id in enumerate(job_df['job_id'])}

        # Save as JSON
        with open(id_output, "w") as f:
            json.dump(id_mapping, f)
        
        # embed all descriptions
        job_df['embedding'] = job_df['job_summary'].apply(self._get_embedding)

        embeddings = np.vstack(job_df['embedding'].values)

        index.add(embeddings)

        faiss.write_index(index, embed_output)

        return
    
    def get_topk_jobs(self, resume_text, job_embeds, k=2):
        resume_embed = self._get_embedding(resume_text)
        D, I = job_embeds.search(np.array([resume_embed]), k=k)
        return D, I

def main():
    with open('api_keys.json', 'r') as f:
        keys = json.load(f)

    api_key = keys['UTS_OPENAI_KEY']

    embedder = Embedder(api_key=api_key)
    summarizer = Summarizer(api_key=api_key)

    with open('resume_prompt.txt', 'r') as f:
        resume_prompt = f.read()

    with open('job_prompt.txt', 'r') as f:
        job_prompt = f.read()

    df = pd.read_csv('datasets/job_descs_exp.csv')

    df['job_summary'] = df['description'].apply(summarizer.summarize_info, prompt=job_prompt)

    '''
    save job embeddings
    '''
    # save_embed_file = 'job_embeds.faiss'
    # save_id_file = 'id_mapping.json'
    # embedder.store_job_embeddings(embed_output=save_embed_file, id_output=save_id_file, job_df=df)
    # return

    '''
    embed resume and get k suitable jobs
    '''
    path = '/Users/hainguyen/Desktop/Harry_Nguyen_Resume.pdf'

    resume_text = Resume(source=path, is_pdf=True).text

    df = pd.read_csv('datasets/job_descs.csv')

    description = df.iloc[2]['description']

    resume_sum = summarizer.summarize_info(prompt=resume_prompt, query=resume_text)

    # read embeddings from .faiss file
    job_embeds = faiss.read_index('job_embeds.faiss')

    D, I = embedder.get_topk_jobs(resume_text=resume_sum, job_embeds=job_embeds, k=5)
    print(D)
    print()
    print(I)

    return

if __name__ == '__main__':
    main()