from openai import OpenAI
import json
from resume import Resume
import pandas as pd
import numpy as np
from summarizer import Summarizer
import faiss
import os.path as osp

class LLM_Embedder:
    def __init__(self, config, embed_dims=1536):
        self.client = OpenAI(api_key=config['UTS_OPENAI_KEY'])
        # embedding dimensions
        self.embed_dims = embed_dims
        
        job_embed_file = osp.join(config['EMBEDDING_DIR'], 'job_embeds.faiss')
        id_mapping_file = osp.join(config['EMBEDDING_DIR'], 'id_mapping.json')
        
        with open(id_mapping_file, 'r') as f:
            self.id_mapping = json.load(f)

        # job embeddings
        self.job_embeds = faiss.read_index(job_embed_file) 

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
    
    def store_job_embeddings(self, embed_output, id_output, job_df):
        index = faiss.IndexFlatL2(self.embed_dims)

        id_mapping = {i: job_id for i, job_id in enumerate(job_df['job_id'])}

        # Save as JSON
        with open(id_output, "w") as f:
            json.dump(id_mapping, f)
        
        # embed all descriptions
        job_df['embedding'] = job_df['job_summary'].apply(self._get_embedding)

        embeddings = np.vstack(job_df['embedding'].values).astype('float32')

        # normalize embeddings
        faiss.normalize_L2(embeddings)

        # add to index
        index.add(embeddings)

        faiss.write_index(index, embed_output)

        return
    
    def get_sim_scores(self, resume_text, job_embeds, k=2):
        resume_embed = np.array(self._get_embedding(resume_text), dtype='float32').reshape(1, -1)
        
        # normalize embedding
        faiss.normalize_L2(resume_embed)

        D, I = job_embeds.search(resume_embed, k=k)
        return D, I

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    embedder = LLM_Embedder(config=config)

    '''
    save job embeddings
    '''
    df = pd.read_csv(osp.join(config['DATA_DIR'], 'job_merged.csv'))

    save_embed_file = osp.join(config['EMBEDDING_DIR'], 'job_embeds.faiss')
    save_id_file = osp.join(config['EMBEDDING_DIR'], 'id_mapping.json')

    embedder.store_job_embeddings(embed_output=save_embed_file, id_output=save_id_file, job_df=df)
    return

if __name__ == '__main__':
    main()