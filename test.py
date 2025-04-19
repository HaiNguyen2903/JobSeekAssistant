import os
import pdfplumber
# import re 
# import spacy
from openai import OpenAI
import json
from IPython import embed
import numpy as np
import openai
import requests
import pandas as pd


def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def query(api_url, headers, payload):
        response = requests.post(api_url, headers, json=payload)
        return response.json()

def main():
    # api = 'xai-XqsoyGjxLcwYoRWeqLYDOPDGmblj9cYTStZP5bSsnk40dx7wuMPj9Rh1CExD91PVV7WtUgYrtEqZbE04'
    # openai_key = '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'
    
    path = '/Users/hainguyen/Desktop/Harry_Nguyen_Resume.pdf'

    # text = extract_text_from_pdf(path)

    with open('api_keys.json', 'r') as f:
        keys = json.load(f)
    
    api_key = keys['UTS_OPENAI_KEY']

    openai.api_key = api_key

    # API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    # headers = {"Authorization": "Bearer ####"}

        
    # output = query(
    #     api_url=API_URL,
    #     headers=headers,
    #     payload=
    #     {
    #     "inputs": "[INST] You are a helpful chatbot assistant which provides answer based on the context given. Do not give any extra information. Do not give the context again in your response\nGenerate a concise and informative answer in less than 100 words for the given question\nSEARCH RESULT 1: The title is Rush, year is 2013, budget is 500000, earning is 300000, genere is action.\nQUESTION: What is the release date of rush?\n[/INST]",
    #     "parameters": {
    #         "return_full_text": False
    #     }
    # })

    # print(output)

    '''
    embedding model
    '''
    
    # resume_info = """
    # Experienced in Power BI and Tableau.
    # """

    # job_description = """
    # We are hiring candidate with strong experience in data visualization tools.
    # """

    # # Get embeddings
    # resume_embedding = get_embedding(resume_info)
    # job_embedding = get_embedding(job_description)

    # # Compute similarity score
    # similarity_score = cosine_similarity(resume_embedding, job_embedding)

    # # Print result
    # print(f"Resume-Job Matching Score: {similarity_score:.4f}")

    '''
    GPT 4.0
    '''

    client = OpenAI(
    api_key=api_key
    )

    # df = pd.read_csv('datasets/UpdatedResumeDataSet.csv')
    # text = df.iloc[1]['Resume']

    # text = extract_text_from_pdf('datasets/resume_pdf/1901841_RESUME.pdf')
    text = extract_text_from_pdf(path)


    # return_format = {
    #     'qualification': 'Education background (e.g bachelor degree, master degree, etc)',
    #     'skills': 'Relevant skills',
    #     'experience': 'A list of crucial working experience (e.g. technologies and skills used), role title and working time at that position'
    # }

    # prompt = f"""
    # Analyze the following resume text and extract the crucial information for finding suitable jobs.
    # Only Return the information in python dictionary format and do not return anything else. If any information is missing,
    # use "Not provided" for strings or empty lists/arrays for skills and experience.

    # ----
    # Return format:\n
    # {return_format}
    # ----
    # Resume text:\n
    # {text}
    # """

    return_format = {
        "key_skills": [],
        "industries_or_job_types": [],
        "unique_strengths": [],
        "suitable_career_levels": []
        }

    prompt = f"""
    You are an AI assistant that helps job seekers find suitable job opportunities by analyzing their resumes.
    I will provide a resume, and your task is to carefully extract and summarize the key strengths, skills, experiences, and potential career directions for this candidate from the job seeker’s perspective.
    Focus on identifying:
    - Key technical and soft skills
    - Most relevant industries or job types for the candidate
    - Highlight accomplishments or unique strengths that would make them stand out
    - Career levels they are suitable for (Entry, Mid, Senior)

    Output the results in structured JSON format like this:
    
    {return_format}
    
    Here is the candidate's resume:
    ----
    {text}
    ----
    """

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information from resumes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
    )

    # Extract the response content
    result = response.choices[0].message.content

    print(result)
    # result = result.replace("```json", "")
    # result = result.replace("```", "")
    
    # # Parse the JSON response (assuming OpenAI returns JSON-like string)
    # # In practice, you might need to clean the response if it's not perfect JSON
    # summary = json.loads(result)
    
    # with open('test.json', 'w', encoding='utf-8') as f:
    #         json.dump(summary, f, indent=2)

    # return summary

if __name__ == '__main__':
    main()
