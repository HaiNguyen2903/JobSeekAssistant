from openai import OpenAI
import json
from resume import Resume
import pandas as pd

class Summarizer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        return
    
    def summarize_info(self, prompt, query):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.2,
                    max_tokens=1000
            )

        # Extract the response content
        result = response.choices[0].message.content

        # Convert the JSON string into a Python dictionary
        try:
            result = json.loads(result)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)

        return result

if __name__ == '__main__':
    with open('api_keys.json', 'r') as f:
        keys = json.load(f)

    summarizer = Summarizer(api_key=keys['UTS_OPENAI_KEY'])

    # path = '/Users/hainguyen/Desktop/Harry_Nguyen_Resume.pdf'

    # resume = Resume(source=path)

    # summarizer = Summarizer(api_key=keys['UTS_OPENAI_KEY'])

    # with open('resume_prompt.txt', 'r') as f:
    #     prompt = f.read()

    # # prompt = prompt.replace('resume_content', resume.text)

    # final = summarizer.summarize_info(prompt, query=resume.text)

    # print(final)

    '''
    summary job
    # '''
    df = pd.read_csv('datasets/job_descs.csv')

    description = df.iloc[2]['description']

    with open('job_prompt.txt', 'r') as f:
        prompt = f.read()

    final = summarizer.summarize_info(prompt, query=description)

    print(final)

