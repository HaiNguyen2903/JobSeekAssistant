from openai import OpenAI
import json
from resume import Resume
import pandas as pd

class Summarizer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        return
    
    def summarize_info(self, query, prompt):
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

        # # Convert the JSON string into a Python dictionary
        # try:
        #     result = json.loads(result)
        # except json.JSONDecodeError as e:
        #     print("Failed to parse JSON:", e)

        return result

def main():
    return 

if __name__ == '__main__':
    main()

