import os
import pdfplumber
# import re 
# import spacy
from openai import OpenAI
import json
from IPython import embed


def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def main():
    # api = 'xai-XqsoyGjxLcwYoRWeqLYDOPDGmblj9cYTStZP5bSsnk40dx7wuMPj9Rh1CExD91PVV7WtUgYrtEqZbE04'
    # openai_key = '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'
    
    path = '/Users/hainguyen/Desktop/Harry_Nguyen_Resume.pdf'

    text = extract_text_from_pdf(path)
    
    client = OpenAI(
    api_key="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"
    )

    prompt = f"""
    Analyze the following resume text and extract the following information:
    - Name
    - Email
    - Skills (as a list)
    - Working Experience (as a list of objects containing: company name, job title, 
      start date, end date, key responsibilities as a list)
    
    Return the information in a structured JSON format. If any information is missing,
    use "Not provided" for strings or empty lists/arrays for skills and experience.
    
    Resume text:
    {text}
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
    result = result.replace("```json", "")
    result = result.replace("```", "")
    
    # Parse the JSON response (assuming OpenAI returns JSON-like string)
    # In practice, you might need to clean the response if it's not perfect JSON
    summary = json.loads(result)
    
    with open('test.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    return summary

if __name__ == '__main__':
    main()
