
- extract info from resume
    - name, email (NER)
    - skills (keyword, NER, stransformer)
- extract info from job posts

- LDA or other techniques to category job industries -> overview insights to job market
- use LLM to gen explanations for matching result
- breakdown embedding for different section -> better understanding results
- resume tailor suggestion (gaps, mismatch, recommendation, etc)
- Q&A:​
    “Why is this job a good fit for me?”
    “Which skills should I highlight more prominently?”
    “What aspects of my experience align with this role?”​
    Leveraging LLMs to handle these queries can provide users with personalized and detailed responses, enhancing their understanding and engagement.

DOCS:
- https://medium.com/thedeephub/resume-scanner-leverage-the-power-of-llm-to-improve-your-resume-401a0cb49cd7
- https://medium.com/before-you-launch/a-broke-b-chs-guide-to-tech-start-up-choosing-llm-api-prices-ad451a2abfd6
- https://github.com/sliday/resume-job-matcher/blob/master/resume_matcher.py
- https://github.com/Aillian/ResumeGPT


recreate faiss, id_mapping and data_merge file