from openai import OpenAI
import os

class Summarizer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def summarize_info(self, prompt, query):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content.strip()

    def generate_skill_gap(self, resume_summary, job_summary):
        prompt = self._load_prompt("prompts/skill_gap_prompt.txt")
        input_text = f"[RESUME]\n{resume_summary}\n\n[JOB]\n{job_summary}"

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text}
            ]
        )
        return response.choices[0].message.content.strip()

    def _load_prompt(self, filepath):
        with open(filepath, "r") as f:
            return f.read().strip()
