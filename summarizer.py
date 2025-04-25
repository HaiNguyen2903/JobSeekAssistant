import os
import openai

class Summarizer:
    def __init__(self, api_key):
        openai.api_key = api_key
        # self.resume_promt_file = os.path.join("prompts", "resume_prompt.txt")
        self.client = openai

    def summarize_info(self, prompt, query):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content.strip()

    # def tailor_cv(self, resume_summary, job_summary):
    #     prompt = self._load_prompt("prompts/tailor_cv_prompt.txt")
    #     input_text = f"[RESUME]\n{resume_summary}\n\n[JOB]\n{job_summary}"
    #     return self._ask_gpt(prompt, input_text)

    # def generate_cover_letter(self, resume_summary, job_summary):
    #     prompt = self._load_prompt("prompts/cover_letter_prompt.txt")
    #     input_text = f"[RESUME]\n{resume_summary}\n\n[JOB]\n{job_summary}"
    #     return self._ask_gpt(prompt, input_text)

    # def get_skill_gap(self, resume_summary, job_summary):
    #     prompt = self._load_prompt("prompts/skill_gap_prompt.txt")
    #     input_text = f"[RESUME]\n{resume_summary}\n\n[JOB]\n{job_summary}"
    #     return self._ask_gpt(prompt, input_text)

    def get_llm_response(self, prompt_file, user_input):
        instruction = self._load_prompt(prompt_file)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content.strip()

    def _load_prompt(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
