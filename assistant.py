import os
import openai
import json
from resume import Resume

class LLM_Assistant:
    def __init__(self, config):
        openai.api_key = config["OPENAI_KEY"]

        resume_prompt_file = os.path.join(config["PROMPT_DIR"], "resume_prompt.txt")
        job_prompt_file = os.path.join(config["PROMPT_DIR"], "job_prompt.txt")
        cover_letter_prompt_file = os.path.join(config["PROMPT_DIR"], "cover_letter_prompt.txt")
        skill_gap_prompt_file = os.path.join(config["PROMPT_DIR"], "skill_gap_prompt.txt")
        fit_explain_prompt_file = os.path.join(config["PROMPT_DIR"], "fit_explain_prompt.txt")

        self.resume_prompt = self._load_prompt(resume_prompt_file)
        self.job_prompt = self._load_prompt(job_prompt_file)
        self.cover_letter_prompt = self._load_prompt(cover_letter_prompt_file)
        self.skill_gap_prompt = self._load_prompt(skill_gap_prompt_file)
        self.fit_explain_prompt = self._load_prompt(fit_explain_prompt_file)

        self.client = openai

    def get_llm_response(self, instruction, user_input):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content.strip()

    def _load_prompt(self, file_path):
        with open(file_path, "r") as f:
            return f.read()

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    llm = LLM_Assistant(config)