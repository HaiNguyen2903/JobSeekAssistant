import pdfplumber
import pandas as pd
# import re
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from transformers import pipeline
# from transformers import BertTokenizer
# from sentence_transformers import SentenceTransformer, util
from sentence_transformers import InputExample, SentenceTransformer, util
import random


# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')#Resources for POS tagging
# nltk.download('tagsets_json')#Resources for POS tagging

# nltk.help.upenn_tagset()


def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def get_resume_example(csv_path, index=0):
    df = pd.read_csv(csv_path)
    # resume = df.sample(n=1)
    return {
        'title': df.iloc[index]['Category'],
        'detail': df.iloc[index]['Resume']
    }

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stop words
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text)

    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)

def extract_resume_skills(resume_detail):
    nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    entities = nlp(resume_detail)
    # Filter the results to capture only skills
    skills = [entity['word'] for entity in entities if entity['entity'] == 'B-LOC']
    print(f"Extracted skills: {skills}")
    return

def extract_resume_exp(resume_detail):
    return

def extract_resume_edu(resume_detail):
    return 

def gen_matching_data(excel_path, pos_samples=100, neg_samples=100):
    df = pd.read_excel(excel_path)
    df = df[['Example', 'Commodity Title']]

    positive_examples = [
        InputExample(texts=[row['Example'], row['Commodity Title']], label=1.0)
        for _, row in df.iterrows()
    ]

    categories = df['Commodity Title'].unique().tolist()
    negative_examples = []

    for _, row in df.iterrows():
        wrong_category = random.choice([c for c in categories if c != row['Commodity Title']])
        negative_examples.append(
            InputExample(texts=[row['Example'], wrong_category], label=0.0)
        )

    all_examples = positive_examples + negative_examples
    return random.shuffle(all_examples)

if __name__ == '__main__':
    print(gen_matching_data('datasets/Technology Skills.xlsx'))

    # Load pre-trained BERT model from Sentence-Transformers
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Small & fast, works well

    sentence1 = "Candidate must be good at visualization software"
    sentence2 = "Proficient at Power BI and Tableau"

    # Get sentence embeddings
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)

    print(f"Similarity Score: {cosine_score.item():.4f}")
