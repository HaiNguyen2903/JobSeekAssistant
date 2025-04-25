from ragas.evaluation import evaluate
from ragas.metrics import answer_relevancy
from datasets import Dataset
import argparse

def evaluate_summary(question, answer, context):
    data = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [[context]],
    })

    result = evaluate(data, metrics=[answer_relevancy])
    return result['answer_relevancy'][0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate summary relevance with RAGAS")
    parser.add_argument('--question', type=str, required=True, help='The question the summary is answering')
    parser.add_argument('--answer', type=str, required=True, help='The summary generated')
    parser.add_argument('--context', type=str, required=True, help='The original context content')
    
    args = parser.parse_args()

    score = evaluate_summary(args.question, args.answer, args.context)
    print(f"✅ RAGAS answer_relevancy score: {score:.4f}")
