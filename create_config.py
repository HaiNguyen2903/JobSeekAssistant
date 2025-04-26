import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate project config")
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key', default='YOUR API KEY')
    
    args = parser.parse_args()

    config = {
    "PROMPT_DIR": "prompts",
    "EMBEDDING_DIR": "embeddings",
    "DATA_DIR": "datasets",
    "OPENAI_KEY": args.api_key,
    }

    with open('config.json', 'w') as f:
        json.dump(config, f)