import requests
import json

def main():
    with open('api_keys.json', 'r') as f:
        keys = json.load(f)

    api_key = keys['PERSONAL_OPENAI_KEY']

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    # Example: Check usage from April 1 to April 19, 2025
    response = requests.get(
        "https://api.openai.com/v1/dashboard/billing/usage?start_date=2025-04-01&end_date=2025-04-19",
        headers=headers
    )

    print(response.json())


if __name__ == '__main__':
    main()
