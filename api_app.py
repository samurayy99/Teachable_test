import requests
import os
import json

def test_openai_api():
    api_key = os.getenv('OPENAI_API_KEY')  # Ensure this is set in your environment
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4-1106-preview",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about Bitcoin."}
        ]
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        # Parse the JSON response
        json_response = response.json()
        if 'choices' in json_response and len(json_response['choices']) > 0:
            # Extract the message from the first choice
            message = json_response['choices'][0].get('message', {}).get('content', '')
            print(f"Response Message: {message}")
        else:
            print("Response does not contain the expected 'choices' structure.")
    else:
        print(f"Status Code: {response.status_code}\nResponse: {response.text}")

test_openai_api()
