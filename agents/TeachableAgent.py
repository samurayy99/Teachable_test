import json
import os
import logging
from dotenv import load_dotenv
from autogen.agentchat.contrib.teachable_agent import TeachableAgent
from autogen import UserProxyAgent
import openai
import requests
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configurations from environment and JSON
def load_configurations():
    load_dotenv()
    with open('config.json') as config_file:
        config = json.load(config_file)

    # Update config dictionary with environment variables
    openai_config = config.get('openai', {})
    config['openai'] = {
        'api_key': os.getenv('OPENAI_API_KEY', openai_config.get('api_key'))
    }

    config['test_mode'] = os.getenv('TEST_MODE', 'False').lower() == 'true'
    config['llm_config'] = config.get('llm_config', {})
    config['teach_config'] = config.get('teach_config', {})

    return config

config = load_configurations()

# Define individual agents
class FinancialDataAgent:
    def fetch_data(self, query):
        return fetch_financial_data(query)

class NewsDataAgent:
    def fetch_data(self, topic):
        return fetch_news_articles(topic)

class QueryProcessingAgent:
    def process_query(self, query):
        return classify_query(query)

# Create a Group Manager
class GroupManager:
    def __init__(self):
        self.financial_agent = FinancialDataAgent()
        self.news_agent = NewsDataAgent()
        self.query_agent = QueryProcessingAgent()

    def handle_query(self, user_input):
        query_type = self.query_agent.process_query(user_input)
        if query_type == "finance":
            return self.financial_agent.fetch_data(user_input)
        elif query_type == "news":
            return self.news_agent.fetch_data(user_input)
        return None

# Define the TeachableAgent with LLM selection and logging
class TeachableAgentWithLLMSelection:
    def __init__(self, name, llm_config, teach_config, group_manager):
        self.name = name
        self.llm_config = llm_config
        self.teach_config = teach_config
        self.group_manager = group_manager
        self.api_key = config['openai']['api_key']

    def respond_to_user(self, user_input):
        response = self.group_manager.handle_query(user_input)
        if response:
            summarized_response = summarize_and_validate(response)
            return visualized_data(summarized_response)
        return "No relevant data found."

    def call_openai_api(self, user_input):
        messages = [{"role": "user", "content": user_input}]
        payload = {"model": "gpt-4-1106-preview", "messages": messages}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            return f"Error in API response: {response.status_code}, {response.text}"
        except Exception as e:
            return f"Exception in API call: {str(e)}"

# Define the functions for query classification, fetching financial data and news, summarizing and validating data
def classify_query(user_input):
    financial_keywords = ["stock", "crypto", "market", "investment", "financial", "economy"]
    news_keywords = ["news", "headline", "current events", "article", "report"]
    if any(keyword in user_input.lower() for keyword in financial_keywords):
        return "finance"
    if any(keyword in user_input.lower() for keyword in news_keywords):
        return "news"
    return "general"

def fetch_financial_data(query):
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={query}&apikey={api_key}'
    try:
        response = requests.get(url)
        return response.json() if response.status_code == 200 else "Error fetching financial data"
    except Exception as e:
        return "Exception in fetching financial data"

def fetch_news_articles(topic):
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={topic}&apikey={api_key}'
    try:
        response = requests.get(url)
        return response.json() if response.status_code == 200 else "Error fetching news articles"
    except Exception as e:
        return "Exception in fetching news articles"

def summarize_and_validate(data):
    summary = mistral_llm(data)
    return validate_with_gpt4(summary)

def validate_with_gpt4(summary):
    try:
        response = openai.Completion.create(
            engine="gpt-4-1106-preview",
            prompt=f"Please review this summary for accuracy: '{summary}'",
            max_tokens=100,
            api_key=config['openai']['api_key']
        )
        review = response.choices[0].text.strip()
        logging.info(f"Summary review: {review}")
        return review if "satisfactory" in review.lower() else "Summary needs improvement."
    except Exception as e:
        logging.exception("Error calling GPT-4 API for summary validation")
        return "Unable to validate summary."

def mistral_llm(query):
    API_URL = "https://z983hozbx3in30io.us-east-1.aws.endpoints.huggingface.cloud"
    auth_token = "Bearer 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"
    headers = {
        "Authorization": auth_token,
        "Content-Type": "application/json"
    }
    data = json.dumps({"inputs": query})

    try:
        response = requests.post(API_URL, headers=headers, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"API call failed: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logging.exception("Exception occurred during API call")
        return None

def visualized_data(data):
    return f"Visualized Data: {data}"

# Define available LLMs and their simulated or actual invocation logic
models_dict = {
    'mistral-llm': mistral_llm,
}

# Simplified function for model selection
# Initialize the group manager and teachable agent
group_manager = GroupManager()
teachable_agent = TeachableAgentWithLLMSelection(
    name="financial_teachable_agent",
    llm_config=config['llm_config'],
    teach_config=config['teach_config'],
    group_manager=group_manager
)

# Create UserProxyAgent
user = UserProxyAgent("user", human_input_mode="ALWAYS")

# Chat with TeachableAgent
teachable_agent.initiate_chat(user, message="Hi, I'm a teachable user assistant! What's on your mind?")

# Update the database
teachable_agent.learn_from_user_feedback()
teachable_agent.close_db()
