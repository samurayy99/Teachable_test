import json
import os
import logging
import requests
import traceback
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from autogen.agentchat.contrib import TeachableAgent as AutoGenTeachableAgent
from db.database import MemoStore  # Ensure this path is correct
# Main logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lenox.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_configurations():
    """
    Load configurations from environment variables and JSON file.
    Returns:
        dict: Configuration dictionary.
    Raises:
        FileNotFoundError: If the config file is not found.
        json.JSONDecodeError: If the config file is not valid JSON.
    """
    load_dotenv()

    config_path = Path('config.json')
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open() as config_file:
        try:
            config = json.load(config_file)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in the configuration file: {e}")
            raise

    openai_api_key = os.getenv('OPENAI_API_KEY') or config.get('openai', {}).get('api_key')
    if not openai_api_key:
        logger.warning("OpenAI API key is missing. Please set it in the environment variables or config.json")

    config['openai'] = {'api_key': openai_api_key}
    config['test_mode'] = os.getenv('TEST_MODE', 'False').lower() == 'true'
    config['llm_config'] = config.get('llm_config', {})
    config['teach_config'] = config.get('teach_config', {})

    logger.info("Configurations loaded successfully.")
    return config

config = load_configurations()


# FinancialDataAgent Implementation
class FinancialDataAgent:
    def fetch_data(self, query):
        try:
            response = requests.get(f"https://financialdata.api/query?param={query}")
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching financial data: {e}")
            return {"status": "error", "data": f"Error fetching data: {str(e)}"}

class NewsDataAgent:
    def fetch_data(self, topic):
        try:
            response = requests.get(f"https://newsdata.api/query?topic={topic}")
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news data: {e}")
            return {"status": "error", "data": f"Error fetching data: {str(e)}"}


# QueryProcessingAgent Implementation
class QueryProcessingAgent:
    def __init__(self, group_manager):
        self.group_manager = group_manager
    def process_query(self, query):
        # Basic implementation - should be enhanced for more complex logic
        if "stock" in query.lower() or "finance" in query.lower():
            return self.group_manager.financial_agent.fetch_data(query)
        elif "news" in query.lower():
            return self.group_manager.news_agent.fetch_data(query)
        else:
            return {"status": "unknown", "data": "Query type unknown"}

# Helper function to parse financial queries
def parse_financial_query(query):
    # Implement logic to parse and extract key information from a financial query
    # Placeholder logic
    return query.split()[0]

# Function to handle financial queries using different models based on query complexity
def handle_financial_query(query, llm_config):
    parsed_query = parse_financial_query(query)
    model_choice = "text-davinci-003" if len(parsed_query.split()) > 3 else "text-curie-001"
    api_key = llm_config['models'][model_choice].get('api_key')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "prompt": f"Financial analysis of: {query}",
        "temperature": 0.5,
        "max_tokens": 100
    }

    try:
        response = requests.post(f"https://api.openai.com/v1/engines/{model_choice}/completions", json=data, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return f"Error fetching response for query: {query}"



# GroupManager Implementation
class GroupManager:
    def __init__(self):
        self.financial_agent = FinancialDataAgent()
        self.news_agent = NewsDataAgent()
        self.query_agent = QueryProcessingAgent(self)

    def handle_query(self, user_input):
        return self.query_agent.process_query(user_input)
    
    def handle_enhanced_query(self, user_input):
        query_type = classify_query(user_input)
        if query_type == "finance":
            return handle_financial_query(user_input, self.llm_config)
        elif query_type == "news":
            # Placeholder for handling news queries
            return f"News response for query: {user_input}"
        else:
            # Fallback to a generic response or another model
            return f"Generic response for query: {user_input}"
        
    def classify_query(user_input):
    # Simple example of classification logic
        if "stock" in user_input.lower() or "finance" in user_input.lower():
            return "finance"
        elif "news" in user_input.lower():
            return "news"
        else:
            return "general"
 
    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of the given text.
        Args:
            text (str): Text to analyze.
        Returns:
            str: Sentiment of the text ('positive', 'negative', 'neutral').
        """
        try:
            response = requests.post(
                "https://sentiment-analysis.api",
                json={"text": text}
            )
            response.raise_for_status()
            sentiment = response.json()["sentiment"]
            return sentiment
        except requests.exceptions.RequestException as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return "neutral"  # Default to neutral if there's an error

    def handle_enhanced_query(self, user_input):
        sentiment = self.analyze_sentiment(user_input)
        # Modify the response based on the sentiment analysis
        if sentiment == "negative":
            return "I understand that you might be upset. Let's see how I can help."

# TeachableAgentWithLLMSelection API Integration
class TeachableAgentWithLLMSelection(AutoGenTeachableAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = config['openai']['api_key']
        self.group_manager = kwargs.get('group_manager', GroupManager())

    def call_openai_api(self, user_input):
        messages = [{"role": "user", "content": user_input}]
        payload = {"model": "gpt-4-1106-preview", "messages": messages}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error in API response: {response.status_code}, {response.text}"
        
    def respond_to_user(self, user_input):
        try:
            response = self.group_manager.handle_query(user_input)
            if response and response["status"] != "unknown":
                return response["data"]
            else:
                return self.call_openai_api(user_input)
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            return "Sorry, I encountered an error while processing your request."
        
    def enhanced_respond_to_user(self, user_input):
        # Use the enhanced query handling method in GroupManager
        response = self.group_manager.handle_enhanced_query(user_input)
        if response:
            return response
        else:
            # Fallback to OpenAI API call
            return self.call_openai_api(user_input)

    def start_chat(self):
        logger.info("Lenox is ready to chat!")
        # Additional startup logic

    def end_chat(self):
        logger.info("Ending chat session with Lenox.")
        self.close_db()   


# Initialize the TeachableAgent
group_manager = GroupManager()
teachable_agent = TeachableAgentWithLLMSelection(
    name="Lenox",
    llm_config=config['llm_config'],
    teach_config=config['teach_config'],
    group_manager=group_manager
)

# Main Interaction Loop
def main():
    try:
        print("Welcome to Lenox, your personal assistant. Type 'exit' to end the session.")
        user_input = input("You: ")

        while user_input.lower() != 'exit':
            response = teachable_agent.respond_to_user(user_input)
            print("Lenox:", response)
            user_input = input("You: ")

        teachable_agent.end_chat()
        print("Thank you for using Lenox. Have a great day!")
    except KeyboardInterrupt:
        print("\nSession interrupted by the user.")
        teachable_agent.end_chat()
    except Exception as e:
        logger.error(f"Unexpected error occurred: {traceback.format_exc()}")
        print("An unexpected error occurred. Please try again later.")
    finally:
        print("Session with Lenox safely closed.")

# Run Lenox
if __name__ == "__main__":
    main()
        



