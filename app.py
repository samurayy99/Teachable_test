# Standard library imports
import json
import os
import logging
import time  # Added import for time

# Third-party imports
from dotenv import load_dotenv
import requests
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt  # Added import for matplotlib
import seaborn as sns  # Added import for seaborn
import nltk  # Added import for nltk
from nltk.tokenize import word_tokenize  # Added specific import for tokenize
from nltk.corpus import stopwords  # Added specific import for stopwords

# Assuming 'openai' and 'autogen' are correctly installed or available in your environment
import openai  # Added import for openai
import autogen  # Added import for autogen

# Additional imports for advanced functionalities
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

import os
from autogen.agentchat import Agent
from autogen.agentchat.assistant_agent import ConversableAgent
from autogen.agentchat.text_analyzer_agent import TextAnalyzerAgent
from chromadb import chromadb


class TeachableAgent(ConversableAgent):
  """(Experimental) TeachableAgent remembers user teachings in a vector database."""

  def __init__(
    self,
    name="teachableagent",
    system_message="You are a helpful AI assistant that remembers user teachings from prior chats.",
    human_input_mode="NEVER",
    llm_config: Optional[Union[Dict, bool]] = None,
    analyzer_llm_config: Optional[Union[Dict, bool]] = None,
    teach_config: Optional[Dict] = None,
  **kwargs,
  ):
    """
    Args:
      - name (str): name of the agent.
      - system_message (str): system message for the ChatCompletion inference.
      - human_input_mode (str): This agent should NEVER prompt the human for input.
      - llm_config (dict or False): llm inference configuration.
      - analyzer_llm_config (dict or False): llm inference configuration passed to TextAnalyzerAgent.
      - teach_config (dict or None): Additional parameters used by TeachableAgent.
      **kwargs (dict): other kwargs in ConversableAgent.
    """
    super().__init__(
      name=name,
      system_message=system_message,
      human_input_mode=human_input_mode,
      llm_config=llm_config,
      analyzer_llm_config=analyzer_llm_config,
      **kwargs,
    )

    # Register a custom reply function.
    self.register_reply(Agent, self._generate_teachable_assistant_reply, position=2)

   # Assemble the parameter settings.
    self._teach_config = {} if teach_config is None else teach_config
    self.verbosity = self._teach_config.get("verbosity", 0)
    self.reset_db = self._teach_config.get("reset_db", False)
    self.path_to_db_dir = self._teach_config.get("path_to_db_dir", "./tmp/teachable_agent_db")

   # Create the analyzer.
   if analyzer_llm_config is None:
     analyzer_llm_config = self.llm_config
   self.analyzer = TextAnalyzerAgent(llm_config=analyzer_llm_config)

   # Create the memo store.
   self.memo_store = MemoStore(verbosity=self.verbosity, reset=self.reset_db, path_to_db_dir=self.path_to_db_dir)

  def learn_from_user_feedback(self):
    """Reviews the user comments from the last chat, and decides what teachings to store as memos."""
    print(colored("\nREVIEWING CHAT FOR USER TEACHINGS TO REMEMBER", "light_yellow"))
    self.user_comments = []

  def consider_memo_storage(self, comment):
    """Decides whether to store something from one user comment in the DB."""
    # Check for a problem-solution pair.
    response = self.analyze(
      comment,
      "Does any part of the TEXT ask the agent to perform a task or solve a problem? Answer with just one word, yes or


# Load configurations from environment and JSON
def load_configurations():
    """ 
    Load configurations from environment and JSON. 
    Returns: 
    dict: The configuration dictionary. 
    """
    load_dotenv()
    with open('config.json') as config_file:
        config = json.load(config_file)
    config['openai'] = {
        'api_key': os.getenv('OPENAI_API_KEY', config.get('api_key'))  # Securely fetch API key
    }
    config['test_mode'] = os.getenv('TEST_MODE', 'False').lower() == 'true'
    config['llm_config'] = config.get('llm_config', {})
    config['teach_config'] = config.get('teach_config', {})
    return config

config = load_configurations()

class FinancialAIAdvisor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.sentiment_analyzer = self._initialize_sentiment_analyzer()

    def _initialize_sentiment_analyzer(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
        return tokenizer, model

    def analyze_market_sentiment(self, text):
        tokenizer, model = self.sentiment_analyzer
        inputs = tokenizer(text, return_tensors="tf")
        outputs = model(inputs)
        prediction = tf.nn.softmax(outputs.logits, axis=-1)
        return prediction.numpy()

    def get_stock_data(self, symbol):
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={self.api_key}"
        response = requests.get(url)
        data = response.json()
        return data

    def compute_var(self, financial_data, confidence_level=0.95):
        portfolio_values = np.array([data['close'] for data in financial_data])  # Simplified example
        var = np.percentile(portfolio_values, (1 - confidence_level) * 100)
        return var

    # Additional methods for financial analysis

class GroupManager:
    def __init__(self, api_key):
        self.financial_advisor = FinancialAIAdvisor(api_key)

    def handle_query(self, user_input):
        # Example of handling different types of queries
        if 'stock' in user_input.lower():
            return self.financial_advisor.get_stock_data(user_input.split()[-1])
        elif 'sentiment' in user_input.lower():
            return self.financial_advisor.analyze_market_sentiment(user_input)
        else:
            return "General financial advice"

class TeachableAgentWithLLMSelection:
    def __init__(self, name, llm_config, teach_config, group_manager):
        self.name = name
        self.llm_config = llm_config
        self.teach_config = teach_config
        self.group_manager = group_manager

    def respond_to_query(self, user_input):
        return self.group_manager.handle_query(user_input)
        pass

    def update_knowledge_base(self, feedback_dataset):
        # Logic to update knowledge base based on user feedback
        pass

    def respond_to_user(self, user_input):
        """
        Provide a response to the user input.
        Args:
            user_input: The query from the user.
        Returns:
            The response string.
        """
        response = self.group_manager.handle_query(user_input)
        return response if response else "No relevant data found."

    def process_and_store_feedback(self, user_input, agent_response):
        # Logic to process and store feedback, if relevant
        pass

    def close_db(self):
        # Close any database connections, if relevant
        pass

def call_openai_api(self, user_input):
    """
    Call the OpenAI API to get a response.
    Args:
        user_input: The user input string.
    Returns:
        The API response.
    """
    try:
        messages = [{"role": "user", "content": user_input}]
        payload = {"model": "gpt-4-1106-preview", "messages": messages}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        elif response.status_code == 429:
            logging.error("API rate limit exceeded. Please try again later.")
            return "API rate limit exceeded. Please try again later."
        else:
            raise Exception(f"Error in API response: {response.status_code}, {response.text}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        return f"Unexpected error: {str(e)}"

def process_feedback(self, feedback):
        """
        Process the feedback.
        Args:
            feedback: The feedback dictionary.
        Returns:
            Processed feedback or None if invalid.
        """
        if not isinstance(feedback, dict):
            logging.error("Invalid feedback format")
            return None
        return {k: v.strip() for k, v in feedback.items() if k in ['user_input', 'agent_response']}

def log_interaction(self, user_input, agent_response):
        """
        Log the interaction between the user and the agent.
        Args:
            user_input: The input from the user.
            agent_response: The response from the agent.
        """
        log_entry = {
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": time.time()
        }
        with open('interaction_log.txt', 'a') as file:
            file.write(json.dumps(log_entry) + '\n')

def clean_and_preprocess_data(self, data):
        """
        Clean and preprocess the data.
        Args:
            data: The data dictionary.
        Returns:
            Cleaned data.
        """
        return {k: v.strip() for k, v in data.items()}

# Custom exception for API errors
class APIError(Exception):
    pass

class CryptoAdvisor:
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

    @staticmethod
    def construct_alpha_vantage_url(function, from_currency, to_currency):
        base_url = 'https://www.alphavantage.co/query'
        return f'{base_url}?function={function}&from_currency={from_currency}&to_currency={to_currency}&apikey={CryptoAdvisor.ALPHA_VANTAGE_API_KEY}'

    def fetch_crypto_data(self, crypto_name, to_currency='USD'):
        url = CryptoAdvisor.construct_alpha_vantage_url('CURRENCY_EXCHANGE_RATE', crypto_name, to_currency)
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error in API call: {str(e)}, URL: {url}")
        return None

    def advise_on_crypto(self, query):
        crypto_name = query.split()[0]  # Simplified extraction
        crypto_data = self.fetch_crypto_data(crypto_name)
        if crypto_data is None:
            return "Unable to fetch cryptocurrency data."
        exchange_rate = crypto_data.get("Realtime Currency Exchange Rate", {}).get("5. Exchange Rate", "No data")
        return f"The current exchange rate of {crypto_name} is {exchange_rate}."
    
class FinancialPlanner:
    def create_financial_plan(self, user_data):
        """
        Create a financial plan based on user data.
        Args:
            user_data: Information about the user's finances.
        Returns:
            A string containing the financial plan.
        """
        savings_plan = "Savings Plan: Save 20% of monthly income."
        investment_plan = "Investment Plan: Diversify investments across stocks and bonds."
        return f"{savings_plan}\n{investment_plan}"

class DebtRepairAdvisor:
    def provide_debt_repair_advice(self, debt_info):
        """
        Provide advice on debt repair.
        Args:
            debt_info: Information about the user's debts.
        Returns:
            A string containing debt repair advice.
        """
        repayment_plan = "Repayment Plan: Focus on high-interest debts first."
        consolidation_advice = "Consider debt consolidation for multiple high-interest debts."
        return f"{repayment_plan}\n{consolidation_advice}"

class FinancialAdvisor:
    def interact_with_user(self, user_data):
        """
        Simulate interaction with the user to obtain financial details.
        Args:
            user_data: Information about the user's finances.
        Returns:
            A string summarizing the interaction results.
        """
        budget_query = "Yes, I have a monthly budget set up." if user_data["budget"] else "No, I haven't set up a budget yet."
        savings_goals = "My long-term savings goal is to save for " + " and ".join(user_data["long_term_goals"]) + "."
        return f"{budget_query}\n{savings_goals}"

def advise(self, user_id, query):
        """
        Provide financial advice based on the user's query.
        Args:
            user_id: The user's identifier.
            query: The financial query from the user.
        Returns:
            A string containing financial advice.
        """
        # Logic for integrating financial models or algorithms based on user data
        return f"Financial advice based on user's query about {query}."

def call_openai_api(self, user_input):
        """
        Call the OpenAI API with the provided user input.
        Args:
            user_input (str): The input from the user.
        Returns:
            The response from the API.
        """
        messages = [{"role": "user", "content": user_input}]
        payload = {"model": "gpt-4-1106-preview", "messages": messages}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logging.error(f"Error in OpenAI API call: {str(e)}")
            return f"Unexpected error: {str(e)}"

def classify_query(user_input):
    """
    Classify the user input into categories.
    Args:
        user_input (str): The input query from the user.
    Returns:
        A string representing the category of the query.
    """
    financial_keywords = ["stock", "crypto", "market", "investment", "financial", "economy"]
    news_keywords = ["news", "headline", "current events", "article", "report"]
    tokens = word_tokenize(user_input.lower())
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    if any(keyword in filtered_tokens for keyword in financial_keywords):
        return "finance"
    elif any(keyword in filtered_tokens for keyword in news_keywords):
        return "news"
    else:
        return "general"

def fetch_financial_data(query):
    """
    Fetch financial data based on the query.
    Args:
        query (str): The financial query.
    Returns:
        The fetched financial data or an error message.
    """
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={query}&apikey={api_key}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            return "API limit exceeded"
        else:
            return f"Error fetching financial data: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
    
    def fetch_news_articles(topic):
     """
    Fetch news articles based on the topic.
    Args:
        topic (str): The news topic.
    Returns:
        The fetched news articles or an error message.
    """
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={topic}&apikey={api_key}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            return "API limit exceeded"
        else:
            return f"Error fetching news articles: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
    
def summarize_and_validate(data, user_input):
    """
    Summarize and validate the given data based on the user's input.
    Args:
        data: The data to be summarized.
        user_input: The user's input for contextual relevance.
    Returns:
        The summary if valid and relevant, otherwise an error message.
    """
    # Use Mistral model to summarize the data
    summary = mistral_llm(data)
    # Validate the summary using GPT-4
    validation = validate_with_gpt4(summary)
    if "accurate" in validation.lower() and "coherent" in validation.lower():
        if user_input.lower() in summary.lower():
            return summary
        else:
            return "The summary was not relevant to your query."
    else:
        return "The summary was not accurate or coherent."

def validate_with_gpt4(summary):
    """
    Validate the given summary using GPT-4.
    Args:
        summary: The summary to be validated.
    Returns:
        Validation result as a string.
    """
    try:
        response = openai.Completion.create(
            engine="gpt-4-1106-preview",
            prompt=f"Please review this summary for accuracy and coherence: '{summary}'",
            max_tokens=100,
            api_key=config['openai']['api_key']
        )
        review = response.choices[0].text.strip()
        logging.info(f"Summary review: {review}")
        return "accurate and coherent" if "accurate" in review.lower() and "coherent" in review.lower() else "not accurate or coherent"
    except Exception as e:
        logging.exception("Error calling GPT-4 API for summary validation")
        return "Unable to validate summary."

def mistral_llm(query):
    """
    Query the Mistral model.
    Args:
        query: The query to be sent to the model.
    Returns:
        The model's response or None if an error occurs.
    """
    API_URL = "https://z983hozbx3in30io.us-east-1.aws.endpoints.huggingface.cloud"
    auth_token = "Bearer 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"
    headers = {"Authorization": auth_token, "Content-Type": "application/json"}
    data = json.dumps({"inputs": query})
    try:
        response = requests.post(API_URL, headers=headers, data=data)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        logging.exception("Exception occurred during API call")
        return None

def visualized_data(dataset):
    """
    Visualize the provided dataset.
    Args:
        dataset: The dataset to visualize.
    Returns:
        The figure object or an error message.
    """
    try:
        if isinstance(dataset, dict):
            # Bar chart for dictionary data
            fig, ax = plt.subplots()
            sns.barplot(x=list(dataset.keys()), y=list(dataset.values()), ax=ax)
        elif isinstance(dataset, pd.DataFrame):
            # Time series or scatter plot for DataFrame
            fig, ax = plt.subplots()
            if "Date" in dataset.columns:
                dataset.set_index("Date", inplace=True)
                sns.lineplot(data=dataset, ax=ax)
                ax.set_title("Time Series Plot")
            else:
                sns.scatterplot(data=dataset, ax=ax)
                ax.set_title("Scatter Plot")
        elif isinstance(dataset, list):
            # Heatmap or clustermap for list data
            fig, ax = plt.subplots()
            sns.heatmap(dataset, ax=ax) if len(dataset) == 2 else sns.clustermap(dataset)
        else:
            return "Unable to visualize data"
        fig.savefig('plot.png')
        return fig
    except Exception as e:
        logging.error("Error in data visualization: {}".format(str(e)))
        return "Error in visualizing data"

# Main execution
if __name__ == "__main__":
    api_key = "your_alpha_vantage_api_key"  # Replace with your actual API key
    group_manager = GroupManager(api_key)
    teachable_agent = TeachableAgentWithLLMSelection(
        name="financial_teachable_agent",
        llm_config={},  # Add actual configuration
        teach_config={},  # Add actual configuration
        group_manager=group_manager
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        response = teachable_agent.respond_to_query(user_input)
        print("Assistant:", response)
    
    # Create UserProxyAgent
    user = UserProxyAgent(
        name="user",
        human_input_mode="ALWAYS",
        llm_config=False  # Add specific configurations if required
    )

    # Setup Group Chat
    groupchat = autogen.GroupChat(agents=[user, teachable_agent.coder, teachable_agent.analyst], messages=[], max_round=10)
    manager = autogen.GroupChatManager(groupchat)

    # Start a continuous interaction loop
    chat_history = []
    while True:
        user_input = input("You: ")
        chat_history.append({"role": "user", "content": user_input})

        if user_input.lower() in ['exit', 'quit']:
            break

        response = teachable_agent.respond_to_user(user_input, chat_history)
        print(response)
        chat_history.append({"role": "assistant", "content": response})

       # Update the database and close connections (if applicable)
    teachable_agent.learn_from_user_feedback()
    teachable_agent.close_db()


