# Standard library imports
import json
import os
import logging
import time
import random
import uuid  # Import the UUID module for generating conversation IDs

# Third-party imports
from dotenv import load_dotenv
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import openai  # Assuming openai is a third-party package
import pandas as pd
from sqlalchemy import create_engine

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Local/application-specific imports
import autogen
from autogen import config_list_from_json
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen import UserProxyAgent
# from tenacity import retry, stop_after_attempt, wait_exponential

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define constants
USER_DATA_STORE = "user_data_store.json"
PREMIUM_THRESHOLD = 1000  # Define the premium account balance threshold

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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



class GroupManager:
    """
    Manages a group of specialized agents.
    """
    def __init__(self):
        """
        Initialize the GroupManager with a set of specialized agents.
        """
        self.logger = logging.getLogger(__name__)
        self.advisors = {}
        self.load_advisor_config()



    def load_advisor_config(self):
        """
        Load advisor configurations and initialize advisors.
        """
        try:
            # Load configuration from a file or database
            with open('advisor_config.json', 'r') as file:
                config = json.load(file)

            for category, advisor_types in config.items():
                self.advisors[category] = {}
                for keyword, advisor_type in advisor_types.items():
                    self.advisors[category][keyword] = AdvisorFactory.create_advisor(advisor_type)

            self.logger.info("Advisors initialized from configuration.")
        except Exception as e:
            self.logger.error(f"Error loading advisor configuration: {e}")

    # def initialize_advisors(self):
    #     """
    #     Initialize advisors with default types.
    #     """
    #     advisors = {
    #         "finance": {
    #             "investing": AdvisorFactory.create_advisor("InvestingAdvisor"),
    #             "trading": AdvisorFactory.create_advisor("TradingAdvisor"),
    #             "budget": AdvisorFactory.create_advisor("BudgetAdvisor"),
    #             "debt": AdvisorFactory.create_advisor("DebtAdvisor"),  
    #             "crypto": AdvisorFactory.create_advisor("CryptoAdvisor"),
    #             "financial_plan": AdvisorFactory.create_advisor("FinancialPlanner") 
    #         }
    #         # Add other specialized agents here...
    #     }
    #     self.logger.info("Initialized advisors: %s", advisors.keys())
    #     return advisors

    def add_advisor(self, category, keyword, advisor_type):
        """
        Add a new advisor to a specific category.
        """
        try:
            advisor = AdvisorFactory.create_advisor(advisor_type)
            if category not in self.advisors:
                self.advisors[category] = {}
            self.advisors[category][keyword] = advisor
            self.logger.info("Added advisor: %s under category %s", keyword, category)
        except ValueError as e:
            self.logger.error("Error adding advisor: %s", e)

    def remove_advisor(self, category, keyword):
        """
        Remove an advisor from a specific category.
        """
        if category in self.advisors and keyword in self.advisors[category]:
            del self.advisors[category][keyword]
            self.logger.info("Removed advisor: %s from category %s", keyword, category)
        else:
            self.logger.warning("Advisor or category not found: %s in %s", keyword, category)

    def get_advisor(self, category, keyword):
        """
        Get an advisor from a specific category.
        """
        if category in self.advisors and keyword in self.advisors[category]:
            return self.advisors[category][keyword]
        else:
            self.logger.warning("Advisor or category not found: %s in %s", keyword, category)
            return None
    
    # Define functions to fetch user and conversation details
    def get_user_metadata(user_id):
        # Fetch user details from the database
        # Example:
        user = user_db.get(user_id)
        is_premium = user.account_balance > PREMIUM_THRESHOLD
        return {
            "user_id": user_id,
            "username": user.name,
            "is_premium": is_premium,
            "user_attributes": user.attributes
        }

    def get_conversation_metadata():
        # Generate a unique ID for the conversation
        # Example:
        return {
            "conversation_id": uuid.uuid4().hex,
            "conversation_topic": detect_conversation_topic()
        }

    def handle_query(self, user_input, chat_history):
        # Fetch user and conversation metadata
        user_id = self.get_user_id()
        user_metadata = self.get_user_metadata(user_id)
        conversation_metadata = self.get_conversation_metadata()

        # Process the query and obtain a response
        response = self.process_user_input(user_input, chat_history)  # Replace with the appropriate method

        # Append user input with metadata to chat history
        chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time(),
            "sentiment": self.analyze_sentiment(user_input),
            "metadata": {**user_metadata, **conversation_metadata}
        })

        # Append assistant response to chat history
        chat_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": time.time(),
            "metadata": {**user_metadata, **conversation_metadata}
        })

        return response 


    def handle_fallback_query(self, user_input, chat_history):
        """
        Handle queries that do not match any specific financial category.
        Args:
            user_input: The input query from the user.
            chat_history: The chat history for context.
        Returns:
            A string response for the user.
        """
        # Check if the query is related to finance
        if is_finance_related(user_input):
            # Provide a generic financial advice or information
            response = "I'm here to help with your financial queries. Could you please provide more details or ask a different finance-related question?"
        else:
            # If the query is unrelated to finance
            response = ("While I specialize in finance, I'm here to help in any way I can. "
                        "For non-finance related queries, I might have limited advice. "
                        "Also, feel free to suggest any features or topics you'd like to see in future updates.")
            # Log the suggestion for future reference
            log_suggestion(user_input, chat_history)

        # Record unresponsive or off-topic interactions
        log_unresponsive_interaction(user_input, response, chat_history)

        return response

    def is_finance_related(user_input):
        # Implement logic to determine if the query is related to finance
        # Example: Check for financial keywords or use a classification model
        return 'finance' in user_input.lower()  # Simplified example

    def log_suggestion(user_input, chat_history):
        # Implement logic to log user suggestions for future updates
        # Example: Store in a database or a file
        logging.info(f"User suggestion logged: {user_input}")

    def log_unresponsive_interaction(user_input, response, chat_history):
        # Log interactions where the agent couldn't provide a relevant response
        chat_history.append({
            "role": "unresponsive",
            "user_input": user_input,
            "response": response,
            "timestamp": time.time()
        })
        logging.warning(f"Unresponsive interaction logged: User input: {user_input}, Response: {response}")

    # Example implementation of AdvisorFactory
class AdvisorFactory:
    """
    Factory class for creating instances of specialized advisors.
    """
    advisor_cache = {}

    @staticmethod
    def create_advisor(advisor_type):
        if advisor_type in AdvisorFactory.advisor_cache:
            return AdvisorFactory.advisor_cache[advisor_type]
        
        elif advisor_type == "InvestingAdvisor":
            advisor = InvestingAdvisor()
            
        elif advisor_type == "TradingAdvisor":
            advisor = TradingAdvisor()
        
        elif advisor_type == "BudgetAdvisor":
            advisor = BudgetAdvisor()
            
        elif advisor_type == "DefaultFinanceAdvisor":
            advisor = DefaultFinanceAdvisor()

        else:
            raise ValueError("Invalid advisor type")
        AdvisorFactory.advisor_cache[advisor_type] = advisor
        return advisor
# Example Advisor classes
class FinancialAdvisor:
    def advise(self, query):
        return f"Financial advice for query: {query}"

class CryptoAdvisor:
    def advise(self, query):
        return f"Crypto advice for query: {query}"

class FinancialPlanner:
    def advise(self, query):
        return f"Financial plan for query: {query}"

class DebtRepairAdvisor:
    def advise(self, query):
        return f"Debt repair advice for query: {query}"
 
class TeachableAgentWithLLMSelection:
    def __init__(self, name, llm_config, teach_config, group_manager):
        self.name = name
        self.llm_config = llm_config
        self.teach_config = teach_config
        self.config = load_configurations()
        self.group_manager = group_manager
        self.api_key = self.config['openai']['api_key']
        self.coder = GPTAssistantAgent("coder", llm_config=self.llm_config, instructions="You are a coder.")
        self.analyst = GPTAssistantAgent("analyst", llm_config=self.llm_config, instructions="You are an analyst.")

    def load_feedback_dataset(self):
        try:
            with open('feedback_data.json', 'r') as file:
                feedback_dataset = json.load(file)
            if not isinstance(feedback_dataset, list) or not all(
                isinstance(entry, dict) and 'user_input' in entry and 'agent_response' in entry
                for entry in feedback_dataset
            ):
                logging.error("Invalid structure in feedback data file.")
                return []
            return feedback_dataset
        except FileNotFoundError:
            logging.error("Feedback data file not found.")
            return []
        except json.JSONDecodeError:
            logging.error("Error decoding feedback data file.")
            return []

    def update_knowledge_base(self, feedback_dataset):
        for feedback in feedback_dataset:
            # Placeholder for updating knowledge base
            pass

    def use_learned_knowledge(self):
        # Placeholder for using learned knowledge
        pass

    def learn_from_user_feedback(self):
        if not self.config['test_mode']:
            feedback_dataset = self.load_feedback_dataset()
            self.update_knowledge_base(feedback_dataset)
        else:
            self.use_learned_knowledge()

    def respond_to_user(self, user_input, chat_history):
        response = self.group_manager.handle_query(user_input, chat_history)
        return response if response else "No relevant data found."

    def process_and_store_feedback(self, user_input, agent_response):
        if not isinstance(user_input, str) or not isinstance(agent_response, str):
            logging.error("Invalid data types for user input or agent response.")
            return

        feedback_entry = {
            'user_input': user_input,
            'agent_response': agent_response
        }

        try:
            with open('feedback_data.json', 'a') as file:
                json.dump(feedback_entry, file)
                file.write('\n')
        except IOError as e:
            logging.error(f"Error writing feedback data: {e}")
    def close_db(self):
    # Close database connections if any
    # Example:
        if self.database_engine:
            self.database_engine.dispose()
        logging.info("Database connections closed.")

    
def call_openai_api(self, user_input):
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
            raise APIError(f"Error in API response: {response.status_code}, {response.text}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        return f"Unexpected error: {str(e)}"




    
    def process_feedback(self, feedback):
        # Example processing logic: Validate and parse the feedback
        if not isinstance(feedback, dict) or 'user_input' not in feedback or 'agent_response' not in feedback:
            logging.error("Invalid feedback format")
            return None
        return {
            'user_input': feedback['user_input'].strip(),
            'agent_response': feedback['agent_response'].strip()
        }

    def log_interaction(self, user_input, agent_response):
        log_entry = {
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": time.time()
        }
        with open('interaction_log.txt', 'a') as file:
            file.write(json.dumps(log_entry) + '\n')

    def perform_health_check(self):
        # Check database connectivity
        try:
            engine = create_engine('postgresql://username:password@host:port/database')
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            return "Database connection is healthy"
        except Exception as e:
            return f"Database health check failed: {e}"

    def clean_and_preprocess_data(self, data):
        # Example cleaning/preprocessing logic
        cleaned_data = {key: value.strip() for key, value in data.items()}
        return cleaned_data

    def process_and_store_feedback(self, user_input, agent_response):
        feedback_entry = {
            'user_input': user_input,
            'agent_response': agent_response
        }
        processed_feedback = self.clean_and_preprocess_data(feedback_entry)
        try:
            engine = create_engine('postgresql://username:password@host:port/database')
            pd.DataFrame([processed_feedback]).to_sql('feedback_data', engine, if_exists='append', index=False)
        except Exception as e:
            logging.error(f"Error storing feedback data: {e}")

    def load_data_for_training(self):
        try:
            engine = create_engine('postgresql://username:password@host:port/database')
            return pd.read_sql('SELECT * FROM feedback_data', engine)
        except Exception as e:
            logging.error(f"Error loading data for training: {e}")
            return pd.DataFrame()

class APIError(Exception):
    pass

class CryptoAdvisor:
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

    @staticmethod
    def construct_alpha_vantage_url(function, from_currency, to_currency):
        base_url = 'https://www.alphavantage.co/query'
        return f'{base_url}?function={function}&from_currency={from_currency}&to_currency={to_currency}&apikey={CryptoAdvisor.ALPHA_VANTAGE_API_KEY}'

    def fetch_crypto_data(self, crypto_name, to_currency='USD'):
        """
        Fetches cryptocurrency data.

        Args:
            crypto_name (str): The name of the cryptocurrency.
            to_currency (str): The target currency (default is USD).

        Returns:
            dict: JSON response from the API or None in case of failure.
        """
        url = CryptoAdvisor.construct_alpha_vantage_url('CURRENCY_EXCHANGE_RATE', crypto_name, to_currency)
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
            return response.json()
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTPError in API call: {e}, URL: {url}")
        except requests.exceptions.ConnectionError as e:
            logging.error(f"ConnectionError in API call: {e}, URL: {url}")
        except requests.exceptions.Timeout as e:
            logging.error(f"TimeoutError in API call: {e}, URL: {url}")
        except requests.exceptions.RequestException as e:
            logging.error(f"RequestException in API call: {e}, URL: {url}")
        except Exception as e:
            logging.error(f"Unexpected error in API call: {str(e)}, URL: {url}")
        return None


    def advise_on_crypto(self, query):
        crypto_name = query.split()[0]  # simplistic extraction
        crypto_data = self.fetch_crypto_data(crypto_name)
        if crypto_data is None:
            return "Unable to fetch cryptocurrency data."
        # Implement logic to analyze data and provide advice
        exchange_rate = crypto_data.get("Realtime Currency Exchange Rate", {}).get("5. Exchange Rate", "No data")
        advice = f"The current exchange rate of {crypto_name} is {exchange_rate}."
        return advice

class FinancialPlanner:
    def create_financial_plan(self, user_data):
        # user_data should contain information like income, expenses, savings, financial goals, etc.
        # Implement logic to analyze user's financial data and suggest a plan
        savings_plan = "Savings Plan: Save 20% of monthly income."
        investment_plan = "Investment Plan: Diversify investments across stocks and bonds."
        return f"{savings_plan}\n{investment_plan}"


class DebtRepairAdvisor:
    def provide_debt_repair_advice(self, debt_info):
        # debt_info should contain details about user's debts
        # Implement logic for creating a repayment plan or debt reduction strategy
        repayment_plan = "Repayment Plan: Focus on high-interest debts first."
        consolidation_advice = "Consider debt consolidation for multiple high-interest debts."
        return f"{repayment_plan}\n{consolidation_advice}"


class FinancialAdvisor:
    """
    Advisor specialized in offering financial advisory services.
    """
    
    def get_user_profile(self, user_id):
        # Fetch user profile from a database or other data source
        # Placeholder for fetching user profile
        return {
            "user_id": user_id,
            "name": "John Doe",
            "budget": True,  # Indicates if the user has a budget setup
            "long_term_goals": ["retirement", "buying a home"]
        }

    def interact_with_user(self, user_id):
        """
        Simulate interaction with the user to obtain additional details.
        Args:
            user_id (str): The user's unique identifier.
        Returns:
            tuple: A tuple containing interaction results and the user's profile.
        """
        user_profile = self.get_user_profile(user_id)

        # Implement logic to interact with the user based on the profile
        # Example: Querying user preferences, financial goals, etc.

        # For now, let's assume these are the results from our simulated interaction
        interaction_results = {
            "budget_query": "Yes, I have a monthly budget set up." if user_profile["budget"] else "No, I haven't set up a budget yet.",
            "savings_goals": "My long-term savings goal is to save for " + " and ".join(user_profile["long_term_goals"]) + "."
        }

        return interaction_results, user_profile
    
    def advise(self, user_id, query):
        # Here we simulate interaction and retrieval of additional info
        interaction_results, user_profile = self.interact_with_user(user_id)
        # Integrate financial models or algorithms based on user data and interaction results
        # Pseudo-code:
        # advice = financial_model.generate_advice(user_profile, interaction_results)
        
        advice = f"Financial advice based on user's query about {query}."
        return advice




 #Function to interact with GPT-4 API
    def call_openai_api(self, user_input):
        """
        Calls the OpenAI API with the provided user input.

        Args:
            user_input (str): The input from the user.

        Returns:
            str: The content of the response message or error message.
        """
        messages = [{"role": "user", "content": user_input}]
        payload = {"model": "gpt-4-1106-preview", "messages": messages}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTPError in OpenAI API call: {e}, URL: {response.url}, Status Code: {response.status_code}")
            return f"Error in API response: {response.status_code}, {response.text}"
        except requests.exceptions.ConnectionError as e:
            logging.error(f"ConnectionError in OpenAI API call: {e}, URL: {response.url}")
            return "Error: Unable to connect to the OpenAI API."
        except requests.exceptions.Timeout as e:
            logging.error(f"TimeoutError in OpenAI API call: {e}, URL: {response.url}")
            return "Error: Timeout while connecting to the OpenAI API."
        except requests.exceptions.RequestException as e:
            logging.error(f"RequestException in OpenAI API call: {e}, URL: {response.url}")
            return "Error: Problem with the request to the OpenAI API."
        except Exception as e:
            logging.error(f"Unexpected error in OpenAI API call: {str(e)}")
            return f"Unexpected error: {str(e)}"

# Define the functions for query classification, fetching financial data and news, summarizing and validating data
def classify_query(self, user_input):
    financial_keywords = ["stock", "crypto", "market", "investment", "financial", "economy"]
    news_keywords = ["news", "headline", "current events", "article", "report"]

    # Tokenize and remove stopwords
    tokens = word_tokenize(user_input.lower())
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]

    # Check for keyword presence
    if any(keyword in filtered_tokens for keyword in financial_keywords):
        return "finance"
    elif any(keyword in filtered_tokens for keyword in news_keywords):
        return "news"
    else:
        return "general"

def fetch_financial_data(query):
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    # Parse the query for more detailed input
    parsed_query = parse_query(query)
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={parsed_query}&apikey={api_key}'
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
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    # Parse the topic for more detailed input
    parsed_topic = parse_topic(topic)
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={parsed_topic}&apikey={api_key}'
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
    # Use Mistral model to summarize the data
    summary = mistral_llm(data)
    # Validate the summary using GPT-4
    validation = validate_with_gpt4(summary)
    # Check for factual accuracy, coherence, and relevance in the validation
    if "accurate" in validation.lower() and "coherent" in validation.lower():
        # Check relevance of the summary to the user's query
        if user_input.lower() in summary.lower():
            return summary
        else:
            return "The summary was not relevant to your query."
    else:
        return "The summary was not accurate or coherent."
def validate_with_gpt4(summary):
    try:
        response = openai.Completion.create(
            engine="gpt-4-1106-preview",
            prompt=f"Please review this summary for accuracy and coherence: '{summary}'",
            max_tokens=100,
            api_key=config['openai']['api_key']
        )
        review = response.choices[0].text.strip()
        logging.info(f"Summary review: {review}")
        # Check for factual accuracy and coherence in the review
        if "accurate" in review.lower() and "coherent" in review.lower():
            return "The summary is accurate and coherent."
        else:
            return "The summary is not accurate or coherent."
    except Exception as e:
        logging.exception("Error calling GPT-4 API for summary validation")
        return "Unable to validate summary."


def mistral_llm(query):
    # Replace with your custom endpoint URL
    API_URL = "https://z983hozbx3in30io.us-east-1.aws.endpoints.huggingface.cloud"
    # Authorization token
    auth_token = "Bearer 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"
    # Prepare the request header
    headers = {
        "Authorization": auth_token,
        "Content-Type": "application/json"
    }
    # Prepare the request body
    data = json.dumps({"inputs": query})

    try:
        # Send POST request to the API
        response = requests.post(API_URL, headers=headers, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"API call failed: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logging.exception("Exception occurred during API call")
        return None


def visualized_data(dataset):
    
    if isinstance(dataset, dict):
        # Bar chart
        fig, ax = plt.subplots()
        sns.barplot(x=list(dataset.keys()), y=list(dataset.values()), ax=ax)

    elif isinstance(dataset, pd.DataFrame):

            fig, ax = plt.subplots()
            
            if "Date" in dataset.columns:
            
                dataset.set_index("Date", inplace=True)  
                sns.lineplot(data=dataset, ax=ax)
                
                ax.set(xlabel="Date", ylabel="Values")
                ax.set_title("Time Series Plot")
                
            elif {"X_Axis", "Y_Axis"} <= set(dataset.columns):  

                sns.scatterplot(data=dataset, x="X_Axis", y="Y_Axis", ax=ax)
                
                ax.set(xlabel="X_Axis", ylabel="Y_Axis") 
                ax.set_title("Scatter Plot")
                
        
    elif isinstance(dataset, list) and len(dataset) == 2 and all(isinstance(i, list) for i in dataset): 
        # Heatmap
        fig, ax = plt.subplots()
        sns.heatmap(dataset, ax=ax)
        
    elif isinstance(dataset, list) and isinstance(dataset[0], list):
        # Clustermap
        fig, ax = plt.subplots()
        sns.clustermap(dataset, ax=ax)
        
    elif isinstance(dataset, list) and all(isinstance(i, tuple) and len(i) == 2 for i in dataset):
        # Line graph
        fig, ax = plt.subplots()
        x = [i[0] for i in dataset]
        y = [i[1] for i in dataset] 
        sns.lineplot(x=x, y=y, ax=ax) 
        
    elif isinstance(dataset, list) and all(isinstance(i, tuple) and len(i) == 2 for i in dataset[0]):
        # Scatter plot
        fig, ax = plt.subplots()
        x = [i[0] for i in dataset[0]]  
        y = [i[1] for i in dataset[0]]
        sns.scatterplot(x=x, y=y, ax=ax)
        
    else:
        return "Unable to visualize data"

    fig.savefig('plot.png')
    return fig

# Main execution
if __name__ == "__main__":
    group_manager = GroupManager()
    teachable_agent = TeachableAgentWithLLMSelection(
        name="financial_teachable_agent",
        llm_config=config['llm_config'],
        teach_config=config['teach_config'],
        group_manager=group_manager
    )
    
    # Create UserProxyAgent
    user = UserProxyAgent(
        name="user",
        human_input_mode="ALWAYS",
        llm_config=False  # or your specific configuration
        # Add other parameters as needed
    )

    # Setup Group Chat
    groupchat = autogen.GroupChat(agents=[user, teachable_agent.coder, teachable_agent.analyst], messages=[], max_round=10)
    manager = autogen.GroupChatManager(groupchat)

    # Start a continuous interaction loop
    chat_history = []
    while True:
        # Get user input
        user_input = input("You: ")
        chat_history.append({"role": "user", "content": user_input})

        # Break the loop if the user types 'exit' or 'quit'
        if user_input.lower() in ['exit', 'quit']:
            break

        # Chat with TeachableAgent
        response = teachable_agent.respond_to_user(user_input, chat_history)
        print(response)
        chat_history.append({"role": "assistant", "content": response})

    # Update the database
    teachable_agent.learn_from_user_feedback()
    teachable_agent.close_db()


