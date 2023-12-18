import os
import sys
import logging
import nltk
import spacy
import html
import json
from dotenv import load_dotenv
from termcolor import colored
from transformers import BertTokenizer, BertModel, pipeline
from autogen import UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.teachable_agent import TeachableAgent
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from nltk.sentiment import SentimentIntensityAnalyzer

# Configuration Loader
class ConfigLoader:
    def __init__(self):
        load_dotenv()  # Load environment variables from a .env file if it exists

    def get_config(self):
        """
        Retrieves configuration settings from environment variables.
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise KeyError("Missing OpenAI API key")

        # Default OpenAI API endpoint
        default_api_base = "https://api.openai.com/v1"

        return {
            "openai_api_key": openai_api_key,
            "model": os.getenv("MODEL", "gpt-4"),
            "api_base": os.getenv("API_BASE", default_api_base),
        }

# Load configurations
config_loader = ConfigLoader()
config = config_loader.get_config()

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Lenox")

# Enhanced LLM configuration
llm_config = {
    "config_list": [{"model": config["model"], "api_key": config["openai_api_key"]}],
    "timeout": 120,
}

# Download necessary resources
nltk.download('vader_lexicon')
spacy.cli.download("en_core_web_sm")

# TextAnalyzerAgent for deep text analysis
text_analyzer = TextAnalyzerAgent(
    name="TextAnalyzer",
    llm_config=llm_config,
    system_message="I specialize in analyzing and structuring text data."
)

# Context Tracker Class
class ContextTracker:
    def __init__(self):
        self.history = []
        self.entity_history = []

    def update_history(self, user_input, bot_response, entities):
        self.history.append((user_input, bot_response))
        self.entity_history.append(entities)

    def get_recent_context(self):
        return self.history[-5:], self.entity_history[-5:]
    

# Enhanced TeachableAgent with TextAnalyzerAgent and spaCy integration
class EnhancedTeachableAgent(TeachableAgent):
    def __init__(self, name, llm_config, teach_config):
        super().__init__(name, llm_config, teach_config)
        self.analyzer = TextAnalyzerAgent(name="TextAnalyzer", llm_config=llm_config)
        self.nlp = spacy.load("en_core_web_sm")
        self.context_tracker = ContextTracker()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.language_model = BertModel.from_pretrained('bert-base-uncased')
        self.sentiment_model = pipeline("sentiment-analysis")

    def generate_response(self):
        last_user_input = self.preprocess_text(self.get_last_user_input())
        relevant_memos, recent_entities = self.retrieve_relevant_memos(last_user_input)
        response = super().generate_response()
        response_context = self.concatenate_memo_texts(relevant_memos)
        enhanced_response = self.compose_response(response, response_context, recent_entities)
        return html.escape(enhanced_response).strip() or "Let me think a bit more about that."

    def process_user_input(self, user_input):
        processed_input, entities = self.advanced_nlp_processing(user_input)
        self.context_tracker.update_history(processed_input, None, entities)
        sentiment_context = self.contextual_sentiment_analysis(user_input)
        logger.info(f"Contextual Sentiment: {sentiment_context}")
        super().process_user_input(user_input)

    def advanced_nlp_processing(self, text):
        try:
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            sentiment_analysis = self.sentiment_model(text)
            sentiment = sentiment_analysis[0]['label'] if sentiment_analysis else "Neutral"
            return {'entities': entities, 'context': doc, 'sentiment': sentiment}
        except Exception as e:
            # Handle any unhandled exceptions here
            # You can log the error or return an error message
            return {'error': str(e)}
    
    def contextual_sentiment_analysis(self, text):
        sentiment_result = self.sentiment_model(text)
        return sentiment_result[0]['label'] if sentiment_result else "Neutral"

    def compose_response(self, response, context, entities):
        # Compose response based on the context and entities identified
        if context:
            response += f" Additionally, based on our previous discussions: {context}"
        if entities:
            response += f" Also, I noticed these entities: {', '.join([e[0] for e in entities])}"
        return response
    

    def retrieve_relevant_memos(self, input_text):
        memo_list = self.memo_store.get_related_memos(input_text, n_results=self.max_num_retrievals)
        return self.filter_memos_by_relevance(memo_list, input_text), self.context_tracker.entity_history[-5:]

    def filter_memos_by_relevance(self, memos, input_text):
        # Enhanced logic to filter memos based on relevance
        return [memo for memo in memos if self.is_memo_relevant(memo, input_text)]

    def is_memo_relevant(self, memo, input_text):
        # Implement more advanced logic to check memo relevance
        return input_text.lower() in memo[0].lower()

    def learn_from_user_feedback(self):
        for comment in self.user_comments:
            feedback_analysis = self.analyze_feedback(comment)
            if feedback_analysis:
                self.memo_store.store_feedback(comment, feedback_analysis)
        self.user_comments = []

    def analyze_feedback(self, comment):
        analysis = self.analyzer.analyze(comment, "Please provide an analysis of this feedback.")
        return analysis if analysis else "No actionable feedback detected."

    def preprocess_text(self, text):
        # Advanced preprocessing for user input
        return ' '.join([token.lemma_ for token in self.nlp(text)])

    def compose_response(self, response, context, entities):
        # Integrate additional NLP insights into the response
        if context:
            response += f" Reflecting on our previous discussions, I remember: {context}"
        if entities:
            response += f" Also, these details caught my attention: {', '.join([f'{e[0]} ({e[1]})' for e in entities])}"
        return response

    def is_memo_relevant(self, memo, input_text):
        # Advanced relevance checking using NLP and context comparison
        memo_text, _ = memo
        doc1 = self.nlp(memo_text)
        doc2 = self.nlp(input_text)
        return doc1.similarity(doc2) > 0.7  # Adjust threshold as needed

    def learn_from_user_feedback(self):
        # Enhanced learning from user interactions
        if self.user_comments:
            for comment in self.user_comments:
                learning_opportunity = self.analyze_feedback(comment)
                if learning_opportunity:
                    self.memo_store.store_learning(comment, learning_opportunity)
            self.user_comments = []

    def analyze_feedback(self, comment):
        # Deep analysis of user feedback for actionable insights
        return self.analyzer.analyze(comment, "Examine this feedback for learning opportunities or insights.")

    def store_learning(self, comment, learning_opportunity):
        # Custom method to store new learnings from feedback
        if learning_opportunity:
            self.memo_store.add_input_output_pair(comment, learning_opportunity)

    def advanced_nlp_processing(self, text):
        # Enhanced NLP processing with entity and sentiment analysis
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        sentiment_analysis = self.sentiment_model(text)
        sentiment = sentiment_analysis[0]['label'] if sentiment_analysis else "Neutral"
        return {'entities': entities, 'context': doc, 'sentiment': sentiment}

    # Additional methods and functionalities as required...

# Instantiate Lenox with enhanced capabilities
lenox = EnhancedTeachableAgent(
    name="Lenox",
    llm_config=llm_config,
    teach_config={"verbosity": 1, "recall_threshold": 1.2}
)

def interact_with_lenox():
    print(colored("Lenox is ready to assist you. Type 'exit' to end the session.", "cyan"))
    user = UserProxyAgent("user", human_input_mode="ALWAYS")
    lenox.initiate_chat(user, message="Hello, I'm here to help. What's on your mind?")

    while True:
        try:
            user_input = input(colored("You: ", "yellow")).strip()
            if user_input.lower() in ['exit', 'quit', 'goodbye']:
                handle_exit_sequence(lenox)
                break

            lenox.process_user_input(user_input)  # Process and analyze user input
            response = lenox.generate_response()  # Generate a response
            print(colored(f"Lenox: {response}", "green"))
        except Exception as e:
            logger.error(f"An error occurred during interaction: {e}")
            print(colored("Sorry, I encountered an error. Let's try that again.", "red"))

def handle_exit_sequence(lenox_agent):
    print(colored("Lenox: Goodbye! Feel free to return if you have more questions.", "green"))
    lenox_agent.learn_from_user_feedback()  # Store learnings from the interaction
    lenox_agent.close_db()  # Close database connections if used

if __name__ == "__main__":
    try:
        interact_with_lenox()
    except KeyboardInterrupt:
        print(colored("\nInterrupted by user, closing Lenox.", "red"))
        handle_exit_sequence(lenox)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
