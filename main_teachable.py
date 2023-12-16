import os
import sys
import asyncio
import logging
import spacy
from typing import Dict, Optional, List, Tuple, Any
from termcolor import colored
from tqdm import tqdm
from autogen import ConversableAgent, config_list_from_json, UserProxyAgent
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from autogen.agentchat.contrib.teachable_agent import TeachableAgent
from chromadb.config import Settings
import pickle  # Ensure pickle is imported for MemoStore functionality
import random  # For generating more dynamic responses

# Logging configuration
logging.basicConfig(filename='teachable_agent_session.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


# MemoStore Class Definition
class MemoStore:
    def __init__(self, path_to_db_dir: str, reset_db: bool = False):
        self.path_to_db_dir = path_to_db_dir
        self.db_file = os.path.join(self.path_to_db_dir, "enhanced_memostore.pkl")
        self.memostore: Dict[str, List[str]] = {}
        self.nlp = spacy.load("en_core_web_sm")

        if reset_db or not os.path.exists(self.db_file):
            self.reset_db()
        else:
            self.load_memos()

    def reset_db(self):
        self.memostore.clear()
        self.save_memos()

    def load_memos(self):
        with open(self.db_file, 'rb') as file:
            self.memostore = pickle.load(file)

    def save_memos(self):
        with open(self.db_file, 'wb') as file:
            pickle.dump(self.memostore, file)

    def add_memo(self, key: str, value: str):
        """Add a memo to the store."""
        if key not in self.memostore:
            self.memostore[key] = []
        self.memostore[key].append(value)
        self.save_memos()

    def get_memo(self, key: str) -> Optional[List[str]]:
        """Retrieve memos by key."""
        return self.memostore.get(key, [])

    def update_memo(self, key: str, new_values: List[str]):
        """Update existing memos."""
        if key in self.memostore:
            self.memostore[key] = new_values
            self.save_memos()

    def delete_memo(self, key: str):
        """Delete a memo from the store."""
        if key in self.memostore:
            del self.memostore[key]
            self.save_memos()

    def search_memos(self, query: str) -> List[str]:
        """Enhanced search for related memos based on a query using advanced NLP techniques."""
        query_doc = self.nlp(query)
        relevant_memos = []
        for key, memos in self.memostore.items():
            key_doc = self.nlp(key)
            # Advanced similarity check (e.g., considering synonyms, context)
            if advanced_similarity_check(query_doc, key_doc):
                relevant_memos.extend(memos)
        return relevant_memos
    
    # Helper function for advanced similarity check (part of MemoStore class or as a separate utility function)
def advanced_similarity_check(doc1, doc2):
    # Simple implementation using spaCy's similarity
    return doc1.similarity(doc2) > 0.7  # Adjust threshold as needed

def retrieve_contextual_memos(self, user_text: str) -> List[str]:
        """Retrieve memos based on contextual relevance to the user's query."""
        query_doc = self.nlp(user_text)
        relevant_memos = []
        for key, memos in self.memostore.items():
            key_doc = self.nlp(key)
            if self.contextual_relevance(query_doc, key_doc):
                relevant_memos.extend(memos)
        return relevant_memos

def contextual_relevance(self, query_doc, key_doc) -> bool:
        """Determines if a memo is contextually relevant to a query."""
        # Implement more advanced NLP techniques for context analysis
        return query_doc.similarity(key_doc) > 0.8  # Adjust the threshold as needed

# CustomTeachableAgent Class for Enhanced Functionality
class CustomTeachableAgent(TeachableAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contextual_memory = {}
        self.nlp = spacy.load("en_core_web_sm")  # Loading the NLP model here for centralized usage
        self.contextual_memory = {}
        

    def _initialize_custom_teachable_agent_config(self, teach_config):
      self._teach_config = teach_config or {}
      self.memo_store = MemoStore(
        path_to_db_dir=self._teach_config.get("path_to_db_dir", "./tmp/teachable_agent_db"),  # Directory, not a file
        reset_db=self._teach_config.get("reset_db", False)
    )

      if self._teach_config.get("prepopulate", True):
                self.memo_store.prepopulate()

    def _generate_custom_teachable_assistant_reply(self, messages, sender, config):
                if self.llm_config is False:
                 raise ValueError("CustomTeachableAgent requires self.llm_config to be set.")

    # Implement additional custom logic for generating a reply, if needed
    # ...

                return super()._generate_teachable_assistant_reply(messages, sender, config)
    
    def context_aware_response_generation(self, user_text: str) -> str:
        """
        Generate responses based on the context and history of user interactions.
        """
        doc = self.nlp(user_text)
        keywords = ['schedule', 'meeting', 'contact', 'info', 'help']
        for keyword in keywords:
            if keyword in user_text.lower():
                return self.generate_keyword_based_response(keyword, doc)

        # Default response for non-keyword scenarios
        return "I'm here to assist you. Could you please provide more details?"

    def generate_keyword_based_response(self, keyword: str, doc) -> str:
        """
        Generate responses based on specific keywords identified in user input.
        """
        if keyword in ['schedule', 'meeting']:
            return "Let's plan your meeting. Please provide the date and time."
        elif keyword in ['contact', 'info']:
            # Extracting entities like names or organizations for contact info
            entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]
            if entities:
                return f"Retrieving info for {', '.join(entities)}."
            else:
                return "Can you specify the contact's name or organization?"
        elif keyword == 'help':
            return random.choice(["How may I assist you today?", "What do you need help with?"])

        return "I'm here to assist you. Please provide more specifics."
    
def advanced_decision_making(self, user_text: str) -> str:
        """Enhanced decision-making using NLP and context analysis."""
        # Example: Use NLP to understand user intent and make decisions
        doc = self.nlp(user_text)
        if any(word.text.lower() in ['schedule', 'meeting'] for word in doc):
            return "Scheduling Module Activated"
        elif any(word.text.lower() in ['contact', 'info', 'email'] for word in doc):
            return "Contact Retrieval Module Activated"
        else:
            return "General Assistance Module Activated"
        
def context_aware_storage_retrieval(self, user_text: str) -> str:
        """Context-aware storage and retrieval considering user history."""
        decision = self.advanced_decision_making(user_text)
        # Implement more sophisticated storage/retrieval logic here
        if decision == "Scheduling Module Activated":
            self.memo_store.add_memo("schedule", user_text)
            return "Scheduling details noted."
        elif decision == "Contact Retrieval Module Activated":
            return " ".join(self.memo_store.get_memo("contact_info"))
        else:
            self.memo_store.add_memo("general", user_text)
            return "General query noted."        


def consider_storage_and_retrieval(self, user_text):
    """
    Manages memory storage and retrieval based on user input.
    """
    decision = self.advanced_decision_making(user_text)
    if decision == "Scheduling Module Activated":
        # Store scheduling related data
        self.memo_store.add_memo("schedule", user_text)
    elif decision == "Contact Information Retrieval Module Activated":
        # Retrieve contact information
        return self.memo_store.get_memo("contact_info")
    else:
        # General memory management
        self.memo_store.add_memo("general", user_text)
        return "Information noted."

def make_context_aware_decision(self, user_text):
    """
    Make decisions based on the context and user interaction history.
    """
    # Simple example logic
    if "specific keyword" in user_text:
        return True
    return False

def generate_personalized_response(self, user_input: str) -> str:
        """Generates personalized responses using user history."""
        # Implement more sophisticated response generation here
        if "how are you" in user_input.lower():
            return "I'm an AI, always ready to assist! What can I do for you?"
        else:
            return f"How can I assist with your query: {user_input}?"

def generate_enhanced_response(self, user_input):
        """
        Generates an enhanced response based on user input.
        """
        # Example implementation: return a simple response for now
        # You can enhance this method with more complex logic later
        return f"Received: {user_input}"

def advanced_contextual_decision_making(self, user_text: str) -> str:
        """Enhanced decision-making using deep NLP analysis."""
        # Example: Implement context analysis for specific tasks
        doc = self.nlp(user_text)
        if 'schedule' in user_text.lower():
            return "Activate Scheduling"
        elif 'contact' in user_text.lower():
            return "Retrieve Contact"
        else:
            return "General Assistance"

def perform_contextual_actions(self, user_text: str) -> str:
        """Perform actions based on the context derived from user input."""
        action = self.advanced_contextual_decision_making(user_text)
        if action == "Activate Scheduling":
            self.memo_store.add_memo("schedule", user_text)
            return "Scheduling process initiated."
        elif action == "Retrieve Contact":
            contacts = self.memo_store.retrieve_contextual_memos(user_text)
            return f"Contact details: {', '.join(contacts)}"
        else:
            return "How can I assist you further?"

def generate_dynamic_response(self, user_input: str) -> str:
        """Generates dynamic responses based on the user's input and history."""
        # More advanced response generation logic
        if "help" in user_input.lower():
            return "Sure, I'm here to help. What do you need assistance with?"
        elif "thank you" in user_input.lower():
            return "You're welcome! Let me know if there's anything else I can do for you."
        else:
            # Custom response based on user's input
            return self.perform_contextual_actions(user_input)

def generate_enhanced_response(self, user_input: str) -> str:
        """Enhanced response generation integrating dynamic responses."""
        return self.generate_dynamic_response(user_input)

# Utility Functions
def create_custom_teachable_agent(reset_db=False, verbosity=0):
    config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict={"model": ["gpt-4-1106-preview"]})
    llm_config = {"config_list": config_list, "timeout": 120}

    return CustomTeachableAgent(
        name="customteachableagent",
        llm_config=llm_config,
        teach_config={
            "verbosity": verbosity,
            "reset_db": reset_db,
            "path_to_db_dir": "./tmp/teachable_agent_db",  # Directory, not a file
            "recall_threshold": 1.5,  # Adjust as needed
        }
    )

def initiate_interactive_session(custom_teachable_agent):
    """
    Initiates an interactive session with the Custom Teachable Agent.
    """
    user = UserProxyAgent("user", human_input_mode="ALWAYS")
    print("Interactive session with Custom Teachable Agent. Type 'exit' to end, 'help' for assistance.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Session ending...")
            break
        elif user_input.lower() == 'help':
            print("Ask anything or teach me. Type 'exit' or 'quit' to end the session.")
            continue

        try:
            # Process user input and generate a response from the Teachable Agent
            response = custom_teachable_agent.generate_enhanced_response(user_input)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

# Main Function with Enhanced Initialization and Interaction
def main():
    custom_teachable_agent = None
    try:
        custom_teachable_agent = create_custom_teachable_agent(verbosity=3)
        initiate_interactive_session(custom_teachable_agent)
    except Exception as e:
        logging.error(f"Critical error in main function: {e}")
        print(f"Critical Error: {e}")
    finally:
        if custom_teachable_agent:
            custom_teachable_agent.learn_from_user_feedback()
            custom_teachable_agent.close_db()
            print("Session concluded. Feedback learned and database closed.")

if __name__ == "__main__":
    main()