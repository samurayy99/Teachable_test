import os
import sys
import pickle
import asyncio
import logging
import spacy
from typing import Callable, Dict, Optional, Union, List, Tuple, Any
from termcolor import colored
from tqdm import tqdm
from autogen import ConversableAgent, config_list_from_json
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from autogen.agentchat.agent import Agent
from autogen import UserProxyAgent
from chromadb.config import Settings

# Basic configuration for logging
logging.basicConfig(filename='teachable_agent_session.log', 
                    level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')

class EnhancedMemoStore:
    def __init__(self, path_to_db_dir: str, reset_db: bool = False):
        self.path_to_db_dir = path_to_db_dir
        self.db_file = os.path.join(self.path_to_db_dir, "memostore.pkl")
        self.memostore: Dict[str, str] = {}

        if reset_db or not os.path.exists(self.db_file):
            self.reset_db()
        else:
            self.load_memos()

    def reset_db(self):
        """Clear the memo store both in memory and on disk."""
        self.memostore.clear()
        self.save_memos()

    def load_memos(self):
        """Load memos from the disk."""
        try:
            with open(self.db_file, 'rb') as file:
                self.memostore = pickle.load(file)
        except (EOFError, FileNotFoundError):
            self.memostore = {}

    def save_memos(self):
        """Save memos to the disk."""
        with open(self.db_file, 'wb') as file:
            pickle.dump(self.memostore, file)

    def add_memo(self, key: str, value: str):
        """Add a memo to the store."""
        self.memostore[key] = value
        self.save_memos()

    def get_memo(self, key: str) -> Optional[str]:
        """Retrieve a memo by key."""
        return self.memostore.get(key)

    def update_memo(self, key: str, new_value: str):
        """Update an existing memo."""
        if key in self.memostore:
            self.memostore[key] = new_value
            self.save_memos()

    def delete_memo(self, key: str):
        """Delete a memo from the store."""
        if key in self.memostore:
            del self.memostore[key]
            self.save_memos()

class TeachableAgent(ConversableAgent):
    def __init__(self, name="teachableagent", system_message=None, human_input_mode="NEVER",
                 llm_config=None, analyzer_llm_config=None, teach_config=None, **kwargs):
        default_system_message = "You are a helpful AI assistant that remembers user teachings from prior chats."
        super().__init__(name=name, system_message=system_message or default_system_message,
                         human_input_mode=human_input_mode, llm_config=llm_config, **kwargs)
        
        self._teach_config = teach_config or {}
        self.llm_config = llm_config
        self.analyzer_llm_config = analyzer_llm_config or llm_config  # Use llm_config if analyzer_llm_config is None
        self.analyzer = TextAnalyzerAgent(llm_config=self.analyzer_llm_config)

        self._initialize_teachable_agent_config()
        self.memo_store = MemoStore(self.verbosity, self.reset_db, self.path_to_db_dir)
        self.user_comments = []
        self.register_reply(Agent, self._generate_teachable_assistant_reply, position=2)

    def _initialize_teachable_agent_config(self):
        self.verbosity = self._teach_config.get("verbosity", 0)
        self.reset_db = self._teach_config.get("reset_db", False)
        self.path_to_db_dir = self._teach_config.get("path_to_db_dir", "./tmp/teachable_agent_db")
        self.prepopulate = self._teach_config.get("prepopulate", True)
        self.recall_threshold = self._teach_config.get("recall_threshold", 1.5)
        self.max_num_retrievals = self._teach_config.get("max_num_retrievals", 10)

        if self.prepopulate:
            self.memo_store.prepopulate_db()

    def _generate_teachable_assistant_reply(self, messages, sender, config):
        """
        Generate a response for the TeachableAgent based on user input and memo retrieval.

        Args:
            messages (list of dict): List of messages in the conversation.
            sender (Agent): The sender of the message.
            config: Additional configuration or context.

        Returns:
            Tuple[bool, Union[str, Dict, None]]: A tuple containing a flag indicating success and the generated response.
        """
        if self.llm_config is False:
            raise ValueError("TeachableAgent requires self.llm_config to be set in its base class.")
        if messages is None:
            messages = self._oai_messages[sender]  # In case of a direct call.

        # Get the last user turn.
        last_message = messages[-1]
        user_text = last_message["content"]
        if (not isinstance(user_text, str)) or ("context" in last_message):
            raise ValueError(
                "TeachableAgent currently assumes that the message content is a simple string. This error serves to flag a test case for relaxing this assumption."
            )

        # Keep track of this user turn as a potential source of memos later.
        self.user_comments.append(user_text)

        # Consider whether to retrieve something from the DB.
        new_user_text = self.memo_store.consider_storage(user_text)
        new_user_text = self.memo_store.consider_retrieval(user_text)
        if new_user_text != user_text:
            # Make a copy of the message list, and replace the last user message with the new one.
            messages = messages.copy()
            messages[-1]["content"] = new_user_text

        # Generate a response by reusing existing generate_oai_reply
        return self.generate_oai_reply(messages, sender, config)

    

def use_question_answer_phrasing():
    teachable_agent = create_teachable_agent(reset_db=True)
    user = ConversableAgent("user", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")
    user.initiate_chat(teachable_agent, "What is the capital of France?")
    user.send(teachable_agent, "The capital of France is Paris.")
    errors = check_agent_response(teachable_agent, user, "Paris")
    teachable_agent.learn_from_user_feedback()
    teachable_agent.close_db()
    return errors



def create_teachable_agent(reset_db=False, verbosity=0):
    config_list = config_list_from_json(env_or_file="/Users/lenox27/AI/TraderBuddy/OAI_CONFIG_LIST")
    llm_config = {"config_list": config_list, "timeout": 120}
    analyzer_llm_config = llm_config

    teachable_agent = TeachableAgent(
        llm_config=llm_config,
        analyzer_llm_config=analyzer_llm_config,
        teach_config={
            "verbosity": verbosity,
            "reset_db": reset_db,
            "path_to_db_dir": "./tmp/teachable_agent_db",
            "recall_threshold": 1.5,
        }
    )
    return teachable_agent

def check_agent_response(teachable_agent, user, correct_answer, assert_on_error=False):
    """
    Checks whether the agent's response contains the correct answer.
    :param assert_on_error: Set to True to raise an assertion error if the check fails.
    """
    agent_response = user.last_message(teachable_agent)["content"]
    if correct_answer not in agent_response:
        print(colored(f"\nTEST FAILED: EXPECTED ANSWER {correct_answer} NOT FOUND IN AGENT RESPONSE", "light_red"))
        if assert_on_error:
            assert correct_answer in agent_response
        return 1
    else:
        print(colored(f"\nTEST PASSED: EXPECTED ANSWER {correct_answer} FOUND IN AGENT RESPONSE", "light_cyan"))
        return 0

def initiate_interactive_session(teachable_agent):
    user = ConversableAgent("user", human_input_mode="ALWAYS")
    print("Interactive session with Teachable Agent. Type 'exit' to end, 'help' for assistance.")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'exit':
            print("Session ending, processing feedback.")
            break
        elif user_input.lower() == 'help':
            print("Ask anything or teach me. Type 'exit' to end the session.")
            continue

        try:
            messages = [{"role": "user", "content": user_input}]
            _, response = teachable_agent._generate_teachable_assistant_reply(messages)
            print("Agent:", response)
        except Exception as e:
            logging.error(f"Session error: {e}, Input: {user_input}")
            print(f"Error: {e}")

def main():
    teachable_agent = None
    try:
        teachable_agent = create_teachable_agent()
        initiate_interactive_session(teachable_agent)
    except Exception as e:
        logging.error(f"Main error: {e}")
        print(f"Error: {e}")
    finally:
        if teachable_agent and hasattr(teachable_agent, 'memo_store'):
            teachable_agent.memo_store.close()
            print("MemoStore data saved and database closed.")
        if teachable_agent and hasattr(teachable_agent, 'learn_from_user_feedback'):
            teachable_agent.learn_from_user_feedback()
        if teachable_agent and hasattr(teachable_agent, 'close_db'):
            teachable_agent.close_db()
            print("Database closed.")

if __name__ == "__main__":
    main()


