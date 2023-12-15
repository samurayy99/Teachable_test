import os
import sys
import pickle
import asyncio
import logging
import spacy
import chromadb
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


class TeachableAgent(ConversableAgent):
    def __init__(self, name="teachableagent", system_message=None, human_input_mode="NEVER",
                 llm_config=None, analyzer_llm_config=None, teach_config=None, **kwargs):
        default_system_message = "You are a helpful AI assistant that remembers user teachings from prior chats."
        super().__init__(name=name, system_message=system_message or default_system_message,
                         human_input_mode=human_input_mode, llm_config=llm_config, **kwargs)
        
        self._teach_config = teach_config or {}
        self.llm_config = llm_config
        self.analyzer_llm_config = analyzer_llm_config or llm_config  # Use llm_config if analyzer_llm_config is None
        self._initialize_teachable_agent_config()
        self.memo_store = MemoStore(self.verbosity, self.reset_db, self.path_to_db_dir)
        self.user_comments = []
        self.register_reply(Agent, self._generate_teachable_assistant_reply, position=2)

    def _initialize_teachable_agent_config(self):
        """
        Initializes the configuration for the teachable agent.
        """
        self.verbosity = self._teach_config.get("verbosity", 0)
        self.reset_db = self._teach_config.get("reset_db", False)
        self.path_to_db_dir = self._teach_config.get("path_to_db_dir", "./tmp/teachable_agent_db")
        self.prepopulate = self._teach_config.get("prepopulate", True)
        self.recall_threshold = self._teach_config.get("recall_threshold", 1.5)
        self.max_num_retrievals = self._teach_config.get("max_num_retrievals", 10)
        self.analyzer = TextAnalyzerAgent(llm_config=self.analyzer_llm_config)
        self.memo_store = MemoStore(self.verbosity, self.reset_db, self.path_to_db_dir)
        self.user_comments = []
        self.register_reply(Agent, self._generate_teachable_assistant_reply, position=2)

    def _generate_teachable_assistant_reply(self, messages, sender, config):
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,  # Persistent state.
    ) -> Tuple[bool, Union[str, Dict, None]]:
        
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
        self.memo_store.consider_storage(user_text, agent_text)
            new_user_text = self.memo_store.consider_retrieval(user_text)
            if new_user_text != user_text:
                # Make a copy of the message list, and replace the last user message with the new one.
                messages = messages.copy()
                messages[-1]["content"] = new_user_text

        # Generate a response by reusing existing generate_oai_reply
        return self.generate_oai_reply(messages, sender, config)


class MemoStore:

    def __init__(self, verbosity, reset, path_to_db_dir):
        """
        Args:
            - verbosity (Optional, int): 1 to print memory operations, 0 to omit them. 3+ to print memo lists.
            - path_to_db_dir (Optional, str): path to the directory where the DB is stored.
        """
        self.verbosity = verbosity
        self.reset = reset
        self.path_to_db_dir = path_to_db_dir

        # Load or create the vector DB on disk.
        settings = Settings(
            anonymized_telemetry=False, allow_reset=True, is_persistent=True, persist_directory=path_to_db_dir
        )
        self.db_client = chromadb.Client(settings)
        self.vec_db = self.db_client.create_collection("memos", get_or_create=True)  # The collection is the DB.
        if reset:
            self.reset_db()

        # Load or create the associated memo dict on disk.
        self.path_to_dict = os.path.join(path_to_db_dir, "uid_text_dict.pkl")
        self.uid_text_dict = {}
        self.last_memo_id = 0
        if (not reset) and os.path.exists(self.path_to_dict):
            print(colored("\nLOADING MEMORY FROM DISK", "light_green"))
            print(colored("    Location = {}".format(self.path_to_dict), "light_green"))
            with open(self.path_to_dict, "rb") as f:
                self.uid_text_dict = pickle.load(f)
                self.last_memo_id = len(self.uid_text_dict)
                if self.verbosity >= 3:
                    self.list_memos()

    def consider_storage(self, input_text, output_text):
        # Analyze the input and output text to decide whether to store the memo
        if self._should_store(input_text, output_text):
            self.add_input_output_pair(input_text, output_text)

    def consider_retrieval(self, user_text):
        # Analyze the user's text to decide whether to retrieve a memo
        if self._should_retrieve(user_text):
            return self.get_nearest_memo(user_text)
        else:
            return None

    def list_memos(self):
        """Prints the contents of MemoStore."""
        print(colored("LIST OF MEMOS", "light_green"))
        for uid, text in self.uid_text_dict.items():
            input_text, output_text = text
            print(
                colored(
                    "  ID: {}\n    INPUT TEXT: {}\n    OUTPUT TEXT: {}".format(uid, input_text, output_text),
                    "light_green",
                )
            )

    def close(self):
        """Saves self.uid_text_dict to disk."""
        print(colored("\nSAVING MEMORY TO DISK", "light_green"))
        print(colored("    Location = {}".format(self.path_to_dict), "light_green"))
        with open(self.path_to_dict, "wb") as file:
            pickle.dump(self.uid_text_dict, file)

    def reset_db(self):
        """Forces immediate deletion of the DB's contents, in memory and on disk."""
        print(colored("\nCLEARING MEMORY", "light_green"))
        self.db_client.delete_collection("memos")
        self.vec_db = self.db_client.create_collection("memos")
        self.uid_text_dict = {}

    def add_input_output_pair(self, input_text, output_text):
        """Adds an input-output pair to the vector DB."""
        self.last_memo_id += 1
        self.vec_db.add(documents=[input_text], ids=[str(self.last_memo_id)])
        self.uid_text_dict[str(self.last_memo_id)] = input_text, output_text
        if self.verbosity >= 1:
            print(
                colored(
                    "\nINPUT-OUTPUT PAIR ADDED TO VECTOR DATABASE:\n  ID\n    {}\n  INPUT\n    {}\n  OUTPUT\n    {}".format(
                        self.last_memo_id, input_text, output_text
                    ),
                    "light_green",
                )
            )
        if self.verbosity >= 3:
            self.list_memos()

    def get_nearest_memo(self, query_text):
        """Retrieves the nearest memo to the given query text."""
        results = self.vec_db.query(query_texts=[query_text], n_results=1)
        uid, input_text, distance = results["ids"][0][0], results["documents"][0][0], results["distances"][0][0]
        input_text_2, output_text = self.uid_text_dict[uid]
        assert input_text == input_text_2
        if self.verbosity >= 1:
            print(
                colored(
                    "\nINPUT-OUTPUT PAIR RETRIEVED FROM VECTOR DATABASE:\n  INPUT1\n    {}\n  OUTPUT\n    {}\n  DISTANCE\n    {}".format(
                        input_text, output_text, distance
                    ),
                    "light_green",
                )
            )
        return input_text, output_text, distance

    def get_related_memos(self, query_text, n_results, threshold):
        """Retrieves memos that are related to the given query text within the specified distance threshold."""
        if n_results > len(self.uid_text_dict):
            n_results = len(self.uid_text_dict)
        results = self.vec_db.query(query_texts=[query_text], n_results=n_results)
        memos = []
        num_results = len(results["ids"][0])
        for i in range(num_results):
            uid, input_text, distance = results["ids"][0][i], results["documents"][0][i], results["distances"][0][i]
            if distance < threshold:
                input_text_2, output_text = self.uid_text_dict[uid]
                assert input_text == input_text_2
                if self.verbosity >= 1:
                    print(
                        colored(
                            "\nINPUT-OUTPUT PAIR RETRIEVED FROM VECTOR DATABASE:\n  INPUT1\n    {}\n  OUTPUT\n    {}\n  DISTANCE\n    {}".format(
                                input_text, output_text, distance
                            ),
                            "light_green",
                        )
                    )
                memos.append((input_text, output_text, distance))
        return memos
    def __init__(self, verbosity, reset, path_to_db_dir):
        self.verbosity = verbosity
        self.reset = reset
        self.path_to_db_dir = path_to_db_dir
        self.path_to_dict = os.path.join(path_to_db_dir, "uid_text_dict.pkl")

        # Initialize the database client and the vector database
        settings = Settings(anonymized_telemetry=False, allow_reset=True, is_persistent=True, persist_directory=path_to_db_dir)
        self.db_client = chromadb.Client(settings)
        self.vec_db = self.db_client.create_collection("memos", get_or_create=True)

        # Handle database reset if required
        if reset:
            self.reset_db()

        # Load existing data from disk if not resetting
        self.uid_text_dict = self._load_data_from_disk() if os.path.exists(self.path_to_dict) and not reset else {}
        self.last_memo_id = len(self.uid_text_dict)

    def _load_data_from_disk(self):
        """Load memo data from disk."""
        with open(self.path_to_dict, "rb") as f:
            return pickle.load(f)

    def add_input_output_pair(self, input_text, output_text):
        """Add a new input-output pair to the database."""
        self.last_memo_id += 1
        self.vec_db.add([input_text], [str(self.last_memo_id)])
        self.uid_text_dict[str(self.last_memo_id)] = (input_text, output_text)
        self._save_data_to_disk()

    def get_related_memos(self, query_text, n_results, threshold):
        """Retrieve related memos based on a query text."""
        results = self.vec_db.query([query_text], n_results)
        return [(self.uid_text_dict[str(uid)], distance) for uid, _, distance in results if distance < threshold]

    def reset_db(self):
        """Reset the memo database."""
        self.db_client.delete_collection("memos")
        self.vec_db = self.db_client.create_collection("memos", get_or_create=True)
        self.uid_text_dict = {}
        self._save_data_to_disk()

    def _save_data_to_disk(self):
        """Save memo data to disk."""
        with open(self.path_to_dict, "wb") as file:
            pickle.dump(self.uid_text_dict, file)

    def list_memos(self):
        """Prints all stored memos for debugging purposes."""
        if self.verbosity >= 1:
            print("Stored Memos:")
            for uid, (input_text, output_text) in self.uid_text_dict.items():
                print(f"ID: {uid}\nInput: {input_text}\nOutput: {output_text}\n")

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


