import os
import sys
import pickle
import chromadb
from tqdm import tqdm
from typing import Callable, Dict, Optional, Union, List, Tuple, Any
from termcolor import colored
from autogen import ConversableAgent, config_list_from_json
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from autogen.agentchat.agent import Agent
from autogen import UserProxyAgent
from autogen.agentchat.contrib.teachable_agent import TeachableAgent
from chromadb.config import Settings

# Additional required imports

# LLM Configuration
filter_dict = {"model": ["gpt-4"]}  # Preference for GPT-4 due to reliability
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict=filter_dict)
llm_config = {"config_list": config_list, "timeout": 120}

# Agent Initialization
def create_teachable_agent():
    teachable_agent = TeachableAgent(
        name="teachableagent",
        llm_config=llm_config,
        teach_config={
            "reset_db": False,
            "path_to_db_dir": "./tmp/interactive/teachable_agent_db"
        }
    )
    return teachable_agent

class TeachableAgent(ConversableAgent):
    def __init__(
        self,
        name="teachableagent",
        system_message: Optional[str] = "You are a helpful AI assistant that remembers user teachings from prior chats.",
        human_input_mode: Optional[str] = "NEVER",
        llm_config: Optional[Union[Dict, bool]] = None,
        analyzer_llm_config: Optional[Union[Dict, bool]] = None,
        teach_config: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            human_input_mode=human_input_mode,
            llm_config=llm_config,
            **kwargs,
        )
        self._teach_config = {} if teach_config is None else teach_config
        self.verbosity = self._teach_config.get("verbosity", 0)
        self.reset_db = self._teach_config.get("reset_db", False)
        self.path_to_db_dir = self._teach_config.get("path_to_db_dir", "./tmp/teachable_agent_db")
        self.prepopulate = self._teach_config.get("prepopulate", True)
        self.recall_threshold = self._teach_config.get("recall_threshold", 1.5)
        self.max_num_retrievals = self._teach_config.get("max_num_retrievals", 10)

        if analyzer_llm_config is None:
            analyzer_llm_config = llm_config
        self.analyzer = TextAnalyzerAgent(llm_config=analyzer_llm_config)

        self.memo_store = MemoStore(self.verbosity, self.reset_db, self.path_to_db_dir)
        self.user_comments = []

        # Register a custom reply function.
        self.register_reply(Agent, TeachableAgent._generate_teachable_assistant_reply, position=2)

    def _generate_teachable_assistant_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        # Implementation of the reply generation logic
        if self.llm_config is False:
            raise ValueError("TeachableAgent requires self.llm_config to be set in its base class.")
        if messages is None:
            messages = self._oai_messages[sender]

        last_message = messages[-1]
        user_text = last_message["content"]
        new_user_text = self.consider_memo_retrieval(user_text)
        if new_user_text != user_text:
            messages = messages.copy()
            messages[-1]["content"] = new_user_text

        return self.generate_oai_reply(messages, sender, config)

    def learn_from_user_feedback(self):
        for comment in self.user_comments:
            self.consider_memo_storage(comment)
        self.user_comments = []

    def consider_memo_storage(self, comment):
        # Example implementation; adapt as needed
        if "store this" in comment.lower():
            self.memo_store.add_input_output_pair(comment, "Stored information")

    def consider_memo_retrieval(self, comment):
        memo_list = self.retrieve_relevant_memos(comment)
        return comment + self.concatenate_memo_texts(memo_list)

    def retrieve_relevant_memos(self, input_text):
        return self.memo_store.get_related_memos(input_text, self.max_num_retrievals, self.recall_threshold)

    def concatenate_memo_texts(self, memo_list):
        return "\n".join(["- " + memo for _, memo, _ in memo_list])

    def analyze(self, text_to_analyze, analysis_instructions):
        return self.analyzer.analyze_text(text_to_analyze, analysis_instructions)

class MemoStore:
    def __init__(self, verbosity, reset, path_to_db_dir):
        self.verbosity = verbosity
        self.reset = reset
        self.path_to_db_dir = path_to_db_dir

        settings = Settings(
            anonymized_telemetry=False, allow_reset=True, is_persistent=True, persist_directory=path_to_db_dir
        )
        self.db_client = chromadb.Client(settings)
        self.vec_db = self.db_client.create_collection("memos", get_or_create=True)

        if reset:
            self.reset_db()

        self.path_to_dict = os.path.join(path_to_db_dir, "uid_text_dict.pkl")
        self.uid_text_dict = {}
        self.last_memo_id = 0

        if (not reset) and os.path.exists(self.path_to_dict):
            with open(self.path_to_dict, "rb") as f:
                self.uid_text_dict = pickle.load(f)
                self.last_memo_id = len(self.uid_text_dict)

    # Include other methods of MemoStore as in your script

# Additional setup or utility functions if needed



class TeachableAgent(ConversableAgent):
    """(Experimental) Teachable Agent, a subclass of ConversableAgent using a vector database to remember user teachings.
    In this class, the term 'user' refers to any caller (human or not) sending messages to this agent.
    Not yet tested in the group-chat setting."""

    def __init__(
        self,
        name="teachableagent",
        system_message: Optional[
            str
        ] = "You are a helpful AI assistant that remembers user teachings from prior chats.",
        human_input_mode: Optional[str] = "NEVER",
        llm_config: Optional[Union[Dict, bool]] = None,
        analyzer_llm_config: Optional[Union[Dict, bool]] = None,
        teach_config: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Args:
            name (str): name of the agent.
            system_message (str): system message for the ChatCompletion inference.
            human_input_mode (str): This agent should NEVER prompt the human for input.
            llm_config (dict or False): llm inference configuration.
                Please refer to [OpenAIWrapper.create](/docs/reference/oai/client#create)
                for available options.
                To disable llm-based auto reply, set to False.
            analyzer_llm_config (dict or False): llm inference configuration passed to TextAnalyzerAgent.
                Given the default setting of None, TeachableAgent passes its own llm_config to TextAnalyzerAgent.
            teach_config (dict or None): Additional parameters used by TeachableAgent.
                To use default config, set to None. Otherwise, set to a dictionary with any of the following keys:
                - verbosity (Optional, int): # 0 (default) for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
                - reset_db (Optional, bool): True to clear the DB before starting. Default False.
                - path_to_db_dir (Optional, str): path to the directory where the DB is stored. Default "./tmp/teachable_agent_db"
                - prepopulate (Optional, int): True (default) to prepopulate the DB with a set of input-output pairs.
                - recall_threshold (Optional, float): The maximum distance for retrieved memos, where 0.0 is exact match. Default 1.5. Larger values allow more (but less relevant) memos to be recalled.
                - max_num_retrievals (Optional, int): The maximum number of memos to retrieve from the DB. Default 10.
            **kwargs (dict): other kwargs in [ConversableAgent](../conversable_agent#__init__).
        """
        super().__init__(
            name=name,
            system_message=system_message,
            human_input_mode=human_input_mode,
            llm_config=llm_config,
            **kwargs,
        )

        
        # Register a custom reply function.
        self.register_reply(Agent, TeachableAgent._generate_teachable_assistant_reply, position=2)

        # Assemble the parameter settings.
        self._teach_config = {} if teach_config is None else teach_config
        self.verbosity = self._teach_config.get("verbosity", 0)
        self.reset_db = self._teach_config.get("reset_db", False)
        self.path_to_db_dir = self._teach_config.get("path_to_db_dir", "./tmp/teachable_agent_db")
        self.prepopulate = self._teach_config.get("prepopulate", True)
        self.recall_threshold = self._teach_config.get("recall_threshold", 1.5)
        self.max_num_retrievals = self._teach_config.get("max_num_retrievals", 10)

        # Create the analyzer.
        if analyzer_llm_config is None:
            analyzer_llm_config = llm_config
        self.analyzer = TextAnalyzerAgent(llm_config=analyzer_llm_config)

        # Create the memo store.
        self.memo_store = MemoStore(self.verbosity, self.reset_db, self.path_to_db_dir)
        self.user_comments = []  # Stores user comments until the end of each chat.

    def close_db(self):
        """Cleanly closes the memo store."""
        self.memo_store.close()

    def prepopulate_db(self):
        """Adds a few arbitrary memos to the DB."""
        self.memo_store.prepopulate()

    def _generate_teachable_assistant_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,  # Persistent state.
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """
        Generates a reply to the last user message, after querying the memo store for relevant information.
        Uses TextAnalyzerAgent to make decisions about memo storage and retrieval.
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
        if self.memo_store.last_memo_id > 0:
            new_user_text = self.consider_memo_retrieval(user_text)
            if new_user_text != user_text:
                # Make a copy of the message list, and replace the last user message with the new one.
                messages = messages.copy()
                messages[-1]["content"] = new_user_text

        # Generate a response by reusing existing generate_oai_reply
        return self.generate_oai_reply(messages, sender, config)

    def learn_from_user_feedback(self):
        """Reviews the user comments from the last chat, and decides what teachings to store as memos."""
        print(colored("\nREVIEWING CHAT FOR USER TEACHINGS TO REMEMBER", "light_yellow"))
        # Look at each user turn.
        if len(self.user_comments) > 0:
            for comment in tqdm(self.user_comments):
                # Consider whether to store something from this user turn in the DB.
                self.consider_memo_storage(comment)
        self.user_comments = []

    def consider_memo_storage(self, comment):
        """Decides whether to store something from one user comment in the DB."""
        # Check for a problem-solution pair.
        response = self.analyze(
            comment,
            "Does any part of the TEXT ask the agent to perform a task or solve a problem? Answer with just one word, yes or no.",
        )
        if "yes" in response.lower():
            # Can we extract advice?
            advice = self.analyze(
                comment,
                "Briefly copy any advice from the TEXT that may be useful for a similar but different task in the future. But if no advice is present, just respond with 'none'.",
            )
            if "none" not in advice.lower():
                # Yes. Extract the task.
                task = self.analyze(
                    comment,
                    "Briefly copy just the task from the TEXT, then stop. Don't solve it, and don't include any advice.",
                )
                # Generalize the task.
                general_task = self.analyze(
                    task,
                    "Summarize very briefly, in general terms, the type of task described in the TEXT. Leave out details that might not appear in a similar problem.",
                )
                # Add the task-advice (problem-solution) pair to the vector DB.
                if self.verbosity >= 1:
                    print(colored("\nREMEMBER THIS TASK-ADVICE PAIR", "light_yellow"))
                self.memo_store.add_input_output_pair(general_task, advice)

        # Check for information to be learned.
        response = self.analyze(
            comment,
            "Does the TEXT contain information that could be committed to memory? Answer with just one word, yes or no.",
        )
        if "yes" in response.lower():
            # Yes. What question would this information answer?
            question = self.analyze(
                comment,
                "Imagine that the user forgot this information in the TEXT. How would they ask you for this information? Include no other text in your response.",
            )
            # Extract the information.
            answer = self.analyze(
                comment, "Copy the information from the TEXT that should be committed to memory. Add no explanation."
            )
            # Add the question-answer pair to the vector DB.
            if self.verbosity >= 1:
                print(colored("\nREMEMBER THIS QUESTION-ANSWER PAIR", "light_yellow"))
            self.memo_store.add_input_output_pair(question, answer)

    def consider_memo_retrieval(self, comment):
        """Decides whether to retrieve memos from the DB, and add them to the chat context."""

        # First, use the user comment directly as the lookup key.
        if self.verbosity >= 1:
            print(colored("\nLOOK FOR RELEVANT MEMOS, AS QUESTION-ANSWER PAIRS", "light_yellow"))
        memo_list = self.retrieve_relevant_memos(comment)

        # Next, if the comment involves a task, then extract and generalize the task before using it as the lookup key.
        response = self.analyze(
            comment,
            "Does any part of the TEXT ask the agent to perform a task or solve a problem? Answer with just one word, yes or no.",
        )
        if "yes" in response.lower():
            if self.verbosity >= 1:
                print(colored("\nLOOK FOR RELEVANT MEMOS, AS TASK-ADVICE PAIRS", "light_yellow"))
            # Extract the task.
            task = self.analyze(
                comment, "Copy just the task from the TEXT, then stop. Don't solve it, and don't include any advice."
            )
            # Generalize the task.
            general_task = self.analyze(
                task,
                "Summarize very briefly, in general terms, the type of task described in the TEXT. Leave out details that might not appear in a similar problem.",
            )
            # Append any relevant memos.
            memo_list.extend(self.retrieve_relevant_memos(general_task))

        # De-duplicate the memo list.
        memo_list = list(set(memo_list))

        # Append the memos to the last user message.
        return comment + self.concatenate_memo_texts(memo_list)
    
    

    def retrieve_relevant_memos(self, input_text):
        """Returns semantically related memos from the DB."""
        memo_list = self.memo_store.get_related_memos(
            input_text, n_results=self.max_num_retrievals, threshold=self.recall_threshold
        )

        if self.verbosity >= 1:
            # Was anything retrieved?
            if len(memo_list) == 0:
                # No. Look at the closest memo.
                print(colored("\nTHE CLOSEST MEMO IS BEYOND THE THRESHOLD:", "light_yellow"))
                self.memo_store.get_nearest_memo(input_text)
                print()  # Print a blank line. The memo details were printed by get_nearest_memo().

        # Create a list of just the memo output_text strings.
        memo_list = [memo[1] for memo in memo_list]
        return memo_list

    def concatenate_memo_texts(self, memo_list):
        """Concatenates the memo texts into a single string for inclusion in the chat context."""
        memo_texts = ""
        if len(memo_list) > 0:
            info = "\n# Memories that might help\n"
            for memo in memo_list:
                info = info + "- " + memo + "\n"
            if self.verbosity >= 1:
                print(colored("\nMEMOS APPENDED TO LAST USER MESSAGE...\n" + info + "\n", "light_yellow"))
            memo_texts = memo_texts + "\n" + info
        return memo_texts

    def analyze(self, text_to_analyze, analysis_instructions):
        """Asks TextAnalyzerAgent to analyze the given text according to specific instructions."""
        if self.verbosity >= 2:
            # Use the messaging mechanism so that the analyzer's messages are included in the printed chat.
            self.analyzer.reset()  # Clear the analyzer's list of messages.
            self.send(
                recipient=self.analyzer, message=text_to_analyze, request_reply=False
            )  # Put the message in the analyzer's list.
            self.send(recipient=self.analyzer, message=analysis_instructions, request_reply=True)  # Request the reply.
            return self.last_message(self.analyzer)["content"]
        else:
            # TODO: This is not an encouraged usage pattern. It breaks the conversation-centric design.
            # consider using the arg "silent"
            # Use the analyzer's method directly, to leave analyzer message out of the printed chat.
            return self.analyzer.analyze_text(text_to_analyze, analysis_instructions)


class MemoStore:
    """(Experimental)
    Provides memory storage and retrieval for a TeachableAgent, using a vector database.
    Each DB entry (called a memo) is a pair of strings: an input text and an output text.
    The input text might be a question, or a task to perform.
    The output text might be an answer to the question, or advice on how to perform the task.
    Vector embeddings are currently supplied by Chroma's default Sentence Transformers.
    """

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

class MemoStore:
    def __init__(self, verbosity):
        self.verbosity = verbosity

    def prepopulate(self):
        if self.verbosity >= 1:
            print(colored("\nPREPOPULATING MEMORY", "light_green"))
        examples = [
            {"text": "When I say papers I mean research papers, which are typically pdfs.", "label": "yes"},
            {"text": "Please verify that each paper you listed actually uses langchain.", "label": "no"},
            {"text": "Tell gpt the output should still be latex code.", "label": "no"},
            {"text": "Hint: convert pdfs to text and then answer questions based on them.", "label": "yes"},
            {"text": "To create a good PPT, include enough content to make it interesting.", "label": "yes"},
            {"text": "No, for this case the columns should be aspects and the rows should be frameworks.", "label": "no"},
            {"text": "When writing code, remember to include any libraries that are used.", "label": "yes"},
            {"text": "Please summarize the papers by Eric Horvitz on bounded rationality.", "label": "no"},
            {"text": "Compare the h-index of Daniel Weld and Oren Etzioni.", "label": "no"},
            {"text": "Double check to be sure that the columns in a table correspond to what was asked for.", "label": "yes"}
        ]
        for example in examples:
            self.add_input_output_pair(example["text"], example["label"])

    def add_input_output_pair(self, text, label):
        # Add implementation here to store the input-output pair
        pass

# Example usage:
verbosity = 1  # You can set the verbosity level as needed
memo_store = MemoStore(verbosity)
memo_store.prepopulate()


# Define the function to create a teachable agent
def create_teachable_agent(reset_db=False, llm_config=None):
    teachable_agent = TeachableAgent(
        name="teachableagent",
        llm_config=llm_config,
        teach_config={
            "reset_db": reset_db,
            "path_to_db_dir": "./tmp/interactive/teachable_agent_db"
        }
    )
    return teachable_agent


# Define the check_agent_response function
def check_agent_response(agent, user, expected_response):
    # Implement the logic to check the response of the agent
    # Compare the agent's response to the expected_response
    # Return errors if there are any discrepancies
    # Otherwise, return None or an appropriate result
    pass  # Replace 'pass' with your implementation


def use_question_answer_phrasing():
    """
    Tests whether the teachable agent can answer a question after being taught the answer in a previous chat.
    """
    teachable_agent = create_teachable_agent(reset_db=True, llm_config=None)
    user = ConversableAgent("user", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")

    # Example interaction
    user.initiate_chat(recipient=teachable_agent, message="What is the capital of France?")
    user.send(recipient=teachable_agent, message="The capital of France is Paris.")
    errors = check_agent_response(teachable_agent, user, "Paris")
    teachable_agent.learn_from_user_feedback()
    teachable_agent.close_db()
    return errors


# Define the TeachableAgent class with necessary methods
class TeachableAgent(ConversableAgent):
    # ... [Existing TeachableAgent class definition]

    def learn_from_user_feedback(self):
        for comment in self.user_comments:
            self.consider_memo_storage(comment)
        self.user_comments = []

    def consider_memo_storage(self, comment):
        if "important" in comment.lower():
            task = "Extracted task from comment"
            advice = "Extracted advice from comment"
            self.memo_store.add_input_output_pair(task, advice)

    def consider_memo_retrieval(self, comment):
        if "what is" in comment.lower() or "how to" in comment.lower():
            return comment + self.concatenate_memo_texts(self.retrieve_relevant_memos(comment))
        return comment

    def handle_query(self, user_input):
        processed_input = self.preprocess_input(user_input)
        response = self.generate_response(processed_input)
        return response

    def preprocess_input(self, user_input):
        return user_input.lower()

    def generate_response(self, processed_input):
        if "weather" in processed_input:
            return "The weather is sunny."
        elif "news" in processed_input:
            return "Here are the latest news updates."
        else:
            return "I'm not sure how to respond to that."


# Define the function to initiate an interactive session with the Teachable Agent
def initiate_interactive_session(teachable_agent):
    user = ConversableAgent("user", human_input_mode="ALWAYS")
    print("Starting interactive session with Teachable Agent. Type 'exit' to end the session.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = teachable_agent.handle_query(user_input)
        print("Agent:", response)


# Main Execution Logic
if __name__ == "__main__":
    teachable_agent = create_teachable_agent()
    user = UserProxyAgent("user", human_input_mode="ALWAYS")
    initiate_interactive_session(teachable_agent)
    teachable_agent.learn_from_user_feedback()
    teachable_agent.close_db()



   
