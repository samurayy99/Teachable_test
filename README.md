Enhanced Inference

autogen.OpenAIWrapper provides enhanced LLM inference for openai>=1. autogen.Completion is a drop-in replacement of openai.Completion and openai.ChatCompletion for enhanced LLM inference using openai<1. There are a number of benefits of using autogen to perform inference: performance tuning, API unification, caching, error handling, multi-config inference, result filtering, templating and so on.

Tune Inference Parameters (for openai<1)​

Find a list of examples in this page: Tune Inference Parameters Examples

Choices to optimize​

The cost of using foundation models for text generation is typically measured in terms of the number of tokens in the input and output combined. From the perspective of an application builder using foundation models, the use case is to maximize the utility of the generated text under an inference budget constraint (e.g., measured by the average dollar cost needed to solve a coding problem). This can be achieved by optimizing the hyperparameters of the inference, which can significantly affect both the utility and the cost of the generated text.

The tunable hyperparameters include:

model - this is a required input, specifying the model ID to use.
prompt/messages - the input prompt/messages to the model, which provides the context for the text generation task.
max_tokens - the maximum number of tokens (words or word pieces) to generate in the output.
temperature - a value between 0 and 1 that controls the randomness of the generated text. A higher temperature will result in more random and diverse text, while a lower temperature will result in more predictable text.
top_p - a value between 0 and 1 that controls the sampling probability mass for each token generation. A lower top_p value will make it more likely to generate text based on the most likely tokens, while a higher value will allow the model to explore a wider range of possible tokens.
n - the number of responses to generate for a given prompt. Generating multiple responses can provide more diverse and potentially more useful output, but it also increases the cost of the request.
stop - a list of strings that, when encountered in the generated text, will cause the generation to stop. This can be used to control the length or the validity of the output.
presence_penalty, frequency_penalty - values that control the relative importance of the presence and frequency of certain words or phrases in the generated text.
best_of - the number of responses to generate server-side when selecting the "best" (the one with the highest log probability per token) response for a given prompt.
The cost and utility of text generation are intertwined with the joint effect of these hyperparameters. There are also complex interactions among subsets of the hyperparameters. For example, the temperature and top_p are not recommended to be altered from their default values together because they both control the randomness of the generated text, and changing both at the same time can result in conflicting effects; n and best_of are rarely tuned together because if the application can process multiple outputs, filtering on the server side causes unnecessary information loss; both n and max_tokens will affect the total number of tokens generated, which in turn will affect the cost of the request. These interactions and trade-offs make it difficult to manually determine the optimal hyperparameter settings for a given text generation task.

Do the choices matter? Check this blogpost to find example tuning results about gpt-3.5-turbo and gpt-4.

With AutoGen, the tuning can be performed with the following information:

Validation data.
Evaluation function.
Metric to optimize.
Search space.
Budgets: inference and optimization respectively.
Validation data​

Collect a diverse set of instances. They can be stored in an iterable of dicts. For example, each instance dict can contain "problem" as a key and the description str of a math problem as the value; and "solution" as a key and the solution str as the value.

Evaluation function​

The evaluation function should take a list of responses, and other keyword arguments corresponding to the keys in each validation data instance as input, and output a dict of metrics. For example,

def eval_math_responses(responses: List[str], solution: str, **args) -> Dict:
    # select a response from the list of responses
    answer = voted_answer(responses)
    # check whether the answer is correct
    return {"success": is_equivalent(answer, solution)}
autogen.code_utils and autogen.math_utils offer some example evaluation functions for code generation and math problem solving.

Metric to optimize​

The metric to optimize is usually an aggregated metric over all the tuning data instances. For example, users can specify "success" as the metric and "max" as the optimization mode. By default, the aggregation function is taking the average. Users can provide a customized aggregation function if needed.

Search space​

Users can specify the (optional) search range for each hyperparameter.

model. Either a constant str, or multiple choices specified by flaml.tune.choice.
prompt/messages. Prompt is either a str or a list of strs, of the prompt templates. messages is a list of dicts or a list of lists, of the message templates. Each prompt/message template will be formatted with each data instance. For example, the prompt template can be: "{problem} Solve the problem carefully. Simplify your answer as much as possible. Put the final answer in \boxed{{}}." And {problem} will be replaced by the "problem" field of each data instance.
max_tokens, n, best_of. They can be constants, or specified by flaml.tune.randint, flaml.tune.qrandint, flaml.tune.lograndint or flaml.qlograndint. By default, max_tokens is searched in [50, 1000); n is searched in [1, 100); and best_of is fixed to 1.
stop. It can be a str or a list of strs, or a list of lists of strs or None. Default is None.
temperature or top_p. One of them can be specified as a constant or by flaml.tune.uniform or flaml.tune.loguniform etc. Please don't provide both. By default, each configuration will choose either a temperature or a top_p in [0, 1] uniformly.
presence_penalty, frequency_penalty. They can be constants or specified by flaml.tune.uniform etc. Not tuned by default.
Budgets​

One can specify an inference budget and an optimization budget. The inference budget refers to the average inference cost per data instance. The optimization budget refers to the total budget allowed in the tuning process. Both are measured by dollars and follow the price per 1000 tokens.

Perform tuning​

Now, you can use autogen.Completion.tune for tuning. For example,

import autogen

config, analysis = autogen.Completion.tune(
    data=tune_data,
    metric="success",
    mode="max",
    eval_func=eval_func,
    inference_budget=0.05,
    optimization_budget=3,
    num_samples=-1,
)
num_samples is the number of configurations to sample. -1 means unlimited (until optimization budget is exhausted). The returned config contains the optimized configuration and analysis contains an ExperimentAnalysis object for all the tried configurations and results.

The tuend config can be used to perform inference.

API unification​

autogen.OpenAIWrapper.create() can be used to create completions for both chat and non-chat models, and both OpenAI API and Azure OpenAI API.

from autogen import OpenAIWrapper
# OpenAI endpoint
client = OpenAIWrapper()
# ChatCompletion
response = client.create(messages=[{"role": "user", "content": "2+2="}], model="gpt-3.5-turbo")
# extract the response text
print(client.extract_text_or_completion_object(response))
# get cost of this completion
print(response.cost)
# Azure OpenAI endpoint
client = OpenAIWrapper(api_key=..., base_url=..., api_version=..., api_type="azure")
# Completion
response = client.create(prompt="2+2=", model="gpt-3.5-turbo-instruct")
# extract the response text
print(client.extract_text_or_completion_object(response))

For local LLMs, one can spin up an endpoint using a package like FastChat, and then use the same API to send a request. See here for examples on how to make inference with local LLMs.

Usage Summary​

The OpenAIWrapper from autogen tracks token counts and costs of your API calls. Use the create() method to initiate requests and print_usage_summary() to retrieve a detailed usage report, including total cost and token usage for both cached and actual requests.

mode=["actual", "total"] (default): print usage summary for all completions and non-caching completions.
mode='actual': only print non-cached usage.
mode='total': only print all usage (including cache).
Reset your session's usage data with clear_usage_summary() when needed. View Notebook

Example usage:

from autogen import OpenAIWrapper

client = OpenAIWrapper()
client.create(messages=[{"role": "user", "content": "Python learning tips."}], model="gpt-3.5-turbo")
client.print_usage_summary()  # Display usage
client.clear_usage_summary()  # Reset usage data
Sample output:

Usage summary excluding cached usage:
Total cost: 0.00015
* Model 'gpt-3.5-turbo': cost: 0.00015, prompt_tokens: 25, completion_tokens: 58, total_tokens: 83

Usage summary including cached usage:
Total cost: 0.00027
* Model 'gpt-3.5-turbo': cost: 0.00027, prompt_tokens: 50, completion_tokens: 100, total_tokens: 150
Caching​

API call results are cached locally and reused when the same request is issued. This is useful when repeating or continuing experiments for reproducibility and cost saving. It still allows controlled randomness by setting the "cache_seed" specified in OpenAIWrapper.create() or the constructor of OpenAIWrapper.

client = OpenAIWrapper(cache_seed=...)
client.create(...)
client = OpenAIWrapper()
client.create(cache_seed=..., ...)
Caching is enabled by default with cache_seed 41. To disable it please set cache_seed to None.

NOTE. openai v1.1 introduces a new param seed. The difference between autogen's cache_seed and openai's seed is that:

autogen uses local disk cache to guarantee the exactly same output is produced for the same input and when cache is hit, no openai api call will be made.
openai's seed is a best-effort deterministic sampling with no guarantee of determinism. When using openai's seed with cache_seed set to None, even for the same input, an openai api call will be made and there is no guarantee for getting exactly the same output.
Error handling​

Runtime error​

One can pass a list of configurations of different models/endpoints to mitigate the rate limits and other runtime error. For example,

client = OpenAIWrapper(
    config_list=[
        {
            "model": "gpt-4",
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
            "api_type": "azure",
            "base_url": os.environ.get("AZURE_OPENAI_API_BASE"),
            "api_version": "2023-08-01-preview",
        },
        {
            "model": "gpt-3.5-turbo",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": "https://api.openai.com/v1",
        },
        {
            "model": "llama2-chat-7B",
            "base_url": "http://127.0.0.1:8080",
        }
    ],
)
client.create() will try querying Azure OpenAI gpt-4, OpenAI gpt-3.5-turbo, and a locally hosted llama2-chat-7B one by one, until a valid result is returned. This can speed up the development process where the rate limit is a bottleneck. An error will be raised if the last choice fails. So make sure the last choice in the list has the best availability.

For convenience, we provide a number of utility functions to load config lists.

get_config_list: Generates configurations for API calls, primarily from provided API keys.
config_list_openai_aoai: Constructs a list of configurations using both Azure OpenAI and OpenAI endpoints, sourcing API keys from environment variables or local files.
config_list_from_json: Loads configurations from a JSON structure, either from an environment variable or a local JSON file, with the flexibility of filtering configurations based on given criteria.
config_list_from_models: Creates configurations based on a provided list of models, useful when targeting specific models without manually specifying each configuration.
config_list_from_dotenv: Constructs a configuration list from a .env file, offering a consolidated way to manage multiple API configurations and keys from a single file.
We suggest that you take a look at this notebook for full code examples of the different methods to configure your model endpoints.

Logic error​

Another type of error is that the returned response does not satisfy a requirement. For example, if the response is required to be a valid json string, one would like to filter the responses that are not. This can be achieved by providing a list of configurations and a filter function. For example,

def valid_json_filter(response, **_):
    for text in OpenAIWrapper.extract_text_or_completion_object(response):
        try:
            json.loads(text)
            return True
        except ValueError:
            pass
    return False

client = OpenAIWrapper(
    config_list=[{"model": "text-ada-001"}, {"model": "gpt-3.5-turbo-instruct"}, {"model": "text-davinci-003"}],
)
response = client.create(
    prompt="How to construct a json request to Bing API to search for 'latest AI news'? Return the JSON request.",
    filter_func=valid_json_filter,
)
The example above will try to use text-ada-001, gpt-3.5-turbo-instruct, and text-davinci-003 iteratively, until a valid json string is returned or the last config is used. One can also repeat the same model in the list for multiple times (with different seeds) to try one model multiple times for increasing the robustness of the final response.

Advanced use case: Check this blogpost to find how to improve GPT-4's coding performance from 68% to 90% while reducing the inference cost.

Templating​

If the provided prompt or message is a template, it will be automatically materialized with a given context. For example,

response = client.create(
    context={"problem": "How many positive integers, not exceeding 100, are multiples of 2 or 3 but not 4?"},
    prompt="{problem} Solve the problem carefully.",
    allow_format_str_template=True,
    **config
)
A template is either a format str, like the example above, or a function which produces a str from several input fields, like the example below.

def content(turn, context):
    return "\n".join(
        [
            context[f"user_message_{turn}"],
            context[f"external_info_{turn}"]
        ]
    )

messages = [
    {
        "role": "system",
        "content": "You are a teaching assistant of math.",
    },
    {
        "role": "user",
        "content": partial(content, turn=0),
    },
]
context = {
    "user_message_0": "Could you explain the solution to Problem 1?",
    "external_info_0": "Problem 1: ...",
}

response = client.create(context=context, messages=messages, **config)
messages.append(
    {
        "role": "assistant",
        "content": client.extract_text(response)[0]
    }
)
messages.append(
    {
        "role": "user",
        "content": partial(content, turn=1),
    },
)
context.append(
    {
        "user_message_1": "Why can't we apply Theorem 1 to Equation (2)?",
        "external_info_1": "Theorem 1: ...",
    }
)
response = client.create(context=context, messages=messages, **config)
Logging (for openai<1)​

When debugging or diagnosing an LLM-based system, it is often convenient to log the API calls and analyze them. autogen.Completion and autogen.ChatCompletion offer an easy way to collect the API call histories. For example, to log the chat histories, simply run:

autogen.ChatCompletion.start_logging()
The API calls made after this will be automatically logged. They can be retrieved at any time by:

autogen.ChatCompletion.logged_history
There is a function that can be used to print usage summary (total cost, and token count usage from each model):

autogen.ChatCompletion.print_usage_summary()
To stop logging, use

autogen.ChatCompletion.stop_logging()
If one would like to append the history to an existing dict, pass the dict like:

autogen.ChatCompletion.start_logging(history_dict=existing_history_dict)
By default, the counter of API calls will be reset at start_logging(). If no reset is desired, set reset_counter=False.

There are two types of logging formats: compact logging and individual API call logging. The default format is compact. Set compact=False in start_logging() to switch.

Example of a history dict with compact logging.
{
    """
    [
        {
            'role': 'system',
            'content': system_message,
        },
        {
            'role': 'user',
            'content': user_message_1,
        },
        {
            'role': 'assistant',
            'content': assistant_message_1,
        },
        {
            'role': 'user',
            'content': user_message_2,
        },
        {
            'role': 'assistant',
            'content': assistant_message_2,
        },
    ]""": {
        "created_at": [0, 1],
        "cost": [0.1, 0.2],
    }
}
Example of a history dict with individual API call logging.
{
    0: {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message_1,
                }
            ],
            ... # other parameters in the request
        },
        "response": {
            "choices": [
                "messages": {
                    "role": "assistant",
                    "content": assistant_message_1,
                },
            ],
            ... # other fields in the response
        }
    },
    1: {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message_1,
                },
                {
                    "role": "assistant",
                    "content": assistant_message_1,
                },
                {
                    "role": "user",
                    "content": user_message_2,
                },
            ],
            ... # other parameters in the request
        },
        "response": {
            "choices": [
                "messages": {
                    "role": "assistant",
                    "content": assistant_message_2,
                },
            ],
            ... # other fields in the response
        }
    },
}
Example of printing for usage summary
Total cost: <cost>
Token count summary for model <model>: prompt_tokens: <count 1>, completion_tokens: <count 2>, total_tokens: <count 3>
It can be seen that the individual API call history contains redundant information of the conversation. For a long conversation the degree of redundancy is high. The compact history is more efficient and the individual API call history contains more details.
-----------------------------------------------------------------------------------------


TEACHABLE AGENT DOCOMENTATION

---
title: AutoGen's TeachableAgent
authors: rickyloynd-microsoft
tags: [LLM, teach]
---

![Teachable Agent Architecture](img/teachable-arch.png)

**TL;DR:**
* We introduce **TeachableAgent** (which uses **TextAnalyzerAgent**) so that users can teach their LLM-based assistants new facts, preferences, and skills.
* We showcase examples of `TeachableAgent` learning and later recalling facts, preferences, and skills in subsequent chats.


## Introduction
Conversational assistants based on LLMs can remember the current chat with the user, and can also demonstrate in-context learning of user teachings during the conversation. But the assistant's memories and learnings are lost once the chat is over, or when a single chat grows too long for the LLM to handle effectively. Then in subsequent chats the user is forced to repeat any necessary instructions over and over.

`TeachableAgent` addresses these limitations by persisting user teachings across chat boundaries in long-term memory implemented as a vector database. Memory is automatically saved to disk at the end of each chat, then loaded from disk at the start of the next. Instead of copying all of memory into the context window, which would eat up valuable space, individual memories (called memos) are retrieved into context as needed. This allows the user to teach frequently used facts and skills to the teachable agent just once, and have it recall them in later chats.

In order to make effective decisions about memo storage and retrieval, `TeachableAgent` calls an instance of `TextAnalyzerAgent` (another AutoGen agent) to identify and reformulate text as needed for remembering facts, preferences, and skills. Note that this adds extra LLM calls involving a relatively small number of tokens, which can add a few seconds to the time a user waits for each response.



## Basic Usage of TeachableAgent
1. Install dependencies

Please install pyautogen with the [teachable] option before using TeachableAgent.
```bash
pip install "pyautogen[teachable]"
```

2. Import agents
```python
from autogen import UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.teachable_agent import TeachableAgent
```

3. Create llm_config
```python
# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
filter_dict = {"model": ["gpt-4"]}  # GPT-3.5 is less reliable than GPT-4 at learning from user feedback.
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict=filter_dict)
llm_config={"config_list": config_list, "timeout": 120}
```

4. Create the agents
```python
teachable_agent = TeachableAgent(
    name="teachableagent",
    llm_config=llm_config,
    teach_config={
        "reset_db": False,  # Use True to force-reset the memo DB, and False to use an existing DB.
        "path_to_db_dir": "./tmp/interactive/teachable_agent_db"  # Can be any path.
    }
)

user = UserProxyAgent("user", human_input_mode="ALWAYS")
```

5. Chat with `TeachableAgent`
```python
# This function will return once the user types 'exit'.
teachable_agent.initiate_chat(user, message="Hi, I'm a teachable user assistant! What's on your mind?")
```

6. Update the database on disk
```python
# Before closing the app, let the teachable agent store things that should be learned from this chat.
teachable_agent.learn_from_user_feedback()
teachable_agent.close_db()
```


## Example 1 - Learning user info

A user can teach the agent facts about themselves.
(Note that due to their finetuning, LLMs can be reluctant to admit that they know personal information.)
```
CLEARING MEMORY
teachableagent (to user):

Hi, I'm a teachable user assistant! What's on your mind?

--------------------------------------------------------------------------------
Provide feedback to teachableagent. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: My name is Ricky
user (to teachableagent):

My name is Ricky

--------------------------------------------------------------------------------
teachableagent (to user):

Hello, Ricky! It's nice to meet you. What can I help you with today?

--------------------------------------------------------------------------------
Provide feedback to teachableagent. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: exit

REVIEWING CHAT FOR USER TEACHINGS TO REMEMBER

SAVING MEMORY TO DISK
    Location = ./tmp/interactive/teachable_agent_db\uid_text_dict.pkl
```

In a later conversation, the user can check whether the teachable agent remembers their name. (For readability, the user prompts and some logged notices are not repeated below.)
```
--------------------------------------------------------------------------------
user (to teachableagent):

what's my name?

--------------------------------------------------------------------------------
teachableagent (to user):

Hi Ricky! How can I help you today?

--------------------------------------------------------------------------------
```


## Example 2 - Learning new facts

A user can teach the agent more complex, related facts.
```
--------------------------------------------------------------------------------
user (to teachableagent):

What is the Vicuna model?

--------------------------------------------------------------------------------
teachableagent (to user):

I'm sorry, but I don't have information about the "Vicuna model" in my current database. Could you please provide some context, description, or details about this model so I can better assist you?

--------------------------------------------------------------------------------
user (to teachableagent):

Vicuna is a 13B-parameter language model released by Meta.

--------------------------------------------------------------------------------
teachableagent (to user):

Thank you for providing the information. The Vicuna model is a large-scale language model developed by Meta. It consists of 13 billion parameters, which is what makes it able to understand and generate human-like text across various topics. Parameters in a neural network are the weights and biases assigned to each connection between the neurons, and having more parameters enables the model to learn and capture more complex patterns and relationships in data. As an AI language model, Vicuna can be utilized in various language-related tasks, such as text completion, translation, summarization, and more.

--------------------------------------------------------------------------------
user (to teachableagent):

What is the Orca model?

--------------------------------------------------------------------------------
teachableagent (to user):

The Orca model is a yet-to-be-released language model by Meta, referring to the Research Preview API release of Meta's advanced language model. While I don't have specific details about the Orca model at this time, it is expected that the model would have an even larger number of parameters to further enhance its capabilities compared to its predecessor, Vicuna.

As AI language models evolve, they generally become more efficient at understanding and generating human-like text. This makes them more effective in completing a wide range of tasks, including text completion, translation, summarization, and Q&A, as well as providing more accurate and coherent results in various applications.

--------------------------------------------------------------------------------
user (to teachableagent):

Orca is a 13B-parameter language model released by Microsoft. It outperforms Vicuna on most tasks.

--------------------------------------------------------------------------------
teachableagent (to user):

Thank you for the clarification. Orca is a 13-billion-parameter language model developed by Microsoft. It is designed to outperform Vicuna, the language model released by Meta. With a large number of parameters, Orca is capable of understanding and generating human-like text across diverse subjects, making it highly useful in various applications.

Orca's advanced capabilities allow it to excel in tasks such as text completion, translation, summarization, Q&A, and more. Its performance improvement over Vicuna highlights the rapid advancements in natural language processing (NLP) research and the development of increasingly capable AI language models.
```

Then in a later chat the teachable agent can answer questions about the facts it has been taught.
(Remember to first close the previous chat by typing 'exit'.)
```
--------------------------------------------------------------------------------
user (to teachableagent):

How does the Vicuna model compare to the Orca model?

--------------------------------------------------------------------------------
teachableagent (to user):

The Vicuna and Orca models are both 13B-parameter language models released by Meta and Microsoft, respectively. In terms of performance, Orca has been reported to outperform Vicuna on most tasks. However, without detailed information about specific tasks and benchmarks, it is difficult to provide a more comprehensive comparison. Generally speaking, both models are advanced language models that aim to provide high-quality natural language understanding and generation, but Orca appears to have an edge in terms of overall performance.

--------------------------------------------------------------------------------
```


## Example 3 - Learning user preferences

A user can teach the agent how they prefer to have things done.

Be aware that a message like the next one cannot be entered as a single message through a command line because it contains a newline character.
Such messages can be entered in a Jupyter notebook, or through some UI layer like that of ChatGPT.
```
--------------------------------------------------------------------------------
user (to teachableagent):

Please summarize this abstract.

AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, Chi Wang
AutoGen is an open-source framework that allows developers to build LLM applications via multiple agents that can converse with each other to accomplish tasks. AutoGen agents are customizable, conversable, and can operate in various modes that employ combinations of LLMs, human inputs, and tools. Using AutoGen, developers can also flexibly define agent interaction behaviors. Both natural language and computer code can be used to program flexible conversation patterns for different applications. AutoGen serves as a generic infrastructure to build diverse applications of various complexities and LLM capacities. Empirical studies demonstrate the effectiveness of the framework in many example applications, with domains ranging from mathematics, coding, question answering, operations research, online decision-making, entertainment, etc.

--------------------------------------------------------------------------------
teachableagent (to user):

AutoGen is an open-source framework designed to enable developers to create LLM applications with multiple customizable agents that can converse with each other to complete tasks. These agents can operate using different combinations of LLMs, human inputs, and tools, allowing developers to define agent interaction behaviors flexibly. AutoGen supports programming flexible conversation patterns using both natural language and code, making it suitable for building diverse applications with varying complexities and LLM capacities. Its effectiveness has been demonstrated through empirical studies across various domains including mathematics, coding, operations research, decision-making, and entertainment.

--------------------------------------------------------------------------------
Please summarize this abstract. When I'm summarizing an abstract, I try to make the summary contain just three short bullet points:  the title, the innovation, and the key empirical results.

AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, Chi Wang
AutoGen is an open-source framework that allows developers to build LLM applications via multiple agents that can converse with each other to accomplish tasks. AutoGen agents are customizable, conversable, and can operate in various modes that employ combinations of LLMs, human inputs, and tools. Using AutoGen, developers can also flexibly define agent interaction behaviors. Both natural language and computer code can be used to program flexible conversation patterns for different applications. AutoGen serves as a generic infrastructure to build diverse applications of various complexities and LLM capacities. Empirical studies demonstrate the effectiveness of the framework in many example applications, with domains ranging from mathematics, coding, question answering, operations research, online decision-making, entertainment, etc.

--------------------------------------------------------------------------------
teachableagent (to user):

- Title: AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
- Innovation: Open-source framework for creating customizable LLM applications through agent conversations, supporting various modes and interaction behaviors.
- Key Empirical Results: Demonstrated effectiveness across diverse application domains, including mathematics, coding, question answering, and more.

--------------------------------------------------------------------------------
```


--------------------------------------------------------------------------------
teachableagent (to user):

- Title: Sparks of Artificial General Intelligence: Early experiments with GPT-4
- Innovation: GPT-4, an LLM with remarkable capabilities, demonstrates human-level performance across various domains, like math, coding, vision, medicine, law, and psychology.
- Key results: GPT-4 significantly surpasses prior models, suggesting it may be an early version of AGI; limitations and challenges toward deeper AGI are also discussed.

--------------------------------------------------------------------------------
```


## Example 4 - Learning new skills

Users can extend the teachable agent's capabilities by teaching it new skills for accomplishing challenging tasks. It usually works best to first describe the task, then (in the same turn) provide a hint or advice for approaching the task.

The [Sparks of AGI](https://arxiv.org/abs/2303.12712) paper evaluated GPT-4 on math problems like the following, which it could only solve 32% of the time. We first show a failure case, then teach the agent a strategy which lifts GPT-4's success rate above 95%.
```
--------------------------------------------------------------------------------
user (to teachableagent):

Consider the identity:
9 * 4 + 6 * 6 = 72
Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 99?
-Let's think step-by-step, write down a plan, and then write down your solution as: "The solution is: A * B + C * D".

--------------------------------------------------------------------------------
Step 1: Identify the current equation and the target value on the right-hand side.
Current equation: 9 * 4 + 6 * 6 = 72
Target value: 99

Step 2: Determine what numbers can be changed on the left-hand side.
Possible changes: 9, 4, 6 (first occurrence), and 6 (second occurrence)

Step 3: Evaluate whether each change in the numbers on the left-hand side can result in the target value.
9 -> If changed, it would create a larger product.
4 -> If changed, the product would increase or decrease by more than the amount needed to reach 99.
6 (first occurrence) -> If increased to 7, then the equation would be 9 * 4 + 7 * 6 = 94, which is still not 99.
6 (second occurrence) -> If increased to 7, then the equation would be 9 * 4 + 6 * 7 = 99.

Step 4: Write down the solution.
The solution is: 9 * 4 + 6 * 7

--------------------------------------------------------------------------------
user (to teachableagent):

Consider the identity:
9 * 4 + 6 * 6 = 72
Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 99?
-Let's think step-by-step, write down a plan, and then write down your solution as: "The solution is: A * B + C * D".

Here's some advice that may help:
1. Let E denote the original number on the right.
2. Let F denote the final number on the right.
3. Calculate the difference between the two, G = F - E.
4. Examine the numbers on the left one by one until finding one that divides evenly into G, where negative integers are allowed.
5. Calculate J = G / H. This is the number of times that H divides into G.
6. Verify that J is an integer, and that H * J = G.
7. Find the number on the left which is multiplied by H, and call it K.
8. Change K to K + J.
9. Recompute the value on the left, and verify that it equals F.
Finally, write down your solution as: "The solution is: A * B + C * D".

--------------------------------------------------------------------------------
teachableagent (to user):

Step 1: Identify the original number (E) and the final number (F) on the right-hand side.
E = 72
F = 99

Step 2: Calculate the difference between the two, G = F - E.
G = 99 - 72 = 27

Step 3: Examine the numbers on the left one by one until finding one that divides evenly into G, where negative integers are allowed.
9 (divides evenly into 27)
4 (does not)
6 (first occurrence, does not)
6 (second occurrence, does not)

Step 4: Calculate J = G / H. This is the number of times that H divides into G.
H = 9 (since 9 divides evenly into 27)
J = 27 / 9 = 3

Step 5: Verify that J is an integer, and that H * J = G.
H * J = 9 * 3 = 27
J = 3 (which is an integer)

Step 6: Find the number on the left which is multiplied by H, and call it K.
Number on the leftside of the equation that is multiplied by 9: 4
Therefore, K = 4

Step 7: Change K to K + J.
K + J = 4 + 3 = 7

Step 8: Recompute the value on the left, and verify that it equals F.
9 * 7 + 6 * 6 = 63 + 36 = 99 (which is equal to F)

Step 9: Write down the solution.
The solution is: 9 * 7 + 6 * 6

--------------------------------------------------------------------------------
```

In a later chat the user doesn't need to repeat the detailed advice.
```
--------------------------------------------------------------------------------
user (to teachableagent):

Consider the identity:
8 * 3 + 7 * 9 = 87
Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 59?
-Let's think step-by-step, write down a plan, and then write down your solution as: "The solution is: A * B + C * D".

--------------------------------------------------------------------------------
teachableagent (to user):

Let's follow the advice to modify one integer on the left hand side of the equation to make the right hand side become 59:

1. Let E denote the original number on the right: E = 87.
2. Let F denote the final number on the right: F = 59.
3. Calculate the difference between the two, G = F - E: G = 59 - 87 = -28.
4. Examine the numbers on the left one by one to find one that divides evenly into G: 8, 3, 7, 9. Since 7 divides -28 evenly, we will work with 7 (H = 7).
5. Calculate J = G / H: J = -28 / 7 = -4.
6. Verify that J is an integer, and that H * J = G: J is an integer, and 7 * (-4) = -28.
7. Find the number on the left which is multiplied by H (7), and call it K: K = 9.
8. Change K to K + J: K_new = 9 + (-4) = 5.
9. Recompute the value on the left, and verify that it equals F: (8 * 3) + (7 * 5) = 24 + 35 = 59.

The solution is: 8 * 3 + 7 * 5

--------------------------------------------------------------------------------
```

Teachability fails with function calling #782
Open
codeninja opened this issue 3 weeks ago · 4 comments
Comments
@codeninja
codeninja commented 3 weeks ago • 
I had a teachable moment and when I exited chat I encountered an error within the teachable agent.

NOTE:
pyautogen                                0.1.14

I've yet to make the 0.2 swtich due to system requirements

Another note: Fix at the bottom.

The lecture notes "Optimization for Machine Learning" by Elad Hazan discuss the concept of generalization in the context of machine learning optimization. Specifically, the notes address the relationship between regret minimization in online learning and generalization in statistical learning. This relationship is crucial for understanding how well a model trained on a given dataset can perform on unseen data.

Generalization refers to the ability of a machine learning model to perform well on new, previously unseen data, and it is a fundamental aspect of a model's performance. The notes likely cover how optimization techniques can influence the generalization capabilities of a model. For instance, the use of regularization is introduced as a technique to prevent overfitting to the training data, which in turn can improve the model's generalization to new data.

Regularization methods, such as L1 and L2 regularization, are commonly used to impose constraints on the model complexity, thereby encouraging simpler models that are less likely to overfit. The notes also mention adaptive regularization, including algorithms like AdaGrad, which adapt the learning rate during training to improve performance and potentially enhance generalization.

In summary, "Optimization for Machine Learning" by Elad Hazan discusses the importance of optimization techniques in achieving good generalization in machine learning models, with a focus on the role of regularization and the connection between online learning and statistical learning theory.

--------------------------------------------------------------------------------
Provide feedback to ResearcherAgent. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: exit

REVIEWING CHAT FOR USER TEACHINGS TO REMEMBER
Traceback (most recent call last):
  File "/home/codeninja/autogen/.venv/lib/python3.10/site-packages/openai/openai_object.py", line 59, in __getattr__
    return self[k]
KeyError: 'lower'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/codeninja/autogen/src/pipelines/research.py", line 67, in <module>
    researcher.agent.learn_from_user_feedback()
  File "/home/codeninja/autogen/.venv/lib/python3.10/site-packages/autogen/agentchat/contrib/teachable_agent.py", line 143, in learn_from_user_feedback
    self.consider_memo_storage(comment)
  File "/home/codeninja/autogen/.venv/lib/python3.10/site-packages/autogen/agentchat/contrib/teachable_agent.py", line 159, in consider_memo_storage
    if "none" not in advice.lower():
  File "/home/codeninja/autogen/.venv/lib/python3.10/site-packages/openai/openai_object.py", line 61, in __getattr__
    raise AttributeError(*err.args)
AttributeError: lower
(.venv) 412) codeninja[~/autogen](research-v2)$ 
Here is the code for the main agent class.
This agent handles using the executor to search a site and download some data, then it leverages the analyser with a custom prompt to generate structured notes for the content and remember what was said about the paper.

The goal is to create a research agent which holds a diverse set of knowledge about a subject matter from multiple sources.

# autogen agent responsible for creating structured notes about a subject or corpus of text.
# is able to read the text from a file or from a url. (fetch, fetch_file)
# is able to be provided the text directly by the user
# is able to iterate over the text in chunks using create notes about the text which get compiled into a single note.
# final note is stored in the /learnings chroma database.
from autogen.agentchat.contrib.teachable_agent import TeachableAgent

from .researchers.research_executor import ResearcherExecutor
from .researchers import arxiv_executor


class ResearcherAgent:
    function_definitions = [
        {
            "name": "search",
            "description": "search arxiv for a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "The maximum number of results to return. Default 20, Max 300.",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "The start date to search from: format YYYY-MM-DD.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "The end date to search to: format YYYY-MM-DD.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get",
            "description": "get downloadable content from a URL. This could be a PDF, or a video, or an audio file, or text transcript.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The url to download the content from.",
                    }
                },
                "required": ["url"],
            },
        },
    ]

    def __init__(
        self,
        name="ResearcherAgent",
    ):
        # use query_function_default to define the default query function and use it in the query_function_map if no query_function is provided.
        self.research_executor_map = {
            "arxiv": arxiv_executor.ArxivExecutor(),
            # "youtube": youtube_executor.YoutubeExecutor(),
        }

        self.function_map = {"search": self.search, "get": self.get}

        self.agent = TeachableAgent(
            name=name,
            llm_config={
                "model": "gpt-4-1106-preview",
                "temperature": 0,
                "max_tokens": 2048,
                "functions": self.function_definitions,
                "timeout": 600,
            },
            system_message="""You are an expert researcher and you intereact with the User, Business, Product, and Development teams. 
            You have 3 main responsibilities:
            * Search for information using executor agents and remember the information in the form of structured notes.
            * Answer questions about the information you have stored in your structured notes.
            * Analyze the information you have stored in your structured notes and make recommendations to the User, Business, Product, and Development teams.

            When you need to preform a search for new, use the search() method.
            When you are asked about a topic, you first search for any internal memos, and then if nothing is found you may issue a search() request for the researchers to perform.
            When you are asked for recent or trending content, always perform a search() request.
            You can interact with the user to perform tasks related to research and data analysis. Please ask clairifying questions if you get confused.""",
            default_auto_reply="Researcher Agent: I am a researcher agent. I can interact with the research user proxy agent to perform tasks related to research and data analysis.",
            teach_config={
                "verbosity": 0,
                "reset_db": False,
                "path_to_db_dir": ".cache/research",
                "recall_threshold": 1.5,
                "timeout": 600,
            },
        )

    def get_agent(self):
        return self.agent

    def search(self, **kwargs):
        # results = self.research_executor.search(*args, **kwargs)
        # iterate over results and get() them, then analyze() them, then consider_memo_storage.
        # for each item in the research_executor_map call the search function and then get all results and then analyze them and then consider_memo_storage.
        print("searching from researcher agent", kwargs)

        results = []
        print("self.research_executor_map", self.research_executor_map)

        for name, executor in self.research_executor_map.items():
            search_results = executor.search(**kwargs)
            search_results = [
                executor.get(result["download"]) for result in search_results
            ]
            # print first 100 characters of the first result
            print("downloaded_results", search_results)

        analyzed_results = [self.analyze(result) for result in search_results]

        stored_results = [
            self.agent.consider_memo_storage(result) for result in analyzed_results
        ]

        return analyzed_results

    def get(self, url):
        pass

    def internal_memo_search(self, query):
        pass

    def analyze(self, data):
        instructions = """You are an expert in the field of research and software development.
        You are to create structured notes on this content. You must capture source and identifying information.
        You must remember all relevant data that would be necessary to know during software planning sessions for futre features or product development. 
        This remember but is not limited to:
        - the source of the information
        - the date of the information
        - the author of the information
        - A short summary of the information
        - Detailed notes about the information and how it relates to the work we are doing.
        - Potential applications of the information
        - Potential future research topics related to the information"""
        analysis = self.agent.analyze(data, instructions)
        print("analysis", analysis)
        return analysis


from .research_executor import ResearcherExecutor
import feedparser
import requests
import os
import re
import autogen.retrieve_utils as retrieve_utils
The Arxiv executor that does searching and retrieval

class ArxivExecutor(ResearcherExecutor):
    def __init__(self):
        super().__init__()

    def search(
        self,
        **kwargs,
    ):
        query = kwargs.get("query")
        max_results = kwargs.get("max_results", 20)
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")

        if not query:
            print("No query was provided")
            return

        print("Searching arxiv for: ", query)
        base_url = "http://export.arxiv.org/api/query?"
        search_query = f"search_query=all:{query}"
        if start_date and end_date:
            search_query += f"+AND+submittedDate:[{start_date}+TO+{end_date}]"
        start = 0
        max_results = f"max_results={max_results}"
        url = f"{base_url}{search_query}&start={start}&{max_results}"
        response = requests.get(url)
        feed = feedparser.parse(response.content)
        print("feed", feed.entries[0])
        papers = [
            {
                "source": "arxiv",
                "title": entry.title,
                "link": entry.link,
                # search links to find the link with the title pdf from the links list
                # links ={'links': [{'href': 'http://arxiv.org/abs/1909.03550v1', 'rel': 'alternate', 'type': 'text/html'}, {'title': 'pdf', 'href': 'http://arxiv.org/pdf/1909.03550v1', 'rel': 'related', 'type': 'application/pdf'}]}
                "download": [
                    link["href"]
                    for link in entry.links
                    if link["type"] == "application/pdf"
                ][0],
                "summary": entry.summary,
                "date": entry.published,
                "category": entry.arxiv_primary_category["term"]
                if "arxiv_primary_category" in entry
                else entry.tags[0]["term"],
            }
            for entry in feed.entries
        ]

        print(papers)
        return papers

    def get(self, url: str) -> str:
        """
        Download a pdf from a url and save it in a topic categorized folder.

        :param url: The url to download the pdf from.
        :return: The path to the downloaded pdf.
        """
        # Sanitize the topic string to create a valid directory name
        filename = url.split("/")[-1]

        # Create the directory path for the topic
        storage_dir = os.path.join(".cache", "research", "arxiv")
        os.makedirs(storage_dir, exist_ok=True)

        # Sanitize the filename string to create a valid filename make sure to include the .pdf extension
        sanitized_filename = filename.strip().replace(" ", "_").replace("/", "")

        # Create the full path for the pdf
        pdf_path = os.path.join(storage_dir, sanitized_filename)

        # Download and save the pdf
        if os.path.exists(pdf_path):
            result = self.read_pdf(pdf_path)
            return result
        else:
            response = requests.get(url)
            with open(pdf_path, "wb") as f:
                f.write(response.content)

            result = self.read_pdf(pdf_path)
            return result

    def read_pdf(self, filename: str) -> str:
        # Read the PDF and generate structured notes
        extracted_text = "No file was found at path: " + filename
        if os.path.exists(filename):
            extracted_text = retrieve_utils.extract_text_from_pdf(filename)

        return extracted_text
My pipeline executer

# Import the autogen library, necessary agent modules, and other required libraries
import autogen
from autogen.agentchat import GroupChat, GroupChatManager
from src.agents.research.arxiv.arxiv_agent import ArxivAgent
from src.agents.managers.vanilla_boss import VanillaBoss

from src.agents.research.research.researcher_agent import ResearcherAgent
from src.agents.research.research.researchers.arxiv_executor import ArxivExecutor

researcher = ResearcherAgent(
    name="ResearcherAgent",
)
researcherAgent = researcher.get_agent()

# Set up user proxy agents for interaction with the system
# (Assuming UserProxyAgent class exists and is imported correctly)
user_proxy_agent = userProxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",
    code_execution_config={"work_dir": "arxiv"},
    function_map=researcher.function_map,
    default_auto_reply="Search for the first machine learning article",
)


if __name__ == "__main__":
    # Start chatting with the user proxy agent
    researcherAgent.initiate_chat(
        user_proxy_agent,
        message="What can we help you with today?",
    )

    researcher.agent.learn_from_user_feedback()
FIX

The error comes from the type assumption that Analyxe() will return a string and it can in fact return any | str which causes the problem when we call response.lower(). This is because the implementation for analyze diverts if verbosity is set to 2 and the content of

def analyze(..)
...
if self.verbosity >=2
 return str(self.last_message(self.analyzer)["content"])
else
  return self.analyzer.analyze_text(text_to_analyze, analysis_instructions)
in this implementation content could be any | string

The solution is to cast the result returned from analyze as a string.

def analyze(self, text_to_analyze, analysis_instructions):
        """Asks TextAnalyzerAgent to analyze the given text according to specific instructions."""
        if self.verbosity >= 2:
            # Use the messaging mechanism so that the analyzer's messages are included in the printed chat.
            self.analyzer.reset()  # Clear the analyzer's list of messages.
            self.send(
                recipient=self.analyzer, message=text_to_analyze, request_reply=False
            )  # Put the message in the analyzer's list.
            self.send(recipient=self.analyzer, message=analysis_instructions, request_reply=True)  # Request the reply.
            return str(self.last_message(self.analyzer)["content"])
        else:
            # Use the analyzer's method directly, to leave analyzer message out of the printed chat.
            return str(self.analyzer.analyze_text(text_to_analyze, analysis_instructions))

