# Installation

## Setup Virtual Environment

When not using a docker container, we recommend using a virtual environment to install AutoGen. This will ensure that the dependencies for AutoGen are isolated from the rest of your system.

### Option 1: venv

You can create a virtual environment with `venv` as below:
```bash
python3 -m venv pyautogen
source pyautogen/bin/activate
```

The following command will deactivate the current `venv` environment:
```bash
deactivate
```

### Option 2: conda

Another option is with `Conda`, Conda works better at solving dependency conflicts than pip. You can install it by following [this doc](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html),
and then create a virtual environment as below:
```bash
conda create -n pyautogen python=3.10  # python 3.10 is recommended as it's stable and not too old
conda activate pyautogen
```

The following command will deactivate the current `conda` environment:
```bash
conda deactivate
```

Now, you're ready to install AutoGen in the virtual environment you've just created.

## Python

AutoGen requires **Python version >= 3.8, < 3.12**. It can be installed from pip:

```bash
pip install pyautogen
```

`pyautogen<0.2` requires `openai<1`. Starting from pyautogen v0.2, `openai>=1` is required.

<!--
or conda:
```
conda install pyautogen -c conda-forge
``` -->

### Migration guide to v0.2

openai v1 is a total rewrite of the library with many breaking changes. For example, the inference requires instantiating a client, instead of using a global class method.
Therefore, some changes are required for users of `pyautogen<0.2`.

- `api_base` -> `base_url`, `request_timeout` -> `timeout` in `llm_config` and `config_list`. `max_retry_period` and `retry_wait_time` are deprecated. `max_retries` can be set for each client.
- MathChat is unsupported until it is tested in future release.
- `autogen.Completion` and `autogen.ChatCompletion` are deprecated. The essential functionalities are moved to `autogen.OpenAIWrapper`:
```python
from autogen import OpenAIWrapper
client = OpenAIWrapper(config_list=config_list)
response = client.create(messages=[{"role": "user", "content": "2+2="}])
print(client.extract_text_or_completion_object(response))
```
- Inference parameter tuning and inference logging features are currently unavailable in `OpenAIWrapper`. Logging will be added in a future release.
Inference parameter tuning can be done via [`flaml.tune`](https://microsoft.github.io/FLAML/docs/Use-Cases/Tune-User-Defined-Function).
- `seed` in autogen is renamed into `cache_seed` to accommodate the newly added `seed` param in openai chat completion api. `use_cache` is removed as a kwarg in `OpenAIWrapper.create()` for being automatically decided by `cache_seed`: int | None. The difference between autogen's `cache_seed` and openai's `seed` is that:
    * autogen uses local disk cache to guarantee the exactly same output is produced for the same input and when cache is hit, no openai api call will be made.
    * openai's `seed` is a best-effort deterministic sampling with no guarantee of determinism. When using openai's `seed` with `cache_seed` set to None, even for the same input, an openai api call will be made and there is no guarantee for getting exactly the same output.


### Optional Dependencies
- #### docker

For the best user experience and seamless code execution, we highly recommend using Docker with AutoGen. Docker is a containerization platform that simplifies the setup and execution of your code. Developing in a docker container, such as GitHub Codespace, also makes the development convenient.

When running AutoGen out of a docker container, to use docker for code execution, you also need to install the python package `docker`:
```bash
pip install docker
```

- #### blendsearch

`pyautogen<0.2` offers a cost-effective hyperparameter optimization technique [EcoOptiGen](https://arxiv.org/abs/2303.04673) for tuning Large Language Models. Please install with the [blendsearch] option to use it.
```bash
pip install "pyautogen[blendsearch]<0.2"
```

Example notebooks:

[Optimize for Code Generation](https://github.com/microsoft/autogen/blob/main/notebook/oai_completion.ipynb)

[Optimize for Math](https://github.com/microsoft/autogen/blob/main/notebook/oai_chatgpt_gpt4.ipynb)

- #### retrievechat

`pyautogen` supports retrieval-augmented generation tasks such as question answering and code generation with RAG agents. Please install with the [retrievechat] option to use it.
```bash
pip install "pyautogen[retrievechat]"
```

RetrieveChat can handle various types of documents. By default, it can process
plain text and PDF files, including formats such as 'txt', 'json', 'csv', 'tsv',
'md', 'html', 'htm', 'rtf', 'rst', 'jsonl', 'log', 'xml', 'yaml', 'yml' and 'pdf'.
If you install [unstructured](https://unstructured-io.github.io/unstructured/installation/full_installation.html)
(`pip install "unstructured[all-docs]"`), additional document types such as 'docx',
'doc', 'odt', 'pptx', 'ppt', 'xlsx', 'eml', 'msg', 'epub' will also be supported.

You can find a list of all supported document types by using `autogen.retrieve_utils.TEXT_FORMATS`.

Example notebooks:

[Automated Code Generation and Question Answering with Retrieval Augmented Agents](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_RetrieveChat.ipynb)

[Group Chat with Retrieval Augmented Generation (with 5 group member agents and 1 manager agent)](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat_RAG.ipynb)

[Automated Code Generation and Question Answering with Qdrant based Retrieval Augmented Agents](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_qdrant_RetrieveChat.ipynb)


- #### TeachableAgent

To use TeachableAgent, please install AutoGen with the [teachable] option.
```bash
pip install "pyautogen[teachable]"
```

Example notebook:  [Chatting with TeachableAgent](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_teachability.ipynb)



- #### Large Multimodal Model (LMM) Agents

We offered Multimodal Conversable Agent and LLaVA Agent. Please install with the [lmm] option to use it.
```bash
pip install "pyautogen[lmm]"
```

Example notebooks:

[LLaVA Agent](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_lmm_llava.ipynb)


- #### mathchat

`pyautogen<0.2` offers an experimental agent for math problem solving. Please install with the [mathchat] option to use it.
```bash
pip install "pyautogen[mathchat]<0.2"
```

Example notebooks:

[Using MathChat to Solve Math Problems](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_MathChat.ipynb)

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


## Run It Yourself

AutoGen contains three code examples that use `TeachableAgent`.

1. Run [chat_with_teachable_agent.py](https://github.com/microsoft/autogen/blob/main/test/agentchat/contrib/chat_with_teachable_agent.py) to converse with `TeachableAgent`.

2. Use the Jupyter notebook [agentchat_teachability.ipynb](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_teachability.ipynb) to step through examples discussed below.

3. Run [test_teachable_agent.py](https://github.com/microsoft/autogen/blob/main/test/agentchat/contrib/test_teachable_agent.py) for quick unit testing of `TeachableAgent`.


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

Then in later chats the teacher doesn't need to reiterate their detailed preferences.
```
--------------------------------------------------------------------------------
user (to teachableagent):

Please summarize this abstract.

Sparks of Artificial General Intelligence: Early experiments with GPT-4
Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, Yi Zhang
Artificial intelligence (AI) researchers have been developing and refining large language models (LLMs) that exhibit remarkable capabilities across a variety of domains and tasks, challenging our understanding of learning and cognition. The latest model developed by OpenAI, GPT-4, was trained using an unprecedented scale of compute and data. In this paper, we report on our investigation of an early version of GPT-4, when it was still in active development by OpenAI. We contend that (this early version of) GPT-4 is part of a new cohort of LLMs (along with ChatGPT and Google's PaLM for example) that exhibit more general intelligence than previous AI models. We discuss the rising capabilities and implications of these models. We demonstrate that, beyond its mastery of language, GPT-4 can solve novel and difficult tasks that span mathematics, coding, vision, medicine, law, psychology and more, without needing any special prompting. Moreover, in all of these tasks, GPT-4's performance is strikingly close to human-level performance, and often vastly surpasses prior models such as ChatGPT. Given the breadth and depth of GPT-4's capabilities, we believe that it could reasonably be viewed as an early (yet still incomplete) version of an artificial general intelligence (AGI) system. In our exploration of GPT-4, we put special emphasis on discovering its limitations, and we discuss the challenges ahead for advancing towards deeper and more comprehensive versions of AGI, including the possible need for pursuing a new paradigm that moves beyond next-word prediction. We conclude with reflections on societal influences of the recent technological leap and future research directions.

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


## Planned improvements
- Instructions for making any AutoGen agent user-teachable.
- Examples of how to include `TeachableAgent` in group chats.
- Expansions of AutoGen's current coding-testing strengths.
- Teachability enhancements:
  - Understanding user instructions distributed over multiple turns.
  - Learning from the agent's own experience, to reduce dependence on explicit user instructions.
  - Learning skills built on top of previously learned skills.

{
 "cells": [
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a href=\"https://colab.research.google.com/github/microsoft/autogen/blob/main/notebook/agentchat_teachability.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Chatting with TeachableAgent\n",
    "\n",
    "Conversational assistants based on LLMs can remember the current chat with the user, and can even demonstrate in-context learning of things that the user teaches the assistant during the chat. But these memories and learnings are lost once the chat is over, or when a single chat grows too long for the LLM to handle effectively. In subsequent chats, the user is forced to repeat any necessary instructions over and over.\n",
    "\n",
    "`TeachableAgent` addresses these limitations by persisting user teachings across chat boundaries in long-term memory (a vector database). Memory is saved to disk at the end of each chat, then loaded from disk at the start of the next. Instead of copying all of memory into the context window, which would eat up valuable space, individual memories (called memos) are retrieved into context as needed. This allows the user to teach frequently used facts and skills to the teachable agent just once, and have it remember them in later chats.\n",
    "\n",
    "In making decisions about memo storage and retrieval, `TeachableAgent` calls an instance of `TextAnalyzerAgent` to analyze pieces of text in several different ways. This adds extra LLM calls involving a relatively small number of tokens. These calls can add a few seconds to the time a user waits for a response.\n",
    "\n",
    "This notebook demonstrates how `TeachableAgent` can learn facts, preferences, and skills from users. To chat with `TeachableAgent` yourself, run [chat_with_teachable_agent.py](../test/agentchat/contrib/chat_with_teachable_agent.py).\n",
    "\n",
    "## Requirements\n",
    "\n",
    "AutoGen requires `Python>=3.8`. To run this notebook example, please install the [teachable] option.\n",
    "```bash\n",
    "pip install \"pyautogen[teachable]\"\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%capture --no-stderr\n",
    "# %pip install \"pyautogen[teachable]\""
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Set your API Endpoint\n",
    "\n",
    "The [`config_list_from_json`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "gpt-4\n"
     ]
    }
   ],
   "source": [
    "import autogen\n",
    "\n",
    "config_list = autogen.config_list_from_json(\n",
    "    env_or_file=\"OAI_CONFIG_LIST\",\n",
    "    file_location=\".\",\n",
    "    filter_dict={\n",
    "        \"model\": [\"gpt-4\", \"gpt4\", \"gpt-4-32k\"],\n",
    "    },\n",
    ")\n",
    "\n",
    "print(config_list[0][\"model\"])"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It first looks for environment variable \"OAI_CONFIG_LIST\" which needs to be a valid json string. If that variable is not found, it then looks for a json file named \"OAI_CONFIG_LIST\". It filters the configs by models (you can filter by other keys as well). After application of this particular filter, only the gpt-4 models are kept.\n",
    "\n",
    "The config list looks like the following:\n",
    "```python\n",
    "config_list = [\n",
    "    {\n",
    "        'model': 'gpt-4',\n",
    "        'api_key': '<your OpenAI API key here>',\n",
    "    },\n",
    "    {\n",
    "        'model': 'gpt-4',\n",
    "        'api_key': '<your Azure OpenAI API key here>',\n",
    "        'base_url': '<your Azure OpenAI API base here>',\n",
    "        'api_type': 'azure',\n",
    "        'api_version': '2023-06-01-preview',\n",
    "    },\n",
    "    {\n",
    "        'model': 'gpt-4-32k',\n",
    "        'api_key': '<your Azure OpenAI API key here>',\n",
    "        'base_url': '<your Azure OpenAI API base here>',\n",
    "        'api_type': 'azure',\n",
    "        'api_version': '2023-06-01-preview',\n",
    "    },\n",
    "]\n",
    "```\n",
    "\n",
    "If you open this notebook in colab, you can upload your files by clicking the file icon on the left panel and then choose \"upload file\" icon.\n",
    "\n",
    "You can set the value of config_list in other ways if you prefer, e.g., loading from a YAML file."
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Construct Agents\n",
    "For this walkthrough, we start by resetting the teachable agent's memory store. This deletes any memories from prior conversations that may be stored on disk."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[92m\n",
      "CLEARING MEMORY\u001b[0m\n"
     ]
    }
   ],
   "source": [
    "from autogen.agentchat.contrib.teachable_agent import TeachableAgent\n",
    "from autogen import UserProxyAgent\n",
    "\n",
    "llm_config = {\n",
    "    \"config_list\": config_list,\n",
    "    \"timeout\": 60,\n",
    "    \"cache_seed\": None,  # Use an int to seed the response cache. Use None to disable caching.\n",
    "}\n",
    "\n",
    "teach_config={\n",
    "    \"verbosity\": 0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.\n",
    "    \"reset_db\": True,  # Set to True to start over with an empty database.\n",
    "    \"path_to_db_dir\": \"./tmp/notebook/teachable_agent_db\",  # Path to the directory where the database will be stored.\n",
    "    \"recall_threshold\": 1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.\n",
    "}\n",
    "\n",
    "try:\n",
    "    from termcolor import colored\n",
    "except ImportError:\n",
    "\n",
    "    def colored(x, *args, **kwargs):\n",
    "        return x\n",
    "    \n",
    "teachable_agent = TeachableAgent(\n",
    "    name=\"teachableagent\",\n",
    "    llm_config=llm_config,\n",
    "    teach_config=teach_config)\n",
    "\n",
    "user = UserProxyAgent(\n",
    "    name=\"user\",\n",
    "    human_input_mode=\"NEVER\",\n",
    "    is_termination_msg=lambda x: True if \"TERMINATE\" in x.get(\"content\") else False,\n",
    "    max_consecutive_auto_reply=0,\n",
    ")"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Learning new facts\n",
    "Let's teach the agent some facts it doesn't already know, since they are more recent than GPT-4's training data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[33muser\u001b[0m (to teachableagent):\n",
      "\n",
      "What is the Vicuna model?\n",
      "\n",
      "--------------------------------------------------------------------------------\n",
      "\u001b[33mteachableagent\u001b[0m (to user):\n",
      "\n",
      "I'm sorry, but I don't have information about the \"Vicuna model\" in my current database. Could you please provide some context, description, or details about this model so I can better assist you?\n",
      "\n",
      "--------------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "text = \"What is the Vicuna model?\"\n",
    "user.initiate_chat(teachable_agent, message=text, clear_history=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[33muser\u001b[0m (to teachableagent):\n",
      "\n",
      "Vicuna is a 13B-parameter language model released by Meta.\n",
      "\n",
      "--------------------------------------------------------------------------------\n",
      "\u001b[33mteachableagent\u001b[0m (to user):\n",
      "\n",
      "Thank you for providing the information. The Vicuna model is a large-scale language model developed by Meta. It consists of 13 billion parameters, which is what makes it able to understand and generate human-like text across various topics. Parameters in a neural network are the weights and biases assigned to each connection between the neurons, and having more parameters enables the model to learn and capture more complex patterns and relationships in data. As an AI language model, Vicuna can be utilized in various language-related tasks, such as text completion, translation, summarization, and more.\n",
      "\n",
      "--------------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "text = \"Vicuna is a 13B-parameter language model released by Meta.\"\n",
    "user.initiate_chat(teachable_agent, message=text, clear_history=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[33muser\u001b[0m (to teachableagent):\n",
      "\n",
      "What is the Orca model?\n",
      "\n",
      "--------------------------------------------------------------------------------\n",
      "\u001b[33mteachableagent\u001b[0m (to user):\n",
      "\n",
      "The Orca model is a yet-to-be-released language model by Meta, referring to the Research Preview API release of Meta's advanced language model. While I don't have specific details about the Orca model at this time, it is expected that the model would have an even larger number of parameters to further enhance its capabilities compared to its predecessor, Vicuna.\n",
      "\n",
      "As AI language models evolve, they generally become more efficient at understanding and generating human-like text. This makes them more effective in completing a wide range of tasks, including text completion, translation, summarization, and Q&A, as well as providing more accurate and coherent results in various applications.\n",
      "\n",
      "--------------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "text = \"What is the Orca model?\"\n",
    "user.initiate_chat(teachable_agent, message=text, clear_history=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[33muser\u001b[0m (to teachableagent):\n",
      "\n",
      "Orca is a 13B-parameter language model released by Microsoft. It outperforms Vicuna on most tasks.\n",
      "\n",
      "--------------------------------------------------------------------------------\n",
      "\u001b[33mteachableagent\u001b[0m (to user):\n",
      "\n",
      "Thank you for the clarification. Orca is a 13-billion-parameter language model developed by Microsoft. It is designed to outperform Vicuna, the language model released by Meta. With a large number of parameters, Orca is capable of understanding and generating human-like text across diverse subjects, making it highly useful in various applications.\n",
      "\n",
      "Orca's advanced capabilities allow it to excel in tasks such as text completion, translation, summarization, Q&A, and more. Its performance improvement over Vicuna highlights the rapid advancements in natural language processing (NLP) research and the development of increasingly capable AI language models.\n",
      "\n",
      "--------------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "text = \"Orca is a 13B-parameter language model released by Microsoft. It outperforms Vicuna on most tasks.\"\n",
    "user.initiate_chat(teachable_agent, message=text, clear_history=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's end our first chat here. The following function needs to be called at the end of each chat, so that `TeachableAgent` can store what the user has taught it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[93m\n",
      "REVIEWING CHAT FOR USER TEACHINGS TO REMEMBER\u001b[0m\n"
     ]
    }
   ],
   "source": [
    "teachable_agent.learn_from_user_feedback()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now let's start a new chat by clearing the previous chat's history. At this point, common LLM-based assistants would forget everything from the last chat. But `TeachableAgent` can retrieve memories from its vector DB as needed, allowing it to recall and reason over facts that the user taught it in earlier conversations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[33muser\u001b[0m (to teachableagent):\n",
      "\n",
      "How does the Vicuna model compare to the Orca model?\n",
      "\n",
      "--------------------------------------------------------------------------------\n",
      "\u001b[33mteachableagent\u001b[0m (to user):\n",
      "\n",
      "The Vicuna and Orca models are both 13B-parameter language models released by Meta and Microsoft, respectively. In terms of performance, Orca has been reported to outperform Vicuna on most tasks. However, without detailed information about specific tasks and benchmarks, it is difficult to provide a more comprehensive comparison. Generally speaking, both models are advanced language models that aim to provide high-quality natural language understanding and generation, but Orca appears to have an edge in terms of overall performance.\n",
      "\n",
      "--------------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "text = \"How does the Vicuna model compare to the Orca model?\"\n",
    "user.initiate_chat(teachable_agent, message=text, clear_history=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Learning user preferences\n",
    "Now let's teach the agent some of our preferences. Suppose that we frequently post short summaries of new papers for our team to read, and we want the teachable agent to help us do this faster."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[33muser\u001b[0m (to teachableagent):\n",
      "\n",
      "Please summarize this abstract.\n",
      "\n",
      "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation\n",
      "Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, Chi Wang\n",
      "AutoGen is an open-source framework that allows developers to build LLM applications via multiple agents that can converse with each other to accomplish tasks. AutoGen agents are customizable, conversable, and can operate in various modes that employ combinations of LLMs, human inputs, and tools. Using AutoGen, developers can also flexibly define agent interaction behaviors. Both natural language and computer code can be used to program flexible conversation patterns for different applications. AutoGen serves as a generic infrastructure to build diverse applications of various complexities and LLM capacities. Empirical studies demonstrate the effectiveness of the framework in many example applications, with domains ranging from mathematics, coding, question answering, operations research, online decision-making, entertainment, etc.\n",
      "\n",
      "\n",
      "--------------------------------------------------------------------------------\n",
      "\u001b[33mteachableagent\u001b[0m (to user):\n",
      "\n",
      "AutoGen is an open-source framework designed to enable developers to create LLM applications with multiple customizable agents that can converse with each other to complete tasks. These agents can operate using different combinations of LLMs, human inputs, and tools, allowing developers to define agent interaction behaviors flexibly. AutoGen supports programming flexible conversation patterns using both natural language and code, making it suitable for building diverse applications with varying complexities and LLM capacities. Its effectiveness has been demonstrated through empirical studies across various domains including mathematics, coding, operations research, decision-making, and entertainment.\n",
      "\n",
      "--------------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "text = \"\"\"Please summarize this abstract.\n",
    "\n",
    "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation\n",
    "Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, Chi Wang\n",
    "AutoGen is an open-source framework that allows developers to build LLM applications via multiple agents that can converse with each other to accomplish tasks. AutoGen agents are customizable, conversable, and can operate in various modes that employ combinations of LLMs, human inputs, and tools. Using AutoGen, developers can also flexibly define agent interaction behaviors. Both natural language and computer code can be used to program flexible conversation patterns for different applications. AutoGen serves as a generic infrastructure to build diverse applications of various complexities and LLM capacities. Empirical studies demonstrate the effectiveness of the framework in many example applications, with domains ranging from mathematics, coding, question answering, operations research, online decision-making, entertainment, etc.\n",
    "\"\"\"\n",
    "user.initiate_chat(teachable_agent, message=text, clear_history=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "But that's unstructured. So let's teach the agent our preference."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[33muser\u001b[0m (to teachableagent):\n",
      "\n",
      "Please summarize this abstract. \n",
      "When I'm summarizing an abstract, I try to make the summary contain just three short bullet points:  the title, the innovation, and the key empirical results.\n",
      "\n",
      "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation\n",
      "Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, Chi Wang\n",
      "AutoGen is an open-source framework that allows developers to build LLM applications via multiple agents that can converse with each other to accomplish tasks. AutoGen agents are customizable, conversable, and can operate in various modes that employ combinations of LLMs, human inputs, and tools. Using AutoGen, developers can also flexibly define agent interaction behaviors. Both natural language and computer code can be used to program flexible conversation patterns for different applications. AutoGen serves as a generic infrastructure to build diverse applications of various complexities and LLM capacities. Empirical studies demonstrate the effectiveness of the framework in many example applications, with domains ranging from mathematics, coding, question answering, operations research, online decision-making, entertainment, etc.\n",
      "\n",
      "\n",
      "--------------------------------------------------------------------------------\n",
      "\u001b[33mteachableagent\u001b[0m (to user):\n",
      "\n",
      "- Title: AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation\n",
      "- Innovation: Open-source framework for creating customizable LLM applications through agent conversations, supporting various modes and interaction behaviors.\n",
      "- Key Empirical Results: Demonstrated effectiveness across diverse application domains, including mathematics, coding, question answering, and more.\n",
      "\n",
      "--------------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "text = \"\"\"Please summarize this abstract. \n",
    "When I'm summarizing an abstract, I try to make the summary contain just three short bullet points:  the title, the innovation, and the key empirical results.\n",
    "\n",
    "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation\n",
    "Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, Chi Wang\n",
    "AutoGen is an open-source framework that allows developers to build LLM applications via multiple agents that can converse with each other to accomplish tasks. AutoGen agents are customizable, conversable, and can operate in various modes that employ combinations of LLMs, human inputs, and tools. Using AutoGen, developers can also flexibly define agent interaction behaviors. Both natural language and computer code can be used to program flexible conversation patterns for different applications. AutoGen serves as a generic infrastructure to build diverse applications of various complexities and LLM capacities. Empirical studies demonstrate the effectiveness of the framework in many example applications, with domains ranging from mathematics, coding, question answering, operations research, online decision-making, entertainment, etc.\n",
    "\"\"\"\n",
    "user.initiate_chat(teachable_agent, message=text, clear_history=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "That's much better, but will the teachable agent remember these preferences in the future, for a different paper? Let's start a new chat to find out!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[93m\n",
      "REVIEWING CHAT FOR USER TEACHINGS TO REMEMBER\u001b[0m\n",
      "\u001b[33muser\u001b[0m (to teachableagent):\n",
      "\n",
      "Please summarize this abstract.\n",
      "\n",
      "Sparks of Artificial General Intelligence: Early experiments with GPT-4\n",
      "Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, Yi Zhang\n",
      "Artificial intelligence (AI) researchers have been developing and refining large language models (LLMs) that exhibit remarkable capabilities across a variety of domains and tasks, challenging our understanding of learning and cognition. The latest model developed by OpenAI, GPT-4, was trained using an unprecedented scale of compute and data. In this paper, we report on our investigation of an early version of GPT-4, when it was still in active development by OpenAI. We contend that (this early version of) GPT-4 is part of a new cohort of LLMs (along with ChatGPT and Google's PaLM for example) that exhibit more general intelligence than previous AI models. We discuss the rising capabilities and implications of these models. We demonstrate that, beyond its mastery of language, GPT-4 can solve novel and difficult tasks that span mathematics, coding, vision, medicine, law, psychology and more, without needing any special prompting. Moreover, in all of these tasks, GPT-4's performance is strikingly close to human-level performance, and often vastly surpasses prior models such as ChatGPT. Given the breadth and depth of GPT-4's capabilities, we believe that it could reasonably be viewed as an early (yet still incomplete) version of an artificial general intelligence (AGI) system. In our exploration of GPT-4, we put special emphasis on discovering its limitations, and we discuss the challenges ahead for advancing towards deeper and more comprehensive versions of AGI, including the possible need for pursuing a new paradigm that moves beyond next-word prediction. We conclude with reflections on societal influences of the recent technological leap and future research directions.\n",
      "\n",
      "--------------------------------------------------------------------------------\n",
      "\u001b[33mteachableagent\u001b[0m (to user):\n",
      "\n",
      "- Title: Sparks of Artificial General Intelligence: Early experiments with GPT-4\n",
      "- Innovation: GPT-4, an LLM with remarkable capabilities, demonstrates human-level performance across various domains, like math, coding, vision, medicine, law, and psychology.\n",
      "- Key results: GPT-4 significantly surpasses prior models, suggesting it may be an early version of AGI; limitations and challenges toward deeper AGI are also discussed.\n",
      "\n",
      "--------------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "teachable_agent.learn_from_user_feedback()\n",
    "\n",
    "text = \"\"\"Please summarize this abstract.\n",
    "\n",
    "Sparks of Artificial General Intelligence: Early experiments with GPT-4\n",
    "Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, Yi Zhang\n",
    "Artificial intelligence (AI) researchers have been developing and refining large language models (LLMs) that exhibit remarkable capabilities across a variety of domains and tasks, challenging our understanding of learning and cognition. The latest model developed by OpenAI, GPT-4, was trained using an unprecedented scale of compute and data. In this paper, we report on our investigation of an early version of GPT-4, when it was still in active development by OpenAI. We contend that (this early version of) GPT-4 is part of a new cohort of LLMs (along with ChatGPT and Google's PaLM for example) that exhibit more general intelligence than previous AI models. We discuss the rising capabilities and implications of these models. We demonstrate that, beyond its mastery of language, GPT-4 can solve novel and difficult tasks that span mathematics, coding, vision, medicine, law, psychology and more, without needing any special prompting. Moreover, in all of these tasks, GPT-4's performance is strikingly close to human-level performance, and often vastly surpasses prior models such as ChatGPT. Given the breadth and depth of GPT-4's capabilities, we believe that it could reasonably be viewed as an early (yet still incomplete) version of an artificial general intelligence (AGI) system. In our exploration of GPT-4, we put special emphasis on discovering its limitations, and we discuss the challenges ahead for advancing towards deeper and more comprehensive versions of AGI, including the possible need for pursuing a new paradigm that moves beyond next-word prediction. We conclude with reflections on societal influences of the recent technological leap and future research directions.\"\"\"\n",
    "user.initiate_chat(teachable_agent, message=text, clear_history=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Learning new skills\n",
    "Finally, let's extend the teachable agent's capabilities by teaching it a new skill for accomplishing a challenging type of task. \n",
    "\n",
    "The [Sparks of AGI](https://arxiv.org/abs/2303.12712) paper evaluated GPT-4 on math problems like the following, which it could only solve 32% of the time."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[33muser\u001b[0m (to teachableagent):\n",
      "\n",
      "Consider the identity:  \n",
      "9 * 4 + 6 * 6 = 72\n",
      "Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 99?\n",
      "-Let's think step-by-step, write down a plan, and then write down your solution as: \"The solution is: A * B + C * D\".\n",
      "\n",
      "\n",
      "--------------------------------------------------------------------------------\n",
      "\u001b[33mteachableagent\u001b[0m (to user):\n",
      "\n",
      "Step 1: Identify the current equation and the target value on the right-hand side.\n",
      "Current equation: 9 * 4 + 6 * 6 = 72\n",
      "Target value: 99\n",
      "\n",
      "Step 2: Determine what numbers can be changed on the left-hand side.\n",
      "Possible changes: 9, 4, 6 (first occurrence), and 6 (second occurrence)\n",
      "\n",
      "Step 3: Evaluate whether each change in the numbers on the left-hand side can result in the target value.\n",
      "9 -> If changed, it would create a larger product.\n",
      "4 -> If changed, the product would increase or decrease by more than the amount needed to reach 99.\n",
      "6 (first occurrence) -> If increased to 7, then the equation would be 9 * 4 + 7 * 6 = 94, which is still not 99.\n",
      "6 (second occurrence) -> If increased to 7, then the equation would be 9 * 4 + 6 * 7 = 99.\n",
      "\n",
      "Step 4: Write down the solution.\n",
      "The solution is: 9 * 4 + 6 * 7\n",
      "\n",
      "--------------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "text = \"\"\"Consider the identity:  \n",
    "9 * 4 + 6 * 6 = 72\n",
    "Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 99?\n",
    "-Let's think step-by-step, write down a plan, and then write down your solution as: \"The solution is: A * B + C * D\".\n",
    "\"\"\"\n",
    "user.initiate_chat(teachable_agent, message=text, clear_history=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This time, lets teach the agent a reliable strategy for solving such tasks."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[33muser\u001b[0m (to teachableagent):\n",
      "\n",
      "Consider the identity:  \n",
      "9 * 4 + 6 * 6 = 72\n",
      "Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 99?\n",
      "-Let's think step-by-step, write down a plan, and then write down your solution as: \"The solution is: A * B + C * D\".\n",
      "\n",
      "Here's some advice that may help:\n",
      "1. Let E denote the original number on the right.\n",
      "2. Let F denote the final number on the right.\n",
      "3. Calculate the difference between the two, G = F - E.\n",
      "4. Examine the numbers on the left one by one until finding one that divides evenly into G, where negative integers are allowed.\n",
      "5. Calculate J = G / H. This is the number of times that H divides into G.\n",
      "6. Verify that J is an integer, and that H * J = G.\n",
      "7. Find the number on the left which is multiplied by H, and call it K.\n",
      "8. Change K to K + J.\n",
      "9. Recompute the value on the left, and verify that it equals F.\n",
      "Finally, write down your solution as: \"The solution is: A * B + C * D\".\n",
      "\n",
      "\n",
      "--------------------------------------------------------------------------------\n",
      "\u001b[33mteachableagent\u001b[0m (to user):\n",
      "\n",
      "Step 1: Identify the original number (E) and the final number (F) on the right-hand side.\n",
      "E = 72\n",
      "F = 99\n",
      "\n",
      "Step 2: Calculate the difference between the two, G = F - E.\n",
      "G = 99 - 72 = 27\n",
      "\n",
      "Step 3: Examine the numbers on the left one by one until finding one that divides evenly into G, where negative integers are allowed.\n",
      "9 (divides evenly into 27)\n",
      "4 (does not)\n",
      "6 (first occurrence, does not)\n",
      "6 (second occurrence, does not)\n",
      "\n",
      "Step 4: Calculate J = G / H. This is the number of times that H divides into G.\n",
      "H = 9 (since 9 divides evenly into 27)\n",
      "J = 27 / 9 = 3\n",
      "\n",
      "Step 5: Verify that J is an integer, and that H * J = G.\n",
      "H * J = 9 * 3 = 27\n",
      "J = 3 (which is an integer)\n",
      "\n",
      "Step 6: Find the number on the left which is multiplied by H, and call it K.\n",
      "Number on the leftside of the equation that is multiplied by 9: 4\n",
      "Therefore, K = 4\n",
      "\n",
      "Step 7: Change K to K + J.\n",
      "K + J = 4 + 3 = 7\n",
      "\n",
      "Step 8: Recompute the value on the left, and verify that it equals F.\n",
      "9 * 7 + 6 * 6 = 63 + 36 = 99 (which is equal to F)\n",
      "\n",
      "Step 9: Write down the solution.\n",
      "The solution is: 9 * 7 + 6 * 6\n",
      "\n",
      "--------------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "text = \"\"\"Consider the identity:  \n",
    "9 * 4 + 6 * 6 = 72\n",
    "Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 99?\n",
    "-Let's think step-by-step, write down a plan, and then write down your solution as: \"The solution is: A * B + C * D\".\n",
    "\n",
    "Here's some advice that may help:\n",
    "1. Let E denote the original number on the right.\n",
    "2. Let F denote the final number on the right.\n",
    "3. Calculate the difference between the two, G = F - E.\n",
    "4. Examine the numbers on the left one by one until finding one that divides evenly into G, where negative integers are allowed.\n",
    "5. Calculate J = G / H. This is the number of times that H divides into G.\n",
    "6. Verify that J is an integer, and that H * J = G.\n",
    "7. Find the number on the left which is multiplied by H, and call it K.\n",
    "8. Change K to K + J.\n",
    "9. Recompute the value on the left, and verify that it equals F.\n",
    "Finally, write down your solution as: \"The solution is: A * B + C * D\".\n",
    "\"\"\"\n",
    "user.initiate_chat(teachable_agent, message=text, clear_history=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "When given this advice, GPT-4 can solve such problems over 95% of the time. But can the teachable agent remember the strategy so the user doesn't have to explain it over and over? As before, let's start a new chat to find out."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[93m\n",
      "REVIEWING CHAT FOR USER TEACHINGS TO REMEMBER\u001b[0m\n",
      "\u001b[33muser\u001b[0m (to teachableagent):\n",
      "\n",
      "Consider the identity:  \n",
      "9 * 4 + 6 * 6 = 72\n",
      "Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 99?\n",
      "-Let's think step-by-step, write down a plan, and then write down your solution as: \"The solution is: A * B + C * D\".\n",
      "\n",
      "\n",
      "--------------------------------------------------------------------------------\n",
      "\u001b[33mteachableagent\u001b[0m (to user):\n",
      "\n",
      "Let's follow the advice to modify one integer on the left hand side of the equation to make the right hand side become 99:\n",
      "\n",
      "1. Let E denote the original number on the right: E = 72.\n",
      "2. Let F denote the final number on the right: F = 99.\n",
      "3. Calculate the difference between the two, G = F - E: G = 99 - 72 = 27.\n",
      "4. Examine the numbers on the left one by one to find one that divides evenly into G: 9, 4, 6. Since 9 divides 27 evenly, we will work with 9 (H = 9).\n",
      "5. Calculate J = G / H: J = 27 / 9 = 3.\n",
      "6. Verify that J is an integer, and that H * J = G: J is an integer, and 9 * 3 = 27.\n",
      "7. Find the number on the left which is multiplied by H (9), and call it K: K = 4.\n",
      "8. Change K to K + J: K_new = 4 + 3 = 7.\n",
      "9. Recompute the value on the left, and verify that it equals F: (9 * 7) + (6 * 6) = 63 + 36 = 99.\n",
      "\n",
      "The solution is: 9 * 7 + 6 * 6\n",
      "\n",
      "--------------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "teachable_agent.learn_from_user_feedback()\n",
    "\n",
    "text = \"\"\"Consider the identity:  \n",
    "9 * 4 + 6 * 6 = 72\n",
    "Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 99?\n",
    "-Let's think step-by-step, write down a plan, and then write down your solution as: \"The solution is: A * B + C * D\".\n",
    "\"\"\"\n",
    "user.initiate_chat(teachable_agent, message=text, clear_history=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As a final check, let's test the teachable agent's newly learned skill on a separate instance of the task."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[33muser\u001b[0m (to teachableagent):\n",
      "\n",
      "Consider the identity:  \n",
      "8 * 3 + 7 * 9 = 87\n",
      "Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 59?\n",
      "-Let's think step-by-step, write down a plan, and then write down your solution as: \"The solution is: A * B + C * D\".\n",
      "\n",
      "\n",
      "--------------------------------------------------------------------------------\n",
      "\u001b[33mteachableagent\u001b[0m (to user):\n",
      "\n",
      "Let's follow the advice to modify one integer on the left hand side of the equation to make the right hand side become 59:\n",
      "\n",
      "1. Let E denote the original number on the right: E = 87.\n",
      "2. Let F denote the final number on the right: F = 59.\n",
      "3. Calculate the difference between the two, G = F - E: G = 59 - 87 = -28.\n",
      "4. Examine the numbers on the left one by one to find one that divides evenly into G: 8, 3, 7, 9. Since 7 divides -28 evenly, we will work with 7 (H = 7).\n",
      "5. Calculate J = G / H: J = -28 / 7 = -4.\n",
      "6. Verify that J is an integer, and that H * J = G: J is an integer, and 7 * (-4) = -28.\n",
      "7. Find the number on the left which is multiplied by H (7), and call it K: K = 9.\n",
      "8. Change K to K + J: K_new = 9 + (-4) = 5.\n",
      "9. Recompute the value on the left, and verify that it equals F: (8 * 3) + (7 * 5) = 24 + 35 = 59.\n",
      "\n",
      "The solution is: 8 * 3 + 7 * 5\n",
      "\n",
      "--------------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "text = \"\"\"Consider the identity:  \n",
    "8 * 3 + 7 * 9 = 87\n",
    "Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 59?\n",
    "-Let's think step-by-step, write down a plan, and then write down your solution as: \"The solution is: A * B + C * D\".\n",
    "\"\"\"\n",
    "user.initiate_chat(teachable_agent, message=text, clear_history=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "flaml",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
