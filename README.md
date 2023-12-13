*GitHub Repository "microsoft/autogen"*

'''--- .github/ISSUE_TEMPLATE.md ---
### Description
<!-- A clear and concise description of the issue or feature request. -->

### Environment
- AutoGen version: <!-- Specify the AutoGen version (e.g., v0.2.0) -->
- Python version: <!-- Specify the Python version (e.g., 3.8) -->
- Operating System: <!-- Specify the OS (e.g., Windows 10, Ubuntu 20.04) -->

### Steps to Reproduce (for bugs)
<!-- Provide detailed steps to reproduce the issue. Include code snippets, configuration files, or any other relevant information. -->

1. Step 1
2. Step 2
3. ...

### Expected Behavior
<!-- Describe what you expected to happen. -->

### Actual Behavior
<!-- Describe what actually happened. Include any error messages, stack traces, or unexpected behavior. -->

### Screenshots / Logs (if applicable)
<!-- If relevant, include screenshots or logs that help illustrate the issue. -->

### Additional Information
<!-- Include any additional information that might be helpful, such as specific configurations, data samples, or context about the environment. -->

### Possible Solution (if you have one)
<!-- If you have suggestions on how to address the issue, provide them here. -->

### Is this a Bug or Feature Request?
<!-- Choose one: Bug | Feature Request -->

### Priority
<!-- Choose one: High | Medium | Low -->

### Difficulty
<!-- Choose one: Easy | Moderate | Hard -->

### Any related issues?
<!-- If this is related to another issue, reference it here. -->

### Any relevant discussions?
<!-- If there are any discussions or forum threads related to this issue, provide links. -->

### Checklist
<!-- Please check the items that you have completed -->
- [ ] I have searched for similar issues and didn't find any duplicates.
- [ ] I have provided a clear and concise description of the issue.
- [ ] I have included the necessary environment details.
- [ ] I have outlined the steps to reproduce the issue.
- [ ] I have included any relevant logs or screenshots.
- [ ] I have indicated whether this is a bug or a feature request.
- [ ] I have set the priority and difficulty levels.

### Additional Comments
<!-- Any additional comments or context that you think would be helpful. -->

'''
'''--- .github/PULL_REQUEST_TEMPLATE.md ---
<!-- Thank you for your contribution! Please review https://microsoft.github.io/autogen/docs/Contribute before opening a pull request. -->

<!-- Please add a reviewer to the assignee section when you create a PR. If you don't have the access to it, we will shortly find a reviewer and assign them to your PR. -->

## Why are these changes needed?

<!-- Please give a short summary of the change and the problem this solves. -->

## Related issue number

<!-- For example: "Closes #1234" -->

## Checks

- [ ] I've included any doc changes needed for https://microsoft.github.io/autogen/. See https://microsoft.github.io/autogen/docs/Contribute#documentation to build and test documentation locally.
- [ ] I've added tests (if relevant) corresponding to the changes introduced in this PR.
- [ ] I've made sure all auto checks have passed.

'''
'''--- CODE_OF_CONDUCT.md ---
# Microsoft Open Source Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

Resources:

- [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
- [Microsoft Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
- Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions or concerns

'''
'''--- README.md ---
[![PyPI version](https://badge.fury.io/py/pyautogen.svg)](https://badge.fury.io/py/pyautogen)
[![Build](https://github.com/microsoft/autogen/actions/workflows/python-package.yml/badge.svg)](https://github.com/microsoft/autogen/actions/workflows/python-package.yml)
![Python Version](https://img.shields.io/badge/3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
[![Downloads](https://static.pepy.tech/badge/pyautogen/week)](https://pepy.tech/project/pyautogen)
[![](https://img.shields.io/discord/1153072414184452236?logo=discord&style=flat)](https://discord.gg/pAbnFJrkgZ)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40pyautogen)](https://twitter.com/pyautogen)

# AutoGen

<!-- <p align="center">
    <img src="https://github.com/microsoft/autogen/blob/main/website/static/img/flaml.svg"  width=200>
    <br>
</p> -->
:fire: Nov 24: pyautogen [v0.2](https://github.com/microsoft/autogen/releases/tag/v0.2.0) is released with many updates and new features compared to v0.1.1. It switches to using openai-python v1. Please read the [migration guide](https://microsoft.github.io/autogen/docs/Installation#python).

:fire: Nov 11: OpenAI's Assistants are available in AutoGen and interoperatable with other AutoGen agents! Checkout our [blogpost](https://microsoft.github.io/autogen/blog/2023/11/13/OAI-assistants) for details and examples.

:fire: Nov 8: AutoGen is selected into [Open100: Top 100 Open Source achievements](https://www.benchcouncil.org/evaluation/opencs/annual.html) 35 days after spinoff.

:fire: Nov 6: AutoGen is mentioned by Satya Nadella in a [fireside chat](https://youtu.be/0pLBvgYtv6U) around 13:20.

:fire: Nov 1: AutoGen is the top trending repo on GitHub in October 2023.

:tada: Oct 03: AutoGen spins off from [FLAML](https://github.com/microsoft/FLAML) on Github and has a major paper update.

:tada: Aug 16: Paper about AutoGen on [arxiv](https://arxiv.org/abs/2308.08155). [📚 Cite paper](#related-papers).

:tada: Mar 29: AutoGen is first created in [FLAML](https://github.com/microsoft/FLAML/pull/968).

<!--
:fire: FLAML is highlighted in OpenAI's [cookbook](https://github.com/openai/openai-cookbook#related-resources-from-around-the-web).

:fire: [autogen](https://microsoft.github.io/autogen/) is released with support for ChatGPT and GPT-4, based on [Cost-Effective Hyperparameter Optimization for Large Language Model Generation Inference](https://arxiv.org/abs/2303.04673).

:fire: FLAML supports Code-First AutoML & Tuning – Private Preview in [Microsoft Fabric Data Science](https://learn.microsoft.com/en-us/fabric/data-science/). -->

## What is AutoGen

AutoGen is a framework that enables the development of LLM applications using multiple agents that can converse with each other to solve tasks. AutoGen agents are customizable, conversable, and seamlessly allow human participation. They can operate in various modes that employ combinations of LLMs, human inputs, and tools.

![AutoGen Overview](https://github.com/microsoft/autogen/blob/main/website/static/img/autogen_agentchat.png)

- AutoGen enables building next-gen LLM applications based on [multi-agent conversations](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat) with minimal effort. It simplifies the orchestration, automation, and optimization of a complex LLM workflow. It maximizes the performance of LLM models and overcomes their weaknesses.
- It supports [diverse conversation patterns](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat#supporting-diverse-conversation-patterns) for complex workflows. With customizable and conversable agents, developers can use AutoGen to build a wide range of conversation patterns concerning conversation autonomy,
  the number of agents, and agent conversation topology.
- It provides a collection of working systems with different complexities. These systems span a [wide range of applications](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat#diverse-applications-implemented-with-autogen) from various domains and complexities. This demonstrates how AutoGen can easily support diverse conversation patterns.
- AutoGen provides [enhanced LLM inference](https://microsoft.github.io/autogen/docs/Use-Cases/enhanced_inference#api-unification). It offers utilities like API unification and caching, and advanced usage patterns, such as error handling, multi-config inference, context programming, etc.

AutoGen is powered by collaborative [research studies](https://microsoft.github.io/autogen/docs/Research) from Microsoft, Penn State University, and the University of Washington.

## Quickstart
The easiest way to start playing is
1. Click below to use the GitHub Codespace

    [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/microsoft/autogen?quickstart=1)

 2. Copy OAI_CONFIG_LIST_sample to ./notebook folder, name to OAI_CONFIG_LIST, and set the correct configuration.
 3. Start playing with the notebooks!

## Installation

AutoGen requires **Python version >= 3.8, < 3.12**. It can be installed from pip:

```bash
pip install pyautogen
```

Minimal dependencies are installed without extra options. You can install extra options based on the feature you need.

<!-- For example, use the following to install the dependencies needed by the [`blendsearch`](https://microsoft.github.io/FLAML/docs/Use-Cases/Tune-User-Defined-Function#blendsearch-economical-hyperparameter-optimization-with-blended-search-strategy) option.
```bash
pip install "pyautogen[blendsearch]"
``` -->

Find more options in [Installation](https://microsoft.github.io/autogen/docs/Installation).

<!-- Each of the [`notebook examples`](https://github.com/microsoft/autogen/tree/main/notebook) may require a specific option to be installed. -->

For [code execution](https://microsoft.github.io/autogen/docs/FAQ/#code-execution), we strongly recommend installing the Python docker package and using docker.

For LLM inference configurations, check the [FAQs](https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints).

## Multi-Agent Conversation Framework

Autogen enables the next-gen LLM applications with a generic [multi-agent conversation](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat) framework. It offers customizable and conversable agents that integrate LLMs, tools, and humans.
By automating chat among multiple capable agents, one can easily make them collectively perform tasks autonomously or with human feedback, including tasks that require using tools via code.

Features of this use case include:

- **Multi-agent conversations**: AutoGen agents can communicate with each other to solve tasks. This allows for more complex and sophisticated applications than would be possible with a single LLM.
- **Customization**: AutoGen agents can be customized to meet the specific needs of an application. This includes the ability to choose the LLMs to use, the types of human input to allow, and the tools to employ.
- **Human participation**: AutoGen seamlessly allows human participation. This means that humans can provide input and feedback to the agents as needed.

For [example](https://github.com/microsoft/autogen/blob/main/test/twoagent.py),

```python
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
# You can also set config_list directly as a list, for example, config_list = [{'model': 'gpt-4', 'api_key': '<your OpenAI API key here>'},]
assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding"})
user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
# This initiates an automated chat between the two agents to solve the task
```

This example can be run with

```python
python test/twoagent.py
```

After the repo is cloned.
The figure below shows an example conversation flow with AutoGen.
![Agent Chat Example](https://github.com/microsoft/autogen/blob/main/website/static/img/chat_example.png)

Alternatively, the [sample code](https://github.com/microsoft/autogen/blob/main/samples/simple_chat.py) here allows a user to chat with an AutoGen agent in ChatGPT style.
Please find more [code examples](https://microsoft.github.io/autogen/docs/Examples#automated-multi-agent-chat) for this feature.

## Enhanced LLM Inferences

Autogen also helps maximize the utility out of the expensive LLMs such as ChatGPT and GPT-4. It offers [enhanced LLM inference](https://microsoft.github.io/autogen/docs/Use-Cases/enhanced_inference#api-unification) with powerful functionalities like caching, error handling, multi-config inference and templating.

<!-- For example, you can optimize generations by LLM with your own tuning data, success metrics, and budgets.

```python
# perform tuning for openai<1
config, analysis = autogen.Completion.tune(
    data=tune_data,
    metric="success",
    mode="max",
    eval_func=eval_func,
    inference_budget=0.05,
    optimization_budget=3,
    num_samples=-1,
)
# perform inference for a test instance
response = autogen.Completion.create(context=test_instance, **config)
```

Please find more [code examples](https://microsoft.github.io/autogen/docs/Examples#tune-gpt-models) for this feature. -->

## Documentation

You can find detailed documentation about AutoGen [here](https://microsoft.github.io/autogen/).

In addition, you can find:

- [Research](https://microsoft.github.io/autogen/docs/Research), [blogposts](https://microsoft.github.io/autogen/blog) around AutoGen, and [Transparency FAQs](https://github.com/microsoft/autogen/blob/main/TRANSPARENCY_FAQS.md)

- [Discord](https://discord.gg/pAbnFJrkgZ)

- [Contributing guide](https://microsoft.github.io/autogen/docs/Contribute)

- [Roadmap](https://github.com/orgs/microsoft/projects/989/views/3)

## Related Papers

[AutoGen](https://arxiv.org/abs/2308.08155)

```
@inproceedings{wu2023autogen,
      title={AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework},
      author={Qingyun Wu and Gagan Bansal and Jieyu Zhang and Yiran Wu and Beibin Li and Erkang Zhu and Li Jiang and Xiaoyun Zhang and Shaokun Zhang and Jiale Liu and Ahmed Hassan Awadallah and Ryen W White and Doug Burger and Chi Wang},
      year={2023},
      eprint={2308.08155},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

[EcoOptiGen](https://arxiv.org/abs/2303.04673)

```
@inproceedings{wang2023EcoOptiGen,
    title={Cost-Effective Hyperparameter Optimization for Large Language Model Generation Inference},
    author={Chi Wang and Susan Xueqing Liu and Ahmed H. Awadallah},
    year={2023},
    booktitle={AutoML'23},
}
```

[MathChat](https://arxiv.org/abs/2306.01337)

```
@inproceedings{wu2023empirical,
    title={An Empirical Study on Challenging Math Problem Solving with GPT-4},
    author={Yiran Wu and Feiran Jia and Shaokun Zhang and Hangyu Li and Erkang Zhu and Yue Wang and Yin Tat Lee and Richard Peng and Qingyun Wu and Chi Wang},
    year={2023},
    booktitle={ArXiv preprint arXiv:2306.01337},
}
```

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

If you are new to GitHub [here](https://help.github.com/categories/collaborating-with-issues-and-pull-requests/) is a detailed help source on getting involved with development on GitHub.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Contributors Wall
<a href="https://github.com/microsoft/autogen/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=microsoft/autogen" />
</a>

# Legal Notices

Microsoft and any contributors grant you a license to the Microsoft documentation and other content
in this repository under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode),
see the [LICENSE](LICENSE) file, and grant you a license to any code in the repository under the [MIT License](https://opensource.org/licenses/MIT), see the
[LICENSE-CODE](LICENSE-CODE) file.

Microsoft, Windows, Microsoft Azure, and/or other Microsoft products and services referenced in the documentation
may be either trademarks or registered trademarks of Microsoft in the United States and/or other countries.
The licenses for this project do not grant you rights to use any Microsoft names, logos, or trademarks.
Microsoft's general trademark guidelines can be found at http://go.microsoft.com/fwlink/?LinkID=254653.

Privacy information can be found at https://privacy.microsoft.com/en-us/

Microsoft and any contributors reserve all other rights, whether under their respective copyrights, patents,
or trademarks, whether by implication, estoppel, or otherwise.

'''
'''--- SECURITY.md ---
<!-- BEGIN MICROSOFT SECURITY.MD V0.0.8 BLOCK -->

## Security

Microsoft takes the security of our software products and services seriously, which includes all source code repositories managed through our GitHub organizations, which include [Microsoft](https://github.com/microsoft), [Azure](https://github.com/Azure), [DotNet](https://github.com/dotnet), [AspNet](https://github.com/aspnet), [Xamarin](https://github.com/xamarin), and [our GitHub organizations](https://opensource.microsoft.com/).

If you believe you have found a security vulnerability in any Microsoft-owned repository that meets [Microsoft's definition of a security vulnerability](https://aka.ms/opensource/security/definition), please report it to us as described below.

## Reporting Security Issues

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them to the Microsoft Security Response Center (MSRC) at [https://msrc.microsoft.com/create-report](https://aka.ms/opensource/security/create-report).

If you prefer to submit without logging in, send email to [secure@microsoft.com](mailto:secure@microsoft.com).  If possible, encrypt your message with our PGP key; please download it from the [Microsoft Security Response Center PGP Key page](https://aka.ms/opensource/security/pgpkey).

You should receive a response within 24 hours. If for some reason you do not, please follow up via email to ensure we received your original message. Additional information can be found at [microsoft.com/msrc](https://aka.ms/opensource/security/msrc).

Please include the requested information listed below (as much as you can provide) to help us better understand the nature and scope of the possible issue:

  * Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
  * Full paths of source file(s) related to the manifestation of the issue
  * The location of the affected source code (tag/branch/commit or direct URL)
  * Any special configuration required to reproduce the issue
  * Step-by-step instructions to reproduce the issue
  * Proof-of-concept or exploit code (if possible)
  * Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

If you are reporting for a bug bounty, more complete reports can contribute to a higher bounty award. Please visit our [Microsoft Bug Bounty Program](https://aka.ms/opensource/security/bounty) page for more details about our active programs.

## Preferred Languages

We prefer all communications to be in English.

## Policy

Microsoft follows the principle of [Coordinated Vulnerability Disclosure](https://aka.ms/opensource/security/cvd).

<!-- END MICROSOFT SECURITY.MD BLOCK -->

'''
'''--- TRANSPARENCY_FAQS.md ---
# AutoGen: Responsible AI FAQs

## What is AutoGen?
AutoGen is a framework for simplifying the orchestration, optimization, and automation of LLM workflows. It offers customizable and conversable agents that leverage the strongest capabilities of the most advanced LLMs, like GPT-4, while addressing their limitations by integrating with humans and tools and having conversations between multiple agents via automated chat.

## What can AutoGen do?
AutoGen is an experimentational framework for building a complex multi-agent conversation system by:
- Defining a set of agents with specialized capabilities and roles.
-	Defining the interaction behavior between agents, i.e., what to reply when an agent receives messages from another agent.

The agent conversation-centric design has numerous benefits, including that it:
-	Naturally handles ambiguity, feedback, progress, and collaboration.
-	Enables effective coding-related tasks, like tool use with back-and-forth troubleshooting.
-	Allows users to seamlessly opt in or opt out via an agent in the chat.
-	Achieves a collective goal with the cooperation of multiple specialists.

## 	What is/are AutoGen’s intended use(s)?
Please note that AutoGen is an open-source library under active development and intended for  use for research purposes. It should not be used in any downstream applications without additional detailed evaluation of robustness, safety issues and assessment of any potential harm or bias in the proposed application.

AutoGen is a generic infrastructure that can be used in multiple scenarios. The system’s intended uses include:

-	Building LLM workflows that solve more complex tasks: Users can create agents that interleave reasoning and tool use capabilities of the latest LLMs such as GPT-4. To solve complex tasks, multiple agents can converse to work together (e.g., by partitioning a complex problem into simpler steps or by providing different viewpoints or perspectives).
-	Application-specific agent topologies: Users can create application specific agent topologies and patterns for agents to interact. The exact topology may depend on the domain’s complexity and semantic capabilities of the LLM available.
-	Code generation and execution: Users can implement agents that can assume the roles of writing code and other agents that can execute code. Agents can do this with varying levels of human involvement. Users can add more agents and program the conversations to enforce constraints on code and output.
-	Question answering: Users can create agents that can help answer questions using retrieval augmented generation.
-	End user and multi-agent chat and debate: Users can build chat applications where they converse with multiple agents at the same time.

While AutoGen automates LLM workflows, decisions about how to use specific LLM outputs should always have a human in the loop. For example, you should not use AutoGen to automatically post LLM generated content to social media.

## How was AutoGen evaluated? What metrics are used to measure performance?
-	Current version of AutoGen was evaluated on six applications to illustrate its potential in simplifying the development of high-performance multi-agent applications. These applications are selected based on their real-world relevance,  problem difficulty and problem solving capabilities enabled by AutoGen, and innovative potential.
-	These applications involve using AutoGen to solve math problems, question answering, decision making in text world environments, supply chain optimization, etc. For each of these domains AutoGen was evaluated on various success based metrics (i.e., how often the AutoGen based implementation solved the task). And, in some cases, AutoGen based approach was also evaluated on implementation efficiency (e.g., to track reductions in developer effort to build). More details can be found at: https://aka.ms/AutoGen/TechReport

## What are the limitations of AutoGen? How can users minimize the impact of AutoGen’s limitations when using the system?
AutoGen relies on existing LLMs. Experimenting with AutoGen would retain common limitations of large language models; including:

- Data Biases: Large language models, trained on extensive data, can inadvertently carry biases present in the source data. Consequently, the models may generate outputs that could be potentially biased or unfair.
-	Lack of Contextual Understanding: Despite their impressive capabilities in language understanding and generation, these models exhibit limited real-world understanding, resulting in potential inaccuracies or nonsensical responses.
-	Lack of Transparency: Due to the complexity and size, large language models can act as `black boxes,' making it difficult to comprehend the rationale behind specific outputs or decisions.
-	Content Harms: There are various types of content harms that large language models can cause. It is important to be aware of them when using these models, and to take actions to prevent them. It is recommended to leverage various content moderation services provided by different companies and institutions.
-	Inaccurate or ungrounded content: It is important to be aware and cautious not to entirely rely on a given language model for critical decisions or information that might have deep impact as it is not obvious how to prevent these models to fabricate content without high authority input sources.
-	Potential for Misuse: Without suitable safeguards, there is a risk that these models could be maliciously used for generating disinformation or harmful content.

Additionally, AutoGen’s multi-agent framework may amplify or introduce additional risks, such as:
-	Privacy and Data Protection: The framework allows for human participation in conversations between agents. It is important to ensure that user data and conversations are protected and that developers use appropriate measures to safeguard privacy.
-	Accountability and Transparency: The framework involves multiple agents conversing and collaborating, it is important to establish clear accountability and transparency mechanisms. Users should be able to understand and trace the decision-making process of the agents involved in order to ensure accountability and address any potential issues or biases.
-	Trust and reliance: The framework leverages human understanding and intelligence while providing automation through conversations between agents. It is important to consider the impact of this interaction on user experience, trust, and reliance on AI systems. Clear communication and user education about the capabilities and limitations of the system will be essential.
-	Security & unintended consequences: The use of multi-agent conversations and automation in complex tasks may have unintended consequences. Especially, allowing LLM agents to make changes in external environments through code execution or function calls, such as install packages, could pose significant risks. Developers should carefully consider the potential risks and ensure that appropriate safeguards are in place to prevent harm or negative outcomes, including keeping a human in the loop for decision making.

## What operational factors and settings allow for effective and responsible use of AutoGen?
-	Code execution: AutoGen recommends using docker containers so that code execution can happen in a safer manner. Users can use function call instead of free-form code to execute pre-defined functions only. That helps increase the reliability and safety. Users can customize the code execution environment to tailor to their requirements.
-	Human involvement: AutoGen prioritizes human involvement in multi agent conversation. The overseers can step in to give feedback to agents and steer them in the correct direction. By default, users get chance to confirm before code is executed.
-	Agent modularity: Modularity allows agents to have different levels of information access. Additional agents can assume roles that help keep other agents in check. For example, one can easily add a dedicated agent to play the role of safeguard.
-	LLMs: Users can choose the LLM that is optimized for responsible use. The default LLM is GPT-4 which inherits the existing RAI mechanisms and filters from the LLM provider. Caching is enabled by default to increase reliability and control cost. We encourage developers to review [OpenAI’s Usage policies](https://openai.com/policies/usage-policies) and [Azure OpenAI’s Code of Conduct](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct) when using GPT-4.
-	Multi-agent setup: When using auto replies, the users can limit the number of auto replies, termination conditions etc. in the settings to increase reliability.

'''
'''--- samples/apps/autogen-assistant/README.md ---
# AutoGen Assistant

![ARA](./docs/ara_stockprices.png)

AutoGen Assistant is an Autogen-powered AI app (user interface) that can converse with you to help you conduct research, write and execute code, run saved skills, create new skills (explicitly and by demonstration), and adapt in response to your interactions.

### Capabilities / Roadmap

Some of the capabilities supported by the app frontend include the following:

- [x] Select fron a list of agents (current support for two agent workflows - `UserProxyAgent` and `AssistantAgent`)
- [x] Modify agent configuration (e.g. temperature, model, agent system message, model etc) and chat with updated agent configurations.
- [x] View agent messages and output files in the UI from agent runs.
- [ ] Support for more complex agent workflows (e.g. `GroupChat` workflows)
- [ ] Improved user experience (e.g., streaming intermediate model output, better summarization of agent responses, etc)

Project Structure:

- _autogenra/_ code for the backend classes and web api (FastAPI)
- _frontend/_ code for the webui, built with Gatsby and Tailwind

### Installation

1.  **Install from PyPi**

    We recommend using a virtual environment (e.g., conda) to avoid conflicts with existing Python packages. With Python 3.10 or newer active in your virtual environment, use pip to install AutoGen Assistant:

    ```bash
    pip install autogenra
    ```

2.  **Install from Source**

    > Note: This approach requires some familiarity with building interfaces in React.

    If you prefer to install from source, ensure you have Python 3.10+ and Node.js (version above 14.15.0) installed. Here's how you get started:

    - Clone the AutoGen Assistant repository and install its Python dependencies:

      ```bash
      pip install -e .
      ```

    - Navigate to the `samples/apps/autogen-assistant/frontend` directory, install dependencies, and build the UI:

      ```bash
      npm install -g gatsby-cli
      npm install --global yarn
      cd frontend
      yarn install
      yarn build
      ```

    For Windows users, to build the frontend, you may need alternative commands to build the frontend.

        ```bash

        gatsby clean && rmdir /s /q ..\\autogenra\\web\\ui && (set \"PREFIX_PATH_VALUE=\" || ver>nul) && gatsby build --prefix-paths && xcopy /E /I /Y public ..\\autogenra\\web\\ui

        ````

### Running the Application

Once installed, run the web UI by entering the following in your terminal:

```bash
autogenra ui --port 8081
```

This will start the application on the specified port. Open your web browser and go to `http://localhost:8081/` to begin using AutoGen Assistant.

Now that you have AutoGen Assistant installed and running, you are ready to explore its capabilities, including defining and modifying agent workflows, interacting with agents and sessions, and expanding agent skills.

## Capabilities

This demo focuses on the research assistant use case with some generalizations:

- **Skills**: The agent is provided with a list of skills that it can leverage while attempting to address a user's query. Each skill is a python function that may be in any file in a folder made availabe to the agents. We separate the concept of global skills available to all agents `backend/files/global_utlis_dir` and user level skills `backend/files/user/<user_hash>/utils_dir`, relevant in a multi user environment. Agents are aware skills as they are appended to the system message. A list of example skills is available in the `backend/global_utlis_dir` folder. Modify the file or create a new file with a function in the same directory to create new global skills.

- **Conversation Persistence**: Conversation history is persisted in an sqlite database `database.sqlite`.

- **Default Agent Workflow**: The default a sample workflow with two agents - a user proxy agent and an assistant agent.

## Example Usage

Let us use a simple query demonstrating the capabilities of the research assistant.

```
Plot a chart of NVDA and TESLA stock price YTD. Save the result to a file named nvda_tesla.png
```

The agents responds by _writing and executing code_ to create a python program to generate the chart with the stock prices.

> Note than there could be multiple turns between the `AssistantAgent` and the `UserProxyAgent` to produce and execute the code in order to complete the task.

![ARA](./docs/ara_stockprices.png)

> Note: You can also view the debug console that generates useful information to see how the agents are interacting in the background.

<!-- ![ARA](./docs/ara_console.png) -->

## FAQ

**Q: How can I add more skills to the AutoGen Assistant?**
A: You can extend the capabilities of your agents by adding new Python functions. The AutoGen Assistant interface also lets you directly paste functions that can be reused in the agent workflow.

**Q: Where can I adjust the agent configurations and settings?**
A: You can modify agent configurations directly from the UI or by editing the default configurations in the `utils.py` file under the `get_default_agent_config()` method (assuming you are building your own UI).

**Q: If I want to reset the conversation with an agent, how do I go about it?**
A: To reset your conversation history, you can delete the `database.sqlite` file. If you need to clear user-specific data, remove the relevant `autogenra/web/files/user/<user_id_md5hash>` folder.

**Q: Is it possible to view the output and messages generated by the agents during interactions?**
A: Yes, you can view the generated messages in the debug console of the web UI, providing insights into the agent interactions. Alternatively, you can inspect the `database.sqlite` file for a comprehensive record of messages.

## Acknowledgements

AutoGen assistant is Based on the [AutoGen](https://microsoft.github.io/autogen) project. It is adapted in October 2023 from a research prototype (original credits: Gagan Bansal, Adam Fourney, Victor Dibia, Piali Choudhury, Saleema Amershi, Ahmed Awadallah, Chi Wang)

'''
'''--- samples/apps/autogen-assistant/frontend/README.md ---
## 🚀 Running UI in Dev Mode

Run the UI in dev mode (make changes and see them reflected in the browser with hotreloading):

- npm install
- npm run start

This should start the server on port 8000.

## Design Elements

- **Gatsby**: The app is created in Gatsby. A guide on bootstrapping a Gatsby app can be found here - https://www.gatsbyjs.com/docs/quick-start/.
  This provides an overview of the project file structure include functionality of files like `gatsby-config.js`, `gatsby-node.js`, `gatsby-browser.js` and `gatsby-ssr.js`.
- **TailwindCSS**: The app uses TailwindCSS for styling. A guide on using TailwindCSS with Gatsby can be found here - https://tailwindcss.com/docs/guides/gatsby.https://tailwindcss.com/docs/guides/gatsby . This will explain the functionality in tailwind.config.js and postcss.config.js.

## Modifying the UI, Adding Pages

The core of the app can be found in the `src` folder. To add pages, add a new folder in `src/pages` and add a `index.js` file. This will be the entry point for the page. For example to add a route in the app like `/about`, add a folder `about` in `src/pages` and add a `index.tsx` file. You can follow the content style in `src/pages/index.tsx` to add content to the page.

Core logic for each component should be written in the `src/components` folder and then imported in pages as needed.

## connecting to front end

the front end makes request to the backend api and expects it at /api on localhost port 8081

## setting env variables for the UI

- please look at env.default
- make a copy of this file and name it `env.development`
- set the values for the variables in this file

'''
'''--- samples/tools/testbed/README.md ---
# Autogen Testbed Environment

The Autogen Testbed environment is a tool for repeatedly running a set of pre-defined Autogen scenarios in a setting with tightly-controlled initial conditions. With each run, Autogen will start from a blank slate, working out what code needs to be written, and what libraries or dependencies to install. The results of each run are logged, and can be ingested by analysis or metrics scripts (see the HumanEval example later in this README). By default, all runs are conducted in freshly-initialized docker containers, providing the recommended level of consistency and safety.

This Testbed sample has been tested in, and is known to work with, Autogen versions 0.1.14 and 0.2.0

## Setup

Before you begin, you must configure your API keys for use with the Testbed. As with other Autogen applications, the Testbed will look for the OpenAI keys in a file in the current working directy, or environment variable named, OAI_CONFIG_LIST. This can be overrriden using a command-line parameter described later.

For some scenarios, additional keys may be required (e.g., keys for the Bing Search API). These can be added to an `ENV` file in the `includes` folder. A sample has been provided in ``includes/ENV.example``. Edit ``includes/ENV`` as needed.

The Testbed also requires Docker (Desktop or Engine) AND the __python docker__ library. **It will not run in codespaces**, unless you opt for native execution (with is strongly discouraged). To install Docker Desktop see [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/). To install the Python library:

``pip install docker``

## Running the Testbed

To run the Testbed, simply execute
``python run_scenarios.py scenarios/Examples``

The default is to run each scenario once. To run each scenario 10 times, use:

``python run_scenarios.py --repeat 10 scenarios/Examples ``

The run_scenarios.py script also allows a number of command-line arguments to control various parameters of execution. Type ``python run_scenarios.py -h`` to explore these options:

```
run_scenarios.py will run the specified autogen scenarios for a given number of repetitions and record all logs and trace information. When running in a Docker environment (default), each run will begin from a common, tightly controlled, environment. The resultant logs can then be further processed by other scripts to produce metrics.

positional arguments:
  scenario      The JSONL scenario file to run. If a directory is specified,
                then all JSONL scenarios in the directory are run. (default:
                ./scenarios)

options:
  -h, --help    show this help message and exit

  -r REPEAT, --repeat REPEAT
                The number of repetitions to run for each scenario (default: 1).

  -c CONFIG, --config CONFIG
                The environment variable name or path to the OAI_CONFIG_LIST (default: OAI_CONFIG_LIST).

  --requirements REQUIREMENTS
                The requirements file to pip install before running the scenario. This file must be found in
                the 'includes' directory. (default: requirements.txt)

  --native      Run the scenarios natively rather than in docker.
                NOTE: This is not advisable, and should be done with great caution.
```

## Results

By default, the Testbed stores results in a folder heirarchy with the following template:

``./results/[scenario]/[instance_id]/[repetition]``

For example, consider the following folders:

``./results/default_two_agents_gpt35/two_agent_stocks/0``
``./results/default_two_agents_gpt35/two_agent_stocks/1``

...

``./results/default_two_agents_gpt35/two_agent_stocks/9``

This folder holds the results for the ``two_agent_stocks`` instance of the ``default_two_agents_gpt35`` scenario. The ``0`` folder contains the results of the first run. The ``1`` folder contains the results of the second run, and so on. You can think of the _instance_ as mapping to a prompt, or a unique set of parameters, while the _scenario_ defines the template in which those parameters are input.

Within each folder, you will find the following files:

- *timestamp.txt*: records the date and time of the run, along with the version of the pyautogen library installed
- *console_log.txt*: all console output produced by Docker when running autogen. Read this like you would a regular console.
- *chat_completions.json*: a log of all OpenAI ChatCompletions, as logged by `autogen.ChatCompletion.start_logging(compact=False)`
- *[agent]_messages.json*: for each Agent, a log of their messages dictionaries
- *./coding*: A directory containing all code written by Autogen, and all artifacts produced by that code.

## Scenario Templating

All scenarios are stored in JSONL files (in subdirectories under `./scenarios`). Each line of a scenario file is a JSON object. The schema varies slightly based on if "template" specifies a _file_ or a _directory_.

If "template" points to a _file_, the format is:
```
{
   "id": string,
   "template": filename,
   "substitutions" {
       "find_string1": replace_string1,
       "find_string2": replace_string2,
       ...
       "find_stringN": replace_stringN
   }
}
```

For example:

```
{
    "id": "two_agent_stocks_gpt4",
    "template": "default_two_agents.py",
    "substitutions": {
        "\__MODEL\__": "gpt-4",
        "\__PROMPT\__": "Plot and save to disk a chart of NVDA and TESLA stock price YTD."
    }
}
```

If "template" points to a _directory_, the format is:

```
{
   "id": string,
   "template": dirname,
   "substitutions" {
       "filename1": {
       	   "find_string1_1": replace_string1_1,
           "find_string1_2": replace_string1_2,
           ...
           "find_string1_M": replace_string1_N
       }
       "filename2": {
       	   "find_string2_1": replace_string2_1,
           "find_string2_2": replace_string2_2,
           ...
           "find_string2_N": replace_string2_N
       }
   }
}
```

For example:

```
{
    "id": "two_agent_stocks_gpt4",
    "template": "default_two_agents",
    "substitutions": {
	"scenario.py": {
            "\__MODEL\__": "gpt-4",
	},
	"prompt.txt": {
            "\__PROMPT\__": "Plot and save to disk a chart of NVDA and TESLA stock price YTD."
        }
    }
}
```

In this example, the string `__MODEL__` will be replaced in the file `scenarios.py`, while the string `__PROMPT__` will be replaced in the `prompt.txt` file.

## Scenario Expansion Algorithm

When the Testbed runs a scenario, it creates a local folder to share with Docker. As noted above, each instance and repetition gets its own folder along the path: ``./results/[scenario]/[instance_id]/[repetition]``

For the sake of brevity we will refer to this folder as the `DEST_FOLDER`.

The algorithm for populating the `DEST_FOLDER` is as follows:

1. Recursively copy the contents of `./incudes` to DEST_FOLDER. This folder contains all the basic starter files for running a scenario, including an ENV file which will set the Docker environment variables.
2. Append the OAI_CONFIG_LIST to the ENV file so that autogen may access these secrets.
3. Recursively copy the scenario folder (if `template` in the json scenario definition points to a folder) to DEST_FOLDER. If the `template` instead points to a file, copy the file, but rename it to `scenario.py`
4. Apply any templating, as outlined in the prior section.
5. Write a run.sh file to DEST_FOLDER that will be executed by Docker when it is loaded.

## Scenario Execution Algorithm

Once the scenario has been expanded it is run (via run.sh). This script will execute the following steps:

1. Read and set the ENV environment variables
2. If a file named `global_init.sh` is present, run it.
3. If a file named `scenario_init.sh` is present, run it.
4. Install the requirements file (if running in Docker)
5. Run the Autogen scenario via `python scenario.py`
6. Clean up (delete cache, etc.)
7. If a file named `scenario_finalize.sh` is present, run it.
8. If a file named `global_finalize.sh` is present, run it.
9. echo "SCENARIO COMPLETE !#!#", signaling that all steps completed.

Notably, this means that scenarios can add custom init and teardown logic by including `scenario_init.sh` and `scenario_finalize.sh` files.

## (Example) Running HumanEval

One sample Testbed scenario type is a variation of the classic [HumanEval](https://github.com/openai/human-eval) benchmark. In this scenario, agents are given access to the unit test results, and are able to continue to debug their code until the problem is solved or they run out of tokens or turns. We can then count how many turns it took to solve the problem (returning -1 if the problem remains unsolved by the end of the conversation, and "" if the run is missing).

Accessing this scenario-type requires downloading and converting the HumanEval dataset, running the Testbed, collating the results, and finally computing the metrics. The following commands will accomplish this, running each test instance 3 times with GPT-3.5-Turbo-16k:

```
python utils/download_humaneval.py
python ./run_scenarios.py scenarios/HumanEval/human_eval_two_agents_gpt35.jsonl
python utils/collate_human_eval.py ./results/human_eval_two_agents_gpt35 | python utils/metrics_human_eval.py > human_eval_results_gpt35.csv
cat human_eval_results_gpt35.csv
```

## (Example) Running GAIA

The Testbed can also be used to run the recently released [GAIA benchmark](https://huggingface.co/gaia-benchmark). This integration is presently experimental, and needs further validation. In this scenario, agents are presented with a series of questions that may include file references, or multi-modal input. Agents then must provide a `FINAL ANSWER`, which is considered correct if it (nearly) exactly matches an unambiguously accepted answer.

Accessing this scenario-type requires downloading and converting the GAIA dataset, running the Testbed, collating the results, and finally computing the metrics. The following commands will accomplish this, running each test instance once with GPT-4:

```
# Clone the GAIA dataset repo (assuming a 'repos' folder in your home directory)
cd ~/repos
git clone https://huggingface.co/datasets/gaia-benchmark/GAIA

# Expand GAIA
cd ~/repos/autogen/samples/tools/testbed
python ./utils/expand_gaia.py ~/repos/GAIA

# Run GAIA
python ./run_scenarios.py ./scenarios/GAIA/gaia_validation_level_1__two_agents_gpt4.jsonl

# Compute Metrics
python utils/collate_gaia_csv.py ./results/gaia_validation_level_1__two_agents_gpt4 | python utils/metrics_gaia.py
```

'''
'''--- samples/tools/testbed/scenarios/AutoGPT/README.md ---
The AutoGPT style tasks are contained in folder `challenges`.

Run `python utils/prepare_data.py` to convert the tasks to jsonl format compatible for evaluation.

'''
'''--- samples/tools/testbed/scenarios/HumanEval/README.md ---
Run `python ../../utils/download_humaneval.py` to populate this folder.

'''
'''--- website/README.md ---
# Website

This website is built using [Docusaurus 2](https://docusaurus.io/), a modern static website generator.

## Prerequisites

To build and test documentation locally, begin by downloading and installing [Node.js](https://nodejs.org/en/download/), and then installing [Yarn](https://classic.yarnpkg.com/en/).
On Windows, you can install via the npm package manager (npm) which comes bundled with Node.js:

```console
npm install --global yarn
```

## Installation

```console
pip install pydoc-markdown
cd website
yarn install
```

## Local Development

Navigate to the website folder and run:

```console
pydoc-markdown
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

'''
'''--- website/docs/Contribute.md ---
# Contributing

This project welcomes and encourages all forms of contributions, including but not limited to:

-  Pushing patches.
-  Code review of pull requests.
-  Documentation, examples and test cases.
-  Readability improvement, e.g., improvement on docstr and comments.
-  Community participation in [issues](https://github.com/microsoft/autogen/issues), [discussions](https://github.com/microsoft/autogen/discussions), [discord](https://discord.gg/pAbnFJrkgZ), and [twitter](https://twitter.com/pyautogen).
-  Tutorials, blog posts, talks that promote the project.
-  Sharing application scenarios and/or related research.

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

If you are new to GitHub [here](https://help.github.com/categories/collaborating-with-issues-and-pull-requests/) is a detailed help source on getting involved with development on GitHub.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## How to make a good bug report

When you submit an issue to [GitHub](https://github.com/microsoft/autogen/issues), please do your best to
follow these guidelines! This will make it a lot easier to provide you with good
feedback:

- The ideal bug report contains a short reproducible code snippet. This way
  anyone can try to reproduce the bug easily (see [this](https://stackoverflow.com/help/mcve) for more details). If your snippet is
  longer than around 50 lines, please link to a [gist](https://gist.github.com) or a GitHub repo.

- If an exception is raised, please **provide the full traceback**.

- Please include your **operating system type and version number**, as well as
  your **Python, autogen, scikit-learn versions**. The version of autogen
  can be found by running the following code snippet:
```python
import autogen
print(autogen.__version__)
```

- Please ensure all **code snippets and error messages are formatted in
  appropriate code blocks**.  See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks)
  for more details.

## Becoming a Reviewer

There is currently no formal reviewer solicitation process. Current reviewers identify reviewers from active contributors. If you are willing to become a reviewer, you are welcome to let us know on discord.

## Guidance for Maintainers

### General

*	Be a member of the community and treat everyone as a member. Be inclusive.
*	Help each other and encourage mutual help.
*	Actively post and respond.
*	Keep open communication.

### Pull Requests
* For new PR, decide whether to close without review. If not, find the right reviewers. The default reviewer is microsoft/autogen. Ask users who can benefit from the PR to review it.
*	For old PR, check the blocker: reviewer or PR creator. Try to unblock. Get additional help when needed.
*	When requesting changes, make sure you can check back in time because it blocks merging.
*	Make sure all the checks are passed.
*	For changes that require running OpenAI tests, make sure the OpenAI tests pass too. Running these tests requires approval.
*	In general, suggest small PRs instead of a giant PR.
*	For documentation change, request snapshot of the compiled website, or compile by yourself to verify the format.
*	For new contributors who have not signed the contributing agreement, remind them to sign before reviewing.
*	For multiple PRs which may have conflict, coordinate them to figure out the right order.
*	Pay special attention to:
    - Breaking changes. Don’t make breaking changes unless necessary. Don’t merge to main until enough headsup is provided and a new release is ready.
    -	Test coverage decrease.
    -	Changes that may cause performance degradation. Do regression test when test suites are available.
    - Discourage **change to the core library** when there is an alternative.

### Issues and Discussions
* For new issues, write a reply, apply a label if relevant. Ask on discord when necessary. For roadmap issues, add to the roadmap project and encourage community discussion. Mention relevant experts when necessary.
* For old issues, provide an update or close. Ask on discord when necessary. Encourage PR creation when relevant.
* Use “good first issue” for easy fix suitable for first-time contributors.
* Use “task list” for issues that require multiple PRs.
* For discussions, create an issue when relevant. Discuss on discord when appropriate.

## Developing

### Setup

```bash
git clone https://github.com/microsoft/autogen.git
pip install -e autogen
```

### Docker

We provide a simple [Dockerfile](https://github.com/microsoft/autogen/blob/main/Dockerfile).

```bash
docker build https://github.com/microsoft/autogen.git#main -t autogen-dev
docker run -it autogen-dev
```

### Develop in Remote Container

If you use vscode, you can open the autogen folder in a [Container](https://code.visualstudio.com/docs/remote/containers).
We have provided the configuration in [devcontainer](https://github.com/microsoft/autogen/blob/main/.devcontainer). They can be used in GitHub codespace too. Developing AutoGen in dev containers is recommended.

### Pre-commit

Run `pre-commit install` to install pre-commit into your git hooks. Before you commit, run
`pre-commit run` to check if you meet the pre-commit requirements. If you use Windows (without WSL) and can't commit after installing pre-commit, you can run `pre-commit uninstall` to uninstall the hook. In WSL or Linux this is supposed to work.

### Write tests

Tests are automatically run via GitHub actions. There are two workflows:
1. [build.yml](https://github.com/microsoft/autogen/blob/main/.github/workflows/build.yml)
1. [openai.yml](https://github.com/microsoft/autogen/blob/main/.github/workflows/openai.yml)

The first workflow is required to pass for all PRs. The second workflow is required for changes that affect the openai tests. The second workflow requires approval to run. When writing tests that require openai, please use [`pytest.mark.skipif`](https://github.com/microsoft/autogen/blob/main/test/test_client.py#L13) to make them run in one python version only when openai is installed. If additional dependency for this test is required, install the dependency in the corresponding python version in [openai.yml](https://github.com/microsoft/autogen/blob/main/.github/workflows/openai.yml).

### Coverage

Any code you commit should not decrease coverage. To run all unit tests, install the [test] option:

```bash
pip install -e."[test]"
coverage run -m pytest test
```

Then you can see the coverage report by
`coverage report -m` or `coverage html`.

### Documentation

To build and test documentation locally, install [Node.js](https://nodejs.org/en/download/). For example,

```bash
nvm install --lts
```

Then:

```console
npm install --global yarn  # skip if you use the dev container we provided
pip install pydoc-markdown  # skip if you use the dev container we provided
cd website
yarn install --frozen-lockfile --ignore-engines
pydoc-markdown
yarn start
```

The last command starts a local development server and opens up a browser window.
Most changes are reflected live without having to restart the server.

Note:
some tips in this guide are based off the contributor guide from [flaml](https://microsoft.github.io/FLAML/docs/Contribute).

'''
'''--- website/docs/Ecosystem.md ---
# Ecosystem
This page lists libraries that have integrations with Autogen for LLM applications using multiple agents in alphabetical order. Including your own integration to this list is highly encouraged. Simply open a pull request with a few lines of text, see the dropdown below for more information.

## MemGPT + AutoGen

![Agent Chat Example](img/ecosystem-memgpt.png)

MemGPT enables LLMs to manage their own memory and overcome limited context windows. You can use MemGPT to create perpetual chatbots that learn about you and modify their own personalities over time. You can connect MemGPT to your own local filesystems and databases, as well as connect MemGPT to your own tools and APIs. The MemGPT + AutoGen integration allows you to equip any AutoGen agent with MemGPT capabilities.

- [MemGPT + AutoGen Documentation with Code Examples](https://memgpt.readthedocs.io/en/latest/autogen/)

'''
'''--- website/docs/Examples.md ---
# Examples

## Automated Multi Agent Chat

AutoGen offers conversable agents powered by LLM, tool or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation via multi-agent conversation.
Please find documentation about this feature [here](/docs/Use-Cases/agent_chat).

Links to notebook examples:

1. **Code Generation, Execution, and Debugging**

   - Automated Task Solving with Code Generation, Execution & Debugging - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_auto_feedback_from_code_execution.ipynb)
   - Automated Code Generation and Question Answering with Retrieval Augmented Agents - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_RetrieveChat.ipynb)
   - Automated Code Generation and Question Answering with [Qdrant](https://qdrant.tech/) based Retrieval Augmented Agents - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_qdrant_RetrieveChat.ipynb)

1. **Multi-Agent Collaboration (>3 Agents)**

   - Automated Task Solving by Group Chat (with 3 group member agents and 1 manager agent) - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb)
   - Automated Data Visualization by Group Chat (with 3 group member agents and 1 manager agent) - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat_vis.ipynb)
   - Automated Complex Task Solving by Group Chat (with 6 group member agents and 1 manager agent) - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat_research.ipynb)
   - Automated Task Solving with Coding & Planning Agents - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_planning.ipynb)
   - Automated Task Solving with agents divided into 2 groups - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_hierarchy_flow_using_select_speaker.ipynb)
   - Automated Task Solving with transition paths specified in a graph - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_graph_modelling_language_using_select_speaker.ipynb)

1. **Applications**

   - Automated Chess Game Playing & Chitchatting by GPT-4 Agents - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_chess.ipynb)
   - Automated Continual Learning from New Data - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_stream.ipynb)
   - [OptiGuide](https://github.com/microsoft/optiguide) - Coding, Tool Using, Safeguarding & Question Answering for Supply Chain Optimization

1. **Tool Use**

   - **Web Search**: Solve Tasks Requiring Web Info - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_web_info.ipynb)
   - Use Provided Tools as Functions - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_function_call.ipynb)
   - Task Solving with Langchain Provided Tools as Functions - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_langchain.ipynb)
   - **RAG**: Group Chat with Retrieval Augmented Generation (with 5 group member agents and 1 manager agent) - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat_RAG.ipynb)
   - Function Inception: Enable AutoGen agents to update/remove functions during conversations. - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_inception_function.ipynb)
   - Agent Chat with Whisper - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_video_transcript_translate_with_whisper.ipynb)
1. **Human Involvement**
   - Simple example in ChatGPT style [View example](https://github.com/microsoft/autogen/blob/main/samples/simple_chat.py)
   - Auto Code Generation, Execution, Debugging and **Human Feedback** - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_human_feedback.ipynb)
   - Automated Task Solving with GPT-4 + **Multiple Human Users** - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_two_users.ipynb)
   - Agent Chat with **Async Human Inputs** - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/Async_human_input.ipynb)
1. **Agent Teaching and Learning**
   - Teach Agents New Skills & Reuse via Automated Chat - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_teaching.ipynb)
   - Teach Agents New Facts, User Preferences and Skills Beyond Coding - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_teachability.ipynb)

1. **Multi-Agent Chat with OpenAI Assistants in the loop**
   - Hello-World Chat with OpenAi Assistant in AutoGen - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_oai_assistant_twoagents_basic.ipynb)
   - Chat with OpenAI Assistant using Function Call - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_oai_assistant_function_call.ipynb)
   - Chat with OpenAI Assistant with Code Interpreter - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_oai_code_interpreter.ipynb)
   - Chat with OpenAI Assistant with Retrieval Augmentation - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_oai_assistant_retrieval.ipynb)
   - OpenAI Assistant in a Group Chat - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_oai_assistant_groupchat.ipynb)

1. **Multimodal Agent**
   - Multimodal Agent Chat with DALLE and GPT-4V   - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_dalle_and_gpt4v.ipynb)
   - Multimodal Agent Chat with Llava  - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_lmm_llava.ipynb)
   - Multimodal Agent Chat with GPT-4V - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_lmm_gpt-4v.ipynb)
1. **Long Context Handling**
   - Conversations with Chat History Compression Enabled - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_compression.ipynb)
1. **Evaluation and Assessment**
   - AgentEval: A Multi-Agent System for Assess Utility of LLM-powered Applications - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agenteval_cq_math.ipynb)
1. **Automatic Agent Building**
   - Automatically Build Multi-agent System with AgentBuilder - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_autobuild.ipynb)

## Enhanced Inferences
### Utilities
- API Unification  - [View Documentation with Code Example](https://microsoft.github.io/autogen/docs/Use-Cases/enhanced_inference/#api-unification)
- Utility Functions to Help Managing API configurations effectively - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/oai_openai_utils.ipynb)
- Cost Calculation - [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/oai_client_cost.ipynb)

### Inference Hyperparameters Tuning

AutoGen offers a cost-effective hyperparameter optimization technique [EcoOptiGen](https://arxiv.org/abs/2303.04673) for tuning Large Language Models. The research study finds that tuning hyperparameters can significantly improve the utility of them.
Please find documentation about this feature [here](/docs/Use-Cases/enhanced_inference).

Links to notebook examples:
* [Optimize for Code Generation](https://github.com/microsoft/autogen/blob/main/notebook/oai_completion.ipynb) | [Open in colab](https://colab.research.google.com/github/microsoft/autogen/blob/main/notebook/oai_completion.ipynb)
* [Optimize for Math](https://github.com/microsoft/autogen/blob/main/notebook/oai_chatgpt_gpt4.ipynb) | [Open in colab](https://colab.research.google.com/github/microsoft/autogen/blob/main/notebook/oai_chatgpt_gpt4.ipynb)

'''
'''--- website/docs/FAQ.md ---
# Frequently Asked Questions

## Set your API endpoints

There are multiple ways to construct configurations for LLM inference in the `oai` utilities:

- `get_config_list`: Generates configurations for API calls, primarily from provided API keys.
- `config_list_openai_aoai`: Constructs a list of configurations using both Azure OpenAI and OpenAI endpoints, sourcing API keys from environment variables or local files.
- `config_list_from_json`: Loads configurations from a JSON structure, either from an environment variable or a local JSON file, with the flexibility of filtering configurations based on given criteria.
- `config_list_from_models`: Creates configurations based on a provided list of models, useful when targeting specific models without manually specifying each configuration.
- `config_list_from_dotenv`: Constructs a configuration list from a `.env` file, offering a consolidated way to manage multiple API configurations and keys from a single file.

We suggest that you take a look at this [notebook](https://github.com/microsoft/autogen/blob/main/notebook/oai_openai_utils.ipynb) for full code examples of the different methods to configure your model endpoints.

### Use the constructed configuration list in agents

Make sure the "config_list" is included in the `llm_config` in the constructor of the LLM-based agent. For example,
```python
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list}
)
```

The `llm_config` is used in the [`create`](/docs/reference/oai/client#create) function for LLM inference.
When `llm_config` is not provided, the agent will rely on other openai settings such as `openai.api_key` or the environment variable `OPENAI_API_KEY`, which can also work when you'd like to use a single endpoint.
You can also explicitly specify that by:
```python
assistant = autogen.AssistantAgent(name="assistant", llm_config={"api_key": ...})
```

### Unexpected keyword argument 'base_url'

In version >=1, OpenAI renamed their `api_base` parameter to `base_url`. So for older versions, use `api_base` but for newer versions use `base_url`.

### Can I use non-OpenAI models?

Yes. Please check https://microsoft.github.io/autogen/blog/2023/07/14/Local-LLMs for an example.

## Handle Rate Limit Error and Timeout Error

You can set `max_retries` to handle rate limit error. And you can set `timeout` to handle timeout error. They can all be specified in `llm_config` for an agent, which will be used in the OpenAI client for LLM inference. They can be set differently for different clients if they are set in the `config_list`.

- `max_retries` (int): the total number of times allowed for retrying failed requests for a single client.
- `timeout` (int): the timeout (in seconds) for a single client.

Please refer to the [documentation](/docs/Use-Cases/enhanced_inference#runtime-error) for more info.

## How to continue a finished conversation

When you call `initiate_chat` the conversation restarts by default. You can use `send` or `initiate_chat(clear_history=False)` to continue the conversation.

## How do we decide what LLM is used for each agent? How many agents can be used? How do we decide how many agents in the group?

Each agent can be customized. You can use LLMs, tools, or humans behind each agent. If you use an LLM for an agent, use the one best suited for its role. There is no limit of the number of agents, but start from a small number like 2, 3. The more capable is the LLM and the fewer roles you need, the fewer agents you need.

The default user proxy agent doesn't use LLM. If you'd like to use an LLM in UserProxyAgent, the use case could be to simulate user's behavior.

The default assistant agent is instructed to use both coding and language skills. It doesn't have to do coding, depending on the tasks. And you can customize the system message. So if you want to use it for coding, use a model that's good at coding.

## Why is code not saved as file?

If you are using a custom system message for the coding agent, please include something like:
`If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line.`
in the system message. This line is in the default system message of the `AssistantAgent`.

If the `# filename` doesn't appear in the suggested code still, consider adding explicit instructions such as "save the code to disk" in the initial user message in `initiate_chat`.
The `AssistantAgent` doesn't save all the code by default, because there are cases in which one would just like to finish a task without saving the code.

## Code execution

We strongly recommend using docker to execute code. There are two ways to use docker:

1. Run autogen in a docker container. For example, when developing in GitHub codespace, the autogen runs in a docker container.
2. Run autogen outside of a docker, while perform code execution with a docker container. For this option, make sure the python package `docker` is installed. When it is not installed and `use_docker` is omitted in `code_execution_config`, the code will be executed locally (this behavior is subject to change in future).

### Enable Python 3 docker image

You might want to override the default docker image used for code execution. To do that set `use_docker` key of `code_execution_config` property to the name of the image. E.g.:
```python
user_proxy = autogen.UserProxyAgent(
    name="agent",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir":"_output", "use_docker":"python:3"},
    llm_config=llm_config,
    system_message=""""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)
```

If you have problems with agents running `pip install` or get errors similar to `Error while fetching server API version: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory')`, you can choose **'python:3'** as image as shown in the code example above and that should solve the problem.

### Agents keep thanking each other when using `gpt-3.5-turbo`

When using `gpt-3.5-turbo` you may often encounter agents going into a "gratitude loop", meaning when they complete a task they will begin congratulating and thanking eachother in a continuous loop. This is a limitation in the performance of `gpt-3.5-turbo`, in contrast to `gpt-4` which has no problem remembering instructions. This can hinder the experimentation experience when trying to test out your own use case with cheaper models.

A workaround is to add an additional termination notice to the prompt. This acts a "little nudge" for the LLM to remember that they need to terminate the conversation when their task is complete. You can do this by appending a string such as the following to your user input string:

```python
prompt = "Some user query"

termination_notice = (
    '\n\nDo not show appreciation in your responses, say only what is necessary. '
    'if "Thank you" or "You\'re welcome" are said in the conversation, then say TERMINATE '
    'to indicate the conversation is finished and this is your last message.'
)

prompt += termination_notice
```

**Note**: This workaround gets the job done around 90% of the time, but there are occurrences where the LLM still forgets to terminate the conversation.

## ChromaDB fails in codespaces because of old version of sqlite3

(from [issue #251](https://github.com/microsoft/autogen/issues/251))

Code examples that use chromadb (like retrieval) fail in codespaces due to a sqlite3 requirement.
```
>>> import chromadb
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/vscode/.local/lib/python3.10/site-packages/chromadb/__init__.py", line 69, in <module>
    raise RuntimeError(
RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.
Please visit https://docs.trychroma.com/troubleshooting#sqlite to learn how to upgrade.
```

Workaround:
1. `pip install pysqlite3-binary`
2. `mkdir /home/vscode/.local/lib/python3.10/site-packages/google/colab`

Explanation: Per [this gist](https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300?permalink_comment_id=4711478#gistcomment-4711478), linked from the official [chromadb docs](https://docs.trychroma.com/troubleshooting#sqlite), adding this folder triggers chromadb to use pysqlite3 instead of the default.

## How to register a reply function

(from [issue #478](https://github.com/microsoft/autogen/issues/478))

See here https://microsoft.github.io/autogen/docs/reference/agentchat/conversable_agent/#register_reply

 For example, you can register a reply function that gets called when `generate_reply` is called for an agent.
```python
def print_messages(recipient, messages, sender, config):
    if "callback" in config and  config["callback"] is not None:
        callback = config["callback"]
        callback(sender, recipient, messages[-1])
    print(f"Messages sent to: {recipient.name} | num messages: {len(messages)}")
    return False, None  # required to ensure the agent communication flow continues

user_proxy.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

assistant.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)
```
In the above, we register a `print_messages` function that is called each time the agent's `generate_reply` is triggered after receiving a message.

## How to get last message ?

Refer to https://microsoft.github.io/autogen/docs/reference/agentchat/conversable_agent/#last_message

## How to get each agent message ?

Please refer to https://microsoft.github.io/autogen/docs/reference/agentchat/conversable_agent#chat_messages

'''
'''--- website/docs/Getting-Started.md ---
# Getting Started

<!-- ### Welcome to AutoGen, a library for enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework! -->

AutoGen is a framework that enables development of LLM applications using multiple agents that can converse with each other to solve tasks. AutoGen agents are customizable, conversable, and seamlessly allow human participation. They can operate in various modes that employ combinations of LLMs, human inputs, and tools.

![AutoGen Overview](/img/autogen_agentchat.png)

### Main Features

- AutoGen enables building next-gen LLM applications based on [multi-agent conversations](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat) with minimal effort. It simplifies the orchestration, automation, and optimization of a complex LLM workflow. It maximizes the performance of LLM models and overcomes their weaknesses.
- It supports [diverse conversation patterns](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat#supporting-diverse-conversation-patterns) for complex workflows. With customizable and conversable agents, developers can use AutoGen to build a wide range of conversation patterns concerning conversation autonomy,
the number of agents, and agent conversation topology.
- It provides a collection of working systems with different complexities. These systems span a [wide range of applications](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat#diverse-applications-implemented-with-autogen) from various domains and complexities. This demonstrates how AutoGen can easily support diverse conversation patterns.
- AutoGen provides [enhanced LLM inference](https://microsoft.github.io/autogen/docs/Use-Cases/enhanced_inference#api-unification). It offers utilities like API unification and caching, and advanced usage patterns, such as error handling, multi-config inference, context programming, etc.

AutoGen is powered by collaborative [research studies](/docs/Research) from Microsoft, Penn State University, and University of Washington.

### Quickstart

Install from pip: `pip install pyautogen`. Find more options in [Installation](/docs/Installation).
For [code execution](/docs/FAQ#code-execution), we strongly recommend installing the python docker package, and using docker.

#### Multi-Agent Conversation Framework
Autogen enables the next-gen LLM applications with a generic multi-agent conversation framework. It offers customizable and conversable agents which integrate LLMs, tools, and humans.
By automating chat among multiple capable agents, one can easily make them collectively perform tasks autonomously or with human feedback, including tasks that require using tools via code. For [example](https://github.com/microsoft/autogen/blob/main/test/twoagent.py),
```python
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample.json
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding"})
user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
# This initiates an automated chat between the two agents to solve the task
```

The figure below shows an example conversation flow with AutoGen.
![Agent Chat Example](/img/chat_example.png)

* [Code examples](/docs/Examples).
* [Documentation](/docs/Use-Cases/agent_chat).

#### Enhanced LLM Inferences
Autogen also helps maximize the utility out of the expensive LLMs such as ChatGPT and GPT-4. It offers enhanced LLM inference with powerful functionalites like tuning, caching, error handling, templating. For example, you can optimize generations by LLM with your own tuning data, success metrics and budgets.
```python
# perform tuning for openai<1
config, analysis = autogen.Completion.tune(
    data=tune_data,
    metric="success",
    mode="max",
    eval_func=eval_func,
    inference_budget=0.05,
    optimization_budget=3,
    num_samples=-1,
)
# perform inference for a test instance
response = autogen.Completion.create(context=test_instance, **config)
```

* [Code examples](/docs/Examples).
* [Documentation](/docs/Use-Cases/enhanced_inference).

### Where to Go Next ?

* Understand the use cases for [multi-agent conversation](/docs/Use-Cases/agent_chat) and [enhanced LLM inference](/docs/Use-Cases/enhanced_inference).
* Find [code examples](/docs/Examples).
* Read [SDK](/docs/reference/agentchat/conversable_agent/).
* Learn about [research](/docs/Research) around AutoGen.
* [Roadmap](https://github.com/orgs/microsoft/projects/989/views/3)
* Chat on [Discord](https://discord.gg/pAbnFJrkgZ).
* Follow on [Twitter](https://twitter.com/pyautogen).

If you like our project, please give it a [star](https://github.com/microsoft/autogen/stargazers) on GitHub. If you are interested in contributing, please read [Contributor's Guide](/docs/Contribute).

<iframe src="https://ghbtns.com/github-btn.html?user=microsoft&amp;repo=autogen&amp;type=star&amp;count=true&amp;size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>

'''
'''--- website/docs/Installation.md ---
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

'''
'''--- website/docs/Research.md ---
# Research

For technical details, please check our technical report and research publications.

* [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework](https://arxiv.org/abs/2308.08155). Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang and Chi Wang. ArXiv 2023.

```bibtex
@inproceedings{wu2023autogen,
      title={AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework},
      author={Qingyun Wu and Gagan Bansal and Jieyu Zhang and Yiran Wu and Shaokun Zhang and Erkang Zhu and Beibin Li and Li Jiang and Xiaoyun Zhang and Chi Wang},
      year={2023},
      eprint={2308.08155},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

* [Cost-Effective Hyperparameter Optimization for Large Language Model Generation Inference](https://arxiv.org/abs/2303.04673). Chi Wang, Susan Xueqing Liu, Ahmed H. Awadallah. AutoML'23.

```bibtex
@inproceedings{wang2023EcoOptiGen,
    title={Cost-Effective Hyperparameter Optimization for Large Language Model Generation Inference},
    author={Chi Wang and Susan Xueqing Liu and Ahmed H. Awadallah},
    year={2023},
    booktitle={AutoML'23},
}
```

* [An Empirical Study on Challenging Math Problem Solving with GPT-4](https://arxiv.org/abs/2306.01337). Yiran Wu, Feiran Jia, Shaokun Zhang, Hangyu Li, Erkang Zhu, Yue Wang, Yin Tat Lee, Richard Peng, Qingyun Wu, Chi Wang. ArXiv preprint arXiv:2306.01337 (2023).

```bibtex
@inproceedings{wu2023empirical,
    title={An Empirical Study on Challenging Math Problem Solving with GPT-4},
    author={Yiran Wu and Feiran Jia and Shaokun Zhang and Hangyu Li and Erkang Zhu and Yue Wang and Yin Tat Lee and Richard Peng and Qingyun Wu and Chi Wang},
    year={2023},
    booktitle={ArXiv preprint arXiv:2306.01337},
}
```

* [EcoAssistant: Using LLM Assistant More Affordably and Accurately](https://arxiv.org/abs/2310.03046). Jieyu Zhang, Ranjay Krishna, Ahmed H. Awadallah, Chi Wang. ArXiv preprint arXiv:2310.03046 (2023).

```bibtex
@inproceedings{zhang2023ecoassistant,
    title={EcoAssistant: Using LLM Assistant More Affordably and Accurately},
    author={Zhang, Jieyu and Krishna, Ranjay and Awadallah, Ahmed H and Wang, Chi},
    year={2023},
    booktitle={ArXiv preprint arXiv:2310.03046},
}
```

'''
'''--- website/docs/Use-Cases/agent_chat.md ---
# Multi-agent Conversation Framework

AutoGen offers a unified multi-agent conversation framework as a high-level abstraction of using foundation models. It features capable, customizable and conversable agents which integrate LLMs, tools, and humans via automated agent chat.
By automating chat among multiple capable agents, one can easily make them collectively perform tasks autonomously or with human feedback, including tasks that require using tools via code.

This framework simplifies the orchestration, automation and optimization of a complex LLM workflow. It maximizes the performance of LLM models and overcome their weaknesses. It enables building next-gen LLM applications based on multi-agent conversations with minimal effort.

### Agents

AutoGen abstracts and implements conversable agents
designed to solve tasks through inter-agent conversations. Specifically, the agents in AutoGen have the following notable features:

- Conversable: Agents in AutoGen are conversable, which means that any agent can send
  and receive messages from other agents to initiate or continue a conversation

- Customizable: Agents in AutoGen can be customized to integrate LLMs, humans, tools, or a combination of them.

The figure below shows the built-in agents in AutoGen.
![Agent Chat Example](images/autogen_agents.png)

We have designed a generic `ConversableAgent` class for Agents that are capable of conversing with each other through the exchange of messages to jointly finish a task. An agent can communicate with other agents and perform actions. Different agents can differ in what actions they perform after receiving messages. Two representative subclasses are `AssistantAgent` and `UserProxyAgent`.

- The `AssistantAgent` is designed to act as an AI assistant, using LLMs by default but not requiring human input or code execution. It could write Python code (in a Python coding block) for a user to execute when a message (typically a description of a task that needs to be solved) is received. Under the hood, the Python code is written by LLM (e.g., GPT-4). It can also receive the execution results and suggest corrections or bug fixes. Its behavior can be altered by passing a new system message. The LLM [inference](#enhanced-inference) configuration can be configured via `llm_config`.

- The `UserProxyAgent` is conceptually a proxy agent for humans, soliciting human input as the agent's reply at each interaction turn by default and also having the capability to execute code and call functions. The `UserProxyAgent` triggers code execution automatically when it detects an executable code block in the received message and no human user input is provided. Code execution can be disabled by setting the `code_execution_config` parameter to False. LLM-based response is disabled by default. It can be enabled by setting `llm_config` to a dict corresponding to the [inference](/docs/Use-Cases/enhanced_inference) configuration. When `llm_config` is set as a dictionary, `UserProxyAgent` can generate replies using an LLM when code execution is not performed.

The auto-reply capability of `ConversableAgent` allows for more autonomous multi-agent communication while retaining the possibility of human intervention.
One can also easily extend it by registering reply functions with the `register_reply()` method.

In the following code, we create an `AssistantAgent` named "assistant" to serve as the assistant and a `UserProxyAgent` named "user_proxy" to serve as a proxy for the human user. We will later employ these two agents to solve a task.

```python
from autogen import AssistantAgent, UserProxyAgent

# create an AssistantAgent instance named "assistant"
assistant = AssistantAgent(name="assistant")

# create a UserProxyAgent instance named "user_proxy"
user_proxy = UserProxyAgent(name="user_proxy")
```

## Multi-agent Conversations

### A Basic Two-Agent Conversation Example

Once the participating agents are constructed properly, one can start a multi-agent conversation session by an initialization step as shown in the following code:

```python
# the assistant receives a message from the user, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""What date is today? Which big tech stock has the largest year-to-date gain this year? How much is the gain?""",
)
```

After the initialization step, the conversation could proceed automatically. Find a visual illustration of how the user_proxy and assistant collaboratively solve the above task autonmously below:
![Agent Chat Example](images/agent_example.png)

1. The assistant receives a message from the user_proxy, which contains the task description.
2. The assistant then tries to write Python code to solve the task and sends the response to the user_proxy.
3. Once the user_proxy receives a response from the assistant, it tries to reply by either soliciting human input or preparing an automatically generated reply. If no human input is provided, the user_proxy executes the code and uses the result as the auto-reply.
4. The assistant then generates a further response for the user_proxy. The user_proxy can then decide whether to terminate the conversation. If not, steps 3 and 4 are repeated.

### Supporting Diverse Conversation Patterns

#### Conversations with different levels of autonomy, and human-involvement patterns

On the one hand, one can achieve fully autonomous conversations after an initialization step. On the other hand, AutoGen can be used to implement human-in-the-loop problem-solving by configuring human involvement levels and patterns (e.g., setting the `human_input_mode` to `ALWAYS`), as human involvement is expected and/or desired in many applications.

#### Static and dynamic conversations

By adopting the conversation-driven control with both programming language and natural language, AutoGen inherently allows dynamic conversation. Dynamic conversation allows the agent topology to change depending on the actual flow of conversation under different input problem instances, while the flow of a static conversation always follows a pre-defined topology. The dynamic conversation pattern is useful in complex applications where the patterns of interaction cannot be predetermined in advance. AutoGen provides two general approaches to achieving dynamic conversation:

- Registered auto-reply. With the pluggable auto-reply function, one can choose to invoke conversations with other agents depending on the content of the current message and context. A working system demonstrating this type of dynamic conversation can be found in this code example, demonstrating a [dynamic group chat](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb). In the system, we register an auto-reply function in the group chat manager, which lets LLM decide who the next speaker will be in a group chat setting.

- LLM-based function call. In this approach, LLM decides whether or not to call a particular function depending on the conversation status in each inference call.
  By messaging additional agents in the called functions, the LLM can drive dynamic multi-agent conversation. A working system showcasing this type of dynamic conversation can be found in the [multi-user math problem solving scenario](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_two_users.ipynb), where a student assistant would automatically resort to an expert using function calls.

### Diverse Applications Implemented with AutoGen

The figure below shows six examples of applications built using AutoGen.
![Applications](images/app.png)

Find a list of examples in this page: [Automated Agent Chat Examples](../Examples.md#automated-multi-agent-chat)

## For Further Reading

_Interested in the research that leads to this package? Please check the following papers._

- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework](https://arxiv.org/abs/2308.08155). Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang and Chi Wang. ArXiv 2023.

- [An Empirical Study on Challenging Math Problem Solving with GPT-4](https://arxiv.org/abs/2306.01337). Yiran Wu, Feiran Jia, Shaokun Zhang, Hangyu Li, Erkang Zhu, Yue Wang, Yin Tat Lee, Richard Peng, Qingyun Wu, Chi Wang. ArXiv preprint arXiv:2306.01337 (2023).

'''
'''--- website/docs/Use-Cases/enhanced_inference.md ---
# Enhanced Inference

`autogen.OpenAIWrapper` provides enhanced LLM inference for `openai>=1`.
`autogen.Completion` is a drop-in replacement of `openai.Completion` and `openai.ChatCompletion` for enhanced LLM inference using `openai<1`.
There are a number of benefits of using `autogen` to perform inference: performance tuning, API unification, caching, error handling, multi-config inference, result filtering, templating and so on.

## Tune Inference Parameters (for openai<1)

Find a list of examples in this page: [Tune Inference Parameters Examples](../Examples.md#tune-inference-hyperparameters)

### Choices to optimize

The cost of using foundation models for text generation is typically measured in terms of the number of tokens in the input and output combined. From the perspective of an application builder using foundation models, the use case is to maximize the utility of the generated text under an inference budget constraint (e.g., measured by the average dollar cost needed to solve a coding problem). This can be achieved by optimizing the hyperparameters of the inference,
which can significantly affect both the utility and the cost of the generated text.

The tunable hyperparameters include:
1. model - this is a required input, specifying the model ID to use.
1. prompt/messages - the input prompt/messages to the model, which provides the context for the text generation task.
1. max_tokens - the maximum number of tokens (words or word pieces) to generate in the output.
1. temperature - a value between 0 and 1 that controls the randomness of the generated text. A higher temperature will result in more random and diverse text, while a lower temperature will result in more predictable text.
1. top_p - a value between 0 and 1 that controls the sampling probability mass for each token generation. A lower top_p value will make it more likely to generate text based on the most likely tokens, while a higher value will allow the model to explore a wider range of possible tokens.
1. n - the number of responses to generate for a given prompt. Generating multiple responses can provide more diverse and potentially more useful output, but it also increases the cost of the request.
1. stop - a list of strings that, when encountered in the generated text, will cause the generation to stop. This can be used to control the length or the validity of the output.
1. presence_penalty, frequency_penalty - values that control the relative importance of the presence and frequency of certain words or phrases in the generated text.
1. best_of - the number of responses to generate server-side when selecting the "best" (the one with the highest log probability per token) response for a given prompt.

The cost and utility of text generation are intertwined with the joint effect of these hyperparameters.
There are also complex interactions among subsets of the hyperparameters. For example,
the temperature and top_p are not recommended to be altered from their default values together because they both control the randomness of the generated text, and changing both at the same time can result in conflicting effects; n and best_of are rarely tuned together because if the application can process multiple outputs, filtering on the server side causes unnecessary information loss; both n and max_tokens will affect the total number of tokens generated, which in turn will affect the cost of the request.
These interactions and trade-offs make it difficult to manually determine the optimal hyperparameter settings for a given text generation task.

*Do the choices matter? Check this [blogpost](/blog/2023/04/21/LLM-tuning-math) to find example tuning results about gpt-3.5-turbo and gpt-4.*

With AutoGen, the tuning can be performed with the following information:
1. Validation data.
1. Evaluation function.
1. Metric to optimize.
1. Search space.
1. Budgets: inference and optimization respectively.

### Validation data

Collect a diverse set of instances. They can be stored in an iterable of dicts. For example, each instance dict can contain "problem" as a key and the description str of a math problem as the value; and "solution" as a key and the solution str as the value.

### Evaluation function

The evaluation function should take a list of responses, and other keyword arguments corresponding to the keys in each validation data instance as input, and output a dict of metrics. For example,

```python
def eval_math_responses(responses: List[str], solution: str, **args) -> Dict:
    # select a response from the list of responses
    answer = voted_answer(responses)
    # check whether the answer is correct
    return {"success": is_equivalent(answer, solution)}
```

`autogen.code_utils` and `autogen.math_utils` offer some example evaluation functions for code generation and math problem solving.

### Metric to optimize

The metric to optimize is usually an aggregated metric over all the tuning data instances. For example, users can specify "success" as the metric and "max" as the optimization mode. By default, the aggregation function is taking the average. Users can provide a customized aggregation function if needed.

### Search space

Users can specify the (optional) search range for each hyperparameter.

1. model. Either a constant str, or multiple choices specified by `flaml.tune.choice`.
1. prompt/messages. Prompt is either a str or a list of strs, of the prompt templates. messages is a list of dicts or a list of lists, of the message templates.
Each prompt/message template will be formatted with each data instance. For example, the prompt template can be:
"{problem} Solve the problem carefully. Simplify your answer as much as possible. Put the final answer in \\boxed{{}}."
And `{problem}` will be replaced by the "problem" field of each data instance.
1. max_tokens, n, best_of. They can be constants, or specified by `flaml.tune.randint`, `flaml.tune.qrandint`, `flaml.tune.lograndint` or `flaml.qlograndint`. By default, max_tokens is searched in [50, 1000); n is searched in [1, 100); and best_of is fixed to 1.
1. stop. It can be a str or a list of strs, or a list of lists of strs or None. Default is None.
1. temperature or top_p. One of them can be specified as a constant or by `flaml.tune.uniform` or `flaml.tune.loguniform` etc.
Please don't provide both. By default, each configuration will choose either a temperature or a top_p in [0, 1] uniformly.
1. presence_penalty, frequency_penalty. They can be constants or specified by `flaml.tune.uniform` etc. Not tuned by default.

### Budgets

One can specify an inference budget and an optimization budget.
The inference budget refers to the average inference cost per data instance.
The optimization budget refers to the total budget allowed in the tuning process. Both are measured by dollars and follow the price per 1000 tokens.

### Perform tuning

Now, you can use `autogen.Completion.tune` for tuning. For example,

```python
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
```

`num_samples` is the number of configurations to sample. -1 means unlimited (until optimization budget is exhausted).
The returned `config` contains the optimized configuration and `analysis` contains an ExperimentAnalysis object for all the tried configurations and results.

The tuend config can be used to perform inference.

## API unification

<!-- `autogen.Completion.create` is compatible with both `openai.Completion.create` and `openai.ChatCompletion.create`, and both OpenAI API and Azure OpenAI API. So models such as "text-davinci-003", "gpt-3.5-turbo" and "gpt-4" can share a common API.
When chat models are used and `prompt` is given as the input to `autogen.Completion.create`, the prompt will be automatically converted into `messages` to fit the chat completion API requirement. One advantage is that one can experiment with both chat and non-chat models for the same prompt in a unified API. -->

`autogen.OpenAIWrapper.create()` can be used to create completions for both chat and non-chat models, and both OpenAI API and Azure OpenAI API.

```python
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

```

For local LLMs, one can spin up an endpoint using a package like [FastChat](https://github.com/lm-sys/FastChat), and then use the same API to send a request. See [here](/blog/2023/07/14/Local-LLMs) for examples on how to make inference with local LLMs.

<!-- When only working with the chat-based models, `autogen.ChatCompletion` can be used. It also does automatic conversion from prompt to messages, if prompt is provided instead of messages. -->

## Usage Summary

The `OpenAIWrapper` from `autogen` tracks token counts and costs of your API calls. Use the `create()` method to initiate requests and `print_usage_summary()` to retrieve a detailed usage report, including total cost and token usage for both cached and actual requests.

- `mode=["actual", "total"]` (default): print usage summary for all completions and non-caching completions.
- `mode='actual'`: only print non-cached usage.
- `mode='total'`: only print all usage (including cache).

Reset your session's usage data with `clear_usage_summary()` when needed. [View Notebook](https://github.com/microsoft/autogen/blob/main/notebook/oai_client_cost.ipynb)

Example usage:
```python
from autogen import OpenAIWrapper

client = OpenAIWrapper()
client.create(messages=[{"role": "user", "content": "Python learning tips."}], model="gpt-3.5-turbo")
client.print_usage_summary()  # Display usage
client.clear_usage_summary()  # Reset usage data
```

Sample output:
```
Usage summary excluding cached usage:
Total cost: 0.00015
* Model 'gpt-3.5-turbo': cost: 0.00015, prompt_tokens: 25, completion_tokens: 58, total_tokens: 83

Usage summary including cached usage:
Total cost: 0.00027
* Model 'gpt-3.5-turbo': cost: 0.00027, prompt_tokens: 50, completion_tokens: 100, total_tokens: 150
```

## Caching

API call results are cached locally and reused when the same request is issued. This is useful when repeating or continuing experiments for reproducibility and cost saving. It still allows controlled randomness by setting the "cache_seed" specified in `OpenAIWrapper.create()` or the constructor of `OpenAIWrapper`.

```python
client = OpenAIWrapper(cache_seed=...)
client.create(...)
```

```python
client = OpenAIWrapper()
client.create(cache_seed=..., ...)
```

Caching is enabled by default with cache_seed 41. To disable it please set `cache_seed` to None.

_NOTE_. openai v1.1 introduces a new param `seed`. The difference between autogen's `cache_seed` and openai's `seed` is that:
* autogen uses local disk cache to guarantee the exactly same output is produced for the same input and when cache is hit, no openai api call will be made.
* openai's `seed` is a best-effort deterministic sampling with no guarantee of determinism. When using openai's `seed` with `cache_seed` set to None, even for the same input, an openai api call will be made and there is no guarantee for getting exactly the same output.

## Error handling

### Runtime error

<!-- It is easy to hit error when calling OpenAI APIs, due to connection, rate limit, or timeout. Some of the errors are transient. `autogen.Completion.create` deals with the transient errors and retries automatically. Request timeout, max retry period and retry wait time can be configured via `request_timeout`, `max_retry_period` and `retry_wait_time`.

- `request_timeout` (int): the timeout (in seconds) sent with a single request.
- `max_retry_period` (int): the total time (in seconds) allowed for retrying failed requests.
- `retry_wait_time` (int): the time interval to wait (in seconds) before retrying a failed request.

Moreover,  -->
One can pass a list of configurations of different models/endpoints to mitigate the rate limits and other runtime error. For example,

```python
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
```

`client.create()` will try querying Azure OpenAI gpt-4, OpenAI gpt-3.5-turbo, and a locally hosted llama2-chat-7B one by one,
until a valid result is returned. This can speed up the development process where the rate limit is a bottleneck. An error will be raised if the last choice fails. So make sure the last choice in the list has the best availability.

For convenience, we provide a number of utility functions to load config lists.
- `get_config_list`: Generates configurations for API calls, primarily from provided API keys.
- `config_list_openai_aoai`: Constructs a list of configurations using both Azure OpenAI and OpenAI endpoints, sourcing API keys from environment variables or local files.
- `config_list_from_json`: Loads configurations from a JSON structure, either from an environment variable or a local JSON file, with the flexibility of filtering configurations based on given criteria.
- `config_list_from_models`: Creates configurations based on a provided list of models, useful when targeting specific models without manually specifying each configuration.
- `config_list_from_dotenv`: Constructs a configuration list from a `.env` file, offering a consolidated way to manage multiple API configurations and keys from a single file.

We suggest that you take a look at this [notebook](https://github.com/microsoft/autogen/blob/main/notebook/oai_openai_utils.ipynb) for full code examples of the different methods to configure your model endpoints.

### Logic error

Another type of error is that the returned response does not satisfy a requirement. For example, if the response is required to be a valid json string, one would like to filter the responses that are not. This can be achieved by providing a list of configurations and a filter function. For example,

```python
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
```

The example above will try to use text-ada-001, gpt-3.5-turbo-instruct, and text-davinci-003 iteratively, until a valid json string is returned or the last config is used. One can also repeat the same model in the list for multiple times (with different seeds) to try one model multiple times for increasing the robustness of the final response.

*Advanced use case: Check this [blogpost](/blog/2023/05/18/GPT-adaptive-humaneval) to find how to improve GPT-4's coding performance from 68% to 90% while reducing the inference cost.*

## Templating

If the provided prompt or message is a template, it will be automatically materialized with a given context. For example,

```python
response = client.create(
    context={"problem": "How many positive integers, not exceeding 100, are multiples of 2 or 3 but not 4?"},
    prompt="{problem} Solve the problem carefully.",
    allow_format_str_template=True,
    **config
)
```

A template is either a format str, like the example above, or a function which produces a str from several input fields, like the example below.

```python
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
```

## Logging (for openai<1)

When debugging or diagnosing an LLM-based system, it is often convenient to log the API calls and analyze them. `autogen.Completion` and `autogen.ChatCompletion` offer an easy way to collect the API call histories. For example, to log the chat histories, simply run:
```python
autogen.ChatCompletion.start_logging()
```
The API calls made after this will be automatically logged. They can be retrieved at any time by:
```python
autogen.ChatCompletion.logged_history
```
There is a function that can be used to print usage summary (total cost, and token count usage from each model):
```python
autogen.ChatCompletion.print_usage_summary()
```
To stop logging, use
```python
autogen.ChatCompletion.stop_logging()
```
If one would like to append the history to an existing dict, pass the dict like:
```python
autogen.ChatCompletion.start_logging(history_dict=existing_history_dict)
```
By default, the counter of API calls will be reset at `start_logging()`. If no reset is desired, set `reset_counter=False`.

There are two types of logging formats: compact logging and individual API call logging. The default format is compact.
Set `compact=False` in `start_logging()` to switch.

* Example of a history dict with compact logging.
```python
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
```

* Example of a history dict with individual API call logging.
```python
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
```

* Example of printing for usage summary
```
Total cost: <cost>
Token count summary for model <model>: prompt_tokens: <count 1>, completion_tokens: <count 2>, total_tokens: <count 3>
```

It can be seen that the individual API call history contains redundant information of the conversation. For a long conversation the degree of redundancy is high.
The compact history is more efficient and the individual API call history contains more details.

'''