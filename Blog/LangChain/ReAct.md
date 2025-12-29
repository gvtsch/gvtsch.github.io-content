---
title: What exactly is ReAct?
date: 2025-07-14
tags: ["python", "langchain", "machine-learning", "llm", "nlp", "agent", "react"]
toc: True
draft: false
author: CKe
---

# ReAct - Reasoning and Acting

ReAct (Reasoning + Acting) is a pretty cool approach that lets large language models (LLMs) both think logically and actually do stuff. Instead of just generating text, ReAct agents can reason through problems step-by-step AND take concrete actions like calling APIs, searching the web, or running code.

It's basically combining the thinking power of LLMs with the ability to interact with the real world through ([[Prompt-Techniques|Chain-of-Thought]]) reasoning.

## Process

The process is actually quite straightforward:

* The agent gets a task
* It thinks about what to do ("Thought")
* It decides on an action ("Action") 
* It gets a result back ("Observation")
* It repeats this cycle until the problem is solved

```mermaid
flowchart TD
    A[Receive task] --&gt; B["Think (Though)"]
    B --&gt; C["Select tool/action (Action)"]
    C --&gt; D["Receive result (Observation)"]
    D --&gt; E{"Goal achieved?"}
    E -- No --&gt; B
    E -- Yes --&gt; F["Give answer"]
```

## What makes ReAct special

* **Reasoning**: The model actually explains its thought process step-by-step. You can see exactly how it's thinking through a problem.
* **Acting**: It doesn't just think - it takes real actions like searching Wikipedia or running Python code.
* **Transparency**: Since you can see the reasoning trace, you understand why the agent made certain decisions.

## Why ReAct is better than traditional approaches

Unlike agents that only reason (think) or only act (perform tasks), ReAct agents combine both approaches. This allows them to solve more complex problems, adapt to new situations, and provide more reliable and explainable results.

## Building a ReAct agent with LangChain

I want to show you how to actually build one of these agents. We'll create an agent that can run Python code, search Wikipedia, and browse the internet. Let me walk you through the key parts.

### What makes an agent a ReAct agent?

It's actually three things working together in the [[What is LangChain|LangChain]] environment:

1. **The ReAct principle**: This is the core idea - the agent cycles through thinking, acting, and observing. It's the theoretical framework that guides the iterative process of "Thought" -> "Action" -> "Observation."
2. **A special prompt**: This is crucial. You need a prompt that teaches the LLM how to behave like a ReAct agent. It's not just an instruction - it's a detailed guide containing examples ([[Prompt-Techniques|Few-shot]] examples) and format specifications. LangChain has a ready-made one:

    ```python
    from langchain import hub
    prompt = hub.pull('hwchase17/react')

    print(prompt.input_variables)
    print(prompt.template)
    ```

    This gives you:

    ```bash
    ['agent_scratchpad', 'input', 'tool_names', 'tools']
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    ```

3. **LangChain's orchestration**: The `create_react_agent` and `AgentExecutor` functions handle all the complex stuff behind the scenes. They're like the technical *managers* that control the ReAct process step by step.

### Tools

This is where it gets interesting. Tools are what give your agent superpowers beyond just text generation. LLMs are great at language, but they have well-known limitations:

* **Knowledge base**: Their knowledge gets outdated quickly
* **Logic and calculations**: They can make mistakes with complex math or strict logical rules
* **Interaction with the real world**: They can't directly access the internet, call APIs, or perform actions

That's where tools come in. They extend the agent with *senses* and *capabilities*. I'll walk you through just a couple of them.

#### Python REPL

Here's how you define the Python REPL tool for your agent:

```python
python_repl = PythonREPLTool()
python_repl_tool = Tool(
    name = 'Python REPL',
    func = python_repl.run,
    description = 'Useful when you need to run Python code. Input should be a valid Python expression or statement.',
)
```

This tool lets the agent execute Python code directly. Super useful for calculations, data manipulation, or any logic where you need precision – things that LLMs can be unreliable at.

#### Wikipedia

Here's how you set up the Wikipedia tool:

```python
api_wrapper = WikipediaAPIWrapper()
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper)
wikipedia_tool = Tool(
    name = 'Wikipedia',
    func = wikipedia.run,
    description = 'Useful for getting information about a topic. Input should be a question or topic.',
)
```

Wikipedia access gives the agent a massive knowledge base for factual information that might not be in the LLM's training data or could be outdated.

#### TavilySearchResults

Here's how you configure the Tavily Search tool:

```python
search = TavilySearchResults(
    max_results=1,
    include_answer=True,
)

tavily_tool = Tool(
    name='Tavily Search',
    func=search.run,
    description='Useful for getting information from the web. Input should be a question or topic.',
)
```

This one's great for real-time web searches when you need current information. Unlike Wikipedia's "curated" knowledge, this finds highly topical stuff, news, or specific website content.

#### Other tools you could add

The possibilities are endless. You could add:
- Database access (SQL queries)
- Email services  
- Calendar APIs
- File system access
- Image generation
- Shell commands

Next, let's put it together and make it work.

### Putting it all together

Alright, time to wire everything together and get this agent working:

You need to utilize the agent with the above shown tools:

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent

tools = [tavily_tool, python_repl_tool, wikipedia_tool]

agent = create_react_agent(llm, tools, prompt)
agent_executer = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
)
```

Now you can ask it to do stuff:

```python
question = 'Generate the first 20 Fibonacci numbers.'
output = agent_executer.invoke({
    'input': prompt_template.format(q=question),
})
```

## Watching the agent think

Here's what's really cool - you can see exactly how the agent works through problems. The agent starts thinking about how to reach its goal and shows you its entire route to the solution:

````bash
> Entering new AgentExecutor chain...
To generate the first 20 Fibonacci numbers, I will write a simple Python function that calculates them. 

Action: Python REPL  
Action Input: `def fibonacci(n): fib_sequence = [0, 1] for i in range(2, n): fib_sequence.append(fib_sequence[-1] + fib_sequence[-2]) return fib_sequence[:n] fibonacci(20)`  
SyntaxError('invalid syntax', ('&lt;string&gt;', 1, 41, 'def fibonacci(n): fib_sequence = [0, 1] for i in range(2, n): fib_sequence.append(fib_sequence[-1] + fib_sequence[-2]) return fib_sequence[:n] fibonacci(20)\n', 1, 44))It seems there was a syntax error in the Python code I attempted to run. I will correct the code and try again to generate the first 20 Fibonacci numbers.

Action: Python REPL  
Action Input: `def fibonacci(n): fib_sequence = [0, 1] for i in range(2, n): fib_sequence.append(fib_sequence[-1] + fib_sequence[-2]) return fib_sequence fib_sequence = fibonacci(20)`  
SyntaxError('invalid syntax', ('&lt;string&gt;', 1, 41, 'def fibonacci(n): fib_sequence = [0, 1] for i in range(2, n): fib_sequence.append(fib_sequence[-1] + fib_sequence[-2]) return fib_sequence fib_sequence = fibonacci(20)\n', 1, 44))It seems that I am encountering a syntax error in the Python code. I will rewrite the function with proper indentation and structure to ensure it runs correctly.

Action: Python REPL  
Action Input: `def fibonacci(n): fib_sequence = [0, 1] for i in range(2, n): fib_sequence.append(fib_sequence[-1] + fib_sequence[-2]) return fib_sequence[:n] fib_sequence = fibonacci(20)`  
SyntaxError('invalid syntax', ('&lt;string&gt;', 1, 41, 'def fibonacci(n): fib_sequence = [0, 1] for i in range(2, n): fib_sequence.append(fib_sequence[-1] + fib_sequence[-2]) return fib_sequence[:n] fib_sequence = fibonacci(20)\n', 1, 44))It appears that I am consistently encountering a syntax error due to the formatting of the code. I will break down the function definition and the call into separate lines to ensure proper execution.

Action: Python REPL  
Action Input: 
```python
def fibonacci(n):
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence[:n]

fib_sequence = fibonacci(20)
fib_sequence
```

I have successfully generated the first 20 Fibonacci numbers. 

Final Answer: The first 20 Fibonacci numbers are: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181].

> Finished chain.
````

Notice how it made a syntax error, realized it, and fixed itself? That's the power of the reasoning loop.

Here's another example showing it doesn't always generate Python functions:


```python
question = 'Who is the last chancellor of Germany?'
output = agent_executer.invoke({
    'input': prompt_template.format(q=question),
})
```

This leads to an interesting thought process where the agent first interprets "last chancellor" as "current chancellor" and then corrects its search:


```bash
> Entering new AgentExecutor chain...
I need to find out who the current chancellor of Germany is, as the question asks for the last chancellor. 

Action: Tavily Search  
Action Input: "current chancellor of Germany 2023"  [{'title': 'Chancellor of Germany - Wikipedia', 'url': 'https://en.wikipedia.org/wiki/Chancellor_of_Germany', 'content': 'The current officeholder is Friedrich Merz of the Christian Democratic Union, sworn in on May 6, 2025.', 'score': 0.6897452}]I have found that the current chancellor of Germany is Friedrich Merz, who was sworn in on May 6, 2025. However, since the question asks for the last chancellor, I need to verify who held the position before him.

Action: Tavily Search  
Action Input: "previous chancellor of Germany before Friedrich Merz"  [{'title': 'List of chancellors of Germany | Names & Years - Britannica', 'url': 'https://www.britannica.com/place/list-of-chancellors-of-Germany-2066807', 'content': 'Konrad Adenauer (1949–63)\n    \n   Ludwig Erhard (1963–66)\n    \n   Kurt Georg Kiesinger (1966–69)\n    \n   Willy Brandt (1969–74)\n    \n   Helmut Schmidt (1974–82)\n    \n   Helmut Kohl (1982–98)\n    \n   Gerhard Schröder (1998–2005)\n    \n   Angela Merkel (2005–21)\n    \n   Olaf Scholz (2021–25)\n    \n   Friedrich Merz (2025– )\n    \n\nThe Editors of Encyclopaedia BritannicaThis article was most recently revised and updated by Amy Tikkanen.\n\nImage 15: Britannica Chatbot logo [...] Philipp Scheidemann (1919)\n    \n   Gustav Bauer (1919–20)\n    \n   Hermann Müller (1920; 1st time)\n    \n   Konstantin Fehrenbach (1920–21)\n    \n   Joseph Wirth (1921–22)\n    \n   Wilhelm Cuno (1922–23)\n    \n   Gustav Stresemann (1923)\n    \n   Wilhelm Marx (1923–25; 1st time)\n    \n   Hans Luther (1925–26)\n    \n   Wilhelm Marx (1926–28; 2nd time)\n    \n   Hermann Müller (1928–30; 2nd time)\n    \n   Heinrich Brüning (1930–32)\n    \n   Franz von Papen (1932)\n    \n   Kurt von Schleicher (1932–33)', 'score': 0.8190992}]I have found that the last chancellor before Friedrich Merz was Olaf Scholz, who served from 2021 to 2025. 

Final Answer: The last chancellor of Germany before Friedrich Merz was Olaf Scholz, who was in office from 2021 to 2025.

> Finished chain.
```

Pretty straightforward to see how the ReAct agent works through different types of problems.

## Why I like ReAct

What I find compelling about ReAct is the transparency. You're not dealing with a black box - you can see exactly how the agent is thinking and what steps it's taking. This makes debugging easier and builds trust in the system.

Plus, with LangChain handling the heavy lifting, building these agents is way more accessible than it used to be. You can focus on defining the right tools and prompts instead of worrying about the orchestration logic.

The combination of reasoning and acting opens up so many possibilities for real-world applications. Pretty exciting stuff!