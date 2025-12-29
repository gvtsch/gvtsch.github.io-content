---
title: Agents at their limits
tags:
- langchain
- langgraph
- python
- machine-learning
- coding
- llm
- nlp
date: 2025-10-05
toc: true
---

## The limits of the LangChain chain

In [[ReAct]], we talked about a versatile agent, coded it, and put it to use. [[ReAct]] agents are great for smaller workflows or tool calls and are also suitable for prototypes.

However, they quickly reach their limits when it comes to more complex tasks. Their architecture is a linear chain with a simple, limited loop. As soon as tasks such as self-correction, collaboration between multiple agents, or complex decision logic are required, this structure is no longer sufficient.

To overcome these challenges, it is worth switching to a so-called graph structure.

## An example: A [[ReAct]] agent reaches its limits

Let's imagine we want to build an agent that tests Python code, corrects errors independently, and, depending on the result, either performs an analysis or escalates the issue to a human. This is not possible with a classic [[ReAct]] agent in LangChain – the limitations quickly become apparent.

**LangChain ReAct agent (classic)**

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

def test_code(code: str) -> str:
    try:
        local_vars = {}
        exec(code, {}, local_vars)
        return "Success"
    except Exception as e:
        return f"Error: {e}"

def analyze_output(output: str) -> str:
    return f"Analysis: {output}"

def escalate_to_human(output: str) -> str:
    return f"Escalated to human: {output}"

tools = [
    Tool(name="CodeTester", func=test_code, description="Tests Python code."),
    Tool(name="Analyzer", func=analyze_output, description="Analyzes output."),
    Tool(name="Escalator", func=escalate_to_human, description="Escalates to human.")
]

llm = OpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

# The agent tries once, cannot retry, cannot branch in code, no shared state
result = agent.run("Test this code and analyze the output or escalate if there is an error:\nprint(x)")
print(result)
```

I think most of the code is understandable to most people. However, a digression on the `test_code` method could provide a little more clarity.

**What is happening, step by step**

1. `local_vars = {}`: Creates an empty dictionary for local variables
2. `exec(code, {}, local_vars)`: Executes the passed Python code
    * `code` = the code to be tested (in our case `"print(x)"`)
* `{}` = global variables (empty)
* `local_vars` = local variables (empty)
1. **If successful**: Returns `"Success"`
2. **If there is an error**: Catches the exception and returns the error message

When the agent tests `print(x)`, the following happens:

* Python tries to find the variable `x`
* However, `x` is not defined anywhere (neither in global nor local variables)
* Python throws a `NameError: name 'x' is not defined`
* The `test_code` function catches this error
* It returns: `"Error: name 'x' is not defined"`

**The problem with the classic agent**

The agent receives this error message, but cannot:

* Automatically try again
* Repair the code (e.g., add `x = 42`)
* Explicitly decide: "Error → escalate" or "Success → analyze"

It simply stops after the first error because the entire retry and branching logic is hidden in the prompt, not in the code.

And that is the core of the problem: Classic [[What is LangChain|LangChain]] agents cannot respond to errors in a structured and traceable manner.

### Conclusion

This example shows that classic [[ReAct]] agents quickly reach their limits when it comes to more complex tasks. They cannot perform iterative error correction, branch flexibly, or manage a shared status.

## Three key limitations of classic agents

### 1. Lack of explicit loop control and self-correction

Classic agent frameworks such as [[What is LangChain|LangChain]] do not offer true programmatic iteration or self-correction. This means that the agent cannot autonomously correct errors or refine results in a targeted manner. The standard agent loop is designed to arrive at a final answer as quickly as possible. Internal iterations are not provided for.

For example, if a tool call fails because a code test returns an error, there is no clear instruction in the code to repeat the error correction cycle. The entire responsibility for corrections lies with the LLM and is hidden in the prompt, which is error-prone and difficult to trace. Debugging is also difficult because the loop and correction logic is hidden in the prompt rather than in the code.

### 2. Complex branching and black box logic

Another problem with classic [[What is LangChain|LangChain]] chains is the handling of complex branches. Often, the workflow must behave differently depending on the result of an intermediate step. In [[What is LangChain|LangChain]], however, the entire branching logic must be formulated in the prompt ("If result X, then tool A, otherwise tool B"). This makes the logic difficult to maintain, error-prone, and inflexible—a small change in the prompt can affect the entire workflow.

There is no clear mechanism in the code to define the next step based on an intermediate result. Control lies in the LLM and not in the code. As complexity increases, such chains quickly become confusing and difficult to maintain. Graph structures, on the other hand, are modular, easier to extend, and allow explicit branching.

**Typical scenarios:** Human-in-the-loop, escalation paths, or the selection between different special tools can hardly be mapped cleanly with classic chains.

### 3. State management and multi-agent collaboration

Especially in projects where multiple agents work together, the classic chain reaches its limits. [[LangChain]] is primarily stateless, which is insufficient for long-term projects or collaborative scenarios. There is no native, clean way to dynamically transfer control between specialized agents (e.g., research agent ↔ analysis agent).
Agents often need a shared, changeable project status (shared state) that they can all access and update before control is passed on. The linear chain does not offer a native mechanism for this. In addition, individual nodes in graph workflows can be parallelized or optimized in a targeted manner. This is not possible with linear chains.

### Further challenges

- **Integration of human-in-the-loop:** Graphs allow human intervention (for example review steps) to be explicitly incorporated.
- **Transparency and debugging**: In classic [[What is LangChain|LangChain]] chains, it is often difficult to understand the exact sequence of events and sources of errors because much of the logic is contained in the prompt rather than in the code. With [[LangGraph]], on the other hand, the sequence of events is explicitly visible in the code and easier to test.

The above mentioned challenges are precisely the challenges I face. In a framework I am working on, we have implemented several agents linearly. In some places, however, it would be helpful if we could integrate a human-in-the-loop or branches and loops. This is also one of the reasons why I am currently working with [[LangGraph]] and why we will convert our framework to [[LangGraph]].

## Conclusion: The solution is the graph

All these problems have a common cause: the linear, static nature of classic [[What is LangChain|LangChain]] agents. What we need is an architecture that can branch flexibly, perform loops, and manage a common state.

This is exactly where [[LangGraph]] comes in – an extension of [[What is LangChain|LangChain]] based on graphs. Instead of a rigid (learned a new english word here, yay!) chain, [[LangGraph]] allows you to explicitly define nodes and edges. This enables true loops, clean branches, and a shared state that all agents can use.

Curious? Then take a closer look at [[LangGraph]] or contact me directly.
