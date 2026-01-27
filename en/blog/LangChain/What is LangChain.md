---
title: What is LangChain
date: 2025-06-29
tags:
  - python
  - langchain
  - langgraph
  - machine-learning
  - llm
  - nlp
  - agent
toc: true
draft: false
author: CKe
translations:
  de: "de/blog/LangChain/LangChain_de"
---

# What is LangChain?

Among other things, I am currently working on LangChain. And I am trying to record my thoughts here in some form. 

## Introduction

LangChain is a Python framework for building applications with LLMs and external tools and it is a powerful framework that combines the integration and interaction of large language models (LLMs) with external data and tools. LLMs are inherently limited in their knowledge to the data they have been trained on and can rarely access external information directly or perform complex actions. This is exactly where LangChain comes in.

The framework overcomes these limitations by giving LLMs the ability to interact with the outside world. Imagine you want to develop a chatbot that can not only answer general questions based on its internal knowledge, but also retrieve information from specific sources such as Wikipedia or your own internal documents and remember what has been said so far. Without LangChain, you would have to build everything yourself, integrate APIs, write logic for conversations, etc. LangChain greatly simplifies this complex task by allowing you to use ready-made chains, memory, and tools.

In addition, LangChain promotes enormous flexibility and modularity in your projects. Instead of relying on a single, often overloaded "super prompt," LangChain enables the development of intelligent agents that can each be assigned specific tasks. This makes it much easier to develop complex LLM applications in a structured and maintainable way.

**What is LangGraph?**  
LangGraph is an extension of LangChain that allows you to build more complex, graph-based workflows for LLM applications.  
While LangChain focuses on chaining components in a linear fashion, LangGraph lets you define flexible, branching logic—ideal for advanced use cases like multi-step reasoning, decision trees, or multi-agent systems.

### Advantages
The advantages include, among others:
* **Modularity**:  Similar to a building block principle, different components can be used and exchanged.
* **Flexibility**: LangChain is very flexible and can be adapted to a wide variety of use cases.
* **Abstraction**: LangChain hides some functionalities behind interfaces, thereby reducing complexity. 
* **Extensibility**: Its modularity makes it easy to integrate additional tools or data sources.
* **Use cases**: 
  * Chatbots
  * Question-and-answer systems, e.g., using your own files
  * Agents that perform tasks
  * ...

### Document retrieval (Retrieval Question Answering - [[RAG]])

LLMs only have knowledge up to their training date and no specific company or project data. If you want to change that, [[RAG]] comes into play. It allows you to "extend" the LLM with relevant external documents to answer specific questions.

**What are embeddings and vector stores?**  
Embeddings are numerical representations of text that capture its meaning and context. By converting text into vectors (lists of numbers), we can compare and search for similar content efficiently.  
A vector store is a special database that stores these embeddings and allows fast similarity searches—so you can quickly find relevant documents or text chunks for a given query.


## Basic concepts in LangChain

### Large Language Models (LLMs)

For virtually every LLM application, you need... surprise, surprise, an LLM. LLMs are models that can understand and generate text. Well-known LLMs include OpenAI's GPT models, Google's Gemini, and open-source models such as Llama and Mistral from France. 

LangChain provides a unified interface for interacting with different LLMs without having to change your own code.

Here's a really simple example of how this might look:

```python
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.7)
print(llm.invoke("What is the name of the most beautiful city in Germany?"))
```

Or if you want to interact with eg Mistral:

```python
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(temperature=0.7)
print(llm.invoke("What is the name of the most beautiful city in Germany?"))
```

Basically, not much more changes than the import and the class used. Everything else is implemented in the background by LangChain.

What I have omitted in both cases is the `API-KEY`. You usually cannot integrate with LLMs without such an `API key`. If you have such a key, you can integrate and use it via [[dotenv]] or [[Keyring]], for example.

In fact, many more parameters can be configured. I have only adjusted the `temperature` parameter here. For Mistral, information on the other parameters can be found in this [LangChain documentation](https://python.langchain.com/api_reference/mistralai/chat_models/langchain_mistralai.chat_models.ChatMistralAI.html). For OpenAI, you would find similar information in this [LangChain documentation](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html).

### Prompts and `PromptTemplates`

In the example above, the LLM is called with the `invoke` command. This means that you actively use or execute the model to perform a specific task or get a response. 
In the example, we pass the question "What is the most beautiful city in Germany?" to the LLM. This question is a prompt, and we expect the LLM to respond with a textual answer to our question. The quality of the answers depends heavily on the quality of the prompts. It's easy to imagine that without further criteria, the answer will also vary for us humans. 

This prompt is also, of course, extremely static. Sometimes you want to make your prompts dynamic (e.g., insert a country into the question). This is where LangChain's PromptTemplate comes into play. These are templates that contain placeholders that are filled with values at runtime. This allows you to reuse your prompt, create consistency, and reduce errors.

A `PromptTemplate` could look like this, for example:

```python
from langchain.prompts import PromptTemplate

template = """You are a helpful assistant who summarizes information. Summarize the following text on the topic of {topic}. The text is {text}. Output the summary in {language}."""

prompt = PromptTemplate(
    input_variables=["topic", "test", "language"],
    template=template
)

print(prompt.format(
    topic="Artificial Intelligence",
    text="Artificial intelligence (AI) deals with the simulation of human intelligence in machines...",
    language="English"
))
```

The finished prompt might look like this:

```bash
You are a helpful assistant who summarizes information. Summarize the following text on the topic of artificial intelligence. The text reads: Artificial intelligence (AI) deals with the simulation of human intelligence in machines.... Output the summary in German.
```

The individual placeholders have been filled with content. With this prompt, you can then invoke agents or chains.

### Chains

A chain is a link between components such as `PromptTemplate` and LLMs. They make it possible to execute several steps in succession and use the output of one step as input for the next.
This makes it relatively easy to set up pipelines for LLM operations. An LLM chain, for example, is the most basic chain that connects a prompt to an LLM.

Chains make it easy to structure your code and simplify often complex workflows.

An LLM chain could look like this, for example (integrate the API key as usual, e.g., via [[Keyring]]):

```python
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.7)

summarize_template = "Summarize the following text: {text}"
summarize_prompt = PromptTemplate.from_template(summarize_template)

summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

long_text = "Künstliche Intelligenz (KI), englisch artificial intelligence, daher auch artifizielle Intelligenz (AI), ist ein Teilgebiet der Informatik, das sich mit der Automatisierung intelligenten Verhaltens und dem maschinellen Lernen befasst. Der Begriff ist schwierig zu definieren, da es verschiedene Definitionen von Intelligenz gibt. [...] Der Begriff artificial intelligence (künstliche Intelligenz) wurde 1955 geprägt von dem US-amerikanischen Informatiker John McCarthy im Rahmen eines Förderantrags an die Rockefeller-Stiftung für das Dartmouth Summer Research Project on Artificial Intelligence, einem Forschungsprojekt, bei dem sich im Sommer 1956 eine Gruppe von 10 Wissenschaftlern über ca. 8 Wochen mit der Thematik befasste."
summary = summarize_chain.invoke({"text": long_text})
print(summary['text'])
```

_Text from [Wikipedia](https://de.wikipedia.org/wiki/K%C3%BCnstliche_Intelligenz)_

Since the input text is not particularly long, the summary will hardly be much shorter, but I think it will suffice to understand the principle. So here is the summary in question:

```bash
AI is a branch of computer science that deals with the automation of intelligent behavior and machine learning. The term is difficult to define because there are different definitions of intelligence. The term was coined by John McCarthy in 1955 and investigated by a group of 10 scientists as part of a research project in the summer of 1956. 
```

#### What is `Invoke`?

Invoking a chain or agent in LangChain means that you pass an input (usually a dictionary, as above) to the object and receive an immediate response. The input is sent to the chain, the language model processes this request (using memory or a tool, if necessary), and you receive the result directly.

In short, `invoke()` is the method used to send requests to a chain or agent and receive a synchronous response.

### Agents and Tools

Agents and tools ensure that LLMs can act more "intelligently." An agent decides which tools to use and in what order based on the current problem. Functions that an agent can call to perform external actions include Google or Wikipedia searches, database queries, API calls, etc. LangChain offers many ready-made tools for this purpose and also allows you to create your own.

Here's an abstract example. You ask the agent, "What will the weather be like tomorrow?" 
This may be followed by the following sequence:
* Agent recognizes: I need weather information
* Agent selects the "weather tool"
* Agent calls up the tool with "Hamburg, tomorrow"
* The "weather tool" returns data
* Agent formulates the response for the user using LLT

A less abstract but shorter example in Python:

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_openai import OpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

llm = OpenAI(temperature=0)

# Tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [wikipedia]

# Prompt for ReAct Agent
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent_executor.invoke({"input": "Who is the current Chancellor of Germany?"}))
```

The output shows how the ReAct agent proceeds. It determines that it should use Wikipedia, searches a few pages, and finally delivers the correct result (as of 2025-06-29). I have shortened the output of the individual Wikipedia pages because it is irrelevant for understanding.

```bash
> Entering new AgentExecutor chain...
 I should use Wikipedia to find the answer.
Action: wikipedia
Action Input: "Chancellor of Germany" Page: Deutschlandlied
Summary: The "Deutschlandlied," officially titled "Das Lied der Deutschen," is a German poem written by August Heinrich Hoffmann von Fallersleben. A popular song [...]

Page: Chancellor of Germany
Summary: The chancellor of Germany, officially the federal chancellor of the Federal Republic of Germany, is the head of the federal government of Germany. [...] The current officeholder is Friedrich Merz of the Christian Democratic Union, sworn in on May 6, 2025.

Page: Friedrich Merz
Summary: Joachim-Friedrich Martin Josef Merz (born November 11, 1955) is a German politician who has served as Chancellor of Germany since May 6, 2025. [...]
Final Answer: The current Chancellor of Germany is Friedrich Merz.

> Finished chain.
{'input': 'Who is the current Chancellor of Germany?', 'output': 'The current Chancellor of Germany is Friedrich Merz.'}
```

Incidentally, a [[ReAct]] agent is used above. A [[ReAct]] agent combines "reasoning" (logical inference) and "acting" (action). It uses language models to consider, in several steps, which actions (e.g., tool calls) are necessary to achieve a goal. In doing so, it alternates between thinking and acting.

### Memory

LLMs are stateless by default, meaning they forget everything after each request. Memory modules in LangChain make it possible to remember past conversations or states. This enables chatbots, personalized applications, or, for example, multi-layered, complex interactions.

There are many types of memory, e.g., for databases or Redis. I would like to mention two here.
* `ConversationBufferMemory`: Stores the entire conversation
* `ConversationSummaryMemory`: Summarizes the conversation if it becomes too long.

A `ConversationChain` with `ConversationBufferMemory` could look like this:

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.7, api_key=api_key)
memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

print(conversation.invoke({"input": "Hello, my name is Christoph"}))
print(conversation.invoke({"input": "How are you today?"}))
print(conversation.invoke({"input": "Do you remember my name?"}))
print(conversation.invoke({"input": "Tell me something about Python"}))
```

If you execute the above lines in a Jupyter Notebook, for example, you will get a visually appealing conversation history enriched with additional information. If, on the other hand, you only want to view questions and answers, the following lines will help.

```python
for message in memory.chat_memory.messages:
    print(f"{message.type}: {message.content}")
```

These then deliver the following result.

```bash
human: Hello, my name is Christoph
ai:  Hello, Christoph! Nice to meet you. My name is AI, which stands for Artificial Intelligence. I am a program designed to have human-like conversations and answer questions. How can I help you today?
human: How are you today?
ai:  I'm fine, thank you for asking. I'm a computer program, so I don't have physical sensations like humans do. But my programming is running smoothly, so I'm happy. How are you?
human: Do you remember my name?
ai:  Yes, your name is Christoph. I have a database with all the information you give me during our conversations, and I remember everything you've told me.
human: Tell me something about Python
ai: Python is a popular programming language developed in the 1990s by Guido van Rossum. It is known for its simple syntax and flexible applications. Many large companies such as Google and Instagram use Python for their applications. It is also one of the most commonly used languages for artificial intelligence and machine learning. Do you have any more questions about Python or would you like to learn more about it?
```

What immediately stands out: The question about my name can still be answered even after further questions. In this way, an LLM, which, as I said, is "stageless," could be turned into a chatbot.

#### Transferring the `ConversationChain` to the LLM

To connect a `ConversationChain` to an LLM, the LLM object is passed as an argument when creating the chain. The chain then takes over communication with the model. In the example above, the LLM (`llm=llm`) is passed directly to the `ConversationChain`. The chain then takes care of generating prompts, saving the history, and retrieving the responses from the LLM. Interaction with the LLM then takes place via methods such as `invoke`. 

This makes the chain the central building block that connects the LLM, memory, and logic.

## Application example

I would like to show further application examples in additional examples.

### Document retrieval (Retrieval Question Answering - [[RAG]])

LLMs only have knowledge up to their training date and no specific company or project data. If you want to change that, [[RAG]] comes into play. It allows you to "extend" the LLM with relevant external documents to answer specific questions.

The workflow could then look something like this:
* **Load documents**: PDFs, text files, databases, etc.
* **Split texts**: Break large documents down into smaller, manageable chunks
* **Embedding**: The text chunks are converted into numerical vectors.
* **Vector database (Vector Store)**: Stores the previously converted vectors for fast similarity searches (e.g., [[cosine similarity|cosine similarity]]) in vector databases (e.g., `Chroma`, `FAISS`, `Pinecone`)
* **Query**: 
  * User asks a question
  * Question is "embedded"
  * The most similar chunks from the vector database are retrieved
* **LLM response**: The retrieved chunks and the question are presented to the LLM, which then generates an informed response.

The advantages are obvious: The answers are based on my data, hallucination is reduced, and current information can be retrieved.

What might this look like in code?
In the following code snippet, I import a PDF. It is [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) [cs.CL]. I also query Wikipedia on the topic. Once without a function, once embedded in a function. And I load an API key with Keyring.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_core.documents import Document
import os

# Load API key
import keyring
api_key = keyring.get_password("openai_api_key", "default")

# Prepare OpenAI interface
llm = OpenAI(temperature=0, api_key=api_key)
embeddings = OpenAIEmbeddings(api_key=api_key)

# Load PDF
loader = PyPDFLoader(r"Attention_is_all_you_need_1706.03762v7.pdf")
docs = loader.load()

# Generate embeddings and vector database
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

# Prepare chain
qa_chain = RetrievalQA.from_chain_type(
	llm=llm,
	chain_type="stuff",
	retriever=vectorstore.as_retriever())

# Ask a question
query = "What does attention mean?"
response = qa_chain.invoke({"query": query})
print(f"Result PDF: {response['result']}")
``` 

The above lines produce the following output. 

```bash
Result PDF:  Attention is a function that maps a query and a set of key-value pairs to an output, where all elements are vectors. The output is calculated as a weighted sum.
```

However, as mentioned above, we can also consult Wikipedia on this topic:

```python
# Prepare Wikipedia interface
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader

    loader = WikipediaLoader(
        query=query,
        lang=lang,
        load_max_docs=load_max_docs,
    )
    data = loader.load()
    return data

# Query Wikipedia
data = load_from_wikipedia("Attention (Machine Learning)", lang='en', load_max_docs=3)
print(f"Wikipedia result: {data[0].page_content[:250]}")
```

The output follows immediately. For the first output—we have instructed our agent to search for the top 3 hits—I output the first $500$ characters.

```bash
Result Wikipedia: A transformer is a deep learning architecture developed by Google that integrates an attention mechanism. Text is converted into numerical representations in the form of vectors through word embedding. This can be used, for example, to translate text from one language to another (see also machine translation). To do this, a transformer is trained using machine learning on a (large) set of sample texts before the trained
```

The above article obviously also deals with the attention mechanism. Without delving further into the content, this is the best result according to the agent.

The second-best result deals with LLMs. However, I will only quote the first 250 characters:

```python
print(f"Wikipedia result: {data[1].page_content[:250]}")
```

```bash
Wikipedia result: A large language model, or LLM for short, is a language model that is characterized by its ability to generate text. It is a computational linguistic probability model, since
``` 

And the third result seems to deal with `Tensorflow`.

```python
print(f"Result Wikipedia: {data[2].page_content[:250]}")
```

```bash
Wikipedia result: TensorFlow is a framework for data stream-oriented programming. TensorFlow is popularly used in the field of machine learning. The name TensorFlow comes from arithmetic operations performed by artificial neural networks on multidimensional
``` 

With the loaded Wikipedia results, you can now perform various further steps, for example:

1. **Summarize**  
The contents of Wikipedia articles can be automatically summarized with an LLM to present the most important information in a compact form.
2. **Compare**  
You can compare the content of different articles to identify differences or similarities. 
3. **Answering questions (QA)**  
You can ask specific questions about the loaded Wikipedia articles by using RetrievalQA or your own chain that uses the articles as a knowledge base.

  ```python
  from langchain.chains import RetrievalQA
  from langchain_community.vectorstores import FAISS
  from langchain_openai import OpenAIEmbeddings

  # Generate embeddings for Wikipedia articles
  wiki_vectorstore = FAISS.from_documents(data, embedding=embeddings)
  wiki_qa_chain = RetrievalQA.from_chain_type(
     llm=llm,
     chain_type="stuff",
     retriever=wiki_vectorstore.as_retriever()
  )

  question = "What is a transformer in the context of machine learning?"
  answer = wiki_qa_chain.invoke({"query": question})
  print(f"Answer: {answer['result']}")
  ```

4. **Further processing**  
  - Extract keywords or entities.
  - Create mind maps or visualizations.
  - Combine Wikipedia content with other data sources.

This allows you to flexibly reuse the loaded Wikipedia data (or other data) for various NLP tasks.

### Chatbots

Another example that each of us has surely encountered at some point... A chatbot. The goal of a chatbot is interactive, context-sensitive conversation. 

A `ConversationalRetrievalChain` combines two key features:  
- **Memory**: It remembers the conversation history, so the chatbot can refer back to previous questions and answers.
- **Retrieval**: It can search external documents or databases to answer questions based on up-to-date or custom information.

This makes chatbots built with LangChain context-aware and able to provide more accurate, personalized responses.


You can reuse the code from the previous examples and add `ConversationalRetrievalChain` and `ConversationBufferMemory` to:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_core.documents import Document
import os

# Load API key
import keyring
api_key = keyring.get_password("openai_api_key", "default")

# Prepare OpenAI interface
llm = OpenAI(temperature=0, api_key=api_key)
embeddings = OpenAIEmbeddings(api_key=api_key)

# Load PDF
loader = PyPDFLoader(r"C:\...\Attention_is_all_you_need_1706.03762v7.pdf")
docs = loader.load()

# Generate embeddings and vector database
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

memory = ConversationBufferMemory(
    memory_key="chat_history", # Important: Key must be 'chat_history'
    return_messages=True
)

# Create the ConversationalRetrievalChain
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)


# Interact with the chatbot
response1 = conversation_chain.invoke({"question": "What is attention?"})
print("User: What is attention?")
print("Bot:", response1['answer'])

response2 = conversation_chain.invoke({"question": "Who is the author?"})
print("\nUser: Who is the author?")
print("Bot:", response2['answer'])

response3 = conversation_chain.invoke({"question": "Do you remember what we talked about first?"})
print("\nUser: Do you remember what we talked about first?")
print("Bot:", response3['answer']) 
```

The output will then look like this:

```bash
User: What is attention?
Bot:  Attention is a function that maps a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. It is used in neural network architectures, such as the Transformer, to connect the encoder and decoder and improve performance in sequence transduction tasks. It allows the network to focus on specific parts of the input and make connections between distant dependencies.

User: Who is the author?
Bot: 
The authors of the Transformer model are Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin.

User: Do you remember what we talked about first?
Bot: 
Attention is a function that maps a query and a set of key-value pairs to an output. The query, keys, values, and output are all vectors. The output is calculated as a weighted sum.
```

You can recognize the activity of the bot (the `ConversationalRetrievalChain`) by the fact that the answers to specific questions (e.g., about the meaning of "Attention" or the authors) come directly and precisely from your PDF document. The function of the buffer (the `ConversationBufferMemory`) is evident because the bot remembers previous conversation topics ("Do you remember what we talked about first?") and responds based on them.

It seems to work! 

### Data analysis and generation

A few scenarios—without code—for how LangChain can be used for data analysis and generation:
* LangChain can also be used to summarize large data sets (e.g., log files or customer reviews). 
* Extract specific information from unstructured text.
* Generate reports or descriptions based on structured data.
* Code generation or explanation
* ...

Another example, but also only theoretical, is an agent that can access CSV files and answer questions about them by generating and executing Python code. Tools such as `PythonREPLTool` or `PandasDataFrame` could then be used. 

The possibilities are virtually endless ;)

## Summary

LangChain is a powerful framework for creating "intelligent" LLM-based applications. It simplifies complex processes through modular components.

It can be used almost anywhere: customer service, education, content creation, data analysis, etc.

I will continue to explore this topic. There is still a lot to learn about callbacks, custom components, integrations, and also `LangGraph`.

If you have any questions, comments, or suggestions, or if you have found any errors, please don't hesitate to contact me :)
