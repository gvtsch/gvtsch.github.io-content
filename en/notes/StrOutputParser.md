---
tags: ["langchain", "llm", "python", 'coding']
author: CKe
title: StrOutputParser
date: 2025-06-29
translations:
  de: "de/notes/StrOutputParser"
---

# What is LangChain's StrOutputParser()?

The `StrOutputParser()` is an implementation of the [[BaseOutputParser|BaseOutputParser]] class in `LangChain`. The main function of the parser is to return the output passed to it as a simple string. What sounds trivial at first glance is crucial, as the outputs within `LangChain` can be complex.

## Why do wi need the parser?

You can use the parser in one of the following scenarios, for example.

* **Standard string output**: If an LLM or chain is supposed to deliver a pure text string as a result and you do not need or expect any further structuring (e.g., JSON, lists, etc.). This is probably the most common use case.
* **Endpoint of a chain**: Often, the parser is the last link in a `LangChain` Expression Language Chain (LCEL). The parser then takes the raw output of the previous step (e.g., the response from an LLM) and ensures that it is passed on to the user or the next process as a simple string.
* **Readability**: Even though LLM output is often already text, explicitly adding the `StrOutputParser()` can clarify the intention that pure string output is expected. It is also a form of type safety in the `LangChain` context. 
* **Combination with other parsers (implicit)**: When using more complex parsers such as `JsonOutputParser` or `CommaSeparatedListOutputParser`, the `StrOutputParser` is often implicitly the starting point. The LLM output is initially a string, which is then further processed by the specialized parsers.
 
Many more parsers can be found in the [LangChain documentation](https://python.langchain.com/api_reference/core/output_parsers.html).

## How do you use the `StrOutputParser`?

The `StrOutputParser()` has a method called `parse()`, which expects a string as input and then returns this string unchanged. In the context of chains, you can use it as follows. The example may not be executable because you still need to specify an API key. You can do this using [[Keyring|Keyring]] or [[dotenv]].

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import keyring


def main():

    OPENAI_API_KEY = keyring.get_password("OPENAI_API_KEY", "default_user")
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Please set it in your keyring.")


    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant.'),
        ('user', '{input}')
    ])

    llm = ChatOpenAI(
        model='gpt-4o', 
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"input": "What is the capital of France?"})
    print(response)

if __name__ == "__main__":
    main()
```

In the example, `prompt` generates a prompt string. `llm` receives this prompt and returns a model response (an `AIMessage` object containing a string). `StrOutputParser()` takes this string from the `AIMessage` object and returns it as a pure Python string.

The output:

```bash
The capital of France is Paris.
```

## What else is important?

* **Default behavior**: In many simple `LangChain` chains, especially if you only have one LLM at the end, `StrOutputParser()` is often the implicit default parser if you don't explicitly specify another one. `LangChain` attempts to convert the LLM's output to a string if no specific parser is available. However, it is usually specified anyway to clarify the intention.
* **Comparison with other parsers**: The `StrOutputParser()` contrasts with other output parsers that need to extract a specific structure from the LLM output: 
  * **JsonOutputParser**: Expects JSON-formatted text and parses it into a Python dictionary or list.
  * **CommaSeparatedListOutputParser**: Expects a comma-separated list and parses it into a Python list of strings.
  * etc. etc.
* **Error Handling**: Because it returns the input unchanged, `StrOutputParser()` has no special error handling for incorrectly formatted outputs. If the LLM response is not what you expect, it is still simply returned as a string. Error handling would then have to take place afterwards.
* **Simplicity is Key**: The strength of `StrOutputParser()` lies in its simplicity. This parser is the building block for all scenarios where you simply need text as output without additional transformations.

**In summary**, `StrOutputParser()` is a fundamental, indispensable tool in LangChain for handling raw text output from LLMs and ensuring that chains deliver clean string results.