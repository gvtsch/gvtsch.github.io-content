---
title: BaseOutputParser
tags:
  - langchain
  - llm
  - python
  - coding
date: 2025-06-29
author: CKe
translations:
  de: "de/notes/BaseOutputParser"
---

# `LangChain`'s `BaseOutputParser`-Class

The `BaseOutputParser` class is the foundation of all output parsers in [[What is LangChain|What is LangChain]]. It defines how the raw text output from a large language model (LLM) is transformed into a useful, structured format. Every specialized parser, such as the [[StrOutputParser]] or the `JsonOutputParser`, inherits from it and implements the `parse`-method. This method takes the LLM text and converts it into, for example, a string, a JSON object, or a list.

The `BaseOutputParser` allows for seamless integration into `LangChain` chains and is crucial when custom logic is needed to process model responses. This way, raw text can be turned into manageable data.