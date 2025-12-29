---
tags: ['machine-learning', 'llm', 'nlp', 'python', 'statistics', 'vector-database', 'rag', 'coding']
author: CKe
title: 'RAG'
date: 2025-07-21
---

# Retrieval Augmented Generation

## What is RAG?

RAG, or Retrieval-Augmented Generation, is a crucial approach in the field of natural language processing (NLP). It is a technique that enables large language models (LLMs) such as ChatGPT or Gemini to access information that is not contained in their original training data.

The main reason for developing RAG is to overcome the weaknesses of traditional LLMs. These models often tend to _hallucinate_ (give factually incorrect or invented answers) or are limited by the data they were trained on. RAG solves this problem by enabling LLMs to access an up-to-date and verifiable external knowledge base.

# How does RAG work?

RAG operates in two basic phases:

1. **Retrieval (information retrieval)**: When you ask a question, the RAG system first searches an external knowledge database for the most relevant information. This knowledge base can be anything from a collection of documents to current news to company-specific data stored in a so-called vector database. The system identifies the most relevant _snippets_ or documents that are useful for answering your question. For more information on how to measure the relevance of theses _snippets_, have a look at [[Cosine similarity|cosine similarity]]
2. **Augmentation and generation**: In the second step, this retrieved information is added to the original user query. The LLM now receives an enriched context that contains all the necessary facts. The model uses this _augmented_ prompt to generate a precise and well-founded response.

## The advantages of RAG

The result of RAG is an AI response that is not only fluent and coherent, but above all factually correct and up-to-date. By accessing verifiable sources, the model minimizes the risk of hallucinations and significantly increases the trustworthiness of the system. RAG makes LLMs reliable tools for specific, knowledge-based tasks and also for searching through your own files, for example (although I would advise against providing OpenAI, Gemini, etc. with this information).