---
date: 2025-07-14
tags: ["llm", "machine-learning", "nlp", "langchain", "agent"]
author: CKe
title: "Map-Reduce Summarizer"
---

# LangChain's Map-Reduce Summarizer

I used the Map-Reduce Summarizer as part of a multi-agent framework, but I wasn't really satisfied with it because I didn't know exactly what the summarizer does.

## What is the Map-Reduce Summarizer?

The **Map-Reduce Summarizer** is a method in LangChain that is designed to summarize very long documents or texts. It uses the classic [[Map-Reduce-Paradigma|Map-Reduce paradigm]] and is particularly effective when the input text is too long to be processed in a single call by a Large Language Model (LLM).

## How does the Map-Reduce Summarizer work?

The process is divided into two main phases.

### 1. Map phase (partial summary)

* **Chunking:** The long input text is divided into smaller, manageable segments (chunks). Each segment is small enough to comply with the token limits of the LLM.
* **Individual summarization:** The LLM is called for each of these chunks to create a separate summary. 

The result is several small summaries, each representing a part of the original text.

### 2. Reduce phase (final summary)

* **Consolidation:** All partial summaries from the map phase are collected.
* **Final summary**: The LLM receives these shorter texts and uses them to create a single, coherent, and final summary of the entire document.

## Advantages and disadvantages

### Advantages:

* **Handling long texts**: Can process documents that exceed the token limits of an LLM.
* **Parallelizability**: The summarization of individual chunks in the map phase can potentially be parallelized.
* **Detailed capture**: Since every part of the text is summarized, theoretically more information is retained than with other methods.

### Disadvantages:

* **Cost**: Requires multiple LLM calls, which can be more expensive.
* **Latency**: The total processing time is longer due to the multiple calls.
* **Potential loss of information across chunk boundaries**: Relationships that extend across the boundaries of individual chunks can be more difficult to capture.

## When should you use the Map-Reduce Summarizer?

This method is suitable when you need to summarize very long documents and the accuracy of the summary across the entire text is important, even if this comes at a higher cost and longer latency.
