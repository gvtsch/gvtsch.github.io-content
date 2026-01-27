---
tags:
  - machine-learning
  - llm
  - nlp
  - python
  - statistics
  - rag
  - coding
author: CKe
title: Cosine similarity
date: 2025-06-29
translations:
  de: "de/notes/Cosine-similarity"
---

# Cosine similarity: A measure of semantic similarity

**Cosine similarity** is a widely used metric for determining the similarity between two non-zero vectors in an inner product space. It is particularly popular in **natural language processing (NLP)** and **large language models (LLMs)**, as it is excellent for measuring the **semantic similarity** of texts.

## What does cosine similarity measure?

Basically, cosine similarity measures the **cosine of the angle between two vectors**.

* A cosine value of **1** means that the angle is 0 degrees, i.e. the vectors point in exactly the same direction. This indicates **maximum similarity**.
* A value of **0** means that the angle is 90 degrees (the vectors are orthogonal). There is **no similarity** in their orientation.
* A value of **-1** means that the angle is 180 degrees, meaning the vectors point in exactly opposite directions. This indicates **maximum dissimilarity**.

It is important to note that cosine similarity is **independent of the magnitude (length)** of the vectors. It only considers their **orientation in vector space**. This is crucial because in many applications (such as text embeddings), the length of a vector does not necessarily play a role in its semantic meaning, only its direction. This is where it differs from the [[Scalar Product]].

## Formula for cosine similarity

The cosine similarity of two vectors $\vec{A}$ and $\vec{B}$ is calculated as:

$$
\text{Cosine similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{||\vec{A}|| \cdot ||\vec{B}||}
$$

Where:
* $\vec{A} \cdot \vec{B}$ is the **scalar product (dot product)** of the vectors $\vec{A}$ and $\vec{B}$.
* $||\vec{A}||$ is the **Euclidean norm (length)** of the vector $\vec{A}$.
* $||\vec{B}||$ is the **Euclidean norm (length)** of the vector $\vec{B}$.


## Areas of application in LLMs and NLP

Cosine similarity is a key component in modern NLP systems:

* **Semantic search and [[RAG|Retrieval Augmented Generation (RAG)]]**: When you ask an LLM a question, the question is converted into a vector (embedding). This embedding is then compared with the embeddings in a knowledge base (often [[Vectordatabases|vector databases]]) to find the most relevant documents or passages. Cosine similarity helps to identify the _closest_ or most semantically similar content.
* **Recommendation systems**: Products, movies, or articles that have similar content descriptions or user profiles can be recommended based on the cosine similarity of their embeddings.
* **Text clustering and classification**: Texts with high cosine similarity are grouped together (clustered) or assigned to specific categories, as they are likely to cover a similar topic.
* **Plagiarism detection:** By comparing the cosine similarity of text segments, similarity can be measured and possible plagiarism identified.

The ability of cosine similarity to capture the _meaning_ or _theme_ of texts through vector alignment makes it an indispensable tool in the age of AI and machine learning.
