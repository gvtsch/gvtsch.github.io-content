---
tags: ['machine-learning', 'llm', 'nlp', 'python', 'statistic', 'vector-database']
author: CKe
title: 'Vectordatabases'
date: 2025-06-28
translations:
  de: "de/notes/Vectordatabases"
---

# Vector databases

**Vector databases** are a special type of database designed to efficiently store, index, and query **vector embeddings**. [[Embeddings and similarity metrics|Embeddings and similarity metrics]] are numerical representations of unstructured data such as text, images, audio files, or videos. They transform this complex data into points in a **high-dimensional space**, typically consisting of hundreds or thousands of dimensions. The core idea here is that similar objects should be **closer to each other** in this space. The degree of similarity between objects can be calculated using, for example, [[Cosine similarity|cosine similarity]] or the [[dot_product|dot product]].

## How do vector databases work?

The functioning of vector databases can be divided into three main steps: the creation of embeddings, their storage and indexing, and the execution of similarity searches.

### Embedding creation

Before unstructured data can be stored in a vector database, it must be converted into [[Embeddings and similarity metrics|Embeddings and similarity metrics]]. This process is typically performed using machine learning models, known as **embedding models**. For text data, natural language processing (NLP) models are used to transform words, sentences, or entire documents into a numerical vector. For images, convolutional neural networks (CNNs) are used to extract visual features. The result is always a high-dimensional numerical vector that mathematically captures the semantic or visual properties of the original data.

### Storage and indexing

Once the data has been converted into vectors, it is stored in the vector database. The key difference to traditional relational databases lies in the type of indexing. Vector databases use special algorithms and data structures to organize the vectors in such a way that a fast **similarity search** is possible. Common indexing methods are:

* **Approximate Nearest Neighbor (ANN)** algorithms: These algorithms, such as _HNSW (Hierarchical Navigable Small Worlds)_ or _IVFFlat (Inverted File Index with Flat Quantization)_, sacrifice minimal accuracy for significantly higher speed when searching very large data sets. They group similar vectors so that not every single vector has to be compared during a search.

### Similarity search

The main purpose of a vector database is to perform **similarity searches** efficiently. When a query is received, it is also converted into a vector (query embedding). The vector database then searches for the vectors in the index that are most similar to this query embedding. The _similarity_ is determined by distance measures in high-dimensional space:

* **[[Cosine similarity|Cosine similarity]]**: Measures the cosine of the angle between two vectors. A value close to $1$ means high similarity, close to $0$ means dissimilarity. This is particularly well suited for comparing the orientation or _direction_ of vectors, regardless of their length.
* **[[dot_product|Dot Product]]**: Calculates the sum of the products of the corresponding components of two vectors. A higher dot product indicates a higher similarity, especially if the vectors are also relevant in terms of their _length_ (magnitude).
* **Euclidean distance**: Measures the _straight line_ distance between two points in vector space. A smaller distance means higher similarity.
The result of a similarity search is typically the `k` most similar vectors (`top_k` results) along with their similarity values.

## Advantages of vector databases

Vector databases offer a number of advantages that make them indispensable for modern AI applications:

* **Efficient similarity search**: They are specially optimized for quickly finding similar data, even in extremely large and high-dimensional datasets. Traditional databases would be hopelessly overwhelmed here.
* **Scalability**: Many vector databases are designed to operate in distributed environments, enabling horizontal scaling to handle huge amounts of data.
* **Flexibility**: They can handle various types of unstructured data (text, image, audio, video) as long as it can be converted into vectors.
* **Integration**: Vector databases can be seamlessly integrated into existing AI workflows and applications, particularly as storage for retrieval-augmented generation ([[RAG]]) in LLM applications.
* **Support for AI applications**: They form the basis for implementing semantic search, recommendation systems, image recognition, and many other AI-powered features that go beyond simple keyword searches.

## Disadvantages of vector databases

Despite their advantages, vector databases also have some challenges:

* **Complexity**: Designing, implementing, and maintaining vector databases requires specialized expertise in embeddings, indexing algorithms, and metric spaces.
* **Resource consumption**: Operating vector databases can be computationally intensive, especially with large datasets and high query rates, as a lot of memory is required for the indexes.
* **Accuracy at high dimensions**: At very high dimensions, the concept of _similarity_ can become diluted (curse of dimensionality), which can affect search accuracy if the right algorithms and metrics are not used.
* **Less context for LLMs**: Vectors alone represent only the meaning, not the full, original context. For LLMs, the original data (the text passage from which the embedding originates) must be retrieved and made available to the LLM along with the similarity result. The vector database itself does not usually store the entire original content, but rather the vectors and metadata.

## Application examples

Vector databases are used in many areas, especially in connection with artificial intelligence and machine learning.

* **Semantic search**: Instead of just searching for keywords, users can search for the _meaning_ of their query. A search for _healthy food for children_ could find recipes for _nutritious meals for toddlers_, even if the exact keywords do not match.
* **Recommendation systems**: Products, movies, music, or content can be recommended to users by comparing their preferences (as vectors) with similar items.
* **Image and video recognition**: Find similar images or scenes in videos based on visual features. For example, _Find all images showing a dog on the beach._
* **Anomaly and fraud detection**: Identify unusual patterns in financial transactions or network activity by detecting deviations from normal behavior vectors.
* **Natural Language Processing (NLP)**: Enables chatbots, question-answering systems (Q&A), and retrieval-augmented generation ([[RAG]]) in LLMs by finding relevant text passages related to a user query, even when the wording varies.
* **Personalized advertising**: Ads can be personalized for users based on their interests and behaviors.
* **Genomics and drug development**: Comparison of genome sequences or molecular structures to identify similarities relevant to research.
* Etc., etc.

## Well-known vector databases

The list of vector databases is constantly growing. Here are the solutions I am most familiar with.

* **Pinecone**: A popular, cloud-native vector database known for its ease of use and scalability.
* **Chroma**: A lightweight vector database often used for local development and smaller projects, which can also be integrated into Python.
* **[[FAISS_en|FAISS]] (Facebook AI Similarity Search)**: An open-source library from Facebook AI for efficient similarity search and clustering of densely populated vectors. It is more of a library than a complete database solution.
