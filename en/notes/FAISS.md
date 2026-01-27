---
title: FAISS Vector Database
tags: ['faiss', 'vector-database', 'llm', 'rag', 'machine-learning', 'nlp', 'python', 'statistics']
author: CKe
date: 2025-07-14
translations:
  de: "de/notes/FAISS"
---

# `FAISS` - Facebook AI Similarity Search

`FAISS` is a library developed by Facebook AI Research (Meta) to perform efficient similarity searches (e.g., [[Cosine similarity|cosine similarity]], [[dot_product|dot producte]]). `FAISS` is a very important tool in artificial intelligence, especially for large language models (LLMs), as it (like other tools such as Pinecone or ChromaDB) forms the basis for ([[RAG|retrieval augmented generation]]).

## What makes `FAISS` so important?

LLMs are known to be limited in their knowledge to their training data. [[RAG]] systems are used to make specific, current, or proprietary knowledge (the opposite of open-source knowledge) accessible. This is where `FAISS` comes into play:

1. **Vector embedding**: Texts (entire documents, individual paragraphs, sentences, etc.) are converted into numerical vectors by embedding models. These vectors represent the semantic meaning of the text. Similar texts have similar vectors and are close to each other in the multidimensional vector space.
2. **Efficient search**: When a user asks a question, that question or query is also vectorized. `FAISS` then uses similarity searches to quickly find the most similar document vectors in what may be a huge database. `FAISS` uses optimized algorithms and data structures to make these searches efficient.
3. **Scalability**: `FAISS` is designed to handle billions of vectors. This becomes particularly important for large databases, because large corpora of documents need to be indexed.
4. **Hardware optimization**: GPUs can be used to further accelerate the search processes.
Without an efficient vector database, searching large data sets would be far too slow to enable real-time [[RAG]]s applications.

## How does `FAISS` work?

I will now simplify the whole process considerably. Imagine you have millions of photos and want to quickly find all those that resemble a specific reference photo. You could, of course, compare each photo visually one by one. That might take a while. The situation is similar with texts and their vectors. `FAISS` proceeds as follows:

1. **Indexing**: `FAISS` uses different index types to organize the vectors. These indexes are optimized to speed up similarity searches.
   * **Flat Index**: Stores vectors directly and performs a brute force search. More accurate but slow with large amounts of data.
   * **Inverted File Index**: Clusters the vectors to limit the search to relevant clusters. Faster, but with a slight loss of accuracy.
   * **Product Quantization**: Compresses vectors to reduce memory requirements and speed up the search. Also leads to accuracy losses.
2. **Search**: When the query - the vector of the user's question - arrives, `FAISS` uses the selected index to find the `k` most similar vectors in the dataset. The _similarity_ is often measured using [[Cosine similarity|cosine similarity]].
Of course, the choice of index now depends on the requirements for speed, accuracy, and memory usage.

## `FAISS` in [[What is LangChain]]

In `LangChain`, it is integrated as a supporting vector database (or vector store). After the documents have been loaded and divided into socalled chunks, these chunks are converted into vectors (for example with `OpenAIEmbeddings`). These vectors are then passed to `FAISS` for indexing.

Let me illustrate this with an example.

```python
import os
from dotenv import load_dotenv # Import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from reportlab.pdfgen import canvas


load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found. Please set it in your .env file or as an environment variable.")

llm = OpenAI(temperature=0, api_key=api_key)

embeddings = OpenAIEmbeddings(api_key=api_key)

pdf_path = 'example_attention.pdf'

if not os.path.exists(pdf_path):
    
    print(f'Creating dummy PDF: {pdf_path}')
    
    c = canvas.Canvas(pdf_path)
    c.drawString(100, 750, 'Attention Is All You Need')
    c.drawString(100, 730, 'Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,')
    c.drawString(100, 710, 'Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.')
    c.drawString(100, 690, 'Attention is a mechanism in neural networks that allows the model,')
    c.drawString(100, 670, 'to focus on relevant parts of the input. It is a function,')
    c.drawString(100, 650, 'that maps a query and a set of key-value pairs to an output.')
    c.drawString(100, 630, 'The transformer uses attention as a core component and completely dispenses with recurrence.')
    c.save()
else:
    print(f'Use existing PDF file: {pdf_path}')

loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs) 
print(f'Number of text chunks created: {len(splits)}')

vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
print('FAISS vector database successfully created.')

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True)

conversation_chain = ConversationalRetrievalChain.from_llm(
    
llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

print('\n--- Start chat interaction ---')
user_question1 = 'What is attention?'
print(f'User: {user_question1}')
response1 = conversation_chain.invoke({'question': user_question1})
print('Bot:', response1['answer'])
user_question2 = 'Who is the author?'
print(f'\nUser: {user_question2}')
response2 = conversation_chain.invoke({'question': user_question2})
print('Bot:', response2['answer'])
print('\n--- Chat interaction ended ---')
```

### How the code implements the [[RAG]] pipeline

The code example demonstrates the basic steps of a RAG pipeline in [[What is LangChain]] and the central role of `FAISS`:

1. **Loading and splitting**: `PyPDFFoader` and `RecursiveCharacterTextSplitter` load the document and split it into manageable text chunks.
2. **Vectorization**: `OpenAIEmbeddings` converts these text chunks into vectors or [[Embeddings and similarity metrics|embeddings]].
3. **Indexing (`FAISS`)**: `FAISS.from_document` indexes the vectors. Here, `FAISS` organizes the database to speed up later searches.
4. **Retrieval and generation**: `ConversationalRetrievalChain` uses `vectorstore.as_retriever()` to retrieve the most similar vectors (the most relevant text chunks) from `FAISS` for a user query (e.g. _'What is attention?'_) and provide them to the LLM for generating the response.

### Persistence of the `FAISS` index: Saving and loading

In the above example, the `FAISS` index is created in memory. For practical applications, it is important to save the index to avoid time-consuming vectorization of the documents each time the program is run. This is actually very easy to do.

#### Save

```python
vectorstore.save_local('faiss_index_directory')
```

#### Load
```python
# Important: The embedding model must be passed when loading
loaded_vectorstore = FAISS.load_local('faiss_index_directory', embeddings)
```

### Output of the example

```
Creating dummy PDF: example_attention.pdf
Number of text chunks created: 1
FAISS vector database successfully created.

--- Start chat interaction ---
User: What is attention?
Bot:  Attention is a mechanism in neural networks that allows the model to focus on relevant parts of the input.

User: Who is the author?
Bot:  The authors of the mechanism of attention in neural networks are Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, ■ukasz Kaiser, and Illia Polosukhin.

--- Chat interaction ended ---
```

The output shows that a dummy PDF file was created and that the information was searched for and found in this file.
 
## Summary

`FAISS` is a crucial component in modern LLM applications. It enables LLMs to quickly and accurately access a constantly growing amount of specific information, for example, to process current or proprietary knowledge.
