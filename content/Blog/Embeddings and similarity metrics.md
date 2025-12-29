---
title: Embeddings and similarity metrics
date: 2025-07-22
tags: [machine-learning, deep-learning, python, llm, nlp, transformer, tokenization, embedding]     
toc: true
---

# Word and Token Embeddings and it's similarity metrics

**Embeddings** - a core concept that drives modern artificial intelligence in language processing. If computers are to understand language, they must convert text into a format they can work with. This is where embeddings come in: they are the key to making words and their meanings _comprehensible_ to machines.

## What are embeddings?

Embeddings are numerical vector representations of words, subwords, or other text units, which we refer to as [[Tokenization|tokens]]. Their main purpose is to capture the semantic (meaning-related) and syntactic (grammatical) relationship between these tokens in a high-dimensional space. This means that words with similar meanings or functions are closer together in the vector space.

Imagine if you could represent each word not just as a string of characters, but as a point in a huge mathematical space. If _king_ and _queen_ are close together in this space and _apple_ is far away, then this reflects their respective similarities in meaning.
To illustrate how this works, I would like to use a simplified example.

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Example vectors
king = np.array([1, 1, 0, 0])
comparison_vectors = {
    'Queen': np.array([1, 1, 1, 0]),
    'Queen_2': np.array([2, 2, 0.1, 0]),
    'Queen_3': np.array([3, 3, 0.2, 0]),
    'apple': np.array([0, 0, 10, 10])
}
```

In this section of code, I have defined a few hypothetical example vectors that represent embeddings for the words _king_, _queen_, and _apple_. In addition, variations of the vector for _queen_ have been created to illustrate the effects of vector size on distance measures. These are really just highly simplified example vectors to illustrate the principle.
In practice, these word embeddings have significantly more dimensions. Modern language models such as _GPT-3_, for example, use embeddings with $12288$ dimensions. These complex embeddings are not created manually, but are learned during the training of language models by analyzing patterns and relationships in huge amounts of text.

### Measuring the similarity of embeddings

Various mathematical metrics are used to measure the semantic similarity between vectors. [[Cosine similarity|Cosine similarity]] and the [[Scalar Product|dot product]] are particularly important in the context of natural language processing (NLP) and large language models (LLMs), as they evaluate the direction of vectors and thus robustly capture semantic similarity. In addition, there are distance measures such as Euclidean or Manhattan distance, which measure the _distance_ between vectors in space.

Here, we will compare [[Cosine similarity|Cosine similarity]], the scalar product together with Euclidean and Manhattan distance in an example to illustrate their differences and relationships in the evaluation of semantic proximity.

```python
def vector_metrics (vector_a: np.ndarray, vector_b: np.ndarray) -> tuple[float, float, float, float]:

    if not isinstance(vector_a, np.ndarray) or not isinstance(vector_b, np.ndarray):
        raise ValueError("vector_a and vector_b must be NumPy arrays.")
    if vector_a.shape != vector_b.shape:
        raise ValueError("vector_a and vector_b must have the same shape.")

    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    cosine = dot_product / (norm_a * norm_b) if (norm_a * norm_b) != 0 else 0
    Euclidean = np.linalg.norm(vector_a - vector_b)
    manhattan = np.linalg.norm(vector_a - vector_b, ord=1)

    return dot_product, cosine, Euclidean, manhattan
```

The function `vector_metrics` calculates the metrics for two given vectors. The following function `compare_vectors` uses `vector_metrics` to compare a reference vector with a series of comparison vectors from a dictionary and output the results in a table.

```python
def compare_vectors(reference_vector: np.ndarray, comparison_vectors_dict: dict[str, np.ndarray]) -> pd.DataFrame:
    if not isinstance(reference_vector, np.ndarray):
        raise ValueError("reference_vector must be a NumPy array.")
    if not isinstance(comparison_vectors_dict, dict):
        raise ValueError("comparison_vectors_dict must be a dictionary.")

    results = []

    for name, vector in comparison_vectors_dict.items():
        if not isinstance(vector, np.ndarray):
            raise ValueError(f'The value for '{name}' in comparison_vectors_dict must be a NumPy array.')

        dot_product, cosine, Euclidean, manhattan = \
            vector_metrics(reference_vector, vector)

        results.append({
            'Comparison word': name,
            'Dot product': dot_product,
            'Cosine similarity': cosine,
            'Euclidean distance': Euclidean,
            'Manhattan distance': Manhattan
        })

    df = pd.DataFrame (results)
    df = df.set_index("Comparison word")

    return df
```

In the following, we compare the vector `King` with the other defined example vectors and output the calculated similarity values.

```python
if __name__ == "__main__":
    df_results = compare_vectors(king, comparison_vectors)
    print(df_results)
```

```bash
                 Dot product  Cosine similarity  Euclidean distance  Manhattan distance
Comparison word
Queen                    2.0           0.816497            1.000000                 1.0
Queen_2                  4.0           0.999376            1.417745                 2.1
Queen_3                  6.0           0.998891            2.835489                 4.2
apple                    0.0           0.000000           14.212670                22.0
```

### Interpretation of metrics in the context of embeddings

The table illustrates the different properties of the metrics when comparing the reference vector for queen with the other words:

#### Dot product
* Measures the _match_ between the directions while also taking into account the length (magnitude) of the vectors. A larger [[Scalar Product|dot product]] indicates a stronger alignment in the same direction and/or longer vectors.
* The higher the value, the more similar the vectors are.
* In the example: `Queen_3` has the highest [[Scalar Product|dot product]] with `King` at $6.0$, followed by `Queen_2` ($4.0$) and `Queen` ($2.0$). As expected, this is a consequence of their increasing vector length, while their semantic direction to `King` remains very similar. `Apple` has a [[Scalar Product|dot product]] of $0$ because the vectors are orthogonal to each other.

#### Cosine similarity
* Measures the angle between vectors and thus purely reflects the semantic direction or similarity, regardless of their length.
* High values (close to $1$) = very similar meaning (vectors point in almost the same direction).
* Low values (close to $0$ or $-1$) = low or opposite similarity.
* In the example: `Queen_2` ($0.999$) and `Queen_3` ($0.998$) are most similar to `King`, as their vectors point almost perfectly in the same direction. `Queen` ($0.816$) is still similar, but the higher entry in the third dimension results in a slightly larger angle. `Apple` is completely dissimilar ($0.000$) because its vector has a completely different direction. 
* Cosine similarity is the preferred metric for LLMs because it robustly captures semantic similarity.

#### Euclidean and Manhattan distance
* Measure the geometric _distance_ between two vectors in vector space.
* Small values indicate high similarity (vectors are close to each other).
* Large values indicate low similarity (vectors are far apart).
* They are sensitive to vector size. Despite their high semantic similarity (high cosine values), `Queen_2` and `Queen_3` show a greater distance to `King` than `Queen`. This is because their vectors are simply _longer_ and therefore further apart, even though their direction is very similar.
* These distance measures are less suitable for quantifying purely semantic relationships in embeddings, as the meaning of words in LLMs is often represented by the direction of their embeddings rather than their length.
 
## Conclusion

The [[Scalar Product|dot product]] is closely related to [[Cosine similarity|Cosine similarity]] and is a key component in the attention mechanism of transformers, where it is used to calculate the similarity (scores) between query and key vectors. It is an efficient way to measure the _match_ of vectors pointing in similar directions. [[Cosine similarity|Cosine similarity]] is the more robust choice when it comes to comparing pure semantic meaning, as it normalizes the vector length.
