---
tags: ["machine-learning", "llm", "nlp", "python", "statistics"]
author: CKe
title: Scalar Product
date: 2025-06-29
---

# Scalar Product (Dot Product)

The **scalar product**, also known as the **dot product**, is a fundamental operation in linear algebra that takes two vectors and returns a single scalar number (hence _scalar product_). This number provides information about the relationship between the vectors, in particular their **relative orientation** and **size**.

## What does the scalar product measure?

The scalar product can be interpreted in two ways:

1.  **Algebraic definition:** It is the sum of the products of the corresponding components of two vectors. For two vectors $\vec{A} = (a_1, a_2, \dots, a_n)$ and $\vec{B} = (b_1, b_2, \dots, b_n)$, the scalar product is:

    $$    
    \vec{A} \cdot \vec{B} = a_1 b_1 + a_2 b_2 + \dots + a_n b_n = \sum_{i=1}^{n} a_i b_i
    $$

2.  **Geometric definition:** It is the product of the lengths (magnitudes) of the vectors multiplied by the cosine of the angle between them.
    
    $$
    \vec{A} \cdot \vec{B} = ||\vec{A}|| \cdot ||\vec{B}|| \cdot \cos(\theta)
    $$
    
    Where:
    * $||\vec{A}||$ and $||\vec{B}||$ are the lengths (magnitudes) of the vectors $\vec{A}$ and $\vec{B}$.
    * $\theta$ is the angle between the vectors $\vec{A}$ and $\vec{B}$.
 
This geometric interpretation is particularly meaningful:
* $\theta = 0^\circ$ (vectors point in the same direction): $\cos(\theta) = 1$. The dot product is at it's maximum and positive. **Meaning**: The vectors are perfectly aligned, their similarity and combined effect are strongest.
* $\theta = 90^\circ$ (vectors are orthogonal): $\cos(\theta) = 0$. The dot product is 0. **Meaning**: The vectors are perpendicular. They have no influence on each other in the direction of the other. 
* $\theta = 180^\circ$ (vectors point in opposite directions): $\cos(\theta) = -1$. The dot product is at it's minimum and negative. **Meaning**: The vectors are completely opposed. Their similarity is negative and their combined effect cancels out.

## Connection to [[Cosine similarity]]

The dot product is closely related to cosine similarity. In fact, cosine similarity is nothing more than the **normalized dot product**. If we solve the geometric formula for $\cos(\theta)$, we obtain exactly the formula for cosine similarity:

$$
\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{||\vec{A}|| \cdot ||\vec{B}||}
$$

This means that the dot product provides the cosine similarity _including_ the lengths of the vectors. If the vectors are already normalized to length 1 (unit vectors), then the dot product is directly equal to the cosine similarity.

## Areas of application

The dot product is widely used in many areas of science and technology, including:

* **Physics:** Calculation of work ($W = \vec{F} \cdot \vec{s}$), power, or flux of fields.
* **Computer graphics:** Determination of lighting effects (e.g., how bright a surface is based on the angle to the light source) and collision detection.
* **Machine learning and neural networks**:
  * **In neural networks**: Dot products are the core operation in many layers where input vectors are multiplied by weight matrices.
  * **Attention mechanisms** (e.g., in transformers): The dot product is used to calculate the similarity between _query_ and _key_ vectors, which determines how strongly different parts of the input are related to each other and how much _attention_ they should pay to each other. See [[The Transformer Architecture I|Transformers Part 1]]
  * **Efficiency:** It is a very efficient operation that can be calculated quickly in high-dimensional spaces.
  
The dot product is therefore an important building block that provides theoretical insights into vector relationships and forms the basis for many practical algorithms in modern data processing and also in areas of machine learning.
