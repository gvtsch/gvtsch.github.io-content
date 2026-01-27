---
title: Prompt Strategies and Techniques
tags: ["python", "machine-learning", "llm", "nlp"]
author: CKe
date: 2025-07-20
translations:
  de: "de/notes/Prompt-Techniques"
---

# Prompt Strategies and Techniques

Designing effective prompts is crucial for getting high-quality, accurate, and useful responses from large language models (LLMs). Below are some of the most important and practical prompt engineering techniques, each with a clear explanation and a concise example.

## Zero-Shot Prompting

**What it is:**  
Ask the model to perform a task without providing any examples.

**Example:**  
```text
Classify the sentiment of this sentence as positive or negative: "The service was excellent."
```

## One-Shot Prompting

**What it is:**  
Provide a single example to show the model the desired format or logic before giving the actual task.

**Example:**  
```text
Q: What is the capital of France?  
A: Paris.

Q: What is the largest planet in our solar system?  
A:
```

## Few-Shot Prompting

**What it is:**  
Give the model several input-output examples to help it learn the pattern or structure you want.

**Example:**  
```text
Q: The car is red.  
Negation: The car is not red.

Q: I am happy.  
Negation: I am not happy.

Q: The sky is blue.  
Negation:
```

## Chain-of-Thought (CoT) Prompting

**What it is:**  
Encourage the model to reason step by step, making its thought process explicit. This is especially useful for complex or multi-step problems.

**Example:**  
```text
Question: If there are 3 apples and you buy 2 more, how many apples do you have in total?  
Let's think step by step.
```

## Tree-of-Thought (ToT) Prompting

**What it is:**  
Ask the model to explore multiple possible solution paths, compare them, and select the best one. This is useful for open-ended or strategic tasks.

**Example:**  
```text
Task: Suggest ways to reduce electricity costs in a household.
- List at least three different approaches.
- For each, briefly discuss pros and cons.
- Choose the most effective approach and explain your reasoning.
```

## Persona Prompting

**What it is:**  
Assign the model a specific role, profession, or personality to influence its tone, style, and expertise.

**Example:**  
```text
You are a physics professor. Explain the theory of relativity in simple terms.
```

## Interview Prompting

**What it is:**  
Structure the interaction as an interview or Q&A, with clear roles for user and model.

**Example:**  
```text
Interviewer: What experience do you have with Python programming?
Candidate:
```
or
```text
Candidate: Please ask me three questions about my experience with Python.
Interviewer:
```

## Self-Consistency Prompting

**What it is:**  
Ask the model to solve the same problem multiple times, each with independent reasoning, and then select the most common answer. This increases reliability for reasoning tasks.

*Note: While self-consistency improves robustness by aggregating multiple independent answers, it differs from Tree-of-Thought prompting, which explores and compares different solution paths within a single reasoning process.*

**Example:**  
```text
Solve this math problem three times, showing your reasoning each time:  
If a bag contains 16 apples and three-quarters are red, how many are green?  
Afterward, state the answer that appears most frequently.
```

## Generated Knowledge Prompting

**What it is:**  
First, have the model generate relevant facts or background knowledge, then use that information to answer the main question.

**Example:**  
```text
Step 1: List three facts about photosynthesis.  
Step 2: Using these facts, explain why plants need sunlight.
```

## Constraint-Based Prompting

**What it is:**  
Specify strict rules or formats that the model's output must follow.

**Example:**  
```text
Summarize the following text in no more than 20 words.
```
or
```text
Return the answer as a JSON object with fields "name" and "age".
```

## Iterative Prompting / Prompt Refinement

**What it is:**  
Start with a basic prompt, review the model's output, and then refine your prompt based on the results to improve the answer.

**Example:**  
```text
Write a product description for a new smartphone.
```
*After reviewing the output:*
```text
Now make it shorter and highlight the battery life.
```

---

These strategies are not exhaustive, but they provide a strong foundation for effective prompt engineering with LLMs. Experimenting with and combining these techniques can help you get the most out of language models in a wide range of