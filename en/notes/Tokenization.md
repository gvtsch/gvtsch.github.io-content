---
title: Tokenization
date: 2025-07-22
tags: [machine-learning, python, llm, nlp, transformer, tokenization, embedding]     
toc: true
translations:
  de: "de/notes/Tokenization"
---

# Tokenization - How computers "read" language

Before large language models (LLMs) such as ChatGPT can understand or generate text, it must be converted into a format that the computer can work with. This is where tokenization comes in – an important first step in natural language processing (NLP).

## What is tokenization?

Tokenization is the process of breaking down a continuous text into smaller units called tokens. A token can be a single word, a punctuation mark, part of a word (subword), or even a single character. The exact type of tokens depends on the chosen tokenization strategy.

Imagine you have a long sentence. For a computer, this is initially just a sequence of characters. Through tokenization, we transform this sentence into an ordered list of "building blocks" that can then be further processed.

## And what is the point of all this?

Computers do not understand text in the same way that we humans do. They need a numerical representation. Tokenization breaks the text down into discrete units. These tokens can then be converted into so-called [[Embeddings and similarity metrics|embeddings]]. [[Embeddings and similarity metrics|Embeddings and similarity metrics]] are numerical vector representations of the text snippets or tokens that NLP models use to process complex language patterns.

Tokenization is crucial because it...

* **... prepares text for models**: Language models do not operate directly with raw text. Tokens are the numerical inputs they need.
* **... reduces complexity**: Instead of processing an infinite number of possible character combinations, models work with a limited number of tokens.
* **... preserves meaning**: Good tokenizers try to preserve the semantic and syntactic integrity of the language as much as possible.
* **... enables efficiency**: Segmenting text into manageable units makes processing by the model more efficient.

## Why different types of tokens?

Why are texts tokenized differently? This depends heavily on the task and the language model used. The following examples and extreme cases should illustrate that there are reasons for the different strategies.

### Word tokenization
The simplest form, in which the text is divided into words based on spaces and punctuation marks. For example, `Hello World!` becomes `["Hello", 'World', "!"]`.
  * **Advantages**: Intuitive, tokens are easy to understand.
  * **Disadvantages**: Handling unknown words (out-of-vocabulary - OOV), large vocabularies, morphemes (word stems, prefixes, suffixes) are not taken into account.

### Character tokenization
Each character becomes a token.
  * **Advantages**: Smallest possible vocabulary size, no OOV problems
  * **Disadvantages**: Very long sequences, loses semantic information at the word level, requires more complex models to learn longer dependencies.

### Subword tokenization
This is the most commonly used method in modern NLP models (such as transformer architectures like BERT, GPT, etc.). Here, words are broken down into smaller, frequently occurring subwords. Examples of algorithms are [Byte Pair Encoding (BPE)](https://huggingface.co/learn/llm-course/chapter6/5), [WordPiece](https://huggingface.co/learn/llm-course/chapter6/6) or [Unigram](https://huggingface.co/learn/llm-course/chapter6/7) Language Model.

The core of these methods lies in a clever mechanism that can be understood as iterative merging or substitution. Algorithms such as BPE typically start with a vocabulary consisting of individual characters. They then search the text for the most frequently occurring pair of adjacent characters (or already formed subwords). This pair is merged into a new, single token and the vocabulary is expanded accordingly. Subsequently, all occurrences of this pair in the text are replaced (substituted) by the new, merged token. This process is repeated until a predefined vocabulary size is reached or no more frequent pairs are found. In this way, tokens are created that can range from individual characters to entire words.

* **Advantages**:
  * **Reduces OOV problems**: Rare and unknown words can be composed of known subwords. For example, `unknownword` could be tokenized as `["unknown", "word"]`. You can test this out [here](https://platform.openai.com/tokenizer). There, `unknownword` becomes exactly that ;)
  * **Optimized memory requirements and efficiency**: By combining frequent character combinations into individual subword tokens, the total number of tokens in a text sequence is often significantly reduced. This results in shorter input sequences for the model, which in turn minimizes computing and memory requirements during training and inference. A smaller but effective vocabulary is more memory-efficient than a huge word vocabulary.
  * **Compromise between word and character tokenization**: Smaller vocabularies than with word tokenization, but more meaningful units than with character tokenization.
  * **Handling morphemes**: Can recognize and use common morphemes. For example, `running` and `runs` could become `["run", "##ning"], ['run', "##s"]`
* **Disadvantages**: Tokens are not always intuitive words, and segmentation is model-dependent.

## Important aspects of tokenization

When talking about tokenization, especially in the context of modern language models, there are some terms you should be familiar with.

* **Vocabulary**: A list of all unique tokens that the tokenizer knows and can use. Each token in the vocabulary is assigned a unique numerical ID.
* **Mapping text to IDs**: The tokenizer gradually converts the raw text into a sequence of its identified tokens, and these tokens are then converted into their corresponding numerical IDs.
* **Special tokens**: Models often use special tokens for specific internal purposes that add additional structure or information to the text:
  
  * `[CLS]`: Classification token (in BERT, often the first token in a sequence whose embedding is used for classification tasks).
  * `[SEP]`: Separator token to separate different segments or sentences within an input sequence.
  * `[PAD]`: Padding token to bring all sequences in a batch to a uniform length. This is necessary for efficient processing in neural networks.
  * `[UNK]`: Unknown token, a placeholder for words or subwords that are not included in the tokenizer's vocabulary.

## Example in Python with `tiktoken`

`tiktoken` is a fast open-source tokenization library from OpenAI that is used for models such as `GPT-3` or `GPT-4`. The library implements a variant of byte pair encoding.
It can be installed with `pip install tiktoken`. 

```python
import tiktoken

def main():
    encoding = tiktoken.get_encoding("cl100k_base")

    text_example = "Hello world! Tokenization is great."

    print(f"Original text: '{text_example}'")
    print("-" * 30)

    token_ids = encoding.encode(text_example)

    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")
    print("-" * 30)

    decoded_text = encoding.decode(token_ids)

    print(f"Decoded text: '{decoded_text}'")
    print("-" * 30)

    print("Individual tokens:")
    for token_id in token_ids:
        token_str = encoding.decode([token_id]) # Decode each ID individually
        print(f"  ID: {token_id}, token: '{token_str}'")

if __name__ == "__main__":
    main()
```

```bash
Original text: 'Hello world! Tokenization is great.'
------------------------------
Token IDs: [9906, 1917, 0, 9857, 2065, 374, 2294, 13]
Number of tokens: 8
------------------------------
Decoded text: 'Hello world! Tokenization is great.'
------------------------------
Individual tokens:
  ID: 9906, token: 'Hello'
  ID: 1917, token: ' world'
  ID: 0, token: '!'
  ID: 9857, token: ' Token'
  ID: 2065, token: 'ization'
  ID: 374, token: ' is'
  ID: 2294, token: ' great'
  ID: 13, token: '.'
```

### Interpretation of the example

The output shows how the input text is broken down into individual tokens and then reassembled.

* **Original text and decoded text**: As expected, the texts are identical after encoding (converting to IDs) and decoding (converting back to text). This shows that no information has been lost.
* **Token IDs and number of tokens**: The sample text was broken down into 8 tokens and each token was given a unique ID.
* **Individual tokens**:
  * `Hello` and `!` are independent tokens.
  * ` world` begins with a space. This is typical for subword tokenization such as BPE. Spaces are often combined with the following word to optimize the number of tokens and facilitate the reconstruction of the original text.
  * The word `Tokenization` is split into ` Token` and `ization`. This is a good example of subword tokenization: when a word is used, it is split into more common syllables or parts of words. This helps to reduce the vocabulary and allows the model to process even unknown or rare words by interpreting their known subword components.
  * ` is`, ` great`, and `.` follow similar patterns, with the words being tokenized with a preceding space.

The example illustrates the key concept of how large, modern language models do not always view text as whole words, but rather divide it into smaller, statistically optimized units to maximize efficiency and understanding.
