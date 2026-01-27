---
title: Python Logging-Module
date: 2025-10-07
tags: ["python", "coding"]
toc: True
draft: false
author: CKe
translations:
  de: "de/blog/Logging_de"
---

# Logging in Python

I used to work with `print` statements a lot to stay informed about what my scripts were currently doing. Depending on how much information I wanted to receive, I set a so-called `verbosity` level. These were things like `VERBOSE_INFO = 1` or `VERBOSE_DEBUG = 2`. I would then set/pass this level when calling my script, and a typical `print` looked something like this:

```python
if self._verbosity >= VERBOSE_DEBUG:
    print(f"DEBUG: {<debug_info>}")
else:
    print(f"INFO: {<info>}")
```

or

```python
print(f"DEBUG: {<debug_info>}" if self._verbosity >= VERBOSE_DEBUG else f"INFO: {<info>}")
```

And basically, this worked quite well. You could run it as a [[callback]] and make it as complicated as you wanted. And because I was comfortable with it, I only stumbled upon the `logging` module by accident. And that's what this is about now...

## Why Logging instead of Print Statements?

After taking a closer look at the `logging` module, I have to admit: It's significantly more elegant than my homemade verbosity solution. The module offers much more flexibility and control over outputs without having to handle all the logic yourself.

## Basic Configuration

```python
import logging

# Basis-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

That's it. These few lines create a logger with:
- Timestamp for each message
- Log level for categorization
- Module name for source identification

## Understanding Log Levels

Python's logging system comes with ready-made levels:

- `DEBUG`: Detailed information for debugging
- `INFO`: General information about program flow
- `WARNING`: Hints about potential problems
- `ERROR`: Errors that affect the program
- `CRITICAL`: Severe errors

This is basically what I used to do with my `VERBOSE_DEBUG = 2` and `VERBOSE_INFO = 1`, just standardized.

## Practical Example

This is what it looks like in practice:

```python
def prepare_titanic_data(file_path='train.csv', test_size=0.2, random_state=42):
    logger.info("Loading data...")
    df = pd.read_csv(file_path)
    X = df.drop('survived', axis=1)
    y = df['survived']
    
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info("Applying preprocessing...")
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    logger.info("Preprocessing completed")
    return results
```

Much cleaner than my if-else constructs from before.

## What I particularly like

* **Flexibility**: You can set the log level at program startup without changing the code – this was possible with my old solution too, but with the logging module it's more standardized.
* **Less code**: No need for custom verbosity logic anymore.
* **Standard**: Every Python developer understands it immediately.

## Setting Log Levels

```python
# Nur Warnungen und Fehler
logging.basicConfig(level=logging.WARNING)

# Alles anzeigen (auch DEBUG)
logging.basicConfig(level=logging.DEBUG)

# In Datei schreiben
logging.basicConfig(filename='app.log', level=logging.INFO)
```

This is significantly easier than my old verbosity parameters.

## What I learned

Previously, I solved it with my own logic:

```python
print("Data loaded")  # Immer sichtbar
if self._verbosity >= VERBOSE_DEBUG:
    print(f"DEBUG: Processing {len(data)} rows...")  # Nur bei hohem Level
```

Now I also use the logging module:

```python
logger.info("Loading data...")  # Klarer Prozess-Schritt
logger.debug(f"Processing {len(data)} rows...")  # Automatisch filterbar
```

## Conclusion

The logging module does exactly what I used to build manually, just better and more standardized. If I had known this earlier, I could have saved myself some homemade solutions. But that's how you learn. 