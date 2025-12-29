---
tags: ["python", 'coding']
author: CKe
title: Regex
date: 2025-06-30

---

# Regex - What are "regular expressions"?

Regular expressions in Python are very powerful, but can be a little confusing at first. One reason to note it down for myself.

## What are regular expressions?

You can think of regex as a special language that can be used to define text patterns. This allows you to search for patterns in texts instead of exact words, for example. Instead of searching for exact email addresses or phone numbers in texts, you can search for _all phone numbers_ or _all email addresses_. This will become even clearer in an example.

They are used for tasks such as searching, replacing, or validating text.

## Why are they so useful?

* **Complex searches**: Search for and find text that does not exactly match the search query, but corresponds to a pattern.
* **Data extraction**: You can extract specific information from large amounts of text (such as all phone numbers).
* **Data validation**: Check whether an input matches a specific format (for example, whether an email address is valid).
* **Text editing**: Find and replace patterns in text.

## Regex in Python: `re` module

Python has a built-in module called `re` that allows you to work with regular expressions.

The most important functions are probably:

* **`re.search()`**: Searches for the **first** occurrence of a pattern in a string and returns a so-called match object if it is found, otherwise `None`.
* **`re.match()`**: Searches for a pattern **at the beginning** of a string. Works similarly to `re.search()`, but is somewhat more limited.
* **`re.findall()`**: Finds all **occurrences** of a pattern in a string and returns a list of strings.
* **`re.sub()`**: Replaces occurrences of a pattern with another string.
* **`re.split()`**: Splits a string based on a pattern.
* **`re.compile()`**: Compiles a regex pattern to improve performance when using the same pattern multiple times.

## Basic regex syntax

Below, I list some of the most common elements you will find in regex patterns.

### Literal characters

Most characters match themselves.

* `a`: Searches for the letter `a`
* `Hello`: Searches for the entire string `Hello`

### Metacharacters - Characters that have a special meaning

* `.`: Matches any single character (except newline)
* `a.b` matches `abc`, `a2l`, ...
* `*`: Matches none or multiple repetitions of the preceding character or group
* `a*` matches ` `, `a`, `aa`, `abc`, ...
* `ab*c` matches `ac`, `abc`, `abbbc`, ...
* `+`: Matches one or more repetitions of the preceding character or group
* `a+`: Matches `a`, `aa`, `aaa`, ... but **not** `ac` like `a*`
  * `ab+c`: Matches `abc`, `abbc`, ... but **not** `ac` like `ab*c`.
* `?`: Matches none or one repetition of the preceding character or group (makes it optional).
* `colou?r`: Matches `color` and `colour`, covering the different spellings in this case.
* `[]`: Matches a single character listed in the brackets.
* `[abc]`: Matches `a`, `b`, or `c`
* `[0-9]`: Matches any digit from `0` to `9`
* `[a-z]`: Matches any lowercase letter
  * `[A-Z]`: Matches any uppercase letter
  * `[a-zA-Z0-9]`: Matches any alphanumeric character
* `[^abc]`: Matches any character that is **not** `a`, `b`, or `c`
* `\`: Removes metacharacters so that they are treated as literals. Also used for special sequences.
  * `\.`: Matches an actual period (`.` would not find a period as described above)
  * `\$`: Matches any actual dollar sign

### Special, frequently used sequences

* `\d`: Finds/matches any digit `0` to `9` (analogous to `[0-9]`)
* `\D`: Matches any non-digit character
* `\w`: Matches any character (alphanumeric characters and underscore) (analogous to `[a-zA-Z0-9_]`)
* `\W`: Matches any non-character
* `\s`: Matches any whitespace character (including spaces, tabs, newlines, etc.)
* `\S`: Matches any non-whitespace character
* `\b`: Matches a word boundary. This is the position between a character (`\w`) and a non-character (`\W`), or between a word character and the beginning/end of a string.  
* `\bHund\b`: Matches `Hund` as in `Der Hund frisst`, but not `hundemüde`.
* `\B`: Matches a non-word boundary, is the opposite of `\b`, and matches anywhere `\b` would not match.
  * `\BHund\B`: Does not match `Hund`, but does match `emüde` in `Hundemüde`.
* `^`: Matches the beginning of a string
* `^Hallo`: Matches `Hallo Welt`, but not `Oh, Hallo Welt`
* `$`: Matches the end of a string.
  * `World$`: Matches `Hello World` but not `The world is beautiful`
* `|`: Logical OR
* `cat|dog`: Matches `cat` or `dog`
* `()`: Grouping patterns. Allows operations to be applied to a group or parts of the match to be extracted.
  * `(ab)+`: Matches `ab`, `abab`, `ababab` , ...

## Examples

I think an example will make it clearer.

```python
import re
text = "Hello world, I am a developer. My email is test@example.com and my phone number is 12-345-6789."

# Example 1: re.search() - Find the first occurrence
match = re.search(r"Developer", text)
if match:
    print(f"Found: '{match.group()}' at position {match.start()} to {match.end()}")
    # match.group() returns the string found
    # match.start() returns the start index of the match 
    # match.end() returns the end index of the match
else:
    print("Not found.")

# Example 2: re.findall() - Find all occurrences of a pattern
numbers = re.findall(r"\d+", text) # \d+ matches one or more digits
print(f"All numbers: {numbers}")

# Example 3: re.sub() - Replace patterns
new_text = re.sub(r"Developer", 'Programmer', text)
print(f"Text after replacement: {new_text}")

# Example 4: Find email address (more complex pattern)
# r"..."is a "raw string" so that backslashes do not have to be double-escaped
email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
email_match = re.search(email_pattern, text)
if email_match:
    print(f"Email found: {email_match.group()}")

# Example 5: Find phone number (with groups)
# (\d{2}) - Group 1: 2 digits
# -? - Optional hyphen
# (\d{3}) - Group 2: 3 digits
# -? - Optional hyphen
# (\d{4}) - Group 3: 4 digits
phone_pattern = r"(\d{2})-?(\d{3})-?(\d{4})"
phone_match = re.search(phone_pattern, text)
if phone_match:
    print(f"Phone number found: {phone_match.group()}")
    print(f"Area code: {phone_match.group(1)}")
    print(f"Middle part: {phone_match.group(2)}")
    print(f"End part: {phone_match.group(3)}")
```

The output:
```bash
Found: 'developer' at position 20 to 29
All numbers: ['12', '345', '6789']
Text after replacement: Hello world, I am a developer. My email is test@example.com and my phone number is 12-345-6789.
Email found: test@example.com
Phone number found: 12-345-6789
Area code: 12
Middle part: 345
End part: 6789
```

## Summary 

Regular expressions are a powerful tool for searchung, extracting, validating and manipulationg text in python. By mastering the basic syntax and understanding how to use the `re` module, you can efficiently handle a wide variety of text processing tasks. While regex can seem complex at first, practice and experimentation will make it an invaluable part of your coding toolkit.

### Key points

* Regex allows flexible pattern matching beyond exact text.
* The `re` module provides essential functions for working with regex in python.
* Understanding metacharacters and special sequences is crucial for building effective patterns. 
* Practical examples help clarify how regex works in real-world scenarios.