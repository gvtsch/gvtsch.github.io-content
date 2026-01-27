---
title: Callbacks
tags:
  - python
  - coding
date: 2025-06-02
author: CKe
translations:
  de: "de/notes/Callbacks"
---

# Callbacks
- The concept is a widely used programming technique.
- It allows you to pass a function as an argument to another function.
- The passed function is then called at a later point in time (e.g., in response to a specific event or action).

## Basic principles of a callback function
1. Definition, e.g. `my_callback(message)`
   * Callback functions are basically regular functions that perform a specific task. However, they are not called directly, but passed to another function.
1. Passing the callback function, e.g. to `process_data(data, callback)`
   * The function that receives the callback stores it and calls it again at a later time, often when a certain condition is met or an event occurs.
1. Calling the callback function `callback(f"Processed item: {result}")`
   * The callback function is called within the function that received it, e.g., to display a message or process data.
  
## Example

```python
def my_callback(message):
    print(f"Callback received message: {message}")

def process_data(data, callback):
    for item in data:
        # Process the item
        result = item * 2
        # Call the callback function and pass the result
        callback(f"Processed item: {result}")

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    process_data(data, my_callback)
```

The output should look like this:

```bash
Callback received message: Processed item: 2
Callback received message: Processed item: 4
Callback received message: Processed item: 6
Callback received message: Processed item: 8
Callback received message: Processed item: 10
```

