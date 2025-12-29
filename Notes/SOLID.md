---
tags: ["python", "coding"]
author: CKe
title: SOLID
date: 2025-09-28
---

SOLID is a set of design principles for Object-Oriented Programming (OOP) that helps developers create software that is more maintainable, flexible, and scalable.

The acronym S.O.L.I.D. stands for five core principles, which are highly relevant when programming with Python classes.

Here is a breakdown of each principle and its application in Python.

## S - Single Responsibility Principle (SRP)

This principle states that a class should have only one reason to change. In simpler terms, a class or module should have only one job or responsibility.

* **Goal**: To prevent "God Classes" that do too much.
* **Python Application/Example**: If you have a class that handles for example both data processing and logging/reporting, you should split it into two separate classes: one for processing and one for logging. This makes each class smaller, easier to understand, and less likely to break when one responsibility changes.
  ```python
  class DataProcessor:
      def process(self, data):
          # process data
          pass

  class Logger:
      def log(self, message):
          print(message)
  ```

  Here, data processing and logging are separated into two classes.

## O - Open/Closed Principle (OCP)

This principle states that software entities (classes, modules, functions) should be open for extension, but closed for modification.

* **Goal**: You should be able to add new functionality without changing the existing, tested code.
* **Python Application/Example**: You achieve this through inheritance and abstraction. Instead of modifying an existing class to add a new feature, you create a new class that extends the existing one or implements a shared Abstract Base Class (ABC).
  ```python
  class Shape:
      def area(self):
          raise NotImplementedError

  class Circle(Shape):
      def __init__(self, radius):
          self.radius = radius
      def area(self):
          return 3.14 * self.radius ** 2

  class Square(Shape):
      def __init__(self, side):
          self.side = side
      def area(self):
          return self.side ** 2
  ```
  You can add new shapes by creating new subclasses, without modifying existing code.


## L - Liskov Substitution Principle (LSP)

This principle states that objects of a derived class must be able to substitute for objects of their base class without altering the correctness of the program.

* **Goal**: Ensures that inheritance is used correctly and that a child class adheres to the contract of its parent.
* **Python Application/Example**: If you have a function that accepts a base class object, it should work perfectly well if you pass an object of any of its subclasses instead. A common violation is a subclass method raising an exception that the base class method does not, or returning unexpected values.
  ```python
  class Bird:
      def fly(self):
          print("Flying")

  class Sparrow(Bird):
      pass

  def make_bird_fly(bird: Bird):
      bird.fly()

  make_bird_fly(Sparrow())  # Works as expected
  ```
  The subclass `Sparrow` can be used wherever `Bird` is expected.

## I - Interface Segregation Principle (ISP)

This principle states that clients should not be forced to depend on interfaces they do not use. It is better to have many small, specific interfaces than one large, "fat" one.

* **Goal**: To reduce coupling by ensuring classes only implement what they actually need.
* **Python Application/Example**: Python doesn't have true interfaces like Java, but you can use Abstract Base Classes (ABCs) from the abc module or Protocols from the typing module to define contracts. This principle suggests you should create multiple small ABCs/Protocols instead of one big one that forces classes to define methods they don't use.
  ```python
  from typing import Protocol

  class Printer(Protocol):
      def print(self, document: str) -> None: ...

  class Scanner(Protocol):
      def scan(self) -> str: ...

  class MultiFunctionDevice(Printer, Scanner):
      def print(self, document: str) -> None:
          print(f"Printing: {document}")
      def scan(self) -> str:
          return "Scanned Document"
  ```
  Devices can implement only the protocols (interfaces) they need.

## D - Dependency Inversion Principle (DIP)

This principle means that high-level modules (your main logic) should not depend directly on low-level modules (details like database or logging). Instead, both should depend on abstractions (like interfaces or protocols).

* **Goal**: Make your code flexible and easy to change by separating logic from implementation details.
* **Python Application/Example**:  
  Instead of creating a database object directly inside your class, you define a protocol or abstract base class for the database. You then pass the actual database object into your class from outside (Dependency Injection).
    ```python
    from typing import Protocol

    class DatabaseConnector(Protocol):
        def save(self, data: str) -> None: ...

    class MySQLDatabase:
        def save(self, data: str) -> None:
            print(f"Saving '{data}' to MySQL")

    class ReportGenerator:
        def __init__(self, db: DatabaseConnector):
            self.db = db

        def generate(self, report: str):
            self.db.save(report)

    # Usage
    db = MySQLDatabase()
    reporter = ReportGenerator(db)
    reporter.generate("Annual Report")
    ```

This way, you can easily swap out `MySQLDatabase` for another database without changing `ReportGenerator`.

By following these principles, you write cleaner code that is easier to maintain, test, and adapt to changing requirements.