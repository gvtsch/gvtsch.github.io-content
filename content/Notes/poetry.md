---
tags: ['python', 'coding']
author: CKe
title: 'Poetry'
date: 2025-06-01
---

# Poetry

Poetry is a tool for Python for managing dependencies and packaging. It allows you to declare project dependencies and manages them. Poetry ensures that everyone working on the project uses the same dependency versions. It simplifies the creation, publication, and management of Python projects.

I use it more often in projects that I share with others. Of course, there is also a solution that works with Python and does not require the installation of additional packages:
 
```bash
pip install -r requirements.txt
```

If you have a corresponding text file with package names (and versions), you can certainly solve this problem. So why use `poetry`?

## Why `Poetry` and not `pip install -r requirements.txt`?

`requirements.txt` has some disadvantages that Poetry remedies:

* **No dependency resolution**: `requirements.txt` only lists direct dependencies. Poetry also resolves transitive dependencies and ensures that all dependencies are compatible with each other.
  * A transitive dependency in Python is a library that a package needs, which your project in turn needs, without you installing it directly.
* **No locking**: `requirements.txt` does not store exact versions of dependencies. This can lead to problems when new versions of dependencies are released that are not compatible with the project. Poetry creates a `poetry.lock` file that stores the exact versions of all dependencies, ensuring deterministic builds.
* **No packaging**: `requirements.txt` is only intended for installing dependencies. Poetry can also be used to create and publish Python packages.
* **Virtual environments**: Poetry manages virtual environments.

Essentially, Poetry offers more comprehensive and robust dependency management.

In addition to the two mentioned above, there are other packages and options that I have not yet explored in depth:

* Pipenv
* venv/virtualenv
* pip-tools
* Hatch
* PDM
* Rye
* Conda (which I am also familiar with, but cannot use everywhere)

## Installing Poetry

Like most other packages, Poetry can be installed using `pip`.

```bash
pip install poetry
```

## Configuring Poetry

Poetry offers various configuration options to customize its behavior to your needs. The configuration can be done via the `pyproject.toml` file or via the command line.
Some important configuration areas are:

* **Dependencies**: Specifying project dependencies with version restrictions.
* **Package information**: Metadata such as the name, version, description, authors, and license of the package.
* **Build settings**: Configuration of the build process, for example which files should be included in the package.
* And much, much more, such as **virtualenv settings**, **repository**, or **scripts**. I have not yet dealt with these in the context of `Poetry`.

### Example of a `pyproject.toml` file

Here is an example of a configuration file. It is certainly not complete, but it is perfectly adequate for my needs.

```toml
[tool.poetry]
name = 'my-library'
version = '0.1.0'
description = 'A short description of my library.'
authors = ['Your name <deine.email@example.com>']
license = 'MIT'
readme = 'README.md'
packages = [{include = 'my_library'}]

[tool.poetry.dependencies]
python = '^3.8'
requests = '^2.28.1'
numpy = '^1.23.4'

[tool.poetry.group.dev.dependencies]
pytest = '^7.2.0'
flake8 = '^5.0.4'
mypy = '^0.982'

[build-system]
requires = ['poetry-core']
build-backend = 'poetry.core.masonry.api'
```

I don't think there's much to say here, the file is relatively self-explanatory.

## Setting up a new project

The next steps explain how to use `Poetry` to set up a new project.

### Creating a new project

You can create a new `Poetry` project with `poetry add <projectname>`.
This will create a new project directory with the basic structure and the `pyproject.toml` file mentioned above.

### Adding dependencies

You can add packages (in the latter case with a defined version) to the project with `poetry add <package name>` or `poetry add <package name>@<version>`. These will then also be listed in `pyproject.toml`.

### Remove dependencies

Of course, you can also remove dependencies: `poetry remove <package name>`

### Update packages

`poetry update` updates the dependencies to the latest compatible versions defined in `pyproject.toml` and updates the `poetry.lock` file.

### Display packages

`poetry show` displays a list of all installed packages and their versions. This is useful for checking which versions are actually installed.

### Build a package

`poetry build` creates a package from the Python project. More specifically, it generates a wheel file (`.whl`) and a source archive file (`.tar.gz`) in the dist folder. These files are the ones you can then use to distribute the package or publish it on PyPI.

### Publishing a package

`poetry publish` is used to publish the package created with poetry build (wheel file and source archive) on a PyPI-compatible server so that other users can install and use it.

## Other useful commands and tips

There are many more commands. I will mention a few more, even though I have only looked at them superficially and have never used them seriously.

* **Defining scripts**: Scripts can be defined in `pyproject.toml` and then executed using `poetry run <scriptname>`. This is very useful for automating recurring tasks (running tests, formatting code, etc.).
    ```toml
    [tool.poetry.scripts]
    test = 'pytest'
    lint = 'flake8 my_library'
    ```
    These scripts can then be easily executed with `poetry run test` or `poetry run lint`.
* **Groups of dependencies**: In addition to the normal dependencies ( `[tool.poetry.dependencies]`) and the development dependencies (`[tool.poetry.group.dev.dependencies]`), additional groups can also be defined. This is useful for managing optional dependencies for example for certain features.
* **Environment variables**: Poetry supports the use of environment variables in the `pyproject.toml` file. This can be useful for managing secret keys or other sensitive information without storing them directly in the configuration file.
* **Plugins**: Poetry supports plugins to extend its functionality. There are a number of community plugins that, for example improve integration with other tools or add additional features.
* **Version management**: Poetry uses semantic versioning. This means that version numbers are specified in the form `MAJOR.MINOR.PATCH`, where:
  
  * `MAJOR`: Incompatible API changes
  * `MINOR`: New functionality, backward compatible
  * `PATCH`: Bug fixes, backward compatible
  
  When defining dependencies, you can specify various version restrictions (e.g., `^1.2.3`, `~1.2.3`, `>1.2.3`, etc.).

## Summary

Poetry is a powerful tool for dependency management in Python projects. It offers many advantages over traditional approaches such as `requirements.txt` and simplifies the creation, publication, and management of Python packages. By using Poetry, you can ensure that all project participants use the same dependency versions and create deterministic builds.
