---
title: "My development environment: Docker, Compose & Codespaces"
date: 2025-09-28
tags:
  - python
  - docker
  - github
  - codespace
toc: true
---

# My development environment: How I work with Docker, Compose, and Codespaces

I decided to write down my workflow with Docker, Compose, etc. You may already be familiar with some or all of the concepts, but I'm also doing this to record my thoughts. Hopefully, one or two of you will find something useful!

The repository can be found here: [Docker Example](https://github.com/gvtsch/docker_example.git) 
It is part of this larger [collection of examples](https://github.com/gvtsch/Blog_Code_Snippets).

## Introduction

I develop privately using Docker, Compose and Codespaces, among other things. In the following chapters, I explain how and why I do this. There will also be a practical Python example.

### What is Docker?

So, what is Docker? Docker is a technology for orchestrating applications. For instance, it enables you to develop and execute Python code in an isolated and portable environment known as a 'container'.

Why containerisation?

Have you ever heard someone say, 'It works on my computer'? Or have you said it yourself? With Docker, that's a thing of the past. Docker enables you to create consistent and reproducible development environments that can be run on a wide variety of systems. For example, I have a dual-boot system on my laptop consisting of Windows 11 and Fedora. When I use Docker, I can easily work on the project on both systems.

## Docker – the cornerstone of containerization

### Basic concept

* With Docker, you create a so-called **image**. This image is a read-only template with instructions for creating a container. To do this, you use the so-called **`Dockerfile`**. More on that later.
* The **`Dockerfile`** is a text file that defines the steps for creating a Docker **image**.
* A Docker **container** is an executable instance of an **image**. It is a lightweight and, as mentioned, isolated development environment. You could also solve all this with virtual machines, but then it would probably no longer be lightweight.
* The **Docker daemon** is the service that runs in the background and manages the containers.

### Some important Docker commands

* `docker build`: Creates an image from a `Dockerfile`.
* `docker run`: Starts a container from an image.
* `docker ps`: Lists the running containers.
* `docker pull`: Downloads an image from a repository.
* `docker push`: Uploads an image to a repository.

## Docker Compose – Simplified Orchestration

### Basic Concepts

**Docker Compose** is a tool for managing multi-container applications (as we will see later in the example). It uses a single configuration file, called `docker-compose.yml`, to define all services (containers), networks, and volumes, allowing the entire application architecture to be orchestrated with a single command.

### Some important Docker Compose Commands

* `docker-compose up`: Builds and starts all services defined in `.yml`.
* `docker-compose down`: Stops and removes all services.

## Practical example with Python

Below I will show a simple web application that displays _"Hello, Docker!"_. I will use Python and the Flask module for this.

### Simple Flask application (with Docker)

Let's start with the `Dockerfile`.
```Dockerfile
# Base image
FROM python:3.9-slim
# Working directory in the container
WORKDIR /app
# Copy dependencies
COPY requirements.txt .
# Install dependencies
RUN pip install -r requirements.txt
# Copy code
COPY . .
# Expose container port
EXPOSE 5000
# Command to start the app
CMD ["python", "app.py"]
```

In the Dockerfile, the `requirements.txt` file is used to install the dependencies. In this case, the file is pretty empty because we only need Flask.

```Text
Flask
```

Finally, we need `app.py`:

```python
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_docker():
  return "Hello, Docker!"

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0")
```

This is basically not a large or complicated app. And to run it, all we have to do is start it in the console.

```bash
docker build -t my-python-app .
docker run -p 5000:5000 my-python-app
```

The first line builds a Docker image named `my-python-app` from the current directory. So we need to be in that folder when we run it. This causes Docker to first perform all the necessary steps, such as installing `Flask`. I've shortened the output to the essentials:

```bash
[+] Building 22.6s (11/11) FINISHED
 => [internal] load build definition from Dockerfile
 => [internal] load metadata for docker.io/library/python:3.9-slim
 => [1/5] FROM docker.io/library/python:3.9-slim
 
=> [internal] load build context
 => [2/5] WORKDIR /app
 => [3/5] COPY requirements.txt .
 => [4/5] RUN pip install -r requirements.txt
 => [5/5] COPY . .
 => exporting to image
 => naming to docker.io/library/my-python-app:latest
 ```

The second line starts a container from this image and forwards port `5000` of the container to port `5000` of the host. Port forwarding is necessary so that you can access services running in the container from your host machine (e.g., your local computer or Codespace).

```bash
(base) $ docker run -p 5000:5000 my-python-app
 
* Serving Flask app "app"
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 
* Running on http://127.0.0.1:5000
 * Running on http://172.17.0.2:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 126-682-939
```

I can now visit the URL `http://127.0.0.1:5000` specified there. Incidentally, `172.17.0.2:5000` is the internal IP of the container in the Docker network and can only be accessed within this network, for example by other Docker containers.
And now, I simply see the text `Hello, Docker!` displayed.

> **Note:** The Flask server used in the examples is intended for development purposes only. For production environments, a WSGI server such as Gunicorn or uWSGI should be used.

### Multi-container application (with Docker Compose)

The second example is again a Flask app (the `web` part in the `yml` file). Now it will communicate with a Redis database (the `redis` part). These are now two separate containers that communicate with each other via the aforementioned internal network.
For this we need a `yml` file: `docker-compose.yml`

```yml
version: "3.8"
services:
  web:
    build: .  # Refers to the Dockerfile in the current directory
    ports:
      - "5000:5000"  # Exposes port 5000 of the web container
    
volumes:
      - .:/app  # Live reload: Links the host directory to the container. Changes to the code are applied immediately.
depends_on:
      - redis  # Only starts the web service when Redis is running
redis:
image: "redis:alpine"  # Uses a ready-made image from Docker Hub
```

And there is also an `app.py` again. It is a small web application with Flask that stores a counter for page views. Inside a Redis database.

```python
from flask import Flask
import redis
import time

app = Flask(__name__)

cache = redis.Redis(host="redis", port=6379)
def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr("hits")
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)

@app.route("/")
def hello():
    count = get_hit_count()
    return 'Hello from Docker! I have been seen {} times.\n'.format(count)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
```

Now let's start our application.

```bash
docker-compose up -d
```

The above command now performs several steps internally.

1. It reads the `docker-compose.yml` file.
2. For the `web` part, the service now recognizes the `build: .` statement, searches the current directory for a `Dockerfile` and recognizes that it needs to create an image. The next section is the same as in the previous example.
3. At the same time the command ensures that the `redis:alpine` image is downloaded and the Redis container is started.
The `-d` means that the containers are started in the background (or in **d**etached mode). If you don't set the flag the console may become a little cluttered because the log outputs of the containers are displayed ;)
 
If you have changed something in a file that is needed to create the image (e.g., `requirements.txt`, `.py` files, `Dockerfile`, etc.), you must rebuild the image. To do this, simply add the `--build` flag:

```bash
docker-compose up -d --build
```

When the image is rebuilt, you will first see an output that is very similar to the one in the first example. If the image has already been built, starting the containers again will produce output like this:

```bash
[+] Running 3/3
✔ Network docker_example_default    Created  0.1s
✔ Container docker_example-redis-1  Started  0.9s
✔ Container docker_example-web-1    Started  1.2s
```

**What does this output mean?**

First, it shows that Docker Compose has created a separate network for the application. Within this network, the containers can communicate with each other. And then it returns that both the `redis` and `web` containers have been started.

I can now access the web interface via `http://127.0.0.1:5000/`. And every time I do so, the counter in the database is incremented, which looks like this: `Hello from Docker! I have been seen 7 times.`

With `docker ps`, you can now see which containers have been started. Of course, you can also do this in the Docker desktop application. There you can also debug the Python code.

```bash
CONTAINER ID   IMAGE                COMMAND                  CREATED         STATUS         PORTS                                         NAMES
b21b0f302613   docker_example-web   "python app.py"          8 minutes ago   Up 8 minutes   0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp   docker_example-web-1
91cea7ff0771   redis:alpine         "docker-entrypoint.s…"   8 minutes ago   Up 8 minutes   6379/tcp                                      docker_example-redis-1
```

To terminate the Docker containers, you can shut them down relatively easily with a single command:

```bash
docker-compose down
```

The output should then look something like this:

```bash
[+] Running 3/3
 ✔ Container docker_example-web-1    Removed  0.6s
 ✔ Container docker_example-redis-1  Removed  0.4s
 ✔ Network docker_example_default    Removed  0.5s
```

## Github Codespaces – The cloud-based part

I don't always have access to my laptop. In such cases, it is helpful to be able to access your project via your browser and run it there. This is where **Codespaces** comes in, for example.

### What is Codespaces?

**Codespaces** is a cloud-based **Visual Studio Code** environment that is directly integrated into **GitHub**. You can use it to launch complete development environments in just a few minutes – directly from your repository, without any local setup. Codespaces uses Docker containers as its basis, so you can ensure an identical environment for all team members.
To start a codespace, simply click on the green "Code" button in the desired GitHub repository and then on "Create codespace on main" (or the desired branch).

### Advantages over local development

* **Quick start:** Just click on "Code → Codespaces" in the GitHub repo, and in a few minutes you'll have a full-fledged development environment at your disposal.
* **No local installation required:** Neither Python, Docker, nor any other dependencies need to be installed on your computer. Everything runs in the browser.
* **Teamwork:** Multiple developers can work in the same codespace at the same time or start their own instances with identical configurations. This ensures consistency and fewer "it only works for me" problems.
* **Resource-independent**: Codespaces run independently of your local hardware. You can work on demanding projects even on weaker devices or tablets.
* **Preconfigured**: Using `.devcontainer/devcontainer.json`, you can specify exactly which tools, extensions, and environment variables are available at startup.
* **Automated environment:** Changes to the development environment (for example new dependencies) are versioned and immediately available to everyone.
* **Integration with Docker and Compose:** You can use Docker containers and Compose setups as usual, including terminal access and port forwarding.

> **Note:** GitHub Codespaces is only available for private accounts with a limited free quota. Additional costs may apply. You can find the current limits in the [GitHub documentation](https://docs.github.com/de/billing/concepts/product-billing/github-codespaces).

> **Note:** Codespaces have their own resource limits (CPU, RAM, runtime). Codespaces is not always suitable for very large, computationally intensive, or GPU-based projects.

### Integration into the Docker workflow

- Use the **`.devcontainer/devcontainer.json`** file to configure the environment. Here you can define extensions, start commands, and environment variables, for example.
- Use the existing `docker-compose.yml` file to start the services. This allows you to develop and test multi-container applications in the codespace as well.
- You can execute all Docker commands as usual via the integrated terminal.
- Open ports (from Flask or other web services) are automatically detected and can be opened in the browser.

## Summary and outlook

### Conclusion

I use the above tools for a modern development environment across systems and hardware.
- **Docker** as the basis for encapsulation.
- **Docker Compose** for local orchestration.
- **GitHub Codespaces** for team-oriented, cloud-based development.

### Next step: Scaling in production (k8s)

There are also other very helpful tools. For example, **Kubernetes** (k8s). **k8s** is one of the leading tools for container orchestration in production environments. It automates the scaling, management, and reliability of containerized applications in a so-called cluster. Kubernetes usually only becomes relevant once a project or team reaches a certain size. I plan to write about this soon.