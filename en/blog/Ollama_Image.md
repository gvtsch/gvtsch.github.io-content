---
title: Create images locally w/ Ollama
date: 2026-01-27
tags: ["machine-learning"]
toc: True
author: CKe
translations:
  en: "de/blog/Ollama_Image"
---

# Local Image Generation on the MacBook – A Short Experience Report

I spent the morning today experimenting with generating images entirely locally on my MacBook. Thanks to Ollama (in experimental status) and models such as FLUX.2 Small and Z-Image Turbo, this now works surprisingly smoothly.

![FLUX.2 Klein (9B)](a-neon-sign-reading-hello-linkedin-in-a-rainy-city-20260125-110511.png)
_FLUX.2 Klein (9B)_

## A few learnings from the process
* **The hurdle**: If you want to try it yourself… the Ollama version of Homebrew currently still doesn’t support image generation. You have to download the version directly from the homepage (link in the comments).
* **Performance**: The image below (FLUX.2 9B) took about 60 seconds. The smaller models (4B or Z‑Image Turbo) take roughly 30 seconds; the images are in the comments.
* **The prompt**: _A neon sign reading ‘Hello LinkedIn!’ in a rainy city alley at night, reflections on wet pavement._

For a local setup on a standard laptop I find the results absolutely fine, even good. I especially find it charming that I don’t incur token costs, the data stays on the device, and even with the current weather the power for rendering in my case comes directly from my own roof.

Sure, it might be faster or higher quality in the cloud, but the independence from third parties is something you shouldn’t underestimate.

![Z-Image Turbo (6B)](a-neon-sign-reading-hello-linkedin-in-a-rainy-city-20260125-103816.png)
_Z-Image Turbo (6B)_

![FLUX.2 Klein (4B)](a-neon-sign-reading-hello-linkedin-in-a-rainy-city-20260125-103655.png)
_FLUX.2 Klein (4B)_