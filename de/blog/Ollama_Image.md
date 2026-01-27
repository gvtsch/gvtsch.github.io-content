---
title: Bilder lokal mit Ollama erzeugen
date: 2026-01-27
tags: ["machine-learning"]
toc: True
draft: false
author: CKe
translations:
  en: "en/blog/Ollama_Image"
---

# Lokale Bildgenerierung auf dem MacBook – Ein kurzer Erfahrungsbericht

Ich habe heute Vormittag damit experimentiert, Bilder komplett lokal auf meinem MacBook zu generieren. Dank Ollama (im Experimental-Status) und Modellen wie FLUX.2 Klein und Z-Image Turbo funktioniert das mittlerweile erstaunlich reibungslos.

![FLUX.2 Klein (9B)](a-neon-sign-reading-hello-linkedin-in-a-rainy-city-20260125-110511.png)
_FLUX.2 Klein (9B)_

## Ein paar Learnings aus dem Prozess
* **Die Hürde**: Wer es selbst testen möchte... die Ollama-Version von Homebrew unterstützt die Bildgenerierung aktuell noch nicht. Man muss direkt die Version von der Homepage laden (Link im Kommentar). 
* **Die Performance**: Das untenstehende Bild (FLUX.2 9B) hat ca. 60 Sekunden gedauert. Die kleineren Modelle (4B oder Z-Image Turbo) liegen bei etwa 30 Sekunden; die Bilder dazu findet ihr in den Kommentaren.
* **Der Prompt**: _A neon sign reading 'Hello LinkedIn!' in a rainy city alley at night, reflections on wet pavement._

Für ein lokales Setup auf einem Standard-Laptop finde ich die Ergebnisse absolut ok, sogar gut. Besonders charmant finde ich, dass mir keine Token-Kosten entstehen, die Daten auf dem Gerät bleiben und selbst bei dem aktuellen Wetter der Strom für das Rendering in meinem Fall sogar direkt vom eigenen Dach kommt.

Sicher geht es in der Cloud schneller oder qualitativ noch hochwertiger, aber die Unabhängigkeit von Drittanbietern ist ein Punkt, den man nicht unterschätzen sollte.

![Z-Image Turbo (6B)](a-neon-sign-reading-hello-linkedin-in-a-rainy-city-20260125-103816.png)
_Z-Image Turbo (6B)_

![FLUX.2 Klein (4B)](a-neon-sign-reading-hello-linkedin-in-a-rainy-city-20260125-103655.png)
_FLUX.2 Klein (4B)_