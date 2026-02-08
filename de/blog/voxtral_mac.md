---
title: "Voxtral auf dem Mac: Lokale Spracherkennung mit Metal-Beschleunigung"
date: 2026-02-08
toc: true
tags: 
  - Apple
  - Voxtral
  - Mistral
  - sovereignAI
  - python
  - C
toc: true
translations:
  en : "en/blog/voxtral_mac"
---

# Voxtral auf dem Mac: Lokale Spracherkennung mit Metal-Beschleunigung

Neulich bin ich auf Voxtral gestoßen — das Open-Source-Spracherkennungsmodell von Mistral AI. Als jemand, der gerne Dinge lokal auf dem eigenen Rechner laufen lässt (und nicht alles in die Cloud schickt), hat mich das sofort interessiert. Die Frage war nur... Läuft das auch auf meinem Mac?

Die kurze Antwort: Ja, aber nicht so, wie man vielleicht denkt. In diesem Beitrag zeige ich dir, wie du Voxtral auf einem Mac mit Apple Silicon zum Laufen bekommst, inklusive aller Stolpersteine, die ich dabei gefunden habe.


## Was ist Voxtral?

Bevor wir in die Einrichtung einsteigen, kurz zum Hintergrund: Voxtral ist ein multimodales Spracherkennungsmodell (auch als ASR bzw. Automatic Speech Recognition bekannt), das Mistral AI im Februar 2026 veröffentlicht hat. Das Besondere daran:

- **Echtzeit-fähig**: Das Modell ist für Streaming-Anwendungen optimiert und kann mit einer Latenz von unter 500ms arbeiten
- **Open Source**: Die Gewichte sind frei verfügbar auf HuggingFace
- **Mehrsprachig**: Unterstützt über 30 Sprachen, darunter auch Deutsch
- **4 Milliarden Parameter**: Kompakt genug für lokale Nutzung, aber leistungsfähig genug für gute Ergebnisse

Der offizielle Name lautet `mistralai/Voxtral-Mini-4B-Realtime-2602`. Die Zahl am Ende ist übrigens das Veröffentlichungsdatum: 26. Februar 2026.

## Der erste Versuch: Python und vLLM

Mein erster Ansatz war der naheliegende: Python mit vLLM. Mistral hat eng mit dem vLLM-Team zusammengearbeitet, um Voxtral-Support zu integrieren. Also habe ich mir ein einfaches Skript geschrieben:

```python
from vllm import LLM, SamplingParams
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

tokenizer = MistralTokenizer.from_file("tekken.json")
llm = LLM(
    model="mistralai/Voxtral-Mini-4B-Realtime-2602",
    tokenizer_mode="mistral",
    trust_remote_code=True,
    max_model_len=8192,
    enforce_eager=True
)
# ... weitere Logik
```

Das Problem: Es funktioniert nicht. Zumindest nicht auf dem Mac.

vLLM läuft auf macOS im CPU-Modus und hat nur eingeschränkte Unterstützung für die spezielle Voxtral-Architektur. Konkret scheitert es an der Whisper-Encoder-Komponente, die RoPE-Positionsembeddings verwendet — ein Feature, das vLLM auf dem Mac (noch) nicht unterstützt.

Die Fehlermeldung sieht ungefähr so aus:

```
ValueError: 'rope' is not a valid WhisperPosEmbedType
```

Wenn du einen Linux-Server mit NVIDIA-GPU hast, wird der Python-Ansatz vermutlich funktionieren. Für den Mac brauchst du aber eine andere Lösung.

## Die Lösung: `voxtral.c`

Nach einiger Recherche bin ich auf `voxtral.c` gestoßen, eine reine C-Implementierung von Antirez (ja, der Redis-Entwickler). Das Projekt hat einige Vorteile:

- **Metal-GPU-Beschleunigung**: Nutzt die GPU des Mac über Apples Metal-Framework
- **Keine Python-Dependencies**: Kompiliert zu einer einzigen Binary
- **Streaming-fähig**: Unterstützt auch Mikrofon-Eingabe in Echtzeit

Kommen wir nun zur Einrichtung...

## Einrichtung Schritt für Schritt

### 1. Repository klonen

Zunächst klonen wir das Repository:

```bash
git clone https://github.com/antirez/voxtral.c.git
cd voxtral.c
```

### 2. Kompilieren mit Metal-Support

Für Apple Silicon kompilierst du mit dem `mps`-Target (Metal Performance Shaders):

```bash
make mps
```

Der Befehl kompiliert die Software so, dass sie statt des Hauptprozessors (CPU) die leistungsstarke Grafikeinheit (GPU) deines Apple-Chips über die Metal-Schnittstelle nutzt.

Dadurch werden rechenintensive Aufgaben, wie das Ausführen von KI-Modellen, direkt auf die GPU verlagert, was durch den schnellen Zugriff auf den gemeinsamen Speicher (Unified Memory) die Geschwindigkeit und Effizienz massiv steigert.

Das sollte ohne Probleme durchlaufen. Am Ende erhältst du eine ausführbare Datei namens `voxtral`.

### 3. Modell herunterladen

Das Modell ist etwa 8.9 GB groß. Das mitgelieferte Skript lädt alle nötigen Dateien von HuggingFace:

```bash
./download_model.sh
```

Je nach Internetverbindung dauert das eine Weile. Am Ende hast du im Ordner `voxtral-model/` folgende Dateien:

- `consolidated.safetensors`: Die eigentlichen Modellgewichte (~8.3 GB)
- `tekken.json`: Der Tokenizer (~14 MB)
- `params.json`: Modellkonfiguration

### 4. Audio-Format beachten

Hier kommt ein wichtiger Punkt: `voxtral.c` erwartet Audio im Format **16-bit PCM WAV, 16kHz, Mono**. Wenn deine Audiodatei in einem anderen Format vorliegt (zum Beispiel 32-bit Float oder 24kHz wie es bei mir zu Beginn der Fall war), musst du sie erst konvertieren.

Mit ffmpeg geht das recht einfach...

```bash
ffmpeg -i original.wav -ar 16000 -ac 1 -acodec pcm_s16le converted.wav
```

Die Parameter im Detail:
- `-ar 16000`: Sample-Rate auf 16kHz setzen
- `-ac 1`: Mono (ein Kanal)
- `-acodec pcm_s16le`: 16-bit PCM Little-Endian

## Nutzung

### Datei transkribieren

Die grundlegende Nutzung ist einfach:

```bash
./voxtral -d voxtral-model -i audio.wav
```

Die Ausgabe erfolgt direkt auf der Konsole. Ich habe das mit einer mit [[Chatterbox]] hergestellten Datei getestet. Bei einer 24-sekündigen Audiodatei sieht das etwa so aus:

<audio controls src="../../Assets/tmpr9b5swn6.wav" title="Title"></audio>

```
Loading weights...
Metal GPU: 8429.2 MB
Model loaded.
Audio: 387840 samples (24.2 seconds)
Die meisten Menschen glauben, dass Erfolg riesige Sprünge und massive
Veränderungen über Nacht erfordert. Doch wahrer Fortschritt basiert
eigentlich auf kleinen, beständigen Gewohnheiten...
Encoder: 2816 mel -> 352 tokens (2110 ms)
Decoder: 114 text tokens (314 steps) in 12290 ms
```

### Mikrofon-Eingabe

Auf dem Mac kannst du auch direkt vom Mikrofon transkribieren:

```bash
./voxtral -d voxtral-model --from-mic
```

Mit Ctrl+C beendest du die Aufnahme. Das ist praktisch für schnelle Notizen oder Experimente. 

### Low-Latency-Modus

Wenn dir die Reaktionszeit wichtig ist, gibt es den `--low-latency`-Modus:

```bash
./voxtral -d voxtral-model --from-mic --low-latency
```

Im Grunde macht dieser Modus zwei Dinge:

1. Das Verarbeitungsintervall wird von $1s$ auf $0.5s$ reduziert
2. Die Stille-Toleranz sinkt von $600ms$ auf $300ms$

Das bedeutet: Der Text erscheint schneller auf dem Bildschirm, aber die GPU hat mehr zu tun. Auf einem M3 oder M4 sollte das kein Problem sein. Auf älteren Macs kann es sein, dass die Verarbeitung nicht ganz mithält.

### Transkription in Datei speichern

Manchmal möchte man die Transkription nicht nur auf der Konsole sehen, sondern auch speichern. Dafür gibt es den `-o`-Parameter:

```bash
./voxtral -d voxtral-model --from-mic -o transkription.txt
```

Die Ausgabe erscheint weiterhin auf der Konsole, wird aber gleichzeitig in die angegebene Datei geschrieben. Das ist besonders praktisch für längere Aufnahmen. Du siehst den Text live und hast am Ende alles in einer Datei.

Natürlich lässt sich das auch kombinieren:

```bash
./voxtral -d voxtral-model --from-mic --low-latency -o notizen.txt
```

### Streaming-Intervall manuell anpassen

Der Parameter `-I` steuert, wie oft der Encoder neue Audio-Chunks verarbeitet. Der Standardwert liegt bei 1 Sekunde für Mikrofon-Eingabe und 2 Sekunden für Datei-Transkription:

```bash
./voxtral -d voxtral-model --from-mic -I 0.5  # Noch responsiver
./voxtral -d voxtral-model --from-mic -I 2.0  # Mehr Effizienz
```

Niedrigere Werte bedeuten weniger Latenz, aber mehr GPU-Overhead. Für Offline-Transkription (also bei Dateien) ist dieser Parameter irrelevant, weil ohnehin alle Daten auf einmal vorliegen.

### Audio über Pipe

Du kannst auch Audio über `stdin` pipen. Das ist nützlich, wenn du andere Formate on-the-fly konvertieren willst:

```bash
ffmpeg -i podcast.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | ./voxtral -d voxtral-model --stdin
```

---

## Bonus: Eine Web-Oberfläche für Voxtral

Die Kommandozeile ist praktisch, aber manchmal möchte man einfach einen Button drücken und sprechen. Also habe ich eine kleine Web-Oberfläche gebaut. Mit Push-to-Talk-Funktion direkt im Browser.

### Die Architektur

Die Lösung besteht aus zwei Teilen:

1. **FastAPI-Backend**: Nimmt Audio vom Browser entgegen, konvertiert es mit `ffmpeg` und ruft `voxtral.c` auf
2. **HTML/JavaScript-Frontend**: Nutzt die MediaRecorder API für die Aufnahme im Browser

Das Ganze läuft komplett lokal. Keine Cloud, keine externen Dienste.

### Das Backend

Das Backend ist bewusst einfach gehalten. Es macht im Wesentlichen drei Dinge:

```python
from fastapi import FastAPI, UploadFile, File
import subprocess
import tempfile

app = FastAPI()

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # 1. Audio speichern
    input_file = save_temp_file(audio)

    # 2. Mit ffmpeg zu 16kHz 16-bit PCM konvertieren
    wav_file = convert_with_ffmpeg(input_file)

    # 3. Voxtral ausführen und Ergebnis zurückgeben
    result = run_voxtral(wav_file)
    return {"transcription": result}
```

Die vollständige Implementierung findest du in `web_app.py`. Der wichtigste Teil ist die ffmpeg-Konvertierung, der Browser sendet Audio im WebM-Format, `voxtral.c` erwartet aber 16-bit PCM WAV.

### Das Frontend

![Voxtral Frontend](voxtral_frontend.png)

Das Frontend nutzt die MediaRecorder API, die in allen modernen Browsern verfügbar ist. Der zentrale Mechanismus ist Push-to-Talk:

- **Mousedown / Touchstart**: Aufnahme starten
- **Mouseup / Touchend**: Aufnahme stoppen und an Server senden
- **Leertaste**: Funktioniert auch als Shortcut

```javascript
recordBtn.addEventListener('mousedown', () => {
    mediaRecorder.start();
    // Visuelles Feedback
});

recordBtn.addEventListener('mouseup', () => {
    mediaRecorder.stop();
    // Audio wird automatisch gesendet
});
```

Das Frontend prüft beim Laden, ob alle Komponenten verfügbar sind (voxtral-Binary, Modell, ffmpeg) und zeigt entsprechende Fehlermeldungen an.

### Starten und Testen

Um die Web-Oberfläche zu starten:

```bash
source .venv/bin/activate
python web_app.py
```

Dann öffnest du http://localhost:8000 im Browser. Du siehst einen großen Button — halte ihn gedrückt, sprich, und lass los. Nach ein paar Sekunden erscheint die Transkription.

*Kurze Anmerkung*: Beim ersten Aufruf fragt der Browser nach Mikrofon-Berechtigung. Die musst du natürlich erlauben.

### Einschränkungen

Die aktuelle Implementierung ist bewusst einfach gehalten:

- **Keine Streaming-Transkription**: Das Audio wird erst nach dem Loslassen des Buttons verarbeitet
- **Keine Authentifizierung**: Für lokale Nutzung kein Problem, aber nicht für öffentliche Server geeignet
- **Sequentielle Verarbeitung**: Nur eine Anfrage gleichzeitig

Für den Hausgebrauch reicht das völlig aus. Wenn du etwas Produktionsreifes brauchst, wäre ein WebSocket-basierter Ansatz mit Streaming-Transkription der nächste Schritt. Dazu vielleicht ein anderes Mal mehr.

---

## Fazit

Voxtral auf dem Mac zum Laufen zu bringen war nicht ganz trivial, denn der offizielle Python-Weg funktioniert nicht. Mit `voxtral.c` gibt es aber eine solide Alternative, die die Metal-GPU nutzt und vernünftige Performance liefert.

Die Einrichtung in Kurzform:

1. `git clone https://github.com/antirez/voxtral.c.git`
2. `cd voxtral.c && make mps`
3. `./download_model.sh`
4. Audio zu 16-bit PCM konvertieren
5. `./voxtral -d voxtral-model -i audio.wav`

Für Live-Transkription vom Mikrofon:

```bash
./voxtral -d voxtral-model --from-mic --low-latency -o transkription.txt
```


Wenn du Fragen hast oder Fehler gefunden hast, melde dich gerne bei mir.

---

*Alle Beispiele wurden auf einem Mac mit Apple Silicon getestet. Die genauen Zeiten können je nach Hardware variieren.*