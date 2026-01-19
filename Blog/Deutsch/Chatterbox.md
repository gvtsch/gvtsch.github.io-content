---
title: Chatterbox
date: 2026-01-19
tags:
  - machine-learning
  - deep-learning
  - python
  - llm
  - nlp
  - sovereignAI
toc: true
---

Ich bin vor kurzem über Chatterbox gestolpert. Chatterbox von Resemble AI ist ein leistungsstarkes Open-Source-Modell für Text-to-Speech. Es ermöglicht hochwertiges Voice-Cloning und auch präzise Emotionssteuerungen. 
Das Tool gibt es hier: https://www.resemble.ai/chatterbox/. Hier kann man sich auch Beispiele anhören. Ist ganz spannend.

Ich habe es lokal auf meinem MacBook installiert und ausgeführt. Ich habe quasi ein wenig herumgespielt. Aber ich bin beim installieren auch über ein paar Steine gestolpert. Wie ich es zum Laufen bekommen habe liest du hier. Im Grunde ganz einfach und wenig aufwendig, wenn man erstmal weiß, wie.

## Installation

Man kann sich bei der Installation am Tutorial entlang hangeln: https://github.com/resemble-ai/chatterbox
Ich gehe es aber auch noch mal Schritt für Schritt durch.
1. Als erstes richtet man sich einen Ordner ein.
2. Dann erzeugt man noch eine Entwicklungsumgebung. Ich nutze dazu **uv**: `uv venv --python3.11`). Diese Umgebung gehört dann natürlich aktiviert: `source .venv/bin/activate`
3. Und dann kann man auch schon Chatterbox installieren (`uv pip install chatterbox-tts`).
	1. Auf meine Mac stieß ich hier auf erste Probleme. Ich musste noch weitere packages installieren: `uv pip install git+https://github.com/resemble-ai/perth.git`
4. Und danach hat man der Anleitung nach zwei Möglichkeiten
	1. Man lässt mit nur eine Sprache laufen.
		1. Erst eine Python Datei erzeugen (`touch clone-gradio.py`) und mit Code befüllen. Den Code kannst du dir aus der Anleitung holen, oder weiter unten in diesem Beitrag.
		2. Und anschließend ausführen mit `python clone-gradio.py`
	2. Oder Multilingual.
		1. Erneut eine Python Datei erzeugen (`touch multi-gradio.py`) und mit Code befüllen
		2. Und wieder ausführen mit `python multi-gradio.py`
5. In beiden Fällen kann man dann die gelistete URL im Browser öffnen und erhält dann die GUI.
	* ![[Bildschirmfoto 2026-01-08 um 06.25.31.png]]
	* Hier kann man dann z.B. die Sprache wählen und etwas einsprechen, wenn man seine eigene Stimme klonen möchte. Ich habe etwas englisches eingesprochen und auch etwas englisches ausgeben lassen. 
		* Hinweis von Chatterbox: 💡 **Note**: Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip's language. To mitigate this, set the CFG weight to 0.
		* **Eingesprochener Text**: `There is a special kind of magic in the early morning, just before the rest of the world wakes up. The air is cool, the streets are silent, and the first light of the sun paints the sky in soft shades of pink and gold. It is a rare moment of peace that allows you to clear your mind and breathe deeply. Starting your day in silence can change your entire perspective on the hours to come.`
		* **Geklonter Text**: `Most people believe that success requires giant leaps and massive changes overnight. However, true progress is actually built on small, consistent habits. By focusing on getting just one percent better every single day, you create a powerful momentum. Remember, it does not matter how slowly you go, as long as you do not stop. Your future self will thank you for the effort you put in today.`
		* **Ausgabe Audio**: 
			* ![[tmpjvbfbs8v.wav]]
			* Ich habe ein paar der Parameter angepasst und bekomme dann die folgende Ausgabe: ![[tmpo6iujdm2.wav]]
			* Oder auf deutsch: ![[tmpr9b5swn6.wav]]
			* Es klingt mir nun nicht zum verwechseln ähnlich, es ist aber auch nicht übertrieben weit weg. Und man hört auch noch ein wenig den Computer heraus. Mehr im Fazit.

## Fazit

Es war super einfach zu installieren. und es läuft auf dem eigenen Rechner. Lokal. Souverän. Und ich habe nur run 10s Text eingesprochen. Dafür erhalte ich ein, wie ich finde, sehr cooles Ergebnis. Ich habe jetzt hier sicher wieder nur an der Oberfläche gekratzt und werde noch mehr Zeit investieren. Ich denke da an ein kleines Projekt, bei dem ich mit Hilfe von Tags oder ähnlichem, zwei Personen miteinander reden lasse, um vielleicht sogar so etwas wie ein Gespräch zu erzeugen.

## Code

Hier noch der Code den man einfach in zwei Python-Dateien ablegen kann.

### `clone-gradio.py`

```python
import torch
import gradio as gr
from chatterbox.vc import ChatterboxVC


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = ChatterboxVC.from_pretrained(DEVICE)
def generate(audio, target_voice_path):
    wav = model.generate(
        audio, target_voice_path=target_voice_path,
    )
    return model.sr, wav.squeeze(0).numpy()


demo = gr.Interface(
    generate,
    [
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input audio file"),
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Target voice audio file (if none, the default voice is used)", value=None),
    ],
    "audio",
)

if __name__ == "__main__":
    demo.launch()
```

### `multi-gradio.py`

```python
import gradio as gr
import torch
import torchaudio
import tempfile
import os

# --- 🛠️ Mac PATCH START 🛠️ ---
# This fixes the "RuntimeError: Attempting to deserialize object on a CUDA device"
# It forces the model to load onto the CPU first, solving the Mac M1 incompatibility.
original_load = torch.load

def safe_load(*args, **kwargs):
    # If the library forgets to specify a device, force it to 'cpu'
    if 'map_location' not in kwargs:
        kwargs['map_location'] = 'cpu'
    return original_load(*args, **kwargs)

torch.load = safe_load
# --- 🛠️ Mac PATCH END 🛠️ ---

# Now we can safely import the library
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("⏳ Loading Chatterbox Multilingual Model... (This downloads ~3GB on first run)")

# 1. Setup Device
if torch.backends.mps.is_available():
    device = "mps"
    print("✅ Using Apple Silicon GPU (MPS)")
else:
    device = "cpu"
    print("⚠️ MPS not available, using CPU")

# 2. Load the Multilingual Model
# The patch above ensures this line no longer crashes
model = ChatterboxMultilingualTTS.from_pretrained(device=device)
print("🚀 Model Ready!")

# 3. Define the Generator Function
def generate_advanced(text, language_code, voice_path, exaggeration, stability):
    if not text.strip():
        return None

    print(f"Generating ({language_code}): '{text[:20]}...' | Emotion: {exaggeration} | Stability: {stability}")

    # Handle the Voice Cloning Input
    prompt_path = voice_path if voice_path else None

    # GENERATE
    audio_tensor = model.generate(
        text,
        language_id=language_code,
        audio_prompt_path=prompt_path,
        exaggeration=exaggeration, # Controls emotion (0.0 - 1.0+)
        cfg_weight=stability       # Controls stability/pacing (0.0 - 1.0)
    )

    # Save to temp file for Gradio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        output_path = fp.name
        torchaudio.save(output_path, audio_tensor, model.sr)

    return output_path

# 4. Build the Interface
LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Japanese": "ja", "Chinese": "zh", "Russian": "ru",
    "Portuguese": "pt", "Polish": "pl", "Korean": "ko", "Dutch": "nl",
    "Turkish": "tr", "Arabic": "ar", "Hindi": "hi", "Swedish": "sv"
}

with gr.Blocks(title="Chatterbox Advanced") as demo:
    gr.Markdown("# Chatterbox Multilingual & Voice Cloning")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text to Speak",
                value="Hello! I can speak many languages with different emotions.",
                lines=3
            )

            lang_dropdown = gr.Dropdown(
                label="Language",
                choices=list(LANGUAGES.keys()),
                value="English",
                type="value"
            )

            audio_input = gr.Audio(label="Reference Voice (Optional - for cloning)", type="filepath")

            with gr.Accordion("Advanced Settings", open=True):
                exaggeration_slider = gr.Slider(
                    minimum=0.0, maximum=1.5, value=0.5, step=0.1,
                    label="Emotion / Exaggeration (Higher = More Dramatic)"
                )
                stability_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.5, step=0.1,
                    label="Stability / Pacing (Lower = Faster/Looser)"
                )

            submit_btn = gr.Button("Generate Audio", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Result")

    def wrap_generate(text, lang_name, voice, exag, stab):
        code = LANGUAGES.get(lang_name, "en")
        return generate_advanced(text, code, voice, exag, stab)

    submit_btn.click(
        fn=wrap_generate,
        inputs=[text_input, lang_dropdown, audio_input, exaggeration_slider, stability_slider],
        outputs=audio_output
    )

if __name__ == "__main__":
    demo.launch()
```
