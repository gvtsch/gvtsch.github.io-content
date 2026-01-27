---
Tags:
translations:
  de: "de/blog/LLM-Architectures"
---


# Übersicht über LLM-Architekturen und ihre Merkmale

## 1. Dense Models (Standard-Transformer-Architektur)
**Beispiele:** GPT-3/4, Llama, Mistral, PaLM
**Merkmale:**
- Klassische Transformer-Architektur mit dichten Attention-Mechanismen.
- Verarbeiten alle Eingabetoken gleichzeitig.
- **Stärken:** Universell einsetzbar, gut für generative Aufgaben.
- **Schwächen:** Hoher Rechenaufwand bei langen Kontexten.

---

## 2. Mixture of Experts (MoE)
**Beispiele:** Mistral Mixtral, Switch-Transformer
**Merkmale:**
- Mehrere spezialisierte Submodelle („Experten“), nur eine Teilmenge pro Eingabe aktiv.
- Router entscheidet über relevante Experten.
- **Stärken:** Effizienter, skalierbar auf große Modellgrößen.
- **Schwächen:** Komplexeres Training.

---

## 3. Reasoning-optimierte Modelle
**Beispiele:** Claude, GPT-4 (Variationen)
**Merkmale:**
- Fokus auf logisches Denken und Schritt-für-Schritt-Problemlösung.
- Oft durch Chain-of-Thought-Prompting optimiert.
- **Stärken:** Besser für komplexe Aufgaben wie Mathematik oder Code.
- **Schwächen:** Weniger „kreativ“.

---

## 4. Retrieval-Augmented Models (RAG)
**Beispiele:** RAG mit FAISS/Weaviate
**Merkmale:**
- Kombination aus LLM und externer Wissensdatenbank.
- **Stärken:** Aktualisierbares Wissen, weniger Halluzinationen.
- **Schwächen:** Abhängig von der Qualität der Datenbank.

---

## 5. Small Language Models (SLMs)
**Beispiele:** TinyLlama, Phi-2
**Merkmale:**
- Kompakte Versionen großer Modelle.
- **Stärken:** Schnell, ressourcenschonend, gut für Edge-Geräte.
- **Schwächen:** Geringere Leistung bei komplexen Aufgaben.

---

## 6. Multimodale Modelle
**Beispiele:** GPT-4V, LLaVA
**Merkmale:**
- Verarbeiten Text, Bilder, Audio oder Video.
- **Stärken:** Universell für multimodale Aufgaben.
- **Schwächen:** Höhere Komplexität.

---

## 7. Hybride Architekturen
**Beispiele:** MoE + RAG, Transformer + symbolische Logik
**Merkmale:**
- Kombination verschiedener Ansätze.
- **Stärken:** Flexibel, leistungsstark für spezifische Anwendungen.
- **Schwächen:** Komplexität in Training und Deployment.

---

## 8. Neuro-Symbolische Modelle
**Beispiele:** DeepProbLog
**Merkmale:**
- Neuronale Netze + symbolische Logik.
- **Stärken:** Interpretierbar, gut für formale Aufgaben.
- **Schwächen:** Weniger flexibel für unstrukturierte Daten.

---

## 9. Energy-Based Models (EBMs)
**Beispiele:** Experimentelle Ansätze
**Merkmale:**
- Nutzen Energie-Funktionen für Wahrscheinlichkeitsmodellierung.
- **Stärken:** Besser für Unsicherheitsquantifizierung.
- **Schwächen:** Komplexes Training.

---

## 10. Diffusionsmodelle für Sprache
**Beispiele:** DiffuseLM
**Merkmale:**
- Diffusionsprozesse für Textgenerierung.
- **Stärken:** Kontrolle über Generierungsprozess.
- **Schwächen:** Langsam, noch nicht ausgereift.

---

## 11. State Space Models (SSMs)
**Beispiele:** S4, H3
**Merkmale:**
- Ersetzen Attention durch rekurrente Zustandsräume.
- **Stärken:** Effizienter für lange Sequenzen.
- **Schwächen:** Weniger etabliert als Transformer.

---

## Vergleichstabelle

| Typ                | Skalierbarkeit | Effizienz | Reasoning | Multimodal | Aktualität |
|--------------------|----------------|-----------|-----------|------------|------------|
| Dense Models       | Hoch           | Mittel    | Mittel    | Nein       | Nein       |
| MoE                | Sehr hoch      | Hoch      | Mittel    | Nein       | Nein       |
| Reasoning-Modelle  | Mittel         | Mittel    | Hoch      | Nein       | Nein       |
| RAG                | Mittel         | Mittel    | Hoch      | Ja*        | Ja         |
| SLMs               | Niedrig        | Hoch      | Niedrig   | Nein       | Nein       |
| Multimodal         | Hoch           | Niedrig   | Mittel    | Ja         | Nein       |

*RAG kann mit multimodalen Datenbanken kombiniert werden.