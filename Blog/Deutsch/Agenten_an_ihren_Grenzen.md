---
title: Agenten an ihren Grenzen
tags:
    - langchain
    - langgraph
    - python
    - machine-learning
    - coding
    - llm
    - nlp
date: 2025-10-05
toc: true
---

# Agenten an ihren Grenzen

## Die Grenzen der LangChain-Kette

In [[ReAct_de|ReAct]] haben wir über einen vielseitigen Agenten gesprochen, ihn programmiert und eingesetzt. [[ReAct_de|ReAct]]-Agenten sind großartig für kleinere Workflows oder Tool-Aufrufe und auch für Prototypen geeignet.

Bei komplexeren Aufgaben stoßen sie jedoch schnell an ihre Grenzen. Ihre Architektur ist eine lineare Kette mit einer einfachen, begrenzten Schleife. Sobald Aufgaben wie Selbstkorrektur, Kollaboration zwischen mehreren Agenten oder komplexe Entscheidungslogik gefragt sind, reicht diese Struktur nicht mehr aus.

Um diese Herausforderungen zu meistern, lohnt sich der Wechsel zu einer sogenannten Graphen-Struktur.

## Ein Beispiel: Ein [[ReAct_de|ReAct]]-Agent stößt an seine Grenzen

Stellen wir uns vor, wir möchten einen Agenten bauen, der Python-Code testen, bei Fehlern selbstständig korrigieren und je nach Ergebnis entweder eine Analyse durchführen oder an einen Menschen eskalieren soll. Mit einem klassischen [[ReAct_de|ReAct]]-Agenten in LangChain ist das nicht möglich – die Grenzen werden schnell sichtbar.

**LangChain ReAct-Agent (klassisch)**

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

def test_code(code: str) -> str:
    try:
        local_vars = {}
        exec(code, {}, local_vars)
        return "Success"
    except Exception as e:
        return f"Error: {e}"

def analyze_output(output: str) -> str:
    return f"Analysis: {output}"

def escalate_to_human(output: str) -> str:
    return f"Escalated to human: {output}"

tools = [
    Tool(name="CodeTester", func=test_code, description="Tests Python code."),
    Tool(name="Analyzer", func=analyze_output, description="Analyzes output."),
    Tool(name="Escalator", func=escalate_to_human, description="Escalates to human.")
]

llm = OpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

# The agent tries once, cannot retry, cannot branch in code, no shared state
result = agent.run("Test this code and analyze the output or escalate if there is an error:\nprint(x)")
print(result)
```

Ich denke der Großteil des Codes ist den meisten verständlich. Ein Exkurs die `test_code`-Methode könnte aber noch etwas mehr Klarheit schaffen. 

**Was passiert, Schritt für Schritt**

1. `local_vars = {}`: Erstellt ein leeres Dictionary für lokale Variablen
2. `exec(code, {}, local_vars)`: Führt den übergebenen Python-Code aus
    * `code` = der zu testende Code (in unserem Fall "print(x)")
    * `{}` = globale Variablen (leer)
    * `local_vars` = lokale Variablen (leer)
3. **Bei Erfolg**: Gibt `"Success"` zurück
4. **Bei Fehler**: Fängt die Exception ab und gibt die Fehlermeldung zurück

**Was passiert mit `print(x)`?**
Wenn der Agent `print(x)` testet, passiert folgendes:

* Python versucht die Variable `x` zu finden
* `x` ist aber nirgends definiert (weder in globalen noch lokalen Variablen)
* Python wirft einen `NameError: name 'x' is not defined`
* Die `test_code`-Funktion fängt diesen Fehler ab
* Sie gibt zurück: `"Error: name 'x' is not defined"`
  
**Das Problem des klassischen Agenten**
Der Agent bekommt diese Fehlermeldung, kann aber:

* Nicht automatisch nochmal versuchen
* Nicht den Code reparieren (z.B. `x = 42` hinzufügen)
* Nicht explizit entscheiden: "Fehler → eskalieren" oder "Erfolg → analysieren"

Er bleibt nach dem ersten Fehler einfach stehen, weil die gesamte Retry- und Branching-Logik im Prompt versteckt ist, nicht im Code.

Und das ist der Kern des Problems: Klassische [[LangChain_de|LangChain]]-Agenten können nicht strukturiert und nachvollziehbar auf Fehler reagieren.

### Fazit

Dieses Beispiel zeigt, dass klassische [[ReAct_de|ReAct]]-Agenten bei komplexeren Aufgaben schnell an ihre Grenzen stoßen. Sie können keine iterativen Fehlerbehebungen durchführen, nicht flexibel verzweigen und keinen gemeinsamen Status verwalten. 


## Drei zentrale Grenzen klassischer Agenten

### 1. Fehlende explizite Schleifensteuerung und Selbstkorrektur

Klassische Agenten-Frameworks wie [[LangChain_de|LangChain]] bieten keine echte programmatische Iteration oder Selbstkorrektur. Der Agent kann Fehler somit nicht autonom beheben oder Ergebnisse gezielt verfeinern. Die Standard-Agenten-Schleife ist darauf ausgelegt, möglichst schnell zu einer finalen Antwort zu gelangen. Interne Iterationen sind nicht vorgesehen.

Wenn beispielsweise ein Tool-Aufruf fehlschlägt, weil ein Code-Test einen Fehler liefert, gibt es keine klare Anweisung im Code, den Fehlerbehebungszyklus zu wiederholen. Die gesamte Verantwortung für Korrekturen liegt beim LLM und ist im Prompt versteckt, was fehleranfällig ist und sich nur schwer nachvollziehen lässt. Auch das Debugging gestaltet sich schwierig, da die Schleifen- und Korrekturlogik nicht im Code, sondern im Prompt verborgen ist.

### 2. Komplexes Branching und Blackbox-Logik

Ein weiteres Problem klassischer [[LangChain_de|LangChain]]-Ketten ist das Handling von komplexen Verzweigungen. Oft muss sich der Workflow je nach Ergebnis eines Zwischenschritts unterschiedlich verhalten. In [[LangChain_de|LangChain]] muss die gesamte Verzweigungslogik jedoch im Prompt formuliert werden ("Wenn Ergebnis X, dann Tool A, sonst Tool B"). Das macht die Logik schwer wartbar, fehleranfällig und unflexibel – eine kleine Änderung im Prompt kann den gesamten Workflow beeinflussen.

Es fehlt ein klarer Mechanismus im Code, um den nächsten Schritt basierend auf einem Zwischenstand zu definieren. Die Steuerung liegt im LLM und nicht im Code. Mit wachsender Komplexität werden solche Ketten schnell unübersichtlich und schwer zu pflegen. Graphenstrukturen hingegen sind modular, leichter zu erweitern und erlauben explizites Branching.

**Typische Szenarien:** Human-in-the-Loop, Eskalationspfade oder die Auswahl zwischen verschiedenen Spezial-Tools lassen sich mit klassischen Ketten kaum sauber abbilden.

### 3. State Management und Multi-Agenten-Kollaboration

Gerade bei Projekten, in denen mehrere Agenten zusammenarbeiten, stößt die klassische Kette an ihre Grenzen. [[LangChain_de|LangChain]] ist primär zustandslos – für Langzeitprojekte oder kollaborative Szenarien ist das unzureichend. Es gibt keine native, saubere Möglichkeit, die Kontrolle dynamisch zwischen spezialisierten Agenten (z.B. Recherche-Agent ↔ Analyse-Agent) zu übergeben.

Agenten benötigen oft einen gemeinsamen, veränderbaren Projektstatus (Shared State), auf den alle zugreifen und den sie aktualisieren können, bevor die Kontrolle weitergegeben wird. Die lineare Kette bietet hierfür keinen nativen Mechanismus. Zudem lassen sich in Graph-Workflows einzelne Knoten gezielt parallelisieren oder optimieren – das ist mit linearen Ketten nicht möglich.

### Weitere Herausforderungen

- **Integration von Human-in-the-Loop:** Mit Graphen können menschliche Eingriffe (z. B. Review-Schritte) explizit eingebaut werden.
- **Transparenz und Debugging:** In klassischen [[LangChain_de|LangChain]]-Ketten ist es oft schwer, den genauen Ablauf und die Fehlerquellen nachzuvollziehen, weil viel Logik im Prompt und nicht im Code steckt. Mit [[LangGraph_de|LangGraph]] ist der Ablauf hingegen explizit im Code sichtbar und besser testbar.

Und vor genau diesen Herausforderungen stehe ich auch. In einem Framework an dem ich mitwirke haben wir mehrere Agenten linear implementiert. An mancher Stelle wäre es aber hilfreich, könnten wir einen Human-in-the-loop oder Verzweigungen und Schleifen integrieren. Das ist auch einer der Gründe warum ich mich aktuell mit [[LangGraph_de|LangGraph]] beschäftige. Wir werden das Framework auf [[LangGraph_de|LangGraph]] umstellen.

## Fazit: Die Lösung ist der Graph

Alle diese Probleme haben eine gemeinsame Ursache: Die lineare, statische Natur klassischer LangChain-Agenten. Was wir brauchen, ist eine Architektur, die flexibel verzweigen, Schleifen durchführen und einen gemeinsamen Zustand verwalten kann.

Genau hier kommt [[LangGraph_de|LangGraph]] ins Spiel – eine Erweiterung von [[LangChain_de|LangChain]], die auf Graphen basiert. Statt einer starren Kette kannst du mit [[LangGraph_de|LangGraph]] Knoten (Nodes) und Kanten (Edges) explizit definieren. Das ermöglicht echte Schleifen, saubere Verzweigungen und einen gemeinsamen State, den alle Agenten nutzen können.

Neugierig geworden? Dann schau dir [[LangGraph_de|LangGraph]] mal genauer an oder kontaktiere mich doch direkt.
