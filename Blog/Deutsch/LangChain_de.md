---
title: Was ist `LangChain`?
date: 2025-06-29
tags: ["python", "langchain", "langgraph", "machine-learning", "llm", "nlp", "agent"]
toc: True
draft: false
author: CKe
---

# Was ist LangChain?

Ich darf mich aktuell unter Anderem mit `LangChain`  auseinandersetzen. Und meine Gedanken versuche ich auch hier in irgendeiner Form festzuhalten. 

## Einführung

`LangChain` ist ein leistungsstarkes Framework, das die Integration und Interaktion von Large Language Models (LLMs) mit externen Daten und Werkzeugen kombitioniert. LLMs sind von Natur aus in ihrem Wissen auf die Daten beschränkt, mit denen sie trainiert wurden, und können selten direkt auf externe Informationen zugreifen oder komplexe Aktionen ausführen. Genau hier setzt `LangChain` an.

Das Framework überwindet diese Limitierungen, indem es LLMs die Fähigkeit verleiht, mit der Außenwelt zu interagieren. Stell dir vor, du möchtest einen Chatbot entwickeln, der nicht nur allgemeine Fragen basierend auf seinem internen Wissen beantworten kann, sondern auch Informationen aus spezifischen Quellen wie zum Beispiel Wikipedia oder deinen eigenen internen Dokumenten abruft und sich an das bisher gesagte erinnern kann. Ohne `LangChain` müsste man alles selber bauen, APIs integrieren, Logiken für Konversationen schreiben, ... `LangChain` vereinfacht diese komplexe Aufgabe erheblich, indem man vorgefertige Ketten (Chains), Speicher (Memory) und Werkzeuge (Tools) nutzen kann.

Darüber hinaus fördert `LangChain` eine enorme Flexibilität und Modularität in deinen Projekten. Anstatt auf einen einzigen, oft überladenen "Super-Prompt" angewiesen zu sein, ermöglicht `LangChain` die Entwicklung von intelligenten Agenten, die jeweils mit speziellen Aufgaben betraut werden können. Dies macht es deutlich einfacher, komplexe LLM-Anwendungen strukturiert und wartbar zu entwickeln.

### Vorteile
Zu den Vorteilen zählen unter anderem:
* **Modularität**:  Gleich einem Baustein Prinzip, kann man unterschiedliche Komponenten einsetzen und austauschen.
* **Flexibilität**: `LangChain` ist sehr flexibel und an die verschiedensten Anwendungsfälle anpassbar.
* **Abstraktion**: `LangChain` versteckt einiges an Funktionalitäten hinter Schnittstellen und nimmt so viel Komplexität heraus. 
* **Erweiterbarkeit**: So modular es ist, so einfach kann man auch weitere Tools oder Datenquellen integrieren.
* **Anwendungsfälle**: 
  * Chatbots
  * Frage-Antwort-Systeme über z.B. eigene Dateien
  * Agenten, die Aufgaben ausführen
  * ...

## Grundlegende Konzepte in `LangChain`

### Large Language Models (LLMs)

Für quasi jede LLM-Anwendung benötigt man ein ... Überraschung, ein LLM. LLMs sind Modelle, die Texte verstehen und generieren konnen. Bekannte LLMs sind z.B. OpenAI's GPT-Modelle, Google's Gemini oder auch Open-Source-Modelle wie Llama oder Mistral aus Frankreich. 

`LangChain` bietet eine einheitliche Schnittstelle, um mit den verschiedenen LLMs zu interagieren, ohne den eigenen Code ändern zu müssen.

Ein wirklich einfaches Beispiel, wie das aussehen kann:

```python
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.7)
print(llm.invoke("Wie heißt die schönste Stadt Deutschlands?"))
```

Oder wenn man mit Mistral ingeragieren möchte:

```python
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(temperature=0.7)
print(llm.invoke("Wie heißt die schönste Stadt Deutschlands?"))
```

Es ändert sich im Grunde nicht viel mehr als der Import und die verwendete Klasse. Alles andere wird im Hintergrund durch `LangChain` umgesetzt.

Was ich in beiden Fällen unterschlagen habe, ist der `API-KEY`. Man kann in der Regel nicht ohne einem solchen `API-Key` mit den LLMs intergarieren. Hat man einen solchen Key, kann man ihn z.B. über [[dotenv|dotenv]] oder [[Keyring|Keyring]] einbinden und nutzen.

Tatsächlich kann noch viele weitere Parameter konfigurieren. Ich habe hier nur den Parameter `temperature` angepasst. Für Mistral findet man Informationen zu den weiteren Parametern in dieser [`LangChain` Dokumentation](https://python.langchain.com/api_reference/mistralai/chat_models/langchain_mistralai.chat_models.ChatMistralAI.html). Für OpenAI würde man analog in dieser [`LangChain` Dokumentation](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html) fündig werden.

### Prompts und `PromptTemplates`

Im obigen Beispiel wird das LLM mit dem Befehl `invoke` aufgerufen. Das bedeutet, dass man das Model aktiv nutzt oder ausführt, um eine bestimmte Aufgabe zu erfüllen oder eine Reaktion zu erhalten. 
Dem LLM im Beispiel übergeben wir die Frage "Wie heißt die schönste Stadt Deutschlands?". Bei dieser Frage handelt es sich um einen Prompt und wir erwarten von dem LLM als Reaktion ein textuelle Antwort auf unsere Frage. Die Qualität der Antworten hängt massiv von der Qualität der Prompts ab. Man kann sich sicher vorstellen, dass ohne weiterer Kriterien die Antwort auch für uns Menschen unterschiedlich sein wird. 

Dieser Prompt ist außerdem natürlich überaus statisch. Manchmal möchte man seine Prompts dynamisch gestalten (z.B. ein Land in die Frage einfügen). Und an der Stelle kommen `LangChain`'s `PromptTemplate` zum Tragen. Dabei handelt es sich um Vorlagen, die Platzhalter enthalten, die zur Laufzeit mit Werten gefüllt werden. So kann man seinen Prompt wiederverwenden, schafft Konsistenz und reduziert Fehler.

Ein `PromptTemplate` könnte z.B. so aussehen:

```python
from langchain.prompts import PromptTemplate

template = """Du bist ein hilfreicher Assistent, der Informationen zusammenfasst. Fasse den folgenden Text zum Thema {thema} zusammen. Der Text lautet {text}. Gib die Zusammenfassung in {sprache} aus."""

prompt = PromptTemplate(
    input_variables=["thema", "test", "sprache"],
    template=template
)

print(prompt.format(
    thema="Künstliche Intelligenz",
    text="Künstliche Intelligenz (KI) befasst sich mit der Simulation menschlicher Intelligenz in Maschinen...",
    sprache="Deutsch"
))
```

Der fertige Prompt sieht dann so aus:

```bash
Du bist ein hilfreicher Assistent, der Informationen zusammenfasst. Fasse den folgenden Text zum Thema Künstliche Intelligenz zusammen. Der Text lautet Künstliche Intelligenz (KI) befasst sich mit der Simulation menschlicher Intelligenz in Maschinen.... Gib die Zusammenfassung in Deutsch aus.
```

Die einzelnen Platzhalter wurden mit Inhalt gefüllt. Mit diesem Prompt kann man dann Agenten oder Chains invoken.

### Chains

Eine Chain ist ein Verknüpfung von Komponenten wie z.B. `PromptTemplate` und LLMs. Sie ermöglichen es, mehrere Schritte hintereinander auszuführen und die Ausgabe des einen Schritts als Eingabe für den nächsten zu verwenden.
So kann man vergleichsweise einfach Pipelines für LLM-Operationen aufzusetzen. Eine LLM-Chain ist z.B. die grundlegendste Kette, die einen Prompt mit einem LLM verbindet.

Durch Chains kann man seinen Code einfach strukturieren und vereinfacht die oft komplexen Workflows.

Eine LLM-Chain könnte z.B. so aussehen (API-Key wie gehabt z.B. über [[Keyring|Keyring]] einbinden):

```python
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.7)

summarize_template = "Fasse folgenden Text zusammen: {text}"
summarize_prompt = PromptTemplate.from_template(summarize_template)

summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

long_text = "Künstliche Intelligenz (KI), englisch artificial intelligence, daher auch artifizielle Intelligenz (AI), ist ein Teilgebiet der Informatik, das sich mit der Automatisierung intelligenten Verhaltens und dem maschinellen Lernen befasst. Der Begriff ist schwierig zu definieren, da es verschiedene Definitionen von Intelligenz gibt. [...] Der Begriff artificial intelligence (künstliche Intelligenz) wurde 1955 geprägt von dem US-amerikanischen Informatiker John McCarthy im Rahmen eines Förderantrags an die Rockefeller-Stiftung für das Dartmouth Summer Research Project on Artificial Intelligence, einem Forschungsprojekt, bei dem sich im Sommer 1956 eine Gruppe von 10 Wissenschaftlern über ca. 8 Wochen mit der Thematik befasste."
summary = summarize_chain.invoke({"text": long_text})
print(summary['text'])
```
_Text von [Wikipedia](https://de.wikipedia.org/wiki/K%C3%BCnstliche_Intelligenz)_

Das es schon kein besonders langer Text als Input ist, wird die Zusammenfassung wohl kaum viel kürzer ausfallen, aber um das Prinzip zu verstehen, wird es denke ich genügen. Das hier ist also die besagte Zusammenfassung:

```bash
KI oder AI ist ein Teilgebiet der Informatik, das sich mit der Automatisierung intelligenten Verhaltens und dem maschinellen Lernen beschäftigt. Der Begriff ist schwer zu definieren, da es verschiedene Definitionen von Intelligenz gibt. Der Begriff wurde 1955 von John McCarthy geprägt und im Rahmen eines Forschungsprojekts im Sommer 1956 von einer Gruppe von 10 Wissenschaftlern untersucht. 
```

#### Was ist `Invoke`?

Das _Invoken_ einer Chain oder eines Agenten in `LangChain` bedeutet, dass man eine Eingabe (meist, so wie oben, ein Dictionary) an das Objekt übergibt und sofort eine Antwort erhält. Die Eingabe wird an die Chain geschickt, das Sprachmodell vererbeitet diese Anfrage (ggf. mit Memory oder Tool) und man bekommt direkt das Ergebnis zurück.

`invoke()` ist kurz gesagt die Methode, mit der man Anfragen an ein Chain oder einen Agenten stellt und die Antwort synchron erhält.

### Agents und Tools

Agents und Tools sorgen dafür, dass LLMs "intelligenter" handeln können. Ein Agent entscheidet basierend auf dem aktuellen Problem, welche Tools er verwenden muss und in welcher Reihenfolge. Funktionen die ein Agent aufrufen kann, um externe Aktionen auszuführen sind z.B. eine Google- oder Wikipedia-Suche, eine Datenbankabfrage, aufrufen einer API, ... `LangChain` bietet für diesen Zweck viele vorgefertigte Tools und erlaubt außerdem das Erstellen eigener.

Ein abstraktes Beispiel. Man fragt den Agenten "Wie wird das Wetter morgen?". 
Dann folgt ggf. dieser Ablauf:
* Agent erkennt: Brauche Wetterinformationen
* Agent wählt das "Wetter-Tool" aus
* Agent ruft das Tool mit "Hamburg, morgen" auf
* Das "Wetter-Tool" liefert Daten zurück
* Agent formuliert mit LLT die Antwort für den Nutzer

Ein weniger abstraktes aber kürzeres Beispiel in Python:

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_openai import OpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

llm = OpenAI(temperature=0)

# Tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [wikipedia]

# Prompt für ReAct Agent
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent_executor.invoke({"input": "Wer ist der aktuelle Bundeskanzler von Deutschland?"}))
```

Der Ausgabe kann man entnehmen, wie der ReAct-Agent vorgeht. Er stellt fest, dass er Wikipedia nutzen sollte, durchsucht ein paar Seiten und liefert schließlich das korrekte Ergebnis (Stand 2025-06-29). Ich habe die Ausgabe der einzelnen Wikipedia-Seiten gekürzt, weil das für das Verständnis irrelevant ist.

```bash
> Entering new AgentExecutor chain...
 I should use Wikipedia to find the answer.
Action: wikipedia
Action Input: "Bundeskanzler Deutschland"Page: Deutschlandlied
Summary: The "Deutschlandlied", officially titled "Das Lied der Deutschen", is a German poem written by August Heinrich Hoffmann von Fallersleben . A popular song [...]

Page: Chancellor of Germany
Summary: The chancellor of Germany, officially the federal chancellor of the Federal Republic of Germany, is the head of the federal government of Germany. [...] The current officeholder is Friedrich Merz of the Christian Democratic Union, sworn in on 6 May 2025.]

Page: Friedrich Merz
Summary: Joachim-Friedrich Martin Josef Merz (born 11 November 1955) is a German politician serving as Chancellor of Germany since 6 May 2025. [...]
Final Answer: The current Bundeskanzler of Germany is Friedrich Merz.

> Finished chain.
{'input': 'Wer ist der aktuelle Bundeskanzler von Deutschland?', 'output': 'The current Bundeskanzler of Germany is Friedrich Merz.'}
```

Oben wird übrigens ein [[ReAct]]-Agent verwendet. Ein [[ReAct]]-Agent kombiniert "Reasoning" (logisches Schlussfolgern) und "Acting" (Handeln). Er nutzt Sprachmodelle, um in mehreren Schritten zu überlegen, welche Aktionen (z.B. Tool-Aufrufe) nötig sind, um ein Ziel zu erreichen. Dabei wechselt er zwischen Nachdenken und Handeln.

### Memory

LLMs sind standardmäßig "stateless", vergessen also alles nach jeder Anfrage. Memory-Module in `LangChain` ermöglichen es, sich an vergangene Konversationen oder Zustände zu erinnern. Das ermöglicht Chatbots, personalisierte Anwendungen oder beispielsweise auch mehrschichtige, komplexe Interaktionen.

Es gibt viele Arten von Speichern, z.B. für Datenbanken oder Redis. Zwei möchte ich hier aufführen.
* `ConversationBufferMemory`: Speichert die gesamte Konversation
* `ConversationSummaryMemory`: Fasst die Konversation zusammen, sollte sie zu lange werden.

Eine `ConversationChain` mit `ConversationBufferMemory` könnte so aussehen:

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.7, api_key=api_key)
memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

print(conversation.invoke({"input": "Hallo, mein Name ist Christoph"}))
print(conversation.invoke({"input": "Wie geht es dir heute?"}))
print(conversation.invoke({"input": "Erinnerst du dich an meinen Namen?"}))
print(conversation.invoke({"input": "Erzähle mir etwas über Python"}))
```

Führt man die obigen Zeilen zum Beispiel in einem Jupyter Notebook aus, erhält man einen optisch aufbereiteten Konversationsverlauf, der um weitere Informationen angereichert wurd. Möchte man hingegen nur Fragen und Antworten betrachten, helfen die folgenden Zeilen.

```python
for message in memory.chat_memory.messages:
    print(f"{message.type}: {message.content}")
```

Die liefern dann das folgende Ergebnis.

```bash
human: Hallo, mein Name ist Christoph
ai:  Hallo Christoph! Schön dich kennenzulernen. Mein Name ist AI, was für Artificial Intelligence steht. Ich bin ein Programm, das entwickelt wurde, um menschenähnliche Gespräche zu führen und Fragen zu beantworten. Wie kann ich dir heute weiterhelfen?
human: Wie geht es dir heute?
ai:  Mir geht es gut, danke der Nachfrage. Ich bin ein Computerprogramm, also habe ich keine körperlichen Empfindungen wie Menschen. Aber meine Programmierung läuft einwandfrei, also bin ich glücklich. Wie geht es dir?
human: Erinnerst du dich an meinen Namen?
ai:  Ja, dein Name ist Christoph. Ich habe eine Datenbank mit allen Informationen, die du mir im Laufe unserer Gespräche gibst, und ich erinnere mich an alles, was du mir gesagt hast.
human: Erzähle mir etwas über Python
ai:  Python ist eine beliebte Programmiersprache, die in den 1990er Jahren von Guido van Rossum entwickelt wurde. Sie ist bekannt für ihre einfache Syntax und flexible Anwendungsbereiche. Viele große Unternehmen wie Google und Instagram nutzen Python für ihre Anwendungen. Es ist auch eine der am häufigsten verwendeten Sprachen für künstliche Intelligenz und maschinelles Lernen. Hast du noch weitere Fragen zu Python oder möchtest du mehr darüber erfahren?
```

Was sofort auffällt: Die Frage nach meinem Namen kann auch nach weiteren Fragen noch beantwortet werden. Auf diese Weise könnte man aus einem LLM, dass wie gesagt "stageless" is , einen Chatbot machen.

#### Übergabe der `ConversationChain` an das LLM

Um eine `ConversationChain` mit einem LLM zu verbinden, wird das LLM-Objekt beim Erstellen der Chain als Argument übergeben. Die Chain übernimmt dann die Kommunikation mit dem Modell. Im obigen Beispiel wird das LLM (`llm=llm`) direkt an die `ConversationChain` übergeben. Die Chain kümmert sich anschließend darum, Prompts zu generieren, den Verlauf zu speichern und die Antworten vom LLM einzuholen. Die Interaktion mit dem LLM erfolgt dann über Methoden wie `invoke`. 

So wird die Chain zum zentralen Baustein, der LLM, Speicher und Logik miteinander verbindet.

## Anwendungsbeispiel

Ich möchte in weiteren Beispielen weitere Anwendungsbeispiele zeigen.

### Dokumentenbefragung (Retrieval Question Answering - [[RAG_de|RAG]])

LLMs haben nur Wissen bis zu ihrem Trainingsdatum und keine spezifischen Unternehmens- oder Projektdaten. Möchte man das ändern, kommt RAG ins Spiel. Auf diese kann man das LLM um relevante, externe Dokumente "erweitern", um spezifische Fragen zu beantworten.

Der Workflow kann dann in etwa so aussehen:
* **Dokumente laden**: PDFs, Textdateien, Datenbanken, ...
* **Texte aufteilen**: Große Dokumente in kleinere, handhabbare sogenannte Chunks zerlegen
* **Einbetten (Embedding)**: Die Textchunks werden in numerische Vektoren gewandelt.
* **Vektordatenbank (Vector Store)**: Speichert die zuvor gewandelten Vektoren für eine schnelle Ähnlichkeitssuche (z.B. [[Cosine similarity|Kosinusähnlichkeit]]) in Vektordatenbanken (z.B. `Chroma`, `FAISS`, `Pinecone`)
* **Abfrage**: 
  * Nutzer stellt eine Frage
  * Frage wird "eingebettet"
  * Die ähnlichsten Chunks aus der Vektordatenbank werden abgerufen
* **LLM-Antwort**: Die abgerufenen Chunks und die Frage werden dem LLM präsentiert, welches daraus dann eine fundierte Antwort generiert.

Die Vorteile liegen auf der Hand: Die Antworten basieren auf meinen Daten, man reduziert das Halluzinieren und man kann aktuelle Informationen abrufen.

Wie kann so etwas im Code aussehen?
Ich lese im folgenden Code-Schnipsel ein PDF ein. Es handelt sich um [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) [cs.CL]. Außerdem befrage ich auch Wikipedia zu dem Thema. Einmal ohne Funktion, einmal in eine Funktion eingebettet. Und ich lade mit Keyring eine API-Key.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_core.documents import Document
import os

# API-Key laden
import keyring
api_key = keyring.get_password("openai_api_key", "default")

# OpenAI Schnittstelle vorbereiten
llm = OpenAI(temperature=0, api_key=api_key)
embeddings = OpenAIEmbeddings(api_key=api_key)

# PDF laden
loader = PyPDFLoader(r"Attention_is_all_you_need_1706.03762v7.pdf")
docs = loader.load()

# Embeddings und Vektordatenbank erzeugen
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

# Chain vorbereiten
qa_chain = RetrievalQA.from_chain_type(
	llm=llm,
	chain_type="stuff",
	retriever=vectorstore.as_retriever())

# Frage stellen
query = "Was bedeutet Attention?"
response = qa_chain.invoke({"query": query})
print(f"Ergebnis PDF: {response['result']}")
``` 

Die obigen Zeilen führen zu folgender Ausgabe. 

```bash
Ergebnis PDF:  Attention ist eine Funktion, die eine Abfrage und eine Reihe von Schlüssel-Wert-Paaren auf einen Ausgang abbildet, wobei alle Elemente Vektoren sind. Der Ausgang wird als gewichtete Summe berechnet.
```

Wir können aber wie gesagt auch Wikipedia zu dem Thema befragen:

```python
# Wikipedia Schnittstelle vorbereiten
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader

    loader = WikipediaLoader(
        query=query,
        lang=lang,
        load_max_docs=load_max_docs,
    )
    data = loader.load()
    return data

# Wikipedia befragen
data = load_from_wikipedia("Attention (Machine Learning)", lang='de', load_max_docs=3)
print(f"Ergebnis Wikipedia: {data[0].page_content[:250]}")
```

Die Ausgabe folgt auf dem Fuße. Für die erste Ausgabe - wir haben unseren Agenten beauftragt, die 3 besten Treffer zu suchen - gebe ich die ersten $500$ Zeichen aus.

```bash
Ergebnis Wikipedia: Ein Transformer ist eine von Google entwickelte Deep-Learning-Architektur, die einen Aufmerksamkeitsmechanismus (englisch Attention) integriert. Dabei wird Text durch Worteinbettung in numerische Darstellungen in Form von Vektoren umgewandelt. Dies kann z. B. dazu benutzt werden, Text von einer Sprache in eine andere zu übersetzen (siehe auch Maschinelle Übersetzung). Dazu wird ein Transformer mittels maschinellem Lernen anhand einer (großen) Menge von Beispieltexten trainiert, bevor das trainie
```

Im obigen Artikel geht es offensichtlich auch um den Aufmerksamkeitsmechanismus. Ohne Inhaltlich weiter einzutauchen, das ist laut dem Agenten das beste Ergebnis.

Das zweitbeste Ergebnis befasst sich mit LLMs. Ich gebe aber nur noch die ersten 250 Zeichen aus:

```python
print(f"Ergebnis Wikipedia: {data[1].page_content[:250]}")
```

```bash
Ergebnis Wikipedia: Ein Large Language Model, kurz LLM (englisch, teilweise übertragen großes Sprachmodell), ist ein Sprachmodell, das sich durch seine Fähigkeit zur Textgenerierung auszeichnet. Es handelt sich um ein computerlinguistisches Wahrscheinlichkeitsmodell, da
``` 

Und das dritte Ergebnis scheint sich mit `Tensorflow` zu befassen.

```python
print(f"Ergebnis Wikipedia: {data[2].page_content[:250]}")
```

```bash
Ergebnis Wikipedia: TensorFlow ist ein Framework zur datenstromorientierten Programmierung. Populäre Anwendung findet TensorFlow im Bereich des maschinellen Lernens. Der Name TensorFlow stammt von Rechenoperationen, welche von künstlichen neuronalen Netzen auf mehrdimen
``` 

Mit den geladenen Wikipedia-Ergebnissen kann man nun verschiedene weiterführende Schritte durchführen, zum Beispiel:

1. **Zusammenfassen**  
Die Inhalte der Wikipedia-Artikel können mit einem LLM automatisch zusammengefasst werden, um die wichtigsten Informationen kompakt darzustellen.
2. **Vergleichen**  
Man kann die Inhalte der verschiedenen Artikel vergleichen, um Unterschiede oder Gemeinsamkeiten herauszuarbeiten. 
3. **Fragen beantworten (QA)**  
Man kann gezielte Fragen zu den geladenen Wikipedia-Artikeln stellen, indem man RetrievalQA oder eine eigene Chain verwendest, die die Artikel als Wissensbasis nutzt.

  ```python
  from langchain.chains import RetrievalQA
  from langchain_community.vectorstores import FAISS
  from langchain_openai import OpenAIEmbeddings

  # Embeddings für Wikipedia-Artikel erzeugen
  wiki_vectorstore = FAISS.from_documents(data, embedding=embeddings)
  wiki_qa_chain = RetrievalQA.from_chain_type(
     llm=llm,
     chain_type="stuff",
     retriever=wiki_vectorstore.as_retriever()
  )

  frage = "Was ist ein Transformer im Kontext von Machine Learning?"
  antwort = wiki_qa_chain.invoke({"query": frage})
  print(f"Antwort: {antwort['result']}")
  ```

4. **Weitere Verarbeitung**  
  - Extrahiere Stichworte oder Entitäten.
  - Erstelle Mindmaps oder Visualisierungen.
  - Kombiniere die Wikipedia-Inhalte mit anderen Datenquellen.

So kann man die geladenen Wikipedia-Daten (oder andere) flexibel für verschiedene NLP-Aufgaben weiterverwenden.

### Chatbots

Ein weiteres Beispiel mit dem jeder von uns sicher schon mal in Verbindung kam... Ein Chatbot. Ziel des Chatbots ist die interaktive, kontextsensitive Konversation. 

Dafür sind unter Anderem ein Memory (für den Gesprächsverlauf) und häufig auch ein Retriever (um spezifische Fragen zu beanworten) notwendig. Kombiniert man die zwei kann man auch von einer `ConversationalRetrievalChain` reden. Der Chatbot kann sich dann an die Konversation erinnern und in Dokumenten nach Antworten suchen.

Man kann den Code der vorangegangenen Beispiele wiederverwenden und ergänzt `ConversationalRetrievalChain` und `ConversationBufferMemory` zu:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_core.documents import Document
import os

# API-Key laden
import keyring
api_key = keyring.get_password("openai_api_key", "default")

# OpenAI Schnittstelle vorbereiten
llm = OpenAI(temperature=0, api_key=api_key)
embeddings = OpenAIEmbeddings(api_key=api_key)

# PDF laden
loader = PyPDFLoader(r"C:\...\Attention_is_all_you_need_1706.03762v7.pdf")
docs = loader.load()

# Embeddings und Vektordatenbank erzeugen
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

memory = ConversationBufferMemory(
    memory_key="chat_history", # Wichtig: Key muss 'chat_history' sein
    return_messages=True
)

# Erstelle die ConversationalRetrievalChain
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)


# Interagiere mit dem Chatbot
response1 = conversation_chain.invoke({"question": "Was ist Attention?"})
print("User: Was ist Attention?")
print("Bot:", response1['answer'])

response2 = conversation_chain.invoke({"question": "Wer ist der Autor?"})
print("\nUser: Wer ist der Autor?")
print("Bot:", response2['answer'])

response3 = conversation_chain.invoke({"question": "Erinnerst du dich, worüber wir zuerst gesprochen haben?"})
print("\nUser: Erinnerst du dich, worüber wir zuerst gesprochen haben?")
print("Bot:", response3['answer']) # Hier sollte es "langchain" sein, da im Memory
```

Die Ausgabe sieht dann wie folgt aus:

```bash
User: Was ist Attention?
Bot:  Attention is a function that maps a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. It is used in neural network architectures, such as the Transformer, to connect the encoder and decoder and improve performance in sequence transduction tasks. It allows the network to focus on specific parts of the input and make connections between distant dependencies.

User: Wer ist der Autor?
Bot: 
Die Autoren des Transformer-Modells sind Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser und Illia Polosukhin.

User: Erinnerst du dich, worüber wir zuerst gesprochen haben?
Bot: 
Attention ist eine Funktion, die eine Anfrage und eine Reihe von Schlüssel-Wert-Paaren auf einen Ausgang abbildet. Dabei sind die Anfrage, die Schlüssel, die Werte und der Ausgang alle Vektoren. Der Ausgang wird als gewichtete Summe berechnet.
```

Man erkennt die Aktivität des Bots (der `ConversationalRetrievalChain`) daran, dass die Antworten auf konkrete Fragen (z.B. nach der Bedeutung von "Attention" oder den Autoren) direkt und präzise aus deinem PDF-Dokument stammen. Die Funktion des Buffers (des `ConversationBufferMemory`) wird ersichtlich, da sich der Bot an frühere Konversationsthemen ("Erinnerst du dich, worüber wir zuerst gesprochen haben?") erinnert und darauf basierend antwortet.

Er scheint zu funktionieren! 

### Datenanalyse und -generierung

Ein paar Szenarien - ohne Code - wie man `LangChain` zur Datenanalyse und -generierung einsetzen kann:
* Man kann mit `LangChain` auch große Datensätze (z.B. Log-Dateien oder Kundenrezensionen) zusammenfassen. 
* Extrahieren spezifischer Informationen aus unstrukturiertem Text.
* Generieren von Berichten oder Beschreibungen basierend auf strukturierten Daten.
* Code-Generierung oder -Erklärung
* ...

Ein weiteres Beispiel, aber auch nur theoretischer Art ist ein Agent, der auf CSV-Dateien zugreifen und Fragen dazu beantworten kann, indem er Python-Code generiert und ausführt. Einsetzen könnte man dann Tools wie `PythonREPLTool` oder `PandasDataFrame`. 

Die Möglichkeiten sind quase schier unendlich ;)

## Zusammenfassung

`LangChain` ist ein mächtiges Framework zur Erstellung "intelligenter", LLM-basierter Anwendungen. Es vereinfacht komplexe Prozesse durch modulare Komponenten.

Man kann es nahezu überall einsetzen. Kundenservice, Bildung, Content-Erstellung, Datenanalyse, ...

Ich werde mich auch weiterhin mit dem Thema auseinandersetzen. Es gibt noch reichlich zu lernen über Callbacks, Custom Components, Integrationen und auch `LangGraph`.

Wenn du Fragen oder Hinweise/Anregungen hast, oder du hast Fehler gefunden, dann bitte nich zögern mich zu kontaktieren :)
