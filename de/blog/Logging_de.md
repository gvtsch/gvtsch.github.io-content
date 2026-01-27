---
title: Python Logging-Modul
date: 2025-10-07
tags: ["python", "coding"]
toc: True
draft: false
author: CKe
translations:
  en: "en/blog/Logging"
---

# Logging in Python

Ich habe in der Vergangenheit häufig mit `print`-Statements gearbeitet, um darüber informiert zu werden, was meine Skripte aktuell machen. Je nachdem, wie viele Informationen ich erhalten wollte, habe ich einen sogenannten `verbosity`-Level gesetzt. Das waren z.B. `VERBOSE_INFO = 1` oder `VERBOSE_DEBUG = 2`. Diesen Level habe ich dann beim Aufrufen meines Skriptes gesetzt/übergeben und ein typischer `print` sah z.B. so aus:

```python
if self._verbosity >= VERBOSE_DEBUG:
    print(f"DEBUG: {<debug_info>}")
else:
    print(f"INFO: {<info>}")
```

oder

```python
print(f"DEBUG: {<debug_info>}" if self._verbosity >= VERBOSE_DEBUG else f"INFO: {<info>}")
```

Und im Grunde hat das auch gut funktioniert. Man konnte es als Callback ausführen und beliebig kompliziert gestalten. Und weil ich damit gut zurecht kam, musste ich erst zufällig über das `logging`-Modul stolpern. Und darum geht es nun ...

## Warum Logging statt Print-Statements?

Nachdem ich mich näher mit dem `logging`-Modul beschäftigt habe, muss ich zugeben: Es ist deutlich eleganter als meine selbstgebaute Verbosity-Lösung. Das Modul bietet viel mehr Flexibilität und Kontrolle über die Ausgaben, ohne dass man sich selbst um die ganze Logik kümmern muss.

## Grundlegende Konfiguration

```python
import logging

# Basis-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

Das war's schon. Diese paar Zeilen erstellen einen Logger mit:
- Zeitstempel bei jeder Nachricht
- Log-Level zur Kategorisierung
- Modulname zur Identifikation der Quelle

## Log-Level verstehen

Python's Logging-System bringt bereits fertige Level mit:

- `DEBUG`: Detaillierte Informationen für Debugging
- `INFO`: Allgemeine Informationen über den Programmablauf
- `WARNING`: Hinweise auf potenzielle Probleme
- `ERROR`: Fehler, die das Programm beeinträchtigen
- `CRITICAL`: Schwerwiegende Fehler

Das ist im Grunde das, was ich früher mit meinen `VERBOSE_DEBUG = 2` und `VERBOSE_INFO = 1` gemacht habe, nur standardisiert.

## Praktisches Beispiel

So sieht das dann in der Praxis aus:

```python
def prepare_titanic_data(file_path='train.csv', test_size=0.2, random_state=42):
    logger.info("Loading data...")
    df = pd.read_csv(file_path)
    X = df.drop('survived', axis=1)
    y = df['survived']
    
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info("Applying preprocessing...")
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    logger.info("Preprocessing completed")
    return results
```

Viel sauberer als meine if-else-Konstrukte von früher.

## Was mir besonders gefällt

* **Flexibilität**: Man kann das Log-Level beim Programmstart setzen, ohne den Code zu ändern – das ging zwar mit meiner alten Lösung auch, aber mit dem Logging-Modul ist es standardisierter.
* **Weniger Code**: Keine eigene Verbosity-Logik mehr nötig.
* **Standard**: Jeder Python-Entwickler versteht es sofort.

## Log-Level einstellen

```python
# Nur Warnungen und Fehler
logging.basicConfig(level=logging.WARNING)

# Alles anzeigen (auch DEBUG)
logging.basicConfig(level=logging.DEBUG)

# In Datei schreiben
logging.basicConfig(filename='app.log', level=logging.INFO)
```

Das ist deutlich einfacher als meine alten Verbosity-Parameter.

## Was ich gelernt habe

Bisher habe ich es mit meiner eigenen Logik gelöst.

```python
print("Data loaded")  # Immer sichtbar
if self._verbosity >= VERBOSE_DEBUG:
    print(f"DEBUG: Processing {len(data)} rows...")  # Nur bei hohem Level
```

Mittlerweile nutze ich auch das Logging-Modul.

```python
logger.info("Loading data...")  # Klarer Prozess-Schritt
logger.debug(f"Processing {len(data)} rows...")  # Automatisch filterbar
```

## Fazit

Das Logging-Modul macht genau das, was ich früher manuell gebaut habe, nur besser und standardisierter. 