---
title: Der Random Forrest
date: 2023-02-09
tags: [machine-learning, python]   
toc: true
translations:
  en: "en/blog/Decision-Tree-and-Random-Forest/The-Random-Forrest"
---
 
 Im letzten Artikel habe ich dir beschrieben, was ein [[2023-01-16-Der Entscheidungsbaum|Entscheidungsbaum]] ist. Es handelt sich dabei um einen strukturierten Prozess, um Entscheidungen zu treffen. Entscheidungsbäume werden häufig für Klassifikations- oder Regressionsverfahren eingesetzt. In diesem Artikel möchte ich dir nun zeigen, was ein Random Forest ist (im deutschen könnte man diesen Namen mit Zufallswald übersetzen). Wir werden uns erneut auf Python Code stürzen. Das bedeutet auch, dass ich dir wieder Quellcode im Github-Repository zur Verfügung gestellt habe.


## Der Random Forest

Bei einem Random Forest handelt es sich um mehrere unkorrelierte Entscheidungsbäume. Jeder einzelne dieser Entscheidungsbäume ist während des Lernprozesses unter einer bestimmten Randomisierung entstanden und wurde mit zufälligen Untermengen der Ausgangsdaten trainiert. Die endgültige Vorhersage des Modells wird durch die Abstimmung der Vorhersagen aller Entscheidungsbäume ermittelt. So wird die Vorhersagegenauigkeit erhöht und das Modell robuster gegenüber Überanpassung (Overfitting). Versuchen wir zum Beispiel ein Klassifikationsproblem zu lösen, darf jeder Baum im Wald eine Entscheidung treffen und es kommt sozusagen zu einer Mehrheitsabstimmung: Die am häufigsten gewählte Klasse bestimmt die endgültige Klassifikation. Random Forests lassen sich aber auch für Regressionsprobleme einsetzen. Wie bei den Entscheidungsbäumen handelt es sich auch bei einem Random Forest um einen Algorithmus des überwachten Lernens bzw. des supervised learnings. Darüber hinaus ist ein Random Forest ein sogenanntes Ensemble-Modell.

* Ein **Ensemble Modell** ist eine Methode des Maschinellen Lernens, bei der mehrere Modelle zusammengeführt werden, um die Vorhersagegenauigkeit zu erhöhen.

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*PuENdSyzCsYguK3XGNsg1g.png)
_Ensemble Model_

Im obigen Bild sind drei Entscheidungsbäume des programmierten Random Forest dargestellt. Der Quellcode für den Random Forest folgt im nächsten Abschnitt.

## Der Programmcode

Einiges in den nächsten Zeilen sollte schon aus dem vorigen Artikel des Entscheidungsbaums bekannt sein. Deswegen werde ich manche Zeilen / Blöcke zusammenfassen und nur kurz beschreiben. Auf Besonderheiten werde ich natürlich näher eingehen.

```python
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
```

Die Imports sind die selben wie beim Entscheidungsbaum. Des Vergleichs wegen greifen wieder auf den Wein-Datensatz zurück.

```python
dataset = load_wine()
x, y = dataset.data, dataset.target
x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size = 0.3, random_state = 42)
```

Es folgt das Aufteilen in Trainings- und Testdatensatz.

```python
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 4, 8, 10],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt', 'log2'],
    'n_estimators': [10, 20, 50, 100, 200]
}
```

Im obigen Code-Block werden die Hyperparameter für die `GridSearchCV` definiert. Diese sind weitestgehend die gleichen wie beim Entscheidungsbaum. Ich habe allerdings `n_estimators` ergänzt.

* `n_estimators` ist ein Hyperparameter beim Random Forest und gibt die Anzahl von Entscheidungsbäumen an, die verwendet werden sollen. Ein höherer Wert von `n_estimators` führt in der Regel zu besseren Ergebnissen, aber es kann auch länger dauern, das Modell zu trainieren. Es ist ein Trade-off zwischen Vorhersagegenauigkeit und Trainingszeit.


```python
clf = RandomForestClassifier(random_state = 42)
grid_cv = GridSearchCV(clf, parameters, cv = 10, n_jobs = -1)
grid_cv.fit(x_train, y_train)
```

Ähnlich dem Entscheidungsbaum wird dann der Klassifizierer — nun der `RandomForestClassifier` — instanziert und die `GridSearchCV` initialisiert und durchgeführt. Die Ergebnisse liefert der folgende Code-Block:

```python
print(f"Parameters of best model: {grid_cv.best_params_}")
print(f"Score of best model: {grid_cv.best_score_}")
```
    Parameters of best model: {'criterion': 'gini', 'max_depth': 4, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 20}
Score of best model: 0.9839743589743589

Das sind die besten Hyperparameter laut obiger `GridSearchCV`. Mit `n_estimators = 20` und demnach $20$ einzelnen Entscheidungsbäumen, erreichen wir einen Score von $0.98…$ Zur Erinnerung: Der Score eines einzelnen Entscheidungsbaums aus dem vorangeganenen Artikel beträgt $0.95…$

Und wenn man das trainierte Modell nun auf die Testdaten loslässt, erhält man einen Score von $0.94…$

```python
score = grid_cv.score(x_test, y_test)
print(f"Accuracy: {score}")
```
    Accuracy: 0.9444444444444444

Und auch hier wieder zum Vergleich mit dem einzelnen Entscheidungsbaum. Dieser lieferte eine Genauigkeit von 0.88…

Die eingangs erwähnte höhere Genauigkeit eines Random Forest gegenüber einem einzelnen Entscheidungsbaum konnte also zumindest für diesen Datensatz bestätigt werden.

## Zusammenfassung

Wir haben in den obigen Abschnitten gelernt, was ein Random Forest ist und wie man diesen mit Sklearn programmieren kann. Wir haben versucht uns dabei möglichst nah am Entscheidungsbaum des vorigen Artikels zu orientieren und so die Ergebnisse vergleichen zu können.
