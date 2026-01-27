---
title: Theorie und Formeln hinter dem Entscheidungsbaum
date: 2023-02-20
tags: [machine-learning]     # TAG names should always be lowercase
toc: true
translations:
  en: "en/blog/Decision-Tree-and-Random-Forest/Theory-and-formulas-behind-the-decision-tree"
---

In den letzten beiden Artikeln haben wir gelernt, wie [[2023-01-16-Der Entscheidungsbaum|Entscheidungsbaum]] und [[v|Random Forests]] programmiert werden können. In diesem Artikel soll es um die Theorie dahinter gehen. Doch zunächst eine kleine Wiederholung.

![Entscheidungsbaum Wein-Datensatz — Quellcode am Ende des Beitrags, Bild vom Autor](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*SYOAspZ_sVzD0oNpCeflTA.png)

Entscheidungsbäume sind ein bekanntes Modell im maschinellen Lernen und werden oft für Klassifikations- und Regressionsprobleme verwendet. Sie stellen eine Hierarchie von Entscheidungen und Vorhersagen dar, die auf bestimmten Merkmalen und Regeln basieren.

Bei einem Random Forest wiederum handelt es sich um mehrere unkorrelierte Entscheidungsbäume, die während des Lernprozesses unter einer bestimmten Randomisierung entstanden sind und mit zufälligen Untermengen der Ausgangsdaten trainiert wurden. Die endgültige Vorhersage des Modells wird durch die Abstimmung der Vorhersagen aller Entscheidungsbäume ermittelt.

Doch nun zu den Konzepten und der Theorie hinter den Entscheidungsbäumen und somit letztlich auch den Random Forests.

## Informationsgewinn und Entropie

Eines der wichtigsten Konzepte bei Entscheidungsbäumen ist die Informationsgewinnung (Information Gain). Dies ist eine Messgröße, die bestimmt, welches Merkmal am besten geeignet ist, um die Daten zu teilen. Der Information Gain wird berechnet, indem man den Unterschied zwischen dem ursprünglichen Entropiewert und dem Entropiewert nach der Teilung berechnet.

### Entropie

Der Entropiewert misst die Unordnung oder Heterogenität einer Gruppe von Beobachtungen. Eine höhere Entropie bedeutet eine höhere Unordnung und eine geringere Vorhersagegenauigkeit. Der Entropiewert wird berechnet, indem man die Wahrscheinlichkeit jeder Klasse in der Gruppe und diese Werte logarithmisch berechnet und summiert.

Die Entropie einer diskreten Zufallsvariable $X$ mit möglichen Ausprägungen ${x_1, x_2, …, x_n}$ und Wahrscheinlichkeiten ${p_1, p_2, …, p_n}$ ist definiert als:

$$ H(X) = - \sum_{i=1}^N P_i \cdot log_2(P_i)$$

### Information Gain

Der Information Gain wird berechnet, indem man den Unterschied zwischen dem ursprünglichen Entropiewert und dem Entropiewert nach der Teilung berechnet. Zum Beispiel teilen wir eine Gruppe von Beobachtungen basierend auf einem bestimmten Merkmal, wie etwa dem Alter. Wenn das Alter ein gutes Teilungsmerkmal ist, werden die Beobachtungen in klarere und homogenere Gruppen unterteilt, was zu einer geringeren Entropie führt. Der Information Gain ist dann der Unterschied zwischen dem ursprünglichen Entropiewert und dem Entropiewert nach der Teilung.

Der Information Gain, der von einer Entscheidung A abhängt und den Unterschied in der Entropie vor und nach der Entscheidung widerspiegelt, ist definiert als:

$$IG(A) = H(S) - \sum p(t) \cdot H(t) $$

wobei $S$ die Menge aller Beispiele ist, die entschieden werden sollen, $t$ die Menge der Beispiele, bei denen die Entscheidung $A$ getroffen wurde, $p(t)$ die Wahrscheinlichkeit, dass ein Beispiel in $t$ ist, und $H(t)$ die Entropie von $t$ ist.

Mit anderen Worten misst der Information Gain, wie viel Entropie verringert wird, wenn eine bestimmte Entscheidung getroffen wird, im Vergleich zur Entropie der ursprünglichen Menge von Beispielen. Ein hoher Information Gain bedeutet, dass eine Entscheidung nützlicher ist, da sie mehr Entropie verringert und damit mehr Information bereitstellt.

## Gini-Impurität
Die Gini-Impurität (Gini-Impurity) ist ein Maß für die Unreinheit einer Menge von Daten. Sie wird in der Entscheidungsbaumklassifikation verwendet, um die Güte von Teilungen (Splits) in den Daten zu bewerten. Die Gini-Impurity misst, wie oft ein zufällig ausgewähltes Element in der Menge falsch klassifiziert würde, wenn es zufällig entsprechend der Verteilung der Klassen in der Menge klassifiziert würde. Ein Wert von $0$ bedeutet, dass die Menge vollständig rein ist (alle Elemente haben die gleiche Klasse), während ein Wert von $1$ bedeutet, dass die Menge vollständig unrein ist (die Elemente sind gleichmäßig auf die Klassen verteilt). Die Gini-Impurity kann wie folgt berechnet werden.

$$ Gini = 1 - \sum_{i=1}^n (p_i)^2 $$

$n$ ist die Anzahl der Klassen und $p_i$ die relative Häufigkeit von Instanzen der Klasse $i$ in der Teilung.

Bei der Berechnung der Gini-Impurity wird angenommen, dass jede Teilung mindestens eine Instanz jeder Klasse enthält. Wenn eine Teilung keine Instanzen einer bestimmten Klasse enthält, ist die Gini-Impurity $0$, was bedeutet, dass die Teilung eine perfekte Klassenseparation bietet.
Die Wahl zwischen der Verwendung von Information Gain oder Gini-Impurity hängt oft von den spezifischen Anforderungen und Vorlieben des Benutzers ab. Information Gain ist eine bekannte Messgröße und bietet eine klare mathematische Basis, aber Gini-Impurity kann schneller berechnet werden und ist in manchen Fällen einfacher zu interpretieren.
Zusammenfassend berechnet die Gini-Impurity die Unreinheit einer bestimmten Teilung in einem Entscheidungsbaum und ist ein wichtiger Indikator für die Vorhersagegenauigkeit. Je niedriger die Gini-Impurität, desto besser ist die Klassenseparation und desto höher ist die Vorhersagegenauigkeit.

## Überanpassung
Ein weiteres wichtiges Konzept oder Problem bei Entscheidungsbäumen ist Überanpassung (Overfitting). Dies tritt auf, wenn ein Modell zu komplex ist und zu sehr auf die Trainingsdaten angepasst wird, anstatt allgemeingültige Vorhersagen zu treffen. Overfitting kann zu einer schlechten Vorhersagegenauigkeit auf neuen, ungesehenen Daten führen.

Es gibt verschiedene Techniken, die Overfitting bei Entscheidungsbäumen verhindern können:
1. **Beschneidung (Pruning)**: Bei dieser Technik wird der Baum nach dem Erstellen beschnitten, um unnötig komplexe Strukturen zu entfernen. Es gibt zwei Hauptarten des Beschneidens: Reduced Error Pruning und Cost Complexity Pruning.
2. **Minimierung der Blätterzahl**: Eine andere Möglichkeit, Overfitting zu vermeiden, besteht darin, die Anzahl der Blätter zu minimieren. Dies kann durch Erhöhung des Schwellenwerts oder durch Anwendung von Regeln erreicht werden, um Blätter zusammenzufassen.
3. **Verwendung von Ensembles**: Eine Kombination mehrerer Entscheidungsbäume, die auf unterschiedlichen Trainingsdaten trainiert wurden, kann die Vorhersagegenauigkeit verbessern und Overfitting verringern. Die bekanntesten Verfahren sind Bagging und Boosting.
4. **Verwendung von Regulierungstermen**: Es ist möglich, Regulierungsterme wie $L1$ und $L2$ zu verwenden, um den Einfluss von bestimmten Variablen auf den Entscheidungsprozess zu verringern und somit Overfitting zu verhindern.
5. **Datenaugmentation**: Ein weiterer Ansatz besteht darin, die Daten zu erweitern, indem künstlich neue Datenpunkte erzeugt werden, die den bereits vorhandenen ähneln. Dies kann helfen, die Varianz zu verringern und Overfitting zu reduzieren.

Es ist wichtig zu beachten, dass es keine einheitliche Methode gibt, um Overfitting zu vermeiden, und dass die Wahl der Methode von der spezifischen Problemstellung abhängt. Sollte dich dieses Thema stärker interessieren, lass es mich wissen und ich werde es aufbereiten.

## Zusammenfassung
Zusammenfassend sind Entscheidungsbäume eine leistungsstarke Methode im maschinellen Lernen und basieren auf Information Gain, Gini-Impurity, Overfitting und Stoppbedingungen. Obwohl sie nicht für jedes Problem geeignet sind, sind sie ein wichtiger Teil des Werkzeugkastens jedes Datenwissenschaftlers.
