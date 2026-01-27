---
title: "Meine Entwicklungsumgebung: Docker, Compose, Codespaces"
date: 2025-09-28
tags:
  - python
  - docker
  - github
  - codespace
toc: true
translations:
  en: "en/blog/Docker,-Compose--and--Codespaces"
---
# Meine Entwicklungsumgebung: Wie ich mit Docker, Compose und Codespaces arbeite

Ich habe mir überlegt, mal aufzuschreiben, wie ich mit Docker usw. arbeite. Manche – oder die meisten/alle – Konzepte dürften euch bekannt sein, aber ich mache das ja auch, um meine Gedanken festzuhalten. Und eventuell kann die eine oder der andere noch etwas mitnehmen.

## Einführung

Ich entwickle (privat) unter anderem mit Docker, Compose und auch Codespaces. In den nächsten Kapiteln habe ich festgehalten, wie und auch warum. Es wird auch wieder ein praktisches Beispiel mit Python folgen.

### Was ist Docker?

Also... Was ist Docker? Bei Docker handelt es sich um eine Technologie zur Orchestrierung von Anwendungen. Man kann z.B. Python-Code in einer isolierten und gleichzeitig portablen Umgebung, dem Container, entwickeln und ausführen.

### Warum Containerisierung?

Hast du schon mal _"Auf meinem Rechner funktioniert es"_ gehört? Oder hast du es selber schon geäußert? Das gehört mit Docker der Vergangenheit an. Mit Docker schafft man konsistente und reproduzierbare Entwicklungsumgebungen, die auf verschiedensten Systemen ausgeführt werden können. Ich arbeite auf meinem Laptop beispielsweise mit einem Dual-Boot-System bestehend aus Windows 11 und Fedora. Wenn ich nun Docker einsetze, dann kann ich ohne weiteres auf beiden Systemen an dem Projekt arbeiten.

## Docker – der Grundstein der Containerisierung

### Grundkonzept

* Mit Docker erzeugt man ein sogenanntes **Image**. Dieses Image ist eine schreibgeschützte Vorlage mit den Anweisungen zum Erstellen eines Containers. Dazu nutzt man das sogenannte **`Dockerfile`**. Dazu später mehr.
* Das **`Dockerfile`** ist eine Textdatei, die die Schritte zum Erzeugen eines Docker-**Images** definiert.
* Ein Docker-**Container** ist eine ausführbare Instanz eines **Images**. Dabei handelt es sich um eine leichtgewichtige und wie erwähnt isolierte Entwicklungsumgebung. Man könnte das alles auch mit Virtual Machines lösen, aber leichtgewichtig ist das dann vermutlich nicht mehr.
* Der **Docker-Daemon** ist der Dienst, der im Hintergrund läuft und die Container verwaltet.

### Wichtige Docker-Befehle

* `docker build`: Erstellt ein Image aus einem `Dockerfile`.
* `docker run`: Startet einen Container aus einem Image.
* `docker ps`: Listet die laufenden Container auf.
* `docker pull`: Lädt ein Image aus einem Repository herunter.
* `docker push`: Lädt ein Image in ein Repository hoch.

## Docker Compose – Vereinfachte Orchestrierung

### Grundkonzepte

Bei **Docker Compose** handelt es sich um ein Tool zur Verwaltung von Multi-Container-Anwendungen (wie wir es später noch im Beispiel sehen werden). Es nutzt eine einzige Konfigurationsdatei, die sogenannte `docker-compose.yml`, um alle Dienste (Container), Netzwerke und Volumes zu definieren und erlaubt die Orchestrierung der gesamten Anwendungsarchitektur mit nur einem einzigen Befehl.

### Wichtige Docker-Compose Befehle

* `docker-compose up`: Baut und startet alle im `.yml` definierten Dienste.
* `docker-compose down`: Stoppt und entfernt alle Dienste.

## Praktisches Beispiel mit Python

Im Folgenden werde ich eine einfache Web-Anwendung zeigen, die _"Hello, Docker!"_ anzeigen soll. Dafür werde ich Python und das Flask-Modul einsetzen.

### Einfache Flask-Anwendung (mit Docker)

Beginnen wir mit dem `Dockerfile`.
```Dockerfile
# Basis-Image
FROM python:3.9-slim
# Arbeitsverzeichnis im Container
WORKDIR /app
# Abhängigkeiten kopieren
COPY requirements.txt .
# Abhängigkeiten installieren
RUN pip install -r requirements.txt
# Code kopieren
COPY . .
# Container-Port freigeben
EXPOSE 5000
# Befehl zum Starten der App
CMD ["python", "app.py"]
```

In dem Dockerfile wird die Datei `requirements.txt` genutzt, um die Abhängigkeiten zu installieren. Im aktuellen Fall ist diese Datei recht leer, weil wir nur Flask benötigen.

```Text
Flask
```

Zuletzt benötigen wir noch `app.py`:

```python
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_docker():
  return "Hello, Docker!"

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0")
```

Das ist im Grunde keine große oder komplizierte App. Und um sie nun auszuführen, müssen wir sie in der Konsole eigentlich nur noch starten.

```bash
docker build -t my-python-app .
docker run -p 5000:5000 my-python-app
```

Die erste Zeile baut ein Docker-Image mit dem Namen `my-python-app` aus dem aktuellen Verzeichnis. Wir müssen uns beim Ausführen also in besagtem Ordner befinden. Dadurch werden zunächst von Docker alle erforderlichen Schritte durchgeführt, wie z.B. das Installieren von `Flask`. Ich habe die Ausgabe mal auf das Wesentliche gekürzt:

```bash
[+] Building 22.6s (11/11) FINISHED
 => [internal] load build definition from Dockerfile
 => [internal] load metadata for docker.io/library/python:3.9-slim
 => [1/5] FROM docker.io/library/python:3.9-slim
 => [internal] load build context
 => [2/5] WORKDIR /app
 => [3/5] COPY requirements.txt .
 => [4/5] RUN pip install -r requirements.txt
 => [5/5] COPY . .
 => exporting to image
 => naming to docker.io/library/my-python-app:latest
 ```

Die zweite Zeile startet einen Container aus diesem Image und leitet Port `5000` des Containers auf Port `5000` des Hosts weiter. Das Weiterleiten des Ports ist erforderlich, damit du von deinem Host-Rechner (z.B. deinem lokalen Computer oder Codespace) auf Dienste zugreifen kannst, die im Container laufen.

```bash
(base) gutsc@G15Christoph:/mnt/d/Coding/Blog_Code_Snippets/docker_compose_codespaces$ docker run -p 5000:5000 my-python-app
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://172.17.0.2:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 126-682-939
```

Die dort angegebene URL `http://127.0.0.1:5000` kann ich nun natürlich besuchen. `172.17.0.2:5000` ist übrigens die interne IP des Containers im Docker-Netzwerk und ist nur innerhalb dieses Netzwerkes von z.B. anderen Docker-Containern erreichbar.

Ich bekomme jetzt übrigens schlicht einen Text `Hello, Docker!` angezeigt.

> **Hinweis:** Der in den Beispielen verwendete Flask-Server ist nur für Entwicklungszwecke gedacht. Für produktive Umgebungen sollte ein WSGI-Server wie Gunicorn oder uWSGI verwendet werden.

### Multi-Container-Anwendung (mit Docker Compose)

Das zweite Beispiel ist wieder eine Flask-App (der `web`-Teil im `yml`-File). Nun wird sie mit einer Redis-Datenbank (dem `redis`-Teil) kommunizieren. Und das sind dann nun zwei getrennte Container, die über das besagte interne Netzwerk miteinander kommunizieren.

Dazu brauchen wir ein `yml`-File: `docker-compose.yml`

```yml
version: '3.8'

services:
  web:
    build: .  # Verweist auf das Dockerfile im aktuellen Verzeichnis
    ports:
      - "5000:5000"  # Exponiert Port 5000 des Web-Containers
    volumes:
      - .:/app  # Live-Reload: Verknüpft das Host-Verzeichnis mit dem Container. Änderungen am Code werden sofort übernommen.
    depends_on:
      - redis  # Startet den Web-Service erst, wenn Redis läuft

  redis:
    image: "redis:alpine"  # Verwendet ein fertiges Image von Docker Hub
```

Und es gibt auch wieder eine `app.py`. Es handelt sich um eine kleine Webanwendung mit Flask, die einen Zähler für Seitenaufrufe speichert – und zwar in einer Redis-Datenbank.

```python
from flask import Flask
import redis
import time

app = Flask(__name__)
# Verbindet sich mit dem Redis-Container, der den Hostnamen "redis" hat
cache = redis.Redis(host='redis', port=6379)

def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)

@app.route('/')
def hello():
    count = get_hit_count()
    return 'Hello from Docker! I have been seen {} times.\n'.format(count)

if __name__ == "__main__":
    # WICHTIGE ÄNDERUNG: Setzen Sie host='0.0.0.0'
    app.run(host="0.0.0.0", debug=True)
```

Und nun starten wir unsere Anwendung.

```bash
docker-compose up -d
```

Der obige Befehl führt intern nun mehrere Schritte aus.

1. Er liest die `docker-compose.yml`-Datei.
2. Für den `web`-Teil erkennt der Service nun die Anweisung `build: .`, sucht im aktuellen Verzeichnis nach einem `Dockerfile` und erkennt, dass er ein Image erzeugen muss. Der nun nächste Abschnitt ist wie im vorigen Beispiel.
3. Gleichzeitig sorgt der Befehl dafür, dass das `redis:alpine`-Image heruntergeladen und der Redis-Container gestartet wird.

Das `-d` bedeutet, dass die Container im Hintergrund (oder im **d**etached Modus) gestartet werden. Wenn man das Flag nicht setzt, wird es in der Konsole gegebenenfalls etwas unübersichtlich, weil die Log-Ausgaben der Container ausgegeben werden ;) 
Sollte man mal etwas an einer Datei verändert haben, die benötigt wird, um das Image zu erstellen (also z.B. `requirements.txt`, `.py`-Dateien, `Dockerfile`, ...) muss man das Image neu bauen lassen. Dazu fügt man einfach das Flag `--build` hinzu:

```bash
docker-compose up -d --build
```

Wenn nun also das Image neu gebaut wird, bekommt man zunächst eine Ausgabe, die der aus dem ersten Beispiel sehr ähnlich ist. Ist das Image bereits erzeugt, also beim zweiten Start, sieht die Ausgabe in etwa wie folgt aus:

```bash
[+] Running 3/3
 ✔ Network docker_example_default    Created  0.1s
 ✔ Container docker_example-redis-1  Started  0.9s
 ✔ Container docker_example-web-1    Started  1.2s
```

**Was bedeutet diese Ausgabe?**
Zunächst wird angezeigt, dass Docker Compose ein eigenes Netzwerk für die Anwendung erstellt hat. Innerhalb dieses Netzwerks können die Container miteinander kommunizieren. Und dann wird noch zurückgegeben, dass sowohl der `redis`- als auch der `web`-Container gestartet wurden.

Über `http://127.0.0.1:5000/` kann ich nun auf die Web-Oberfläche zugreifen. Und jedes Mal, wenn ich das tue, wird der Counter in der Datenbank inkrementiert, was sich so darstellt: `Hello from Docker! I have been seen 7 times.`

Mit `docker ps` kannst du dir nun noch anschauen, welche Container gestartet wurden. Das geht aber natürlich auch in der Desktop-Anwendung von Docker. Dort kann man ggf. auch z.B. den Python-Code debuggen, wenn etwas nicht startet.

```bash
CONTAINER ID   IMAGE                COMMAND                  CREATED         STATUS         PORTS                                         NAMES
b21b0f302613   docker_example-web   "python app.py"          8 minutes ago   Up 8 minutes   0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp   docker_example-web-1
91cea7ff0771   redis:alpine         "docker-entrypoint.s…"   8 minutes ago   Up 8 minutes   6379/tcp                                      docker_example-redis-1
```

Um die Docker-Container zu beenden, kannst du sie relativ einfach mit einem einzelnen Befehl herunterfahren:

```bash
docker-compose down
```

Die Ausgabe sollte dann in etwa so etwas zeigen:

```bash
[+] Running 3/3
 ✔ Container docker_example-web-1    Removed  0.6s
 ✔ Container docker_example-redis-1  Removed  0.4s
 ✔ Network docker_example_default    Removed  0.5s
```

## Github Codespaces – Der cloudbasierte Teil

Nicht immer habe ich Zugriff auf meinen Laptop. Dann ist es hilfreich, wenn man über den Browser auf sein Projekt zugreifen und es eben auch dort ausführen kann. Hier kommt z.B. **Codespaces** ins Spiel.

### Was ist Codespaces?

**Codespaces** ist eine cloud-basierte **Visual Studio Code**-Umgebung, die direkt in **GitHub** integriert ist. Du kannst damit komplette Entwicklungsumgebungen in wenigen Minuten starten – direkt aus deinem Repository heraus, ohne lokale Einrichtung. Codespaces nutzt Docker-Container als Grundlage, sodass du eine identische Umgebung für alle Teammitglieder sicherstellen kannst.

Um einen Codespace zu starten, klickst du im gewünschten GitHub-Repository einfach auf den grünen "Code"-Button und dann auf "Create codespace on main" (oder dem gewünschten Branch).

### Vorteile gegenüber lokaler Entwicklung

* **Schneller Einstieg:** Ein Klick auf "Code → Codespaces" im GitHub-Repo genügt, und nach wenigen Minuten steht dir eine vollwertige Entwicklungsumgebung zur Verfügung.
* **Keine lokale Installation nötig:** Weder Python, noch Docker, noch sonstige Abhängigkeiten müssen auf deinem Rechner installiert sein – alles läuft im Browser.
* **Teamarbeit:** Mehrere Entwickler:innen können gleichzeitig im selben Codespace arbeiten oder eigene Instanzen mit identischer Konfiguration starten. Das sorgt für Konsistenz und weniger "funktioniert nur bei mir"-Probleme.
* **Ressourcenunabhängig:** Codespaces laufen unabhängig von deiner lokalen Hardware. Auch auf schwächeren Geräten oder Tablets kannst du anspruchsvolle Projekte bearbeiten.
* **Vorkonfiguriert:** Über die `.devcontainer/devcontainer.json` kannst du genau festlegen, welche Tools, Extensions und Umgebungsvariablen beim Start bereitstehen.
* **Automatisierte Umgebung:** Änderungen an der Entwicklungsumgebung (z.B. neue Abhängigkeiten) werden versioniert und stehen sofort allen zur Verfügung.
* **Integration mit Docker und Compose:** Du kannst Docker-Container und Compose-Setups wie gewohnt nutzen, inklusive Terminalzugriff und Port-Forwarding.

> **Hinweis:** GitHub Codespaces ist für private Accounts nur mit einem begrenzten kostenlosen Kontingent nutzbar. Darüber hinaus können Kosten entstehen. Die aktuellen Limits findest du in der [GitHub-Dokumentation](https://docs.github.com/de/billing/concepts/product-billing/github-codespaces).

> **Hinweis:** Codespaces haben eigene Ressourcenlimits (CPU, RAM, Laufzeit). Für sehr große, rechenintensive oder GPU-basierte Projekte ist Codespaces nicht immer geeignet.


### Integration in den Docker-Workflow

- Verwendet die **`.devcontainer/devcontainer.json`**-Datei zur Konfiguration der Umgebung. Hier kannst du z.B. Extensions, Startbefehle und Umgebungsvariablen definieren.
- Nutzt die vorhandene `docker-compose.yml`-Datei, um die Dienste zu starten. So kannst du auch im Codespace Multi-Container-Anwendungen entwickeln und testen.
- Über das integrierte Terminal kannst du alle Docker-Befehle wie gewohnt ausführen.
- Geöffnete Ports (z.B. von Flask oder anderen Webdiensten) werden automatisch erkannt und können im Browser geöffnet werden.


## Zusammenfassung und Ausblick

### Fazit

Ich nutze die oben genannten Werkzeuge für eine moderne Entwicklungsumgebung über Systeme und Hardware hinweg.
- **Docker** als Grundlage für die Kapselung.
- **Docker Compose** für die lokale Orchestrierung.
- **GitHub Codespaces** für die teamorientierte, cloud-basierte Entwicklung.

### Nächster Schritt: Skalierung in der Produktion (k8s)

Darüber hinaus gibt es noch weitere sehr hilfreiche Tools. Zum Beispiel **Kubernetes** (k8s). **k8s** ist eines der führenden Tools für die Container-Orchestrierung in Produktionsumgebungen. Es automatisiert die Skalierung, das Management und die Ausfallsicherheit von containerisierten Anwendungen in einem sogenannten Cluster. Kubernetes wird meist erst ab einer gewissen Projekt- oder Teamgröße relevant. In Kürze möchte ich auch darüber schreiben.