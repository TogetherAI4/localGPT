```md
# LocalGPT: Sichere, lokale Unterhaltungen mit Ihren Dokumenten 🌐

🚨🚨 Sie können localGPT auf einer vorkonfigurierten [Virtual Machine](https://bit.ly/localGPT) ausführen. Vergewissern Sie sich, den Code PromptEngineering zu verwenden, um 50% Rabatt zu erhalten. Ich werde eine kleine Provision erhalten!

**LocalGPT** ist eine Open-Source-Initiative, die es Ihnen ermöglicht, mit Ihren Dokumenten zu interagieren, ohne Ihre Privatsphäre zu gefährden. Mit allem, was lokal läuft, können Sie sicher sein, dass keine Daten Ihren Computer verlassen. Tauchen Sie ein in die Welt der sicheren, lokalen Dokumenteninteraktionen mit LocalGPT.

## Funktionen 🌟
- **Höchste Privatsphäre**: Ihre Daten bleiben auf Ihrem Computer, was eine 100%ige Sicherheit gewährleistet.
- **Vielseitige Modellunterstützung**: Integrieren Sie nahtlos eine Vielzahl von Open-Source-Modellen, darunter HF, GPTQ, GGML und GGUF.
- **Vielfältige Einbettungen**: Wählen Sie aus einer Reihe von Open-Source-Einbettungen.
- **Verwenden Sie Ihr LLM erneut**: Nach dem Download können Sie Ihr LLM ohne wiederholte Downloads wiederverwenden.
- **Chatverlauf**: Erinnert sich an Ihre vorherigen Unterhaltungen (in einer Sitzung).
- **API**: LocalGPT verfügt über eine API, die Sie für den Aufbau von RAG-Anwendungen verwenden können.
- **Grafische Benutzeroberfläche**: LocalGPT wird mit zwei GUIs geliefert, eines verwendet die API und das andere ist eigenständig (basierend auf Streamlit).
- **GPU-, CPU- und MPS-Unterstützung**: Unterstützt mehrere Plattformen out-of-the-box. Unterhalten Sie sich mit Ihren Daten unter Verwendung von `CUDA`, `CPU` oder `MPS` und mehr!

## Tauchen Sie tiefer ein mit unseren Videos 🎥
- [Detaillierte Code-Durchlauf](https://youtu.be/MlyoObdIHyo)
- [Llama-2 mit LocalGPT](https://youtu.be/lbFmceo4D5E)
- [Hinzufügen von Chatverlauf](https://youtu.be/d7otIM_MCZs)
- [LocalGPT - Aktualisiert (17.09.2023)](https://youtu.be/G_prHSKX9d4)

## Technische Details 🛠️
Durch Auswahl der richtigen lokalen Modelle und der Leistung von `LangChain` können Sie die gesamte RAG-Pipeline lokal ausführen, ohne dass Daten Ihre Umgebung verlassen, und mit vernünftiger Leistung.

- `ingest.py` verwendet `LangChain`-Tools, um das Dokument zu analysieren und lokal Einbettungen mit `InstructorEmbeddings` zu erstellen. Die Ergebnisse werden dann in einer lokalen Vektordatenbank mit `Chroma`-Vektorspeicher gespeichert.
- `run_localGPT.py` verwendet ein lokales LLM, um Fragen zu verstehen und Antworten zu erstellen. Der Kontext für die Antworten wird aus dem lokalen Vektorstore extrahiert, indem eine Ähnlichkeitssuche durchgeführt wird, um das richtige Stück Kontext aus den Dokumenten zu lokalisieren.
- Sie können dieses lokale LLM durch ein beliebiges anderes LLM von HuggingFace ersetzen. Stellen Sie sicher, dass das von Ihnen ausgewählte LLM im HF-Format vorliegt.

Dieses Projekt wurde von dem ursprünglichen [privateGPT](https://github.com/imartinez/privateGPT) inspiriert.

## Aufgebaut mit 🧩
- [LangChain](https://github.com/hwchase17/langchain)
- [HuggingFace LLMs](https://huggingface.co/models)
- [InstructorEmbeddings](https://instructor-embedding.github.io/)
- [LLAMACPP](https://github.com/abetlen/llama-cpp-python)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)

# Umgebung einrichten 🌍

1. 📥 Klonen Sie das Repository mit Git:

```shell
git clone https://github.com/PromtEngineer/localGPT.git
```

2. 🐍 Installieren Sie [conda](https://www.anaconda.com/download) für die Verwaltung von virtuellen Umgebungen. Erstellen und aktivieren Sie eine neue virtuelle Umgebung.

```shell
conda create -n localGPT python=3.10.0
conda activate localGPT
```

3. 🛠️ Installieren Sie die Abhängigkeiten mit pip

Um Ihre Umgebung einzurichten, um den Code auszuführen, installieren Sie zunächst alle Anforderungen:

```shell
pip install -r requirements.txt
```

***LLAMA-CPP installieren:***

LocalGPT verwendet [LlamaCpp-Python](https://github.com/abetlen/llama-cpp-python) für GGML (Sie benötigen llama-cpp-python <=0.1.76) und GGUF-Modelle (llama-cpp-python >=0.1.83).

Wenn Sie BLAS oder Metal mit [llama-cpp](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal) verwenden möchten, können Sie entsprechende Flags setzen:

Für `NVIDIA`-GPU-Unterstützung, verwenden Sie `cuBLAS`

```shell
# Beispiel: cuBLAS
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```

Für Apple Metal (`M1/M2`) Unterstützung, verwenden Sie

```shell
# Beispiel: METAL
CMAKE_ARGS="-DLLAMA_METAL=on"  FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```
Für weitere Details, siehe [llama-cpp](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal)

## Docker 🐳

Die Installation der erforderlichen Pakete für die GPU-Inferenz auf NVIDIA-GPUs wie gcc 11 und CUDA 11 kann Konflikte mit anderen Paketen in Ihrem System verursachen.
Als Alternative zu Conda können Sie Docker mit der bereitgestellten Dockerdatei verwenden.
Es enthält CUDA, Ihr System benötigt lediglich Docker, BuildKit, Ihren NVIDIA-GPU-Treiber und das NVIDIA-Container-Toolkit.
Erstellen

 Sie mit `docker build -t localgpt .`, BuildKit ist erforderlich.
Docker BuildKit unterstützt derzeit keine GPU während der *docker build*-Zeit, nur während der *docker run*-Zeit.
Führen Sie mit `docker run -it --mount src="$HOME/.cache",target=/root/.cache,type=bind --gpus=all localgpt` aus.

## Testdatensatz

Zu Testzwecken wird dieses Repository mit der [Verfassung der USA](https://constitutioncenter.org/media/files/constitution.pdf) als Beispieldatei mitgeliefert.

## Importieren Ihrer EIGENEN Daten.
Legen Sie Ihre Dateien in den Ordner `SOURCE_DOCUMENTS`. Sie können mehrere Ordner innerhalb des Ordners `SOURCE_DOCUMENTS` platzieren, und der Code wird Ihre Dateien rekursiv lesen.

### Unterstützte Dateiformate:
LocalGPT unterstützt derzeit die folgenden Dateiformate. LocalGPT verwendet `LangChain` zum Laden dieser Dateiformate. Der Code in `constants.py` verwendet ein `DOCUMENT_MAP`-Wörterbuch, um ein Dateiformat auf den entsprechenden Loader zuzuordnen. Um ein anderes Dateiformat zu unterstützen, fügen Sie einfach dieses Wörterbuch mit dem Dateiformat und dem entsprechenden Loader aus [LangChain](https://python.langchain.com/docs/modules/data_connection/document_loaders/) hinzu.

```shell
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}
```

### Eingestellt

Führen Sie den folgenden Befehl aus, um alle Daten einzulesen.

Wenn Sie `cuda` auf Ihrem System eingerichtet haben.

```shell
python ingest.py
```
Sie sehen eine Ausgabe wie diese:
<img width="1110" alt="Screenshot 2023-09-14 at 3 36 27 PM" src="https://github.com/PromtEngineer/localGPT/assets/134474669/c9274e9a-842c-49b9-8d95-606c3d80011f">


Verwenden Sie das Gerätetypargument, um ein bestimmtes Gerät anzugeben.
Zum Ausführen auf `cpu`

```sh
python ingest.py --device_type cpu
```

Um auf `M1/M2` auszuführen

```sh
python ingest.py --device_type mps
```

Verwenden Sie die Hilfe für eine vollständige Liste der unterstützten Geräte.

```sh
python ingest.py --help
```

Dies erstellt einen neuen Ordner namens `DB` und verwendet ihn für den neu erstellten Vektorstore. Sie können so viele Dokumente einlesen, wie Sie möchten, und alle werden in der lokalen Einbettungsdatenbank akkumuliert.
Wenn Sie von einer leeren Datenbank aus starten möchten, löschen Sie `DB` und lesen Sie Ihre Dokumente erneut ein.

Hinweis: Wenn Sie dies zum ersten Mal ausführen, benötigt es Internetzugriff, um das Einbettungsmodell herunterzuladen (Standard: `Instructor Embedding`). Bei den nachfolgenden Ausführungen verlässt keine Daten Ihre lokale Umgebung, und Sie können Daten ohne Internetverbindung einlesen.

## Stellen Sie Fragen an Ihre Dokumente, lokal!

Um mit Ihren Dokumenten zu chatten, führen Sie den folgenden Befehl aus (standardmäßig wird es auf `cuda` ausgeführt).

```shell
python run_localGPT.py
```
Sie können auch den Gerätetyp angeben, genau wie bei `ingest.py`

```shell
python run_localGPT.py --device_type mps # um auf Apple-Silizium auszuführen
```

Dies lädt den eingelesenen Vektorstore und das Einbettungsmodell. Sie erhalten eine Aufforderung:

```shell
> Geben Sie eine Abfrage ein:
```

Nachdem Sie Ihre Frage eingegeben haben, drücken Sie die Eingabetaste. LocalGPT benötigt einige Zeit basierend auf Ihrer Hardware. Sie erhalten eine Antwort wie unten dargestellt.
<img width="1312" alt="Screenshot 2023-09-14 at 3 33 19 PM" src="https://github.com/PromtEngineer/localGPT/assets/134474669/a7268de9-ade0-420b-a00b-ed12207dbe41">

Sobald die Antwort generiert wurde, können Sie eine weitere Frage stellen, ohne das Skript erneut auszuführen. Warten Sie einfach auf die erneute Aufforderung.

***Hinweis:*** Wenn Sie dies zum ersten Mal ausführen, benötigt es eine Internetverbindung, um das LLM herunterzuladen (Standard: `TheBloke/Llama-2-7b-Chat-GGUF`). Danach können Sie Ihre Internetverbindung trennen, und die Skriptinferenz funktioniert trotzdem. Keine Daten verlassen Ihre lokale Umgebung.

Geben Sie `exit` ein, um das Skript zu beenden.

### Zusätzliche Optionen mit run_localGPT.py

Sie können das Flag `--show_sources` mit `run_localGPT.py` verwenden, um anzuzeigen, welche Abschnitte vom Einbettungsmodell abgerufen wurden. Standardmäßig werden 4 verschiedene Quellen/Abschnitte angezeigt. Sie können die Anzahl der Quellen/Abschnitte ändern

```shell
python run_localGPT.py --show_sources
```

Eine andere Option besteht darin, den Chatverlauf zu aktivieren. ***Hinweis***: Dies ist standardmäßig deaktiviert und kann mit dem Flag `--use_history` aktiviert werden. Das Kontextfenster ist begrenzt, daher verwendet das Aktivieren von History es und kann überlaufen.

```shell
python run_localGPT.py --use_history
```

Sie können Benutzerfragen und Modellantworten mit dem Flag `--save_qa` in eine csv-Datei `/local_chat_history/qa_log.csv` speichern. Jede Interaktion wird gespeichert.

```shell
python run_localGPT.py --save_qa
```

# Führen Sie die grafische Benutzeroberfläche aus

1. Öffnen Sie `constants.py` in einem Editor Ihrer Wahl, und je nach Auswahl fügen Sie das LLM hinzu, das Sie verwenden möchten. Standardmäßig wird das folgende Modell verwendet:

   ```shell
   MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
   MODEL_BASENAME = "llama-2-

7b-chat.Q4_K_M.gguf"
   ```

3. Öffnen Sie ein Terminal und aktivieren Sie Ihre Python-Umgebung, die die Abhängigkeiten aus requirements.txt installiert hat.

4. Navigieren Sie zum Verzeichnis `/LOCALGPT`.

5. Führen Sie den folgenden Befehl aus: `python run_localGPT_API.py`. Die API sollte gestartet werden.

6. Warten Sie, bis alles geladen ist. Sie sollten etwas wie `INFO:werkzeug:Press CTRL+C to quit.` sehen.

7. Öffnen Sie ein zweites Terminal und aktivieren Sie dieselbe Python-Umgebung.

8. Navigieren Sie zum Verzeichnis `/LOCALGPT/localGPTUI`.

9. Führen Sie den Befehl `python localGPTUI.py` aus.

10. Öffnen Sie einen Webbrowser und gehen Sie zur Adresse `http://localhost:5111/`.


# Wie wählt man verschiedene LLM-Modelle aus?

Um die Modelle zu ändern, müssen sowohl `MODEL_ID` als auch `MODEL_BASENAME` festgelegt werden.

1. Öffnen Sie `constants.py` in einem Editor Ihrer Wahl.
2. Ändern Sie die `MODEL_ID` und `MODEL_BASENAME`. Wenn Sie ein quantisiertes Modell (`GGML`, `GPTQ`, `GGUF`) verwenden, müssen Sie `MODEL_BASENAME` angeben. Für unquantisierte Modelle setzen Sie `MODEL_BASENAME` auf `NONE`
5. Es gibt eine Reihe von Beispielmodellen von HuggingFace, die bereits getestet wurden, um mit dem original trainierten Modell verwendet zu werden (enden mit HF oder haben eine .bin in ihren "Files and versions"), und quantisierte Modelle (enden mit GPTQ oder haben eine .no-act-order oder .safetensors in ihren "Files and versions").
6. Für Modelle, die mit HF enden oder eine .bin in ihren "Files and versions" auf ihrer HuggingFace-Seite haben.

   - Stellen Sie sicher, dass Sie eine `MODEL_ID` ausgewählt haben. Zum Beispiel -> `MODEL_ID = "TheBloke/guanaco-7B-HF"`
   - Gehen Sie zum [HuggingFace Repo](https://huggingface.co/TheBloke/guanaco-7B-HF)

7. Für Modelle, die GPTQ in ihrem Namen enthalten und/oder eine .no-act-order oder .safetensors-Erweiterung in ihren "Files and versions" auf ihrer HuggingFace-Seite haben.

   - Stellen Sie sicher, dass Sie eine `MODEL_ID` ausgewählt haben. Zum Beispiel -> model_id = `"TheBloke/wizardLM-7B-GPTQ"`
   - Gehen Sie zum entsprechenden [HuggingFace Repo](https://huggingface.co/TheBloke/wizardLM-7B-GPTQ) und wählen Sie "Files and versions".
   - Wählen Sie einen der Modellnamen aus und setzen Sie ihn als `MODEL_BASENAME`. Zum Beispiel -> `MODEL_BASENAME = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"`

8. Befolgen Sie die gleichen Schritte für `GGUF`- und `GGML`-Modelle.

# GPU- und VRAM-Anforderungen

Nachfolgend finden Sie die VRAM-Anforderung für verschiedene Modelle, abhängig von ihrer Größe (Milliarden von Parametern). Die Schätzungen in der Tabelle enthalten nicht den VRAM, der von den Einbettungsmodellen verwendet wird - diese verwenden zusätzlich 2 GB-7 GB VRAM, abhängig vom Modell.

| Modellgröße (B) | float32   | float16   | GPTQ 8bit      | GPTQ 4bit          |
| ------- | --------- | --------- | -------------- | ------------------ |
| 7B      | 28 GB     | 14 GB     | 7 GB - 9 GB    | 3,5 GB - 5 GB      |
| 13B     | 52 GB     | 26 GB     | 13 GB - 15 GB  | 6,5 GB - 8 GB      |
| 32B     | 130 GB    | 65 GB     | 32,5 GB - 35 GB| 16,25 GB - 19 GB   |
| 65B     | 260,8 GB  | 130,4 GB  | 65,2 GB - 67 GB| 32,6 GB - 35 GB    |


# Systemanforderungen

## Python-Version

Um diese Software verwenden zu können, muss Python 3.10 oder höher installiert sein. Frühere Versionen von Python werden nicht kompiliert.

## C++-Compiler

Wenn Sie beim Erstellen eines Rades während des `pip install`-Vorgangs einen Fehler erhalten, müssen Sie möglicherweise einen C++-Compiler auf Ihrem Computer installieren.

### Für Windows 10/11

Um einen C++-Compiler unter Windows 10/11 zu installieren, befolgen Sie diese Schritte:

1. Installieren Sie Visual Studio 2022.
2. Stellen Sie sicher,

 dass während der Installation die Workload "Desktopentwicklung mit C++" ausgewählt ist.
3. Aktivieren Sie in der Workload-Einstellung "Desktopentwicklung mit C++" das Kontrollkästchen "C++-CMake-Tools für Windows" unter "Einzelne Komponenten".
4. Starten Sie Ihren Computer neu, nachdem Sie die Installation abgeschlossen haben.

### Für macOS

macOS verfügt standardmäßig über den C++-Compiler Clang. Sie sollten keine zusätzlichen Schritte ausführen müssen.

### Für Linux

Unter Linux können Sie den C++-Compiler GCC installieren, indem Sie den folgenden Befehl in Ihrem Terminal ausführen:

```bash
sudo apt-get update
sudo apt-get install build-essential
```

## Numpy-Abhängigkeit

Stellen Sie sicher, dass Sie das Numpy-Paket installiert haben. Wenn Sie Numpy nicht haben, können Sie es mit dem folgenden Befehl installieren:

```shell
pip install numpy
```

# Beitrag

LocalGPT ist ein Open-Source-Projekt und wir begrüßen Beiträge von der Community! Fühlen Sie sich frei, [ein Issue zu öffnen](https://github.com/PromtEngineer/localGPT/issues), einen Vorschlag zu machen oder einen Pull-Request zu senden.

# Lizenz

LocalGPT ist unter der [MIT-Lizenz](https://github.com/PromtEngineer/localGPT/blob/main/LICENSE) lizenziert.
```

Ich hoffe, diese Anleitung ist hilfreich für Sie! Wenn Sie weitere Fragen haben oder Unterstützung benötigen, lassen Sie es mich wissen!
