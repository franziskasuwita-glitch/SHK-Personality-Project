# SHK-Personality-Project
Project to assess Personality in a naturalistic paradigm 
Dieses Projekt dokumentiert den vollständigen Prozess von der Vorbereitung psychometrischer Items über die semantische Validierung bis hin zur KI-basierten Generierung klinisch-relevanter Alltagsszenen.
Projektübersicht
Die Pipeline ist in vier aufeinanderfolgende Phasen unterteilt:

    Item-Augmentierung: Standardisierung der Textlänge zur Optimierung von SBERT-Embeddings.

    Item-Ähnlichkeitsanalyse: Validierung der Konstruktkohäsion der Original-Items.

    Szenengenerierung: KI-gestützte Erstellung von narrativen Stimuli mit integrierter Validierungsschleife.

    Szenen-Validierung: Statistische Überprüfung der finalen Stimuli auf konvergente und diskriminante Validität.

Installation & Setup

git clone https://github.com/franziskasuwita-glitch/SHK-Personality-Project.git
cd SHK-Personality-Project

pip install -r requirements.txt

Hinweis: Für das Skript zur Item-Augmentation und zur Szenengenerierung sind API-Keys erforderlich.

1. Item-Augmentierung (item_augmentation.py)
Dieses Skript nutzt GPT-4o-mini, um kurze psychometrische Items semantisch konsistent auf eine Ziel-Wortanzahl zu erweitern. Dies minimiert Längeneffekte bei der anschließenden Vektorisierung.
    Input: Original-Items (direkt im Skript editierbar).
    Output: Konsolidierte Textblöcke für stabilere Embeddings.

2. Item-Ähnlichkeitsanalyse (item_similarity_analysis.py)

Berechnung der internen Konsistenz der Item-Gruppen mittels Sentence-BERT (SBERT).

    Modell: paraphrase-multilingual-MiniLM-L12-v2.

    Metrik: Cosinus-Ähnlichkeit.

    Visualisierung: Heatmaps der Within-Group und Between-Group Ähnlichkeiten.

3. Validierte Szenengenerierung (scene_generator.py)

Generiert narrative Alltagsszenen aus der Ich-Perspektive, die subklinische Ausprägungen von Impulsivität darstellen.

    Technik: Ein rekursiver Algorithmus prüft jede generierte Szene gegen einen Schwellenwert (Cosine Similarity > 0.3).

    KI-Modell: Google Gemini 2.5 Pro.

4. Szenen-Validierungsanalyse (scene_validation.py)

Abschließende statistische Evaluation der generierten Szenen.

    Diskriminanzprüfung: Vergleich der Ähnlichkeit von Szenen zu ihren eigenen Facetten-Definitionen vs. Fremd-Definitionen.

    Output: Export von PNG-Grafiken zur Dokumentation der Konstruktvalidität.

Anforderungen (requirements.txt)

Für die Ausführung werden folgende Bibliotheken benötigt:

    sentence-transformers

    openai

    google-generativeai

    pandas, numpy

    matplotlib, seaborn

    scikit-learn

    langdetect
