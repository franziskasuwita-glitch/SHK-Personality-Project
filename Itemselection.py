
#!/usr/bin/env python
# -*- coding: utf-8 -*-
## TEST Item Selection

# Version ohne Clustering & Faktorenanalyse
# Mit vollständiger L2-Normierung + Within-Group- & Between-Group-Kohäsion
# als HEATMAPS, plus Center-Text-Similarity-Heatmap und Gruppenpaar-Heatmap

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


def load_items(filepath):
    """
    Lädt die Item-Gruppen aus der angegebenen Datei.
    Jede Gruppe enthält verschiedene Versionen eines Items.
    Enthält die bereitgestellten Item-Gruppen und ein Check-Item.
    Falls die Datei nicht gefunden wird, werden Standard-Beispielitems verwendet.
    """
    item_groups = []
    current_group = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("--- Item") or line.startswith("--- Check Item"):
                    if current_group:
                        item_groups.append(current_group)
                    current_group = []
                elif line and not line.startswith(("1)", "2)", "3)", "4)")):
                     current_group.append(line.strip('"'))
                elif line and line.startswith(("1)", "2)", "3)", "4)")):
                     current_group.append(line.split(")", 1)[1].strip().strip('"'))

            if current_group:
                item_groups.append(current_group)

        # Labels for items loaded from file
        if item_groups:
            group_labels = [f"Item {i+1}" for i in range(len(item_groups) - 1)] + ["Check Item"]
        else:
            print(f"Warning: The file '{filepath}' was found but contains no items. Using fallback items.")
            raise FileNotFoundError # Trigger fallback to example items

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found or is empty. Using fallback example items.")
        item_groups = [
            ["Ich treffe häufig spontane Entscheidungen ohne lange darüber nachzudenken.", "Oft lasse ich mich von der Situation leiten und handle impulsiv.", "Ich neige dazu, im Moment zu handeln, ohne alles gründlich abzuwägen."],
            ["Ich handle häufig impulsiv und überlege selten, welche Folgen mein Verhalten haben könnte.", "Oft sage und mache ich Dinge, ohne mir Gedanken über die möglichen Konsequenzen zu machen.", "Ich neige dazu, spontan zu handeln und die Auswirkungen meiner Entscheidungen nicht zu berücksichtigen."],
            ["Wenn ein Vorhaben sich als zu schwierig erweist, neige ich dazu etwas Neues anzufangen.", "Wenn ein Projekt zu herausfordernd wird, tendiere ich dazu, etwas anderes zu beginnen.", "Wenn ich merke, dass ein Ziel zu schwer zu erreichen ist, starte ich oft einen neuen Ansatz.", "Wenn ein Vorhaben zu kompliziert erscheint, beginne ich häufig mit einer neuen Idee."],
            ["Alle meine Entchen schwimmen auf dem See"] # Kontrollitems
        ]
        group_labels = [f"Item {i+1}" for i in range(len(item_groups) - 1)] + ["Check Item"]

    return item_groups, group_labels


def extract_embeddings_for_groups(item_groups, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    """
    Berechnet SBERT-Embeddings für jeden Text in jeder Gruppe und gibt gruppierte Embeddings zurück.
    Führt eine L2-Normierung der Einzel-Embeddings durch (normalize_embeddings=True).
    """
    model = SentenceTransformer(model_name)
    # Optional stärkeres Modell:
    # model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    grouped_embeddings = []
    for group in item_groups:
        # L2-normiert
        embeddings = model.encode(group, convert_to_numpy=True, normalize_embeddings=True)
        grouped_embeddings.append(embeddings)
    return grouped_embeddings, model


def compute_mean_embeddings(grouped_embeddings):
    """
    Berechnet das mittlere Embedding für jede Item-Gruppe.
    WICHTIG: Nach der Mittelwertbildung wird erneut L2-normiert.
    """
    mean_embeddings = []
    for embeddings in grouped_embeddings:
        m = np.mean(embeddings, axis=0)
        # Erneute L2-Normierung
        m = m / (np.linalg.norm(m) + 1e-12)
        mean_embeddings.append(m)
    return np.array(mean_embeddings)


def save_embeddings(embeddings, filepath):
    """
    Speichert die Embeddings in einer .npy-Datei.
    """
    np.save(filepath, embeddings)
    print(f"Saved embeddings: {filepath}")


def compute_cosine_matrix(embeddings, labels, save_csv=None):
    """
    Berechnet die Cosine-Ähnlichkeitsmatrix und speichert sie optional als CSV-Datei.
    Erwartet L2-normierte Vektoren (dann ist Cosine = Dotprodukt).
    """
    matrix = cosine_similarity(embeddings)
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    if save_csv:
        df.to_csv(save_csv)
        print(f"Saved cosine matrix to {save_csv}")
    return df


def plot_matrix(df, title, pdf, cmap=None, vmin=None, vmax=None):
    """
    Plottet eine Matrix als Heatmap und speichert sie in einem PDF-Dokument.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(df.values, interpolation='nearest', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.xticks(range(len(df.columns)), df.columns, rotation=90)
    plt.yticks(range(len(df.index)), df.index)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


def fuse_bilingual_center_vector(center_texts_en, center_texts_de, model):
    """
    Bildet Fusionsvektoren aus englischen und deutschen Center-Texten.
    1. Beide Texte einzeln einbetten (bereits L2-normalisiert).
    2. Den Mittelwert der beiden Embeddings bilden.
    3. Das Ergebnis erneut L2-normalisieren.
    Erwartet Listen von Texten (center_texts_en[i] wird mit center_texts_de[i] fusioniert).
    """
    fused_vectors = []
    for text_en, text_de in zip(center_texts_en, center_texts_de):
        # (2, d) – jede Zeile ist bereits L2-normalisiert (normalize_embeddings=True)
        C = model.encode([text_en, text_de], convert_to_numpy=True, normalize_embeddings=True)
        fused = C.mean(axis=0)
        fused = fused / (np.linalg.norm(fused) + 1e-12)
        fused_vectors.append(fused)
    # Gibt ein Array von Fusionsvektoren zurück, Form: (Anzahl_Vektoren, Dimensionalität)
    return np.array(fused_vectors)


def compute_and_plot_center_text_similarity(mean_item_embeddings, group_labels, fused_center_vectors, center_vector_labels, pdf=None):
    """
    Berechnet und plottet die absolute Cosine-Ähnlichkeit zwischen den mittleren Item-Embeddings
    und den fusionierten bilingualen Center-Text-Vektoren.
    """
    # fused_center_vectors shape: (n_vectors, d)
    # mean_item_embeddings shape: (n_items, d)

    # Berechnet die Cosine-Ähnlichkeitsmatrix
    similarity_matrix = cosine_similarity(mean_item_embeddings, fused_center_vectors)  # (n_items, n_vectors)
    # Verwendet den Absolutwert der Ähnlichkeit
    absolute_similarity_matrix = np.abs(similarity_matrix)

    # Erstellt einen DataFrame mit den Ähnlichkeitswerten
    df = pd.DataFrame(absolute_similarity_matrix, index=group_labels, columns=center_vector_labels)

    # Plottet die Matrix als Heatmap im PDF, falls ein PDF-Objekt übergeben wurde
    if pdf is not None:
        plot_matrix(df, 'Absolute Cosine Similarity: Item Groups → Fused Center Texts (EN+DE)', pdf, cmap='viridis', vmin=0, vmax=1)

    # Gibt den DataFrame der Ähnlichkeitswerte zurück
    return df


# ----------------- Within-Group-Kohäsion -----------------
def compute_within_group_cohesion(grouped_embeddings, group_labels, exclude_check_item=True):
    """
    Berechnet die mittlere paarweise Cosine-Ähnlichkeit innerhalb jeder Gruppe (Within-Group-Kohäsion).
    Schließt optional das Check-Item aus.
    """
    rows = []
    for E, lbl in zip(grouped_embeddings, group_labels):
        if exclude_check_item and lbl == "Check Item":
            continue
        # Berechnet paarweise Ähnlichkeit innerhalb der Gruppe
        S = cosine_similarity(E)  # (Anzahl_Varianten, Anzahl_Varianten)
        # Wählt die oberen Dreieckselemente aus (um Duplikate und Diagonale zu vermeiden)
        iu = np.triu_indices_from(S, 1)
        # Berechnet den Mittelwert der Ähnlichkeiten; falls nur ein Element, Kohäsion = 1.0
        coh = float(S[iu].mean()) if len(iu[0]) else 1.0
        rows.append((lbl, coh))
    # Erstellt einen DataFrame und sortiert nach Kohäsion
    df = pd.DataFrame(rows, columns=["Group", "WithinGroupCohesion"]).sort_values("WithinGroupCohesion")
    return df


def plot_vector_as_heatmap(series_or_df, title, pdf):
    """
    Plottet einen 1D-Vektor (Series oder 1-Spalten-DataFrame) als Heatmap (n x 1).
    """
    if isinstance(series_or_df, pd.Series):
        data = series_or_df.values.reshape(-1, 1)
        index = series_or_df.index.tolist()
        columns = [series_or_df.name or "Value"]
    else:
        data = series_or_df.values
        index = series_or_df.index.tolist()
        columns = series_or_df.columns.tolist()

    df = pd.DataFrame(data, index=index, columns=columns)
    plot_matrix(df, title, pdf)


# ----------------- Between-Group-Kohäsion -----------------
def compute_between_group_cohesion(mean_embeddings_filtered, labels_filtered):
    """
    Berechnet für jede Gruppe die mittlere Cosine-Ähnlichkeit zu allen anderen Gruppen (ohne die Ähnlichkeit zu sich selbst).
    Erwartet Embeddings und Labels, bei denen das Check-Item bereits ausgeschlossen wurde.
    """
    # Berechnet die paarweise Ähnlichkeit zwischen den mittleren Gruppen-Embeddings
    S = cosine_similarity(mean_embeddings_filtered)  # (Anzahl_Gruppen, Anzahl_Gruppen)
    # Setzt die Diagonale auf NaN, um die Ähnlichkeit einer Gruppe zu sich selbst auszuschließen
    np.fill_diagonal(S, np.nan)
    # Berechnet den Mittelwert der Ähnlichkeiten für jede Zeile (jede Gruppe) unter Ignorierung von NaN
    means = np.nanmean(S, axis=1)
    # Erstellt einen DataFrame mit der berechneten Between-Group-Kohäsion
    df = pd.DataFrame({"BetweenGroupCohesion": means}, index=labels_filtered)
    return df


def analyze_similarity_matrix(cos_df, item_threshold=0.9, center_threshold=0.5):
    """
    Analysiert die Cosine-Ähnlichkeitsmatrix (ohne Check-Item) und listet Paare mit Ähnlichkeit >= item_threshold auf.
    Analysiert auch die Ähnlichkeit zu den fusionierten Center-Texten und listet Items mit Ähnlichkeit >= center_threshold auf.
    """
    print("\n--- Cosine Similarity Analysis ---")

    # Analyse der paarweisen Gruppenähnlichkeit (ohne Check-Item)
    high_similarity_pairs = []

    # Erstellt eine gefilterte Matrix ohne das Check-Item, falls vorhanden
    if 'Check Item' in cos_df.index:
        cos_df_filtered = cos_df.drop('Check Item', axis=0).drop('Check Item', axis=1)
    else:
        cos_df_filtered = cos_df

    # Durchläuft die obere Dreiecksmatrix, um paarweise Ähnlichkeiten zu prüfen
    for i in range(len(cos_df_filtered.index)):
        for j in range(i + 1, len(cos_df_filtered.columns)):
            item1 = cos_df_filtered.index[i]
            item2 = cos_df_filtered.columns[j]
            similarity = cos_df_filtered.iloc[i, j]
            # Fügt Paare hinzu, deren Ähnlichkeit den Schwellenwert erreicht oder überschreitet
            if similarity >= item_threshold:
                high_similarity_pairs.append((item1, item2, similarity))

    # Gibt die Paare mit hoher Ähnlichkeit aus
    print(f"\nItem-Paare mit Cosine-Ähnlichkeit bei oder über {item_threshold} (ohne Check-Item):")
    if high_similarity_pairs:
        for item1, item2, similarity in high_similarity_pairs:
            print(f"- {item1} und {item2}: {similarity:.2f}")
    else:
        print("Keine Paare bei oder über dem Schwellenwert gefunden.")

    # Analyse der Ähnlichkeit zu den fusionierten Center-Texten (inkl. Check-Item)
    # Überprüft, ob der DataFrame der Center-Text-Ähnlichkeit global verfügbar ist
    if 'center_text_sim_df' in globals():
        print("\n--- Ähnlichkeit zu fusionierten Center-Texten Analyse ---")
        # Durchläuft jede Spalte (jeden fusionierten Center-Text) im DataFrame der Center-Text-Ähnlichkeit
        for col in center_text_sim_df.columns:
            # Filtert Items heraus, deren Ähnlichkeit zum aktuellen Center-Text den Schwellenwert erreicht oder überschreitet
            high_sim_items = center_text_sim_df[center_text_sim_df[col] >= center_threshold]
            # Gibt die Item-Gruppen mit hoher Ähnlichkeit zum aktuellen Center-Text aus
            print(f"\nItem-Gruppen mit absoluter Cosine-Ähnlichkeit zu '{col}' bei oder über {center_threshold}:")
            if not high_sim_items.empty:
                for index, row in high_sim_items.iterrows():
                    print(f"- {index}: {row[col]:.2f}")
            else:
                print("Keine Item-Gruppen bei oder über dem Schwellenwert gefunden.")

    else:
        print("\nCenter-Text-Ähnlichkeitsdaten sind nicht verfügbar für die Analyse.")


def main(filepath, output_dir='output', pdf_file='results.pdf'):
    # Erstellt das Ausgabeverzeichnis, falls es nicht existiert
    os.makedirs(output_dir, exist_ok=True)

    # 1. Item-Gruppen laden
    item_groups, group_labels = load_items(filepath)

    # Prüfen, ob Item-Gruppen geladen wurden
    if item_groups is None:
        return

    # 2. Embeddings extrahieren (L2-normiert) + Modell zurückgeben
    grouped_embeddings, model = extract_embeddings_for_groups(item_groups)

    # 3. Mittlere Embeddings für jede Gruppe berechnen (erneut L2-normieren)
    mean_embeddings = compute_mean_embeddings(grouped_embeddings)
    mean_emb_file = os.path.join(output_dir, 'mean_embeddings.npy')
    save_embeddings(mean_embeddings, mean_emb_file)

    # 3b. Within-Group-Kohäsion berechnen
    cohesion_df = compute_within_group_cohesion(grouped_embeddings, group_labels, exclude_check_item=True)
    cohesion_csv = os.path.join(output_dir, 'within_group_cohesion.csv')
    cohesion_df.to_csv(cohesion_csv, index=False)

    # 4. Cosine-Matrix (Between auf Paar-Ebene) ohne Check-Item berechnen
    filtered_group_labels = [label for label in group_labels if label != 'Check Item']
    if 'Check Item' in group_labels:
        check_item_index = group_labels.index('Check Item')
        filtered_mean_embeddings = np.delete(mean_embeddings, check_item_index, axis=0)
    else:
        filtered_mean_embeddings = mean_embeddings
        print("Kein 'Check Item' in den Gruppen-Labels gefunden.")

    cos_csv = os.path.join(output_dir, 'cosine_similarity_groups_no_check.csv')
    cos_df = compute_cosine_matrix(filtered_mean_embeddings, filtered_group_labels, save_csv=cos_csv)

    # 4b. Between-Group-Kohäsion (Mittelwert zu allen anderen) berechnen
    between_coh_df = compute_between_group_cohesion(filtered_mean_embeddings, filtered_group_labels)
    between_coh_csv = os.path.join(output_dir, 'between_group_cohesion.csv')
    between_coh_df.to_csv(between_coh_csv)

    # 5. Center-Text Ähnlichkeit (inkl. Check-Item) via EN+DE-Fusionsvektor berechnen
    center_text_en1 =  (
        "Behavior characterized by little or no forethought, relfection, or consideration of the consequences of an action,"
        "particularly one that involves taking risks."
    )
    center_text_en2 = (
        "Inability to complete projects and to work under conditions, that require resistance"
        "to obstacles and to distracting stimuli."
    )

    # Deutsche Übersetzung (Platzhalter – bitte anpassen):
    center_text_de1 = (
        "Verhalten, charakterisiert durch wenig oder keine vorausschauende Planung,"
        "Reflexion oder Berücksichtigung der Folgen einer Handlung, insbesondere einer,"
        "die das Eingehen von Risiken beinhaltet."
    )
    center_text_de2 = (
        "Unfähigkeit, Projekte abzuschließen und unter Bedingungen zu arbeiten, die Widerstandsfähigkeit"
        "gegenüber Hindernissen und ablenkenden Reizen erfordern."
    )

    center_texts_en = [center_text_en1, center_text_en2]
    center_texts_de = [center_text_de1, center_text_de2]
    center_vector_labels = ['lack of premeditation', 'lack of perseverance']

    fused_center_vectors = fuse_bilingual_center_vector(center_texts_en, center_texts_de, model)

    global center_text_sim_df
    center_text_sim_df = compute_and_plot_center_text_similarity(
        mean_embeddings, group_labels, fused_center_vectors, center_vector_labels
    )
    center_text_sim_csv = os.path.join(output_dir, 'center_text_similarity.csv')
    center_text_sim_df.to_csv(center_text_sim_csv)

    # 6. PDF erstellen: Heatmaps – Paarweise, Center (Fusionsvektor), Within, Between
    with PdfPages(os.path.join(output_dir, pdf_file)) as pdf:
        # (a) Paarweise Gruppenähnlichkeiten (Between auf Matrix-Ebene)
        plot_matrix(cos_df, 'Cosine Similarity Matrix: Between-Group (pairwise, excluding Check Item)', pdf, cmap='viridis', vmin=0, vmax=1)

        # (b) Center-Text Ähnlichkeit (n x 2) – Fusionsvektoren
        plot_matrix(center_text_sim_df, 'Absolute Cosine Similarity: Item Groups → Fused Center Texts (EN+DE)', pdf, cmap='viridis', vmin=0, vmax=1)

        # (c) Within-Group Kohäsion (n x 1 Heatmap)
        coh_series = cohesion_df.set_index("Group")["WithinGroupCohesion"]
        plot_vector_as_heatmap(coh_series, 'Within-Group Cohesion (mean pairwise cosine)', pdf)

        # (d) Between-Group Kohäsion (Mittel zu allen anderen; n x 1 Heatmap)
        plot_vector_as_heatmap(between_coh_df["BetweenGroupCohesion"], 'Between-Group Cohesion (mean cosine to others)', pdf)

        # Neue Seite für Items mit hoher Ähnlichkeit zu Center-Texten in der PDF
        item_analysis_threshold = 0.9 # Threshold for item-item similarity analysis output
        center_analysis_threshold = 0.5 # Threshold for center-item similarity analysis output


        if 'center_text_sim_df' in globals():
            pdf.attach_note("Items mit absoluter Cosine-Ähnlichkeit >= {} zu fusionierten Center-Texten".format(center_analysis_threshold))
            for col in center_text_sim_df.columns:
                high_sim_items = center_text_sim_df[center_text_sim_df[col] >= center_analysis_threshold]
                if not high_sim_items.empty:
                    plt.figure(figsize=(8, high_sim_items.shape[0] * 0.5))
                    plt.text(0.05, 0.95, f"Item-Gruppen mit absoluter Cosine-Ähnlichkeit zu '{col}' bei oder über {center_analysis_threshold}:",
                             fontsize=12, ha='left', va='top', wrap=True)
                    text_content = ""
                    for index, row in high_sim_items.iterrows():
                         text_content += f"- {index}: {row[col]:.2f}\n"
                    plt.text(0.05, 0.9, text_content, fontsize=10, ha='left', va='top', wrap=True)
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()
                else:
                    plt.figure(figsize=(8, 2))
                    plt.text(0.05, 0.5, f"Keine Item-Gruppen mit absoluter Cosine-Ähnlichkeit zu '{col}' bei oder über {center_analysis_threshold} gefunden.",
                             fontsize=12, ha='left', va='center', wrap=True)
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()


    # 7. Konsolen-Analyse (optional)
    analyze_similarity_matrix(cos_df, item_threshold=item_analysis_threshold, center_threshold=center_analysis_threshold)

    print(f"\nAlle Ergebnisse wurden in {output_dir}/ gespeichert.")


if __name__ == '__main__':
    # Dateipfad hier definieren oder als Argument an main übergeben
    filepath = '/content/augmenteditems.txt'
    main(filepath, output_dir='output')
