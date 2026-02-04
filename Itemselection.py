import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_custom_items():
    """
    Initialisiert die Item-Gruppen und zugehörige Labels für die Analyse.
    Jede Unterliste repräsentiert eine semantische Gruppe von Items.
    """
    # Beispielhafte Definition von Item-Clustern
    custom_item_groups = [
        ["Beispiel Item A1", "Beispiel Item A2", "Beispiel Item A3"], 
        ["Beispiel Item B1", "Beispiel Item B2", "Beispiel Item B3"], 
        ["Validierungs-Item", "Kontroll-Item"]
    ]
    custom_group_labels = ["Konstrukt_A", "Konstrukt_B", "Test_Cluster"]

    return custom_item_groups, custom_group_labels

def extract_embeddings_for_groups(item_groups, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    """
    Berechnet L2-normalisierte SBERT-Embeddings für die übergebenen Item-Gruppen.
    Verwendet standardmäßig das multilingual-MiniLM Modell für sprachübergreifende Konsistenz.
    """
    model = SentenceTransformer(model_name)
    grouped_embeddings = []
    for group in item_groups:
        # Erzeugung der Embeddings inklusive Normalisierung für Cosinus-Ähnlichkeit
        embeddings = model.encode(group, convert_to_numpy=True, normalize_embeddings=True)
        grouped_embeddings.append(embeddings)
    return grouped_embeddings, model

def compute_mean_embeddings(grouped_embeddings):
    """
    Berechnet den aggregierten Zentroid-Vektor (Mean Embedding) für jede Gruppe.
    Inklusive Re-Normalisierung zur Sicherstellung der Vektorlänge 1.
    """
    mean_embeddings = []
    for embeddings in grouped_embeddings:
        m = np.mean(embeddings, axis=0)
        m = m / (np.linalg.norm(m) + 1e-12) 
        mean_embeddings.append(m)
    return np.array(mean_embeddings)

def fuse_bilingual_center_vector(center_text_en, center_text_de, model):
    """
    Erstellt einen hybriden Referenzvektor durch Mittelung der Embeddings 
    einer englischen und deutschen Konstruktdefinition.
    """
    C = model.encode([center_text_en, center_text_de], convert_to_numpy=True, normalize_embeddings=True)
    fused = C.mean(axis=0)
    fused = fused / (np.linalg.norm(fused) + 1e-12)
    return fused

def run_analysis_with_custom_items(output_dir='analysis_output', pdf_file='similarity_report.pdf'):
    """
    Hauptroutine zur Durchführung der semantischen Ähnlichkeitsanalyse.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Datenakquise
    item_groups, group_labels = load_custom_items()

    # 2. Modellgestützte Feature-Extraktion
    grouped_embeddings, model = extract_embeddings_for_groups(item_groups)
    mean_embeddings = compute_mean_embeddings(grouped_embeddings)

    # 3. Definition des theoretischen Referenzrahmens (Center Texts)
    # Diese Texte definieren den semantischen Ankerpunkt der Analyse.
    ref_text_en = "Definition of the theoretical construct in English..."
    ref_text_de = "Theoretische Definition des Konstrukts auf Deutsch..."

    # 4. Berechnung der Konstrukt-Validität (Similarity to Center)
    center_text_sim_df = compute_and_plot_center_text_similarity(
        mean_embeddings, group_labels, ref_text_en, ref_text_de, model
    )

    # 5. Dokumentation der Ergebnisse (Speicherung der CSV-Daten)
    print("Speichere numerische Ergebnisse...")
    
    # Speichern der Center-Text Similarity
    center_text_sim_csv = os.path.join(output_dir, 'center_text_similarity.csv')
    center_text_sim_df.to_csv(center_text_sim_csv)

    # Speichern der Kohäsions-Daten
    cohesion_csv = os.path.join(output_dir, 'within_group_cohesion.csv')
    cohesion_df.to_csv(cohesion_csv, index=False)

    # 6. Grafische Aufbereitung (PDF-Export)
    print(f"Erstelle grafischen Report in {output_dir}/{pdf_file}...")
    
    with PdfPages(os.path.join(output_dir, pdf_file)) as pdf:
        # (a) Darstellung der Gruppen-Ähnlichkeitsmatrix
        plot_matrix(cos_df, 'Semantic Similarity Matrix (Between-Groups)', pdf, cmap='viridis', vmin=0, vmax=1)

        # (b) Darstellung der Center-Text Validität
        plot_matrix(center_text_sim_df, 'Construct Validity: Item Groups to Fused Center Text', pdf, cmap='viridis', vmin=0, vmax=1)

        # (c) Visualisierung der Within-Group Kohäsion
        coh_series = cohesion_df.set_index("Group")["WithinGroupCohesion"]
        plot_vector_as_heatmap(coh_series, 'Internal Group Cohesion (Mean Pairwise Cosine)', pdf)

    # 7. Konsolen-Log für die schnelle Überprüfung
    analyze_similarity_matrix(cos_df, threshold=0.5)
    analyze_center_text_similarity(center_text_sim_df, threshold=0.5)

    print(f"\nAnalyse erfolgreich abgeschlossen. Alle Dateien wurden im Ordner '{output_dir}' hinterlegt.")
