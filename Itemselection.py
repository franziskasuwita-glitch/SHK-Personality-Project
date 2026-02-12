## Test Item selection
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns # Added for heatmaps

def load_custom_items():
    """
    Initialisiert die Item-Gruppen und zugehörige Labels für die Analyse.
    Jede Unterliste repräsentiert eine semantische Gruppe von Items.
    """
    # Beispielhafte Definition von Item-Clustern
    custom_item_groups = [
        ["Ich treffe häufig spontane Entscheidungen ohne lange darüber nachzudenken.", "Oft lasse ich mich von der Situation leiten und handle impulsiv.", "Ich neige dazu, im Moment zu handeln, ohne alles gründlich abzuwägen."],
        ["Ich handle häufig impulsiv und überlege selten, welche Folgen mein Verhalten haben könnte.",  "Oft sage und mache ich Dinge, ohne mir Gedanken über die möglichen Konsequenzen zu machen.", "Ich neige dazu, spontan zu handeln und die Auswirkungen meiner Entscheidungen nicht zu berücksichtigen."],
        ["Wenn ein Vorhaben sich als zu schwierig erweist, neige ich dazu etwas Neues anzufangen.", "Wenn ein Projekt zu herausfordernd wird, tendiere ich dazu, etwas anderes zu beginnen.", "Wenn ich merke, dass ein Ziel zu schwer zu erreichen ist, starte ich oft einen neuen Ansatz.", "Wenn ein Vorhaben zu kompliziert erscheint, beginne ich häufig mit einer neuen Idee."]
        ["Alle meine Entchen schwimmen auf dem See"] #Kontrollitems
    ]
    custom_group_labels = ["Originalitem_1", "Originalitem_2", "Test_Cluster"]

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

