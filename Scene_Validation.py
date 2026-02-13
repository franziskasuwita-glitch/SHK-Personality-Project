# SZENENANALYSE: SEMANTISCHE VALIDIERUNG VON SZENEN MIT SBERT UND COSINE-ÄHNLICHKEIT
# ============================================================================
# BIBLIOTHEKEN IMPORTIEREN
# ============================================================================
# Import der erforderlichen Bibliotheken für Text-Embedding, numerische Operationen,
# Ähnlichkeitsberechnungen, Datenvisualisierung und statistische Tests.
from sentence_transformers import SentenceTransformer # Für die Erstellung von semantischen Text-Embeddings.
import numpy as np # Für numerische Operationen, insbesondere mit Arrays und Vektoren.
from sklearn.metrics.pairwise import cosine_similarity # Für die Berechnung der Cosine-Ähnlichkeit zwischen Vektoren.
import matplotlib.pyplot as plt # Für die Erstellung von Diagrammen und Visualisierungen.
import seaborn as sns # Eine auf Matplotlib basierende Bibliothek für ansprechende statistische Grafiken.
from scipy import stats # Für statistische Tests, hier speziell der Wilcoxon Signed-Rank Test.

# ============================================================================
# SZENEN UND DEFINITIONEN EINFÜGEN
# ============================================================================

# Diese Sektion enthält die vordefinierten Szenen, die als Beispielmaterial für die Analyse dienen.
# Sie sind in zwei Hauptkategorien unterteilt, die den psychologischen Subfacetten 'Mangelnde Voraussicht'
# (Lack of Premeditation) und 'Mangelndes Durchhaltevermögen' (Lack of Perseverance) entsprechen.
# Jede Szene wurde so formuliert, dass sie die jeweilige Disposition subklinisch darstellt.

# Szenen zur Messung von 'Mangelnde Voraussicht' (Lack of Premeditation):
premeditation_182 = "Statt die Anleitung zu lesen, lege ich sofort mit dem neuen Schrank los und schraube die Teile nach Gefühl zusammen. Erst als fast alles steht, merke ich, dass ich die Türen falsch herum montiert habe. Jetzt muss ich die meiste Arbeit noch einmal von vorn machen und bin genervt."
premeditation_7 = "Ein kurzer Adrenalinkick schießt durch mich, als ich den steilen Abhang auf dem Mountainbike-Trail sehe. Ohne die Landung richtig anzusehen, lasse ich die Bremsen los und stürze mich hinunter. Der Aufprall ist härter als gedacht, und nun hängt mein Schaltwerk verbogen herunter und die Kette schleift nutzlos am Boden."
premeditation_12 = "Meine Kollegin erzählt mir voller Stolz von ihrer neuen Projektidee und ohne nachzudenken, platze ich damit heraus, dass ihr Ansatz aus meiner Sicht niemals funktionieren wird. Jetzt sitzt sie mir mit versteinertem Gesicht gegenüber und ich sehe, wie die ganze Begeisterung aus ihren Augen weicht. Ich wünschte, ich hätte einfach erst einmal nachgedacht, bevor ich den Mund aufgemacht habe."
premeditation_198 = "Als mein Chef mich heute Morgen kritisierte, bin ich einfach aufgestanden und habe gekündigt. Jetzt sitze ich zu Hause auf dem Sofa und starre auf die unbezahlten Rechnungen auf dem Tisch. Der kurze Moment der Genugtuung ist verflogen und weicht einer kalten Panik bei dem Gedanken an die Miete nächsten Monat."
premeditation_36 = "Im Streit mit einem Freund tippe ich wütend eine lange Nachricht und drücke auf Senden, ohne sie noch einmal zu lesen. Ein paar Sekunden später, als der erste Zorn verflogen ist, lese ich meine eigenen Worte noch einmal durch. Mir wird klar, wie unfair und verletzend meine Sätze sind, und ich weiß genau, dass ich die Situation damit nur verschlimmert habe."
premeditation_112 = 'Als eine enge Freundin anruft erzählt sie mir von ihrem stressigen Tag und erwähnt, dass sie einen wichtigen Rat von mir bräuchte. Doch während sie spricht, schießt mir ein Gedanke durch den Kopf: Ich könnte spontan heute Abend noch mit meinem alten Schulfreund ausgehen! Aus dem Augenblick heraus, ohne zu planen oder die Folgen zu bedenken, greife ich zum Handy und schreibe ihm: "Heute Abend Bar? Ich zahle die erste Runde!" "Hörst du mir überhaupt zu?", fragt meine Freundin plötzlich verletzt. Erst jetzt wird mir bewusst: Sie wollte mit mir über etwas Ernstes sprechen, und ich habe aus dem Moment heraus gehandelt und sie ignoriert. Ich bin das Risiko eingegangen, eine wichtige Freundschaft zu beschädigen, ohne vorauszudenken, was mein Verhalten bedeutet."Tut mir leid", stammle ich. "Mir kam gerade ein Gedanke und ich hab sofort... ohne nachzudenken..." Sie seufzt tief. "Das ist genau das Problem. Du handelst immer aus dem Augenblick heraus, ohne zu reflektieren, was das für andere bedeutet. Irgendwann wird das Konsequenzen haben." Sie legt auf. Mir wird klar: Ich bin gerade das Risiko eingegangen, eine meiner engsten Freundschaften zu gefährden.'

# Szenen zur Messung von 'Mangelndes Durchhaltevermögen' (Lack of Perseverance):
perseverance_9 = "Voller Begeisterung erstelle ich eine Chat-Gruppe für das große Grillfest, das ich meinen Freunden letztes Wochenende versprochen habe. Doch als die ersten Fragen zur Organisation und zum Einkaufen kommen, merke ich, wie die ganze Planung mich überfordert. Seitdem habe ich die Gruppe nicht mehr geöffnet und reagiere nicht auf die Nachfragen, was nun aus dem Fest wird."
perseverance_78 = "Die Reparatur der Wasserleitung ist komplizierter, als ich dachte, und erfordert genaues Arbeiten und Geduld. Nach einem frustrierenden Versuch, das Leck selbst zu dichten, beschließe ich, das ganze Renovierungsprojekt sei eine dumme Idee. Stattdessen beginne ich online zu recherchieren, wie man am besten eine neue Sprache lernt."
perseverance_175 = "Ich sitze am Schreibtisch und starre auf die halbfertige Präsentation für morgen, als mir plötzlich einfällt, dass ich unbedingt noch die Flugpreise für den Sommerurlaub vergleichen muss. Eine Stunde später finde ich mich auf einer Reise-Website wieder, habe unzählige Hotels angesehen, aber an der Präsentation kein einziges Wort mehr geschrieben. Ein Gefühl der Panik steigt in mir auf, weil die Zeit jetzt richtig knapp wird."
perseverance_176 = "Ich sitze vor meinem Laptop und starre auf den Online-Kurs, für den ich mich letzte Woche voller Elan angemeldet habe. Das neue Modul wirkt komplizierter als das letzte und nach zehn Minuten merke ich, wie meine Gedanken abschweifen. Ich klappe den Rechner zu und beschließe, es morgen noch einmal zu versuchen, obwohl ich weiß, dass er wahrscheinlich bis zur nächsten monatlichen Abbuchung wieder unberührt bleibt."
perseverance_27 = "Die Planung für das Nachbarschaftsfest zieht sich und ich merke, wie meine anfängliche Begeisterung verfliegt. Seit einer Woche ignoriere ich die E-Mails der anderen Organisatoren, weil mir die ständigen Diskussionen zu anstrengend sind. Gerade hat mich meine Nachbarin enttäuscht angesprochen, dass sie sich jetzt allein um alles kümmern muss."
perseverance_26 = 'Ich lasse die angefangene Bewerbung seit Wochen auf meinem Desktop liegen und denke mir, dass ich sie irgendwann fertigmache – es stört mich nicht wirklich, dass sie unvollendet ist. Als die Bewerbungsfrist abläuft, zucke ich nur mit den Schultern, aber meine Freundin ist fassungslos, weil es um meinen Traumjob ging und ich jetzt ein weiteres Jahr warten muss.'

# Definitionen der psychologischen Facetten:
# Diese Definitionen dienen als Referenzpunkte für die semantische Ähnlichkeitsberechnung.
Premeditation_definition = "Verhalten, charakterisiert durch wenig oder keine vorausschauende Planung, Reflexion oder Berücksichtigung der Folgen einer Handlung, insbesondere einer, die das Eingehen von Risiken beinhaltet."
Perseverance_definition = "Unfähigkeit, Projekte abzuschließen und unter Bedingungen zu arbeiten, die Widerstandsfähigkeit gegenüber Hindernissen und ablenkenden Reizen erfordern."

# ============================================================================
# MODELL LADEN
# ============================================================================
print("Lade Sentence Transformer Modell...")
# Hier wird ein vortrainiertes Sentence Transformer Modell geladen.
# 'paraphrase-multilingual-MiniLM-L12-v2' ist ein leistungsstarkes Modell, das für die Erstellung
# von semantischen Embeddings in mehreren Sprachen (inkl. Deutsch) optimiert ist.
# Es wandelt Text in hochdimensionale Vektoren (Embeddings) um, die die semantische Bedeutung
# des Textes erfassen, sodass ähnliche Texte auch ähnliche Vektoren erhalten.
sentence_encoder_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Modell erfolgreich geladen!\n")

# ============================================================================
# EMBEDDINGS ERSTELLEN (Schritt 1: Text zu Vektoren konvertieren)
# ============================================================================
print("Erstelle Embeddings für alle Szenen und Definitionen...")

# Embeddings für die Premeditation Szenen erstellen:
# Jede Szene wird einzeln in einen numerischen Vektor umgewandelt.
# 'encode' normalisiert die Vektoren standardmäßig auf die L2-Norm, was für die
# Cosine-Ähnlichkeitsberechnung vorteilhaft ist, da diese dann dem Skalarprodukt entspricht.
Premeditation_scenes_text = [
    premeditation_182, premeditation_7, premeditation_12,
    premeditation_198, premeditation_36, premeditation_112
]
embeddings_Premeditation_scenes = sentence_encoder_model.encode(Premeditation_scenes_text)

# Embeddings für die Perseverance Szenen erstellen:
Perseverance_scenes_text = [
    perseverance_9, perseverance_78, perseverance_175,
    perseverance_176, perseverance_27, perseverance_26
]
embeddings_Perseverance_scenes = sentence_encoder_model.encode(Perseverance_scenes_text)

# Embeddings für die Definitionen erstellen:
embedding_def_Premeditation = sentence_encoder_model.encode(Premeditation_definition)
embedding_def_Perseverance = sentence_encoder_model.encode(Perseverance_definition)

print("Alle Embeddings erfolgreich erstellt!\n")

# Funktion zur L2-Normierung (Erklärung: Die Cosine-Ähnlichkeit funktioniert am besten mit
# L2-normierten Vektoren. Die 'encode'-Funktion von Sentence-Transformers führt dies
# standardmäßig durch. Diese Hilfsfunktion ist hier nur zur Verdeutlichung oder
# für den Fall, dass man nicht-normierte Vektoren verarbeiten müsste. Für die
# nachfolgenden 'cosine_similarity'-Aufrufe ist sie nicht explizit erforderlich,
# da sklearn dies intern handhabt, wenn Vektoren nicht bereits normiert sind.)
def l2_normalize(vector):
    """
    L2-Normierung eines Vektors (Euklidische Norm).
    Der Vektor wird durch seine Länge geteilt, sodass seine neue Länge 1 beträgt.
    Dies ist wichtig für die Cosine-Ähnlichkeit, da sie dann dem Skalarprodukt entspricht.
    """
    norm = np.linalg.norm(vector)
    if norm == 0: # Vermeidung von Division durch Null bei Nullvektoren
        return vector
    return vector / norm

# ============================================================================
# COSINE-ÄHNLICHKEIT ZWISCHEN SZENEN INNERHALB JEDES SETS (Schritt 2)
# ============================================================================
print("="*70)
print("TEIL 1: ÄHNLICHKEIT ZWISCHEN SZENEN INNERHALB JEDES SETS")
print("="*70)

# Berechnung der paarweisen Cosine-Ähnlichkeit für alle Szenen innerhalb der 'Premeditation'-Gruppe.
# Eine hohe Ähnlichkeit (Werte nahe 1) deutet darauf hin, dass die Szenen semantisch kohärent sind
# und die gleiche Facette konsistent repräsentieren.
# Die Funktion `cosine_similarity` von scikit-learn erwartet eine 2D-Array-Struktur.
similarity_matrix_Premeditation = cosine_similarity(embeddings_Premeditation_scenes)

print("\nCosine Similarity Matrix für Premeditation Szenen:")
# Die Ausgabe wird zur besseren Lesbarkeit auf zwei Dezimalstellen gerundet.
print(similarity_matrix_Premeditation.round(2))

# Berechnung der paarweisen Cosine-Ähnlichkeit für alle Szenen innerhalb der 'Perseverance'-Gruppe.
similarity_matrix_Perseverance = cosine_similarity(embeddings_Perseverance_scenes)

print("\n\nCosine Similarity Matrix für Perseverance Szenen:")
print(similarity_matrix_Perseverance.round(2))

# --- Labels für Heatmaps erstellen ---
# Diese Labels werden für die Achsenbeschriftung der Heatmaps verwendet,
# um die einzelnen Szenen eindeutig zu identifizieren.
Premeditation_scene_labels = [
    'Premeditation_182', 'Premeditation_7', 'Premeditation_12',
    'Premeditation_198', 'Premeditation_36', 'Premeditation_112'
]
Perseverance_scene_labels = [
    'Perseverance_9', 'Perseverance_78', 'Perseverance_175',
    'Perseverance_176', 'Perseverance_27', 'Perseverance_26'
]

# ============================================================================
# VISUALISIERUNG: SZENEN-ÄHNLICHKEIT (HEATMAPS)
# ============================================================================
# Zwei Heatmaps werden erstellt, um die semantische Ähnlichkeit zwischen den Szenen
# innerhalb jeder Facette visuell darzustellen. Dies erleichtert die Erkennung von
# besonders ähnlichen oder unähnlichen Szenenpaaren.
fig, axes = plt.subplots(1, 2, figsize=(16, 7)) # Erstellt eine Figur mit zwei Unterplots nebeneinander.

# Heatmap für Premeditation Szenen:
# `sns.heatmap` visualisiert die Ähnlichkeitsmatrix.
# `annot=True` zeigt die Ähnlichkeitswerte in den Zellen an.
# `fmt='.2f'` formatiert die Werte auf zwei Dezimalstellen.
# `cmap='YlOrRd'` verwendet eine Farbskala von Gelb über Orange zu Rot, wobei dunklere Farben höhere Ähnlichkeit anzeigen.
# `vmin=0, vmax=1` setzt den Bereich der Farbskala von 0 (keine Ähnlichkeit) bis 1 (perfekte Ähnlichkeit).
sns.heatmap(similarity_matrix_Premeditation, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=Premeditation_scene_labels,
            yticklabels=Premeditation_scene_labels,
            vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'Cosine Similarity'})
axes[0].set_title('Ähnlichkeit zwischen Premeditation Szenen', fontsize=14, fontweight='bold')

# Heatmap für Perseverance Szenen (analog zur Premeditation Heatmap):
sns.heatmap(similarity_matrix_Perseverance, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=Perseverance_scene_labels,
            yticklabels=Perseverance_scene_labels,
            vmin=0, vmax=1, ax=axes[1], cbar_kws={'label': 'Cosine Similarity'})
axes[1].set_title('Ähnlichkeit zwischen Perseverance Szenen', fontsize=14, fontweight='bold')

plt.tight_layout() # Passt das Layout an, um Überlappungen zu vermeiden.
plt.savefig('szenen_aehnlichkeit_heatmaps.png', dpi=300, bbox_inches='tight') # Speichert die Visualisierung als PNG-Datei.
plt.show() # Zeigt die Visualisierung an.


# ============================================================================
# COSINE-ÄHNLICHKEIT ZWISCHEN SZENEN UND DEFINITIONEN (Schritt 3)
# ============================================================================
print("\n" + "="*70)
print("TEIL 2: ÄHNLICHKEIT ZWISCHEN SZENEN UND DEFINITIONEN")
print("="*70)

# Jede Premeditation Szene wird mit der Definition von 'Mangelnder Voraussicht' verglichen.
# Eine hohe Ähnlichkeit bestätigt, dass die Szene die intendierte psychologische Facette gut repräsentiert.
sim_Premeditation_scenes_def = cosine_similarity(embeddings_Premeditation_scenes, [embedding_def_Premeditation])

print("\nÄhnlichkeit Premeditation Szenen mit Premeditation Definition:")
for i, sim in enumerate(sim_Premeditation_scenes_def):
    print(f"{Premeditation_scene_labels[i]} vs Definition Premeditation: {sim[0]:.2f}")

# Jede Perseverance Szene wird mit der Definition von 'Mangelndem Durchhaltevermögen' verglichen.
sim_Perseverance_scenes_def = cosine_similarity(embeddings_Perseverance_scenes, [embedding_def_Perseverance])

print("\nÄhnlichkeit Perseverance Szenen mit Perseverance Definition:")
for i, sim in enumerate(sim_Perseverance_scenes_def):
    print(f"{Perseverance_scene_labels[i]} vs Definition Perseverance: {sim[0]:.2f}")

# Berechnung der durchschnittlichen Cross-Facette Ähnlichkeiten:
# Dies dient zur Überprüfung der diskriminanten Validität: Szenen einer Facette sollten
# eine geringere Ähnlichkeit zur Definition der *anderen* Facette aufweisen.
avg_Premeditation_scenes_vs_Perseverance_def = np.mean(cosine_similarity(embeddings_Premeditation_scenes, [embedding_def_Perseverance]))
print(f"\nDurchschnittliche Ähnlichkeit aller Premeditation Szenen zur Perseverance Definition: {avg_Premeditation_scenes_vs_Perseverance_def:.2f}")

avg_Perseverance_scenes_vs_Premeditation_def = np.mean(cosine_similarity(embeddings_Perseverance_scenes, [embedding_def_Premeditation]))
print(f"Durchschnittliche Ähnlichkeit aller Perseverance Szenen zur Premeditation Definition: {avg_Perseverance_scenes_vs_Premeditation_def:.2f}")


# ============================================================================
# VISUALISIERUNG: SZENEN VS DEFINITIONEN (HEATMAPS)
# ============================================================================
# Diese Heatmaps visualisieren die Ähnlichkeit jeder einzelnen Szene zu ihrer
# zugeordneten Definition. Die Farbskala (RdYlGn) zeigt hohe Ähnlichkeit in Grüntönen
# und geringere Ähnlichkeit in Rottönen.
fig, axes = plt.subplots(1, 2, figsize=(10, 7))

# Heatmap für Premeditation Szenen vs. Premeditation Definition:
sns.heatmap(sim_Premeditation_scenes_def, annot=True, fmt='.2f', cmap='RdYlGn',
            xticklabels=['Definition Premeditation'],
            yticklabels=Premeditation_scene_labels,
            vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'Cosine Similarity'})
axes[0].set_title('Premeditation Szenen vs Definition', fontsize=14, fontweight='bold')

# Heatmap für Perseverance Szenen vs. Perseverance Definition:
sns.heatmap(sim_Perseverance_scenes_def, annot=True, fmt='.2f', cmap='RdYlGn',
            xticklabels=['Definition Perseverance'],
            yticklabels=Perseverance_scene_labels,
            vmin=0, vmax=1, ax=axes[1], cbar_kws={'label': 'Cosine Similarity'})
axes[1].set_title('Perseverance Szenen vs Definition', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('szenen_definitionen_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# GEMITTELTE VEKTOREN MIT L2-NORMIERUNG ERSTELLEN (Schritt 4)
# ============================================================================
print("\n" + "="*70)
print("TEIL 3: GEMITTELTE UND L2-NORMIERTE VEKTOREN DER SZENEN")
print("="*70)

# Zur Repräsentation jeder Facette als Ganzes wird ein gemittelter Vektor gebildet.
# Dieser Vektor stellt den 'Durchschnitt' aller Szenen dieser Facette dar.
# Wichtig ist, dass nach der Mittelwertbildung eine erneute L2-Normierung erfolgt,
# um sicherzustellen, dass der gemittelte Vektor ebenfalls eine Länge von 1 hat,
# was für konsistente Cosine-Ähnlichkeitsberechnungen erforderlich ist.

# Gemittelten Vektor für Premeditation Szenen berechnen und normieren:
mean_vector_Premeditation_scenes = np.mean(embeddings_Premeditation_scenes, axis=0)
mean_vector_Premeditation_scenes_normalized = l2_normalize(mean_vector_Premeditation_scenes)

# Gemittelten Vektor für Perseverance Szenen berechnen und normieren:
mean_vector_Perseverance_scenes = np.mean(embeddings_Perseverance_scenes, axis=0)
mean_vector_Perseverance_scenes_normalized = l2_normalize(mean_vector_Perseverance_scenes)

# Überprüfung der L2-Norm, um die korrekte Normierung zu bestätigen.
print(f"\nL2-Norm des gemittelten Vektors Premeditation Szenen: {np.linalg.norm(mean_vector_Premeditation_scenes_normalized):.2f}")
print(f"L2-Norm des gemittelten Vektors Perseverance Szenen: {np.linalg.norm(mean_vector_Perseverance_scenes_normalized):.2f}")
print("\n(Beide Normen sollten ≈ 1.0 sein, was bestätigt, dass die Normierung korrekt ist)")


# ============================================================================
# COSINE-ÄHNLICHKEIT ZWISCHEN GEMITTELTEN SZENEN VEKTOREN UND DEFINITIONEN (Schritt 5)
# ============================================================================
print("\n" + "="*70)
print("TEIL 4: ÄHNLICHKEIT ZWISCHEN GEMITTELTEN SZENEN VEKTOREN UND DEFINITIONEN")
print("="*70)

# Vergleich der gemittelten Facetten-Vektoren mit den jeweiligen Definitionen.
# Dies gibt ein aggregiertes Maß für die semantische Passung der gesamten Facette zur Definition.

# Gemittelten Premeditation Vektor mit der Premeditation Definition vergleichen:
sim_mean_Premeditation_def = cosine_similarity([mean_vector_Premeditation_scenes_normalized], [embedding_def_Premeditation])[0][0]
print(f"\nGemittelter Vektor Premeditation Szenen vs Definition Premeditation: {sim_mean_Premeditation_def:.2f}")

# Gemittelten Perseverance Vektor mit der Perseverance Definition vergleichen:
sim_mean_Perseverance_def = cosine_similarity([mean_vector_Perseverance_scenes_normalized], [embedding_def_Perseverance])[0][0]
print(f"Gemittelter Vektor Perseverance Szenen vs Definition Perseverance: {sim_mean_Perseverance_def:.2f}")

# --- CROSS-FACETTE DURCHSCHNITTSÄHNLICHKEITEN FÜR GEMITTELTE VEKTOREN ---
# Diese Werte überprüfen, ob ein gemittelter Facetten-Vektor eine höhere Ähnlichkeit
# zur Definition seiner eigenen Facette als zur Definition der anderen Facette aufweist.
# Dies ist ein Indikator für die diskriminante Validität auf Facettenebene.
sim_mean_Premeditation_vs_def_Perseverance = cosine_similarity([mean_vector_Premeditation_scenes_normalized], [embedding_def_Perseverance])[0][0]
sim_mean_Perseverance_vs_def_Premeditation = cosine_similarity([mean_vector_Perseverance_scenes_normalized], [embedding_def_Premeditation])[0][0]

print(f"Gemittelter Vektor Premeditation Szenen vs Definition Perseverance: {sim_mean_Premeditation_vs_def_Perseverance:.2f}")
print(f"Gemittelter Vektor Perseverance Szenen vs Definition Premeditation: {sim_mean_Perseverance_vs_def_Premeditation:.2f}")


# ============================================================================
# FACETTEN-ANALYSE: WITHIN VS BETWEEN FACETTE (DURCHSCHNITTE)
# ============================================================================
print("\n" + "="*70)
print("TEIL 5: FACETTEN-ANALYSE (Vergleich der Durchschnittsähnlichkeiten)")
print("="*70)
print("Frage: Sind Szenen innerhalb einer Facette im Durchschnitt ähnlicher als zwischen Facetten?")

# Berechnung der durchschnittlichen Ähnlichkeit innerhalb der Facetten ('Within-Facette Kohäsion').
# Hierbei wird der Mittelwert aller paarweisen Ähnlichkeiten innerhalb einer Gruppe berechnet,
# wobei die Ähnlichkeit einer Szene zu sich selbst (die immer 1 ist) ausgeschlossen wird.
within_facette_Premeditation_avg = np.mean(similarity_matrix_Premeditation[np.triu_indices_from(similarity_matrix_Premeditation, k=1)])
within_facette_Perseverance_avg = np.mean(similarity_matrix_Perseverance[np.triu_indices_from(similarity_matrix_Perseverance, k=1)])

# Gesamtdurchschnitt der Within-Facette Kohäsion.
within_facette_avg = (within_facette_Premeditation_avg + within_facette_Perseverance_avg) / 2

print(f"\nDurchschnittliche Ähnlichkeit innerhalb Facette Premeditation: {within_facette_Premeditation_avg:.2f}")
print(f"Durchschnittliche Ähnlichkeit innerhalb Facette Perseverance: {within_facette_Perseverance_avg:.2f}")
print(f"Gesamtdurchschnitt innerhalb Facetten: {within_facette_avg:.2f}")

# Berechnung der durchschnittlichen Ähnlichkeit zwischen den Facetten ('Between-Facette Separierbarkeit').
# Dies misst, wie stark sich die Szenen der einen Facette von denen der anderen Facette unterscheiden.
between_similarities_matrix = cosine_similarity(embeddings_Premeditation_scenes, embeddings_Perseverance_scenes)
between_facette_avg = np.mean(between_similarities_matrix)

print(f"\nDurchschnittliche Ähnlichkeit zwischen Facetten: {between_similarities_matrix.mean():.2f}")

# Vergleich der Durchschnittswerte:
# Wenn die Within-Facette Ähnlichkeit deutlich höher ist als die Between-Facette Ähnlichkeit,
# deutet dies auf eine gute interne Konsistenz der Facetten und eine klare Abgrenzung voneinander hin.
print(f"\n{'='*70}")
print("ERGEBNIS DES DURCHSCHNITTSVERGLEICHS:")
if within_facette_avg > between_facette_avg:
    diff = within_facette_avg - between_facette_avg
    print(f"✓ JA, Szenen innerhalb einer Facette sind im Durchschnitt ähnlicher!")
    print(f"  Differenz: {diff:.2f} ({diff/between_facette_avg*100:.1f}% höher als Between-Durchschnitt)")
else:
    diff = between_facette_avg - within_facette_avg
    print(f"✘ NEIN, Szenen zwischen Facetten sind im Durchschnitt ähnlicher oder gleich ähnlich!")
    print(f"  Differenz: {diff:.2f}")
print(f"{'='*70}")

# Visualisierung des Vergleichs der Durchschnittswerte:
# Ein Balkendiagramm vergleicht die durchschnittlichen Ähnlichkeiten visuell.
fig, ax = plt.subplots(figsize=(10, 6))
kategorien = ['Within\nPremeditation\n(Avg)', 'Within\nPerseverance\n(Avg)', 'Within\nFacetten\n(Overall Avg)', 'Between\nFacetten\n(Overall Avg)']
werte = [within_facette_Premeditation_avg, within_facette_Perseverance_avg, within_facette_avg, between_facette_avg]
farben = ['#27AE60', '#27AE60', '#2ECC71', '#E74C3C'] # Grüne Farben für 'Within', Rot für 'Between'

bars = ax.bar(kategorien, werte, color=farben, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Durchschnittliche Cosine Similarity', fontsize=12)
ax.set_title('Vergleich: Durchschnittliche Ähnlichkeit innerhalb vs. zwischen Facetten', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=within_facette_avg, color='green', linestyle='--', alpha=0.5, label='Within-Durchschnitt')
ax.axhline(y=between_facette_avg, color='red', linestyle='--', alpha=0.5, label='Between-Durchschnitt')

for i, v in enumerate(werte):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold', fontsize=11)

ax.legend()
plt.tight_layout()
plt.savefig('facetten_vergleich_erweitert.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# VISUALISIERUNG: GEMITTELTE SZENEN VEKTOREN VS DEFINITIONEN (BALKENDIAGRAMM)
# ============================================================================
print("\n" + "="*70)
print("TEIL 6: VISUALISIERUNG GEMITTELTE SZENEN VEKTOREN VS DEFINITIONEN")
print("="*70)

# Dieses Balkendiagramm visualisiert die aggregierte Ähnlichkeit der gemittelten
# Facetten-Vektoren zu den jeweiligen (und den Kreuz-)Definitionen. Dies bietet
# eine schnelle Übersicht über die konvergente und diskriminante Validität auf Facettenebene.
fig, ax = plt.subplots(figsize=(10, 6)) # Angepasste Größe für mehr Kategorien

kategorien_avg_def = [
    'Mean Premeditation Scenes\nvs Definition Premeditation',
    'Mean Perseverance Scenes\nvs Definition Perseverance',
    'Mean Premeditation Scenes\nvs Definition Perseverance',
    'Mean Perseverance Scenes\nvs Definition Premeditation'
]
werte_avg_def = [
    sim_mean_Premeditation_def,
    sim_mean_Perseverance_def,
    sim_mean_Premeditation_vs_def_Perseverance,
    sim_mean_Perseverance_vs_def_Premeditation
]
farben_avg_def = ['#3498DB', '#9B59B6', '#F1C40F', '#E67E22'] # Erweiterte Farbpalette

bars = ax.bar(kategorien_avg_def, werte_avg_def, color=farben_avg_def, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Cosine Similarity', fontsize=12)
ax.set_title('Ähnlichkeit gemittelter Szenen Vektoren zu Definitionen', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

for i, v in enumerate(werte_avg_def):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('mean_scenes_definitionen_balken.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# WILCOXON SIGNED-RANK TEST FÜR SZENENVALIDIERUNG (Schritt 6)
# ============================================================================
print("\n" + "="*70)
print("TEIL 7: WILCOXON SIGNED-RANK TEST (Szenenvalidierung)")
print("="*70)

# Der Wilcoxon Signed-Rank Test ist ein nicht-parametrischer statistischer Test,
# der verwendet wird, um zu prüfen, ob es einen signifikanten Unterschied zwischen
# zwei abhängigen Stichproben gibt. Hier wird er eingesetzt, um zu untersuchen,
# ob die Szenen einer bestimmten Facette signifikant ähnlicher zu ihrer *eigenen* Definition sind
# als zur Definition der *anderen* Facette. Dies ist ein Maß für die diskriminante Validität.

# 1. Daten für Premeditation Szenen vorbereiten:
# Stichprobe 1: Ähnlichkeiten jeder Premeditation Szene zur Premeditation Definition.
# Stichprobe 2: Ähnlichkeiten jeder Premeditation Szene zur Perseverance Definition (Cross-Facette).
sample_premed_own_def = sim_Premeditation_scenes_def.flatten()
sample_premed_other_def = cosine_similarity(embeddings_Premeditation_scenes, [embedding_def_Perseverance]).flatten()

# 2. Durchführung des Wilcoxon Tests für Premeditation Szenen:
# 'alternative='greater'' testet die Hypothese, dass die Ähnlichkeiten zur eigenen Definition
# *größer* sind als die Ähnlichkeiten zur anderen Definition.
if len(sample_premed_own_def) > 0 and len(sample_premed_own_def) == len(sample_premed_other_def):
    statistic_premed, p_value_premed = stats.wilcoxon(sample_premed_own_def, sample_premed_other_def, alternative='greater')
    print("\n--- Wilcoxon Test für Premeditation Szenen ---")
    print(f"  Statistic: {statistic_premed:.2f}")
    print(f"  P-value: {p_value_premed:.3f}")
    if p_value_premed < 0.05:
        print("  -> Result: Premeditation Szenen sind signifikant ähnlicher zur Premeditation Definition als zur Perseverance Definition.")
    else:
        print("  -> Result: Kein signifikanter Unterschied in der Ähnlichkeit von Premeditation Szenen zu den beiden Definitionen.")
else:
    print("\n--- Wilcoxon Test für Premeditation Szenen ---")
    print("  Kann Wilcoxon Test nicht durchführen: Ungleich große Stichproben oder keine Daten.")

# 3. Daten für Perseverance Szenen vorbereiten (analog zu Premeditation):
sample_persev_own_def = sim_Perseverance_scenes_def.flatten()
sample_persev_other_def = cosine_similarity(embeddings_Perseverance_scenes, [embedding_def_Premeditation]).flatten()

# 4. Durchführung des Wilcoxon Tests für Perseverance Szenen:
if len(sample_persev_own_def) > 0 and len(sample_persev_own_def) == len(sample_persev_other_def):
    statistic_persev, p_value_persev = stats.wilcoxon(sample_persev_own_def, sample_persev_other_def, alternative='greater')
    print("\n--- Wilcoxon Test für Perseverance Szenen ---")
    print(f"  Statistic: {statistic_persev:.2f}")
    print(f"  P-value: {p_value_persev:.3f}")
    if p_value_persev < 0.05:
        print("  -> Result: Perseverance Szenen sind signifikant ähnlicher zur Perseverance Definition als zur Premeditation Definition.")
    else:
        print(
            "  -> Result: Kein signifikanter Unterschied in der Ähnlichkeit von Perseverance Szenen zu den beiden Definitionen."
        )
else:
    print("\n--- Wilcoxon Test für Perseverance Szenen ---")
    print("  Kann Wilcoxon Test nicht durchführen: Ungleich große Stichproben oder keine Daten.")


print("\n✓ Analyse abgeschlossen! Alle Visualisierungen wurden erstellt und gespeichert.")
print("  - szenen_aehnlichkeit_heatmaps.png")
print("  - szenen_definitionen_heatmaps.png")
print("  - facetten_vergleich_erweitert.png")
print("  - mean_scenes_definitionen_balken.png")
