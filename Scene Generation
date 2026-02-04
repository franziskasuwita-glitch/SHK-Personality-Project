import os
import time
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# SCHRITT 1: API KONFIGURATION 
# -----------------------------------------------------------------------------

def get_api_key():
    # Versucht Key aus Umgebungsvariablen zu lesen (Lokal/IDE)
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        # Falls in Colab, versucht userdata (dein Original-Weg)
        try:
            from google.colab import userdata
            api_key = userdata.get('GEMINI_API_KEY')
        except:
            api_key = input("Bitte Gemini API-Key eingeben: ").strip()
    return api_key

GEMINI_API_KEY = get_api_key()
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Beibehaltung deines Modell-Aufrufs (angepasst auf aktuell verfügbare Namen)
    model = genai.GenerativeModel('gemini-2.5-pro') 
else:
    print("❌ Kein API-Key gefunden. Skript wird abgebrochen.")
    exit()

# -----------------------------------------------------------------------------
# SCHRITT 2: INITIALISIERUNG
# -----------------------------------------------------------------------------

# Modell für Ähnlichkeit (Original: MiniLM-L3-v2)
sentence_transformer_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

definition_lack_of_premeditation = "Verhalten, charakterisiert durch wenig oder keine vorausschauende Planung, Reflexion oder Berücksichtigung der Folgen einer Handlung, insbesondere einer, die das Eingehen von Risiken beinhaltet."
definition_lack_of_perseverance = "Unfähigkeit, Projekte abzuschließen und unter Bedingungen zu arbeiten, die Widerstandsfähigkeit gegenüber Hindernissen und ablenkenden Reizen erfordern."

embedding_lack_of_premeditation = sentence_transformer_model.encode(definition_lack_of_premeditation)
embedding_lack_of_perseverance = sentence_transformer_model.encode(definition_lack_of_perseverance)

def calculate_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None: return -1
    return cosine_similarity([embedding1], [embedding2])[0][0]

# ausgewählte Items einfügen
scenes_info = [
    {"item_number": 182, "item_description": "Ich handle üblicherweise aus einem Impuls heraus, ohne die möglichen Folgen zu bedenken.", "subfacet": "Mangelnde Voraussicht"},
    {"item_number": 9, "item_description": "Ich beteilige mich oft an Projekten, ohne vorher mögliche Probleme zu bedenken.", "subfacet": "Mangelndes Durchhaltevermögen"},
    {"item_number": 7, "item_description": "Ich gerate oft in Schwierigkeiten, weil ich nicht nachdenke, bevor ich handle.", "subfacet": "Mangelnde Voraussicht"},
    {"item_number": 78, "item_description": "Wenn ein Vorhaben sich als zu schwierig erweist, neige ich dazu etwas Neues anzufangen", "subfacet": "Mangelndes Durchhaltevermögen"},
    {"item_number": 12, "item_description": "Ich sage und tue oft Dinge, ohne die Konsequenzen zu bedenken.", "subfacet": "Mangelnde Voraussicht"},
    {"item_number": 175, "item_description": "Ich kann Ziele nicht erreichen, weil andere Dinge meine Aufmerksamkeit auf sich ziehen.", "subfacet": "Mangelndes Durchhaltevermögen"},
    {"item_number": 198, "item_description": "Ich habe die Konsequenzen einer Handlung nicht bedacht.", "subfacet": "Mangelnde Voraussicht"},
    {"item_number": 176, "item_description": "Ich habe Schwierigkeiten, bestimmte Ziele zu verfolgen, selbst über kurze Zeiträume.", "subfacet": "Mangelndes Durchhaltevermögen"},
    {"item_number": 5, "item_description": "Gelegentlich handele ich zuerst und denke dann erst darüber nach.", "subfacet": "Mangelnde Voraussicht"},
    {"item_number": 27, "item_description": "Wenn ich einmal mit etwas angefangen habe, höre ich leicht wieder auf.", "subfacet": "Mangelndes Durchhaltevermögen"},
    {"item_number": 6, "item_description": "Ich handele oft aus dem Augenblick heraus.", "subfacet": "Mangelnde Voraussicht"},
    {"item_number": 26, "item_description": "Es macht mir nichts aus, wenn Aufgaben unerledigt bleiben.", "subfacet": "Mangelndes Durchhaltevermögen"}
]

# -----------------------------------------------------------------------------
# SCHRITT 3: GENERIERUNGSSCHLEIFE (Prompt an KI)
# -----------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.3
MAX_RETRIES = 3
final_scenes_with_similarity = []

for i, scene_info in enumerate(scenes_info):
    scene_prompt = f"""Schreibe eine Szene bestehend aus 3-4 Sätzen.

Die Szene soll die Disposition "Impulsivität" über die Subfacette "{scene_info['subfacet']}" darstellen:

SUBFACETTE: {scene_info['subfacet']}
Definition: {"Verhalten, charakterisiert durch wenig oder keine vorausschauende Planung, Reflexion oder Berücksichtigung der Folgen einer Handlung, insbesondere einer, die das Eingehen von Risiken beinhaltet." if scene_info['subfacet'] == 'Mangelnde Voraussicht' else "Unfähigkeit, Projekte abzuschließen und unter Bedingungen zu arbeiten, die Widerstandsfähigkeit gegenüber Hindernissen und ablenkenden Reizen erfordern."}

Eine Disposition der Persönlichkeit bezeichnet "eine individuell unterschiedliche, zeitlich relativ stabile Bereitschaft, auf Umweltbedingungen mit bestimmten Gedanken, Gefühlen, Emotionen oder Verhaltensweisen zu reagieren."

Formatiere die Szene wie folgt:

SZENE {i+1} (Item {scene_info['item_number']}): "{scene_info['item_description']}"
[Text der Szene]

In der Szene soll die psychologische Disposition der Impulsivität SUBKLINISCH ausgeprägt sein:
- Deutlich überdurchschnittlich, auffällig außerhalb der Norm
- Spürbare negative Konsequenzen im Alltag
- NICHT behandlungsbedürftig

Kriterien für die Szene:
- Impulsives Verhalten PLUS dessen negative Folgen schildern
- Ich-Perspektive
- Neutrale Perspektive
- Geschrieben im Präsens
- Realistische Alltagssituationen mit universalen Erfahrungen
- Vermeide Fremdwörter und verwende Sprache in ihrem natürlichen Gebrauch des Alltags
- Die Szene = eigenständige Situation"""

    retries = 0
    success = False
    while retries < MAX_RETRIES:
        try:
            print(f"Generiere Szene {i+1} (Item {scene_info['item_number']}), Versuch {retries+1}...")
            response = model.generate_content(scene_prompt)
            generated_text = response.text
            
            scene_embedding = sentence_transformer_model.encode(generated_text)
            def_embedding = embedding_lack_of_premeditation if scene_info['subfacet'] == 'Mangelnde Voraussicht' else embedding_lack_of_perseverance
            
            score = calculate_similarity(scene_embedding, def_embedding)
            print(f"   Ähnlichkeit: {score:.4f}")

            if score >= SIMILARITY_THRESHOLD:
                final_scenes_with_similarity.append({
                    "scene_number": i + 1,
                    "item_number": scene_info['item_number'],
                    "item_description": scene_info['item_description'],
                    "subfacet": scene_info['subfacet'],
                    "generated_text": generated_text,
                    "similarity": score
                })
                success = True
                break
            retries += 1
        except Exception as e:
            print(f"Fehler: {e}")
            time.sleep(2)
            retries += 1

    if not success:
        final_scenes_with_similarity.append({
            "scene_number": i + 1, "item_number": scene_info['item_number'],
            "item_description": scene_info['item_description'], "subfacet": scene_info['subfacet'],
            "generated_text": "Generierung nach Fehlversuchen abgebrochen.", "similarity": -1
        })

# -----------------------------------------------------------------------------
# SCHRITT 4: SPEICHERN (UTF-8)
# -----------------------------------------------------------------------------
filename = f"ergebnisse_geschichte_{int(time.time())}.txt"
with open(filename, 'w', encoding='utf-8') as f:
    for s in final_scenes_with_similarity:
        f.write(f"SZENE {s['scene_number']} (Item {s['item_number']})\nScore: {s['similarity']:.4f}\n{s['generated_text']}\n{'-'*40}\n")

print(f"Fertig! Datei gespeichert als: {filename}")
