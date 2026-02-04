#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ITEM AUGMENTATION TOOL - PSYCHOMETRIC RESEARCH
==============================================
Dieses Skript dient der automatisierten Augmentierung psychologischer Items.
Ziel ist es, durch semantische Umformulierungen (mittels GPT-4o-mini) die 
Wortanzahl pro Item-Block zu standardisieren, um stabilere Embeddings zu erhalten.
"""

import os
import time
import openai
from langdetect import detect

# ============================================================================
# KONFIGURATION & ITEM-INPUT
# ============================================================================

# Hier können Items direkt eingefügt werden. 
ORIGINAL_ITEMS = [
    "Ich handle oft aus dem Augenblick heraus.",
    "Ich sage und tue oft Dinge, ohne die Konsequenzen zu bedenken.",
    "Wenn ein Vorhaben sich als zu schwierig erweist, neige ich dazu etwas Neues anzufangen"
]

def get_api_key():
    """Sicherer Bezug des API-Keys aus Umgebungsvariablen oder manueller Eingabe."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback für Umgebungen ohne gesetzte Variable (z.B. lokaler PC)
        api_key = input("Bitte geben Sie Ihren OpenAI API-Key ein: ").strip()
    return api_key

# ============================================================================
# AUGMENTIERUNGS-LOGIK
# ============================================================================

class ItemAugmenter:
    def __init__(self, api_key, target_words=45):
        self.client = openai.OpenAI(api_key=api_key)
        self.target_words = target_words
        self.max_retries = 3

    def generate_versions(self, item_text, retries=None):
        if retries is None: retries = self.max_retries
        
        words = item_text.split()
        original_length = len(words)
        
        # Bestimmung der Anzahl benötigter Varianten basierend auf Wortlänge
        approx_variants = 3 if (self.target_words - original_length) >= 20 else 2

        # Spracherkennung für lokalisierte Prompts
        try:
            lang = detect(item_text)
        except:
            lang = 'de'

        if lang == 'de':
            prompt = (
                f"Das folgende psychologische Item besteht aus {original_length} Wörtern:\n"
                f"\"{item_text}\"\n\n"
                f"Erstelle {approx_variants} alternative Formulierungen, die dieselbe Eigenschaft betonen. "
                f"Ziel: Gesamtlänge (Original + Alternativen) ca. {self.target_words} Wörter. "
                "Format: 1), 2), 3)."
            )
        else:
            prompt = (
                f"The following item has {original_length} words:\n"
                f"\"{item_text}\"\n\n"
                f"Create {approx_variants} alternative formulations to reach a total of "
                f"approx. {self.target_words} words. Format: 1), 2), 3)."
            )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a specialist in psychometric item construction."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7,
            )
            content = response.choices[0].message.content.strip()
            # Bereinigung der Ausgabe (Entfernung von Aufzählungszeichen)
            variants = [line.strip('123456789) \t') for line in content.splitlines() if line.strip()]
            return [item_text] + variants[:approx_variants]

        except Exception as e:
            if retries > 0:
                print(f"⚠️ Fehler/Limit erreicht. Re-try in 5s... ({retries} übrig)")
                time.sleep(5)
                return self.generate_versions(item_text, retries - 1)
            print(f"❌ Fehler bei Item '{item_text[:20]}': {e}")
            return [item_text]

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    api_key = get_api_key()
    if not api_key:
        print("Abbruch: Kein API-Key vorhanden.")
        return

    augmenter = ItemAugmenter(api_key)
    
    print(f"Starte Augmentierung für {len(ORIGINAL_ITEMS)} Items...\n")
    print("="*50)

    for idx, item in enumerate(ORIGINAL_ITEMS, 1):
        results = augmenter.generate_versions(item)
        
        print(f"\nITEM {idx}: {item}")
        print(f"Ergebnis (Zusammengefügt):")
        # Zusammenfügen der Varianten zu einem langen Textblock für das Embedding
        combined_text = " ".join(results)
        print(combined_text)
        print(f"Wortanzahl: {len(combined_text.split())}")
        print("-" * 30)

if __name__ == "__main__":
    main()
