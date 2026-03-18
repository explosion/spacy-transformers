"""Smoke test: load en_core_web_hftrf and run NER end-to-end."""
import spacy

nlp = spacy.load("en_core_web_hftrf")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
assert len(doc) > 0
assert len(doc.ents) > 0
print(f"Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
print(f"Tokens: {len(doc)}")
print("Smoke test passed.")
