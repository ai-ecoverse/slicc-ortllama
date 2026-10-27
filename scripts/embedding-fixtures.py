import json
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer

SOURCE = "google/embeddinggemma-2"
EXPORT = "onnx-community/embeddinggemma-2-ONNX"
DIMENSIONS = [768, 256, 128]
OUT = Path(__file__).resolve().parent.parent / "fixtures" / f"{EXPORT}.json"

LIGHTHOUSE = (
    "The old lighthouse stood at the edge of the cliff for more than a hundred years. "
    "Every night its keeper climbed the narrow stairs, trimmed the wick, and watched the beam "
    "sweep across the dark water. Ships passing in the fog relied on that steady light to find "
    "their way around the rocks."
)

REVERSE = """def reverse(head):
    prev = None
    while head:
        head.next, prev, head = prev, head, head.next
    return prev"""

DEBOUNCE = """export function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}"""

TOP = """SELECT customer_id, SUM(total) AS spent
FROM orders
GROUP BY customer_id
ORDER BY spent DESC
LIMIT 10;"""

QUERY = {"input_type": "query"}
DOCUMENT = {"input_type": "document"}

TEXTS = [
    ("mars", "Retrieval-query", {"task": "search result", **QUERY}, "Which planet is known as the Red Planet?"),
    ("venus", "Retrieval-document", DOCUMENT, "Venus is often called Earth's twin because of its similar size and proximity."),
    ("red-planet", "Retrieval-document", DOCUMENT, "Mars, known for its reddish appearance, is often referred to as the Red Planet."),
    ("jupiter", "Retrieval-document", DOCUMENT, "Jupiter, the largest planet in our solar system, has a prominent red spot."),
    ("saturn", "Retrieval-document", DOCUMENT, "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."),
    ("code", "CodeRetrieval", {"task": "code retrieval", **QUERY}, "reverse a singly linked list in place"),
    ("reverse.py", None, None, f"title: reverse.py | text: {REVERSE}"),
    ("debounce.js", None, None, f"title: debounce.js | text: {DEBOUNCE}"),
    ("top.sql", None, None, f"title: top_customers.sql | text: {TOP}"),
    ("guitar", "STS", {"task": "sentence similarity"}, "A man is playing a guitar on stage."),
    ("gitarre", "STS", {"task": "sentence similarity"}, "Ein Mann spielt auf der Bühne Gitarre."),
    ("lighthouse", None, {"input_type": "document", "title": "The lighthouse"}, LIGHTHOUSE),
]

RANKS = {"mars": ["venus", "red-planet", "jupiter", "saturn"], "code": ["reverse.py", "debounce.js", "top.sql"]}


def prompt_of(name, extension, text):
    if name:
        return model.prompts[name] + text
    if extension and extension.get("title"):
        return f"title: {extension['title']} | text: {text}"
    return text


model = SentenceTransformer(SOURCE, device="cpu", model_kwargs={"dtype": torch.float32})
inputs = [prompt_of(name, extension, text) for _, name, extension, text in TEXTS]
vectors = {dim: model.encode(inputs, truncate_dim=dim, normalize_embeddings=True, batch_size=1) for dim in DIMENSIONS}
names = [t[0] for t in TEXTS]
fixtures = []
for i, (name, prompt_name, extension, text) in enumerate(TEXTS):
    entry = {"name": name}
    if prompt_name:
        entry["prompt_name"] = prompt_name
    if extension:
        entry["request"] = extension
    entry["text"] = text
    entry["input"] = inputs[i]
    entry["tokens"] = len(model.tokenizer(inputs[i])["input_ids"])
    if name in RANKS:
        ranked = {}
        for dim in DIMENSIONS:
            scores = {doc: float(vectors[dim][i] @ vectors[dim][names.index(doc)]) for doc in RANKS[name]}
            ranked[str(dim)] = sorted(scores, key=scores.get, reverse=True)
        entry["ranks"] = ranked
        entry["scores"] = {doc: round(float(vectors[768][i] @ vectors[768][names.index(doc)]), 4) for doc in RANKS[name]}
    entry["embeddings"] = {str(dim): [round(float(x), 5) for x in vectors[dim][i]] for dim in DIMENSIONS}
    fixtures.append(entry)

about = {
    "model": EXPORT,
    "reference": SOURCE,
    "reference_revision": model_info(SOURCE).sha,
    "dtype": "float32",
    "device": "cpu",
    "sentence_transformers": version("sentence-transformers"),
    "transformers": version("transformers"),
    "torch": version("torch"),
    "generator": "scripts/embedding-fixtures.py",
}
OUT.write_text(json.dumps({**about, "fixtures": fixtures}) + "\n")
print(f"{SOURCE} -> {OUT}", file=sys.stderr)
