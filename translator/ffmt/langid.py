"""Tiny language detector for chat lines (own model, trained from our data).

Chinese, Hindi (Devanagari) and Arabic are told apart by script. Latin-script
text (English, Spanish, Portuguese, French, romanized Hindi) goes through a
character n-gram naive Bayes model with hashed features.
"""
import random
import re
import zlib
from pathlib import Path

import numpy as np

LATIN_CLASSES = ["en", "es", "pt", "fr", "hi"]
BUCKETS = 1 << 18
_HAN = re.compile(r"[㐀-䶿一-鿿]")
_DEV = re.compile(r"[ऀ-ॿ]")
_ARA = re.compile(r"[؀-ۿݐ-ݿ]")
_LAT = re.compile(r"[A-Za-zÀ-ɏ]")


def _features(text):
    t = " " + re.sub(r"\s+", " ", text.lower()).strip() + " "
    idx = []
    for n in (1, 2, 3, 4):
        for i in range(len(t) - n + 1):
            idx.append(zlib.crc32(t[i:i + n].encode("utf-8")) & (BUCKETS - 1))
    return idx


class LangId:
    def __init__(self, logp=None, prior=None):
        self.logp = logp      # (classes, BUCKETS) float32
        self.prior = prior    # (classes,)

    def detect(self, text):
        """Returns (lang, confidence 0..1)."""
        han, dev, ara, lat = (len(r.findall(text)) for r in (_HAN, _DEV, _ARA, _LAT))
        top = max(han, dev, ara, lat)
        if top == 0:
            return None, 0.0
        if top == han:
            return "zh", han / max(han + lat, 1)
        if top == dev:
            return "hi", 1.0
        if top == ara:
            return "ar", 1.0
        f = _features(text)
        scores = self.prior + self.logp[:, f].sum(axis=1)
        scores = scores - scores.max()
        p = np.exp(scores)
        p /= p.sum()
        k = int(p.argmax())
        return LATIN_CLASSES[k], float(p[k])

    def save(self, path):
        np.savez_compressed(path, logp=self.logp.astype(np.float16), prior=self.prior)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["logp"].astype(np.float32), d["prior"])

    @classmethod
    def train(cls, shard_paths, per_lang=200_000, seed=0):
        from .noise import chatify, romanize_hindi
        rng = random.Random(seed)
        counts = np.ones((len(LATIN_CLASSES), BUCKETS), dtype=np.float64)  # add-one smoothing
        seen = {l: 0 for l in LATIN_CLASSES}
        for p in shard_paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 4:
                        continue
                    for lang, text in ((parts[0], parts[2]), (parts[1], parts[3])):
                        if lang not in seen or seen[lang] >= per_lang:
                            continue
                        if lang == "hi":
                            text = romanize_hindi(text, rng)
                        else:
                            text = chatify(text, lang, rng)
                        k = LATIN_CLASSES.index(lang)
                        np.add.at(counts[k], _features(text), 1)
                        seen[lang] += 1
            if all(v >= per_lang for v in seen.values()):
                break
        logp = np.log(counts / counts.sum(axis=1, keepdims=True)).astype(np.float32)
        prior = np.log(np.full(len(LATIN_CLASSES), 1.0 / len(LATIN_CLASSES))).astype(np.float32)
        return cls(logp, prior), seen


def default_path():
    return Path(__file__).resolve().parents[1] / "exports" / "latest" / "langid.npz"
