"""Offline inference with the exported CTranslate2 int8 model (CPU).

    from ffmt.translator import ChatTranslator
    tr = ChatTranslator("exports/latest")
    tr.translate(["where is the blacksmith?"], "es")   # -> ["¿dónde está el herrero?"]
    tr.detect("bhai mujhe gold chahiye")                # -> ("hi", 0.97)
"""
import re
from pathlib import Path

import ctranslate2
import sentencepiece as spm

from .chat import expand, peel
from .langid import LangId
from .names import Lexicon, NameBook, bare, protect, restore, spelling_jobs, take_spellings

LANGS = ["en", "zh", "hi", "es", "ar", "pt", "fr"]
# The model was trained mostly on single sentences and tends to drop everything after the
# first one, so each message is split into sentences, translated in one batch, and rejoined.
_SENT_END = re.compile(r"(?<=[.!?…؟।])\s+|(?<=[。！？])\s*|\s*\n+\s*")
_JOIN = {"zh": ""}  # Chinese sentences are written without a space between them


def split_sentences(text):
    parts = [p.strip() for p in _SENT_END.split(text.strip())]
    return [p for p in parts if p] or [text.strip()]


class ChatTranslator:
    def __init__(self, export_dir="exports/latest", threads=2, parallel=6, device="cpu", names=None):
        """threads: CPU threads per batch. parallel: batches run at the same time (each target
        language is its own batch, so six languages take about as long as one).
        names: a NameBook to share remembered name spellings (a new one by default)."""
        d = Path(export_dir)
        self.lex = Lexicon(d / "words.npz")
        self.book = names if names is not None else NameBook()
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(d / "spm.model"))
        self.model = ctranslate2.Translator(
            str(d / "ct2"), device=device, compute_type="int8" if device == "cpu" else "int8_float16",
            inter_threads=parallel if device == "cpu" else 1, intra_threads=threads,
        )
        lid = d / "langid.npz"
        self.lid = LangId.load(lid) if lid.exists() else None

    def detect(self, text):
        return self.lid.detect(text) if self.lid else (None, 0.0)

    def _run(self, jobs, beam=1, max_len=128):
        """jobs: [(target, sentence)] -> one decoded string per job. One batch per target
        language (or per sentence when there is only one language), all submitted at once
        so they run in parallel."""
        toks = [[f"<2{t}>"] + self.sp.encode(s, out_type=str)[:200] + ["</s>"] for t, s in jobs]
        groups = {}
        for i, (t, _) in enumerate(jobs):
            groups.setdefault(t, []).append(i)
        if len(groups) == 1:
            groups = {i: [i] for i in range(len(jobs))}
        futures = [(idx, self.model.translate_batch(
            [toks[i] for i in idx], beam_size=beam, max_decoding_length=max_len, max_batch_size=64,
            repetition_penalty=1.1, no_repeat_ngram_size=0, return_scores=False, asynchronous=True,
        )) for idx in groups.values()]
        out = [""] * len(jobs)
        for idx, res in futures:
            for i, r in zip(idx, res):
                out[i] = self.sp.decode(r.result().hypotheses[0])
        return out

    def _many(self, texts, targets, sources, names=(), learn=True, beam=1, max_len=128):
        """texts[i] (in sources[i]) into each of targets[i] -> [{target: text}].
        Splits sentences, keeps universal chat words and names out of the model, expands
        abbreviations, and runs every sentence (and any new name spellings) in one batch."""
        plans, jobs, used = [], [], []
        for text, tgts, src in zip(texts, targets, sources):
            sents = []
            for s in split_sentences(text):
                pre, core, suf = peel(s)
                held, slots = protect(core, src, self.lex, self.book, names) if core else ("", [])
                used += [n for n, _ in slots]
                sents.append((pre, expand(held, src) if held else "", suf, slots, expand(core, src) if core else ""))
            plans.append((sents, tgts))
            jobs += [(t, s[1]) for t in tgts for s in sents if s[1]]
        spell = spelling_jobs(used, {t for _, tgts in plans for t in tgts}, self.book)
        outs = self._run(jobs + [(t, s) for t, _, s in spell], beam, max_len) if jobs or spell else []
        take_spellings(spell, outs[len(jobs):], self.book)
        if learn:
            for n in dict.fromkeys(bare(n) for n in used):
                self.book.learn(n, self.lex)
        flat = iter(outs[:len(jobs)])
        results, retry = [], []
        for sents, tgts in plans:
            res = {}
            for t in tgts:
                parts = []
                for pre, core, suf, slots, raw in sents:
                    mid = ""
                    if core:
                        mid, ok = restore(next(flat), slots, t, self.book)
                        if not ok:
                            retry.append((parts, len(parts), t, pre, suf, core, slots, raw))
                    parts.append(" ".join(x for x in (pre, mid, suf) if x))
                res[t] = parts
            results.append(res)
        if retry:
            # The model dropped a name: once more with a wider search, then as typed.
            last = []
            for r, mid in zip(retry, self._run([(r[2], r[5]) for r in retry], 4, max_len)):
                parts, i, t, pre, suf, _, slots, raw = r
                mid, ok = restore(mid, slots, t, self.book)
                if ok:
                    parts[i] = " ".join(x for x in (pre, mid, suf) if x)
                else:
                    last.append(r)
            for (parts, i, t, pre, suf, _, _, raw), mid in zip(last, self._run([(r[2], r[7]) for r in last], beam, max_len) if last else []):
                parts[i] = " ".join(x for x in (pre, mid, suf) if x)
        return [{t: _JOIN.get(t, " ").join(p for p in parts if p) for t, parts in res.items()} for res in results]

    def translate(self, texts, target, source=None, names=(), learn=True, beam=1, max_len=128):
        """Translate a batch of chat lines into one target language.
        source: the lines' language (enables abbreviation expansion); detected if langid exists.
        names: player/NPC names to keep out of the model. learn: remember spotted names."""
        srcs = [source or self.detect(t)[0] for t in texts]
        return [r[target] for r in self._many(texts, [[target]] * len(texts), srcs, names, learn, beam, max_len)]

    def translate_for(self, text, targets, source=None, names=(), learn=True):
        """One chat line into several languages; skips its own language."""
        src = source or self.detect(text)[0]
        need = [t for t in targets if t != src]
        out = {t: text for t in targets if t == src}
        out.update(self._many([text], [need], [src], names, learn)[0])
        return src, out
