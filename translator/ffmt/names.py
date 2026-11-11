"""Player and NPC names: kept away from the model and written the same way every time.

Before translating, each name in a sentence is swapped for a common stand-in name the model
knows well (Kevin, Brandon, Steven, Patrick), so grammar around it comes out right ("matar a
Kevin", "la espada de Kevin"), and swapped back afterwards. In en/es/fr/pt the name is
written as typed; in zh/ar/hi in that language's script. That spelling is made once by the
model inside short frame sentences ("My name is Lekai." -> "اسمي ليكاي") and remembered, so a
name is always written the same way. The memory also works backwards: ليكاي in an Arabic
message comes back as Lekai.

Names come from the caller (the game knows its players and NPCs) or are spotted:
@mentions, words with digits, underscores or inner capitals (xXSlayer99, Dark_Knight,
ShadowBlade), capitalized words the training data never had (Lekai), and Latin words inside
zh/ar/hi text that are not everyday words.
"""
import hashlib
import re
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from .chat import ABBREV, UNIVERSAL

LATIN_LANGS = {"en", "es", "fr", "pt"}
SCRIPT_LANGS = ("zh", "ar", "hi")
LEXICON_LANGS = ("en", "es", "fr", "pt", "ar", "hi")
_SCRIPT = {"zh": re.compile(r"[㐀-鿿]"), "ar": re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-ﻼ]"),
           "hi": re.compile(r"[ऀ-ॿ]")}
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_LETTER = re.compile(r"[^\W\d_]")
_STRIP = " \t.,!?;:\"'()[]{}«»“”‘’「」『』（）【】。，！？、；：…؟،؛·।-‎‏‪‫‬‭‮"
_LATIN_TOKEN = re.compile(r"@?[A-Za-zÀ-ɏ0-9_]+(?:['’-][A-Za-zÀ-ɏ0-9_]+)*")
_AR_PREFIX = "وبلفك"
# What counts as part of a word next to a name: letters plus the vowel marks \w misses
# (not punctuation such as ؟ or ।).
_BOUND = {"ar": r"\wً-ٰٟ", "hi": r"\wऀ-ःऺ-ॏ॑-ॗॢॣ"}

# Stand-ins and every spelling the model uses for them (checked on 20 chat sentences).
STANDINS = {
    "Kevin": {"zh": ["凯文"], "ar": ["كيفين", "كيفن"], "hi": ["केविन"]},
    "Brandon": {"zh": ["布兰登"], "ar": ["براندون", "براندن"], "hi": ["ब्रैंडन"]},
    "Steven": {"zh": ["史蒂芬", "史蒂文"], "ar": ["ستيفن"], "hi": ["स्टीवन", "स्टीव"]},
    "Patrick": {"zh": ["帕特里克"], "ar": ["باتريك"], "hi": ["पैट्रिक"]},
}

# A name is written in zh/ar/hi by translating it inside these sentences and cutting the
# frame off again; the spelling most frames agree on wins (a name alone is often misread
# as a word: "Mira" alone comes out as "look").
FRAMES = ["My name is {}.", "{} is here.", "I am {}."]
_CUT = {
    "zh": [r"^我叫(.+)$", r"^(.+?)(?:来了|在这里|在这儿|在这)$", r"^我(?:就)?是(.+)$"],
    "ar": [r"^اسمي (.+)$", r"^(.+?) (?:هنا|موجود هنا|موجودة هنا|موجود|موجودة)$", r"^أنا (.+)$"],
    "hi": [r"^मेरा नाम (.+) है$", r"^(.+?) यहाँ (?:है|हैं)$", r"^मैं (.+) (?:हूँ|हूं)$"],
}
_CUT = {t: [re.compile(p) for p in ps] for t, ps in _CUT.items()}


def word_hash(w):
    return int.from_bytes(hashlib.blake2b(w.encode("utf-8"), digest_size=8).digest(), "little")


def words_of(text):
    """Lowercase words of a line, punctuation stripped (same rule for building and lookup)."""
    out = []
    for tok in text.lower().split():
        tok = tok.strip(_STRIP)
        if tok and _LETTER.search(tok):
            out.append(tok)
    return out


def bare(name):
    return name[1:] if name.startswith("@") else name


def _is_script(text, t):
    letters = _LETTER.findall(text)
    return bool(letters) and all(_SCRIPT[t].match(c) for c in letters)


def _mostly_script(text, lang):
    letters = _LETTER.findall(text)
    return bool(letters) and sum(1 for c in letters if _SCRIPT[lang].match(c)) / len(letters) >= 0.5


class Lexicon:
    """Word lists from scripts/lexicon.py: known words, and the most common ones."""

    def __init__(self, path=None):
        self.known, self.common = {}, {}
        if path and Path(path).exists():
            z = np.load(path)
            for l in LEXICON_LANGS:
                self.known[l] = z[f"{l}_known"]
                self.common[l] = z[f"{l}_common"]

    @staticmethod
    def _has(arr, w):
        if arr is None or not len(arr):
            return False
        h = np.uint64(word_hash(w.lower()))
        i = np.searchsorted(arr, h)
        return bool(i < len(arr) and arr[i] == h)

    def is_known(self, w, lang):
        return self._has(self.known.get(lang), w)

    def is_common(self, w, lang):
        return self._has(self.common.get(lang), w)

    def __bool__(self):
        return bool(self.known)


class NameBook:
    """Remembered spellings: name -> {lang: text}, plus the way back (text -> name)."""

    def __init__(self, cap=50_000):
        self.cap = cap
        self.spell = OrderedDict()
        self.back = {t: {} for t in SCRIPT_LANGS}
        self.lower = {}
        self._rx = {}

    def get(self, name, t):
        d = self.spell.get(name)
        return d.get(t) if d else None

    def put(self, name, t, text):
        self.spell.setdefault(name, {})[t] = text
        self.spell.move_to_end(name)
        while len(self.spell) > self.cap:
            old, d = self.spell.popitem(last=False)
            for lang, txt in d.items():
                if self.back.get(lang, {}).get(txt) == old:
                    del self.back[lang][txt]
                    self._rx.pop(lang, None)
            self.lower.pop(old.lower(), None)

    def learn(self, name, lex):
        """Let this name be found again: its zh/ar/hi spellings, and lowercase typing."""
        for t, txt in self.spell.get(name, {}).items():
            if t not in self.back or self.back[t].get(txt) == name or not _is_script(txt, t):
                continue
            if t == "zh" and len(txt) < 2:
                continue  # one character is part of too many words
            if t != "zh" and (len(txt) < 3 or lex.is_common(txt, t)):
                continue  # would hijack an everyday word
            self.back[t][txt] = name
            self._rx.pop(t, None)
        if len(name) >= 4 and not name.islower() and not any(lex.is_known(name, l) for l in LATIN_LANGS):
            self.lower[name.lower()] = name

    def finder(self, t):
        """Regex for remembered spellings in a zh/ar/hi text (group 1: Arabic prefix letter)."""
        if t not in self._rx:
            keys = sorted(self.back[t], key=len, reverse=True)
            if not keys:
                self._rx[t] = None
            elif t == "zh":
                self._rx[t] = re.compile("()(" + "|".join(map(re.escape, keys)) + ")")
            else:
                pre = f"[{_AR_PREFIX}]?" if t == "ar" else ""
                b = _BOUND[t]
                self._rx[t] = re.compile(rf"(?<![{b}])({pre})({'|'.join(map(re.escape, keys))})(?![{b}])")
        return self._rx[t]


def looks_like_name(tok, src, lex, keep):
    """tok: one Latin-script token (no possessive). True when it should be kept away from the model."""
    if tok in keep or tok.startswith("@") and len(tok) > 1:
        return True
    low = tok.lower()
    if low in UNIVERSAL or low in ABBREV.get(src, {}) or not _LETTER.search(tok):
        return False
    if tok.isupper():
        return False  # NPC, DPS, shouting: the model handles these
    if "_" in tok or (tok[0].isalpha() and any(c.isdigit() for c in tok)):
        return True
    if any(a.islower() and b.isupper() for a, b in zip(tok, tok[1:])):
        return True  # xXSlayerXx, ShadowBlade
    return bool(tok[0].isupper() and len(tok) >= 3 and lex and not lex.is_known(tok, src))


def _latin_word_is_name(tok, lex, keep):
    """A Latin word inside zh/ar/hi text: a name unless it is an everyday word ("boss", "OK")."""
    if tok in keep or tok.startswith("@"):
        return True
    if tok.isupper() or tok.lower() in UNIVERSAL:
        return False
    return looks_like_name(tok, "en", lex, keep) or bool(lex) and not any(lex.is_known(tok, l) for l in LATIN_LANGS)


def _standin_in(st, src):
    return st if src in LATIN_LANGS or src not in SCRIPT_LANGS else STANDINS[st][src][0]


def protect(text, src, lex, book, keep=()):
    """-> (text with stand-ins in place of names, [(name, stand-in)])."""
    keep_low = {k.lower(): k for k in keep}
    keep = set(keep)
    spans = []
    latin_src = src not in SCRIPT_LANGS or not _mostly_script(text, src)
    for m in _LATIN_TOKEN.finditer(text):
        tok, end = m.group(0), m.end()
        if not _LATIN.search(tok):
            continue
        for poss in ("'s", "’s"):
            if tok.endswith(poss) and len(tok) > 3:
                tok, end = tok[:-2], end - 2
        if tok.lower() in keep_low:
            name = keep_low[tok.lower()]
        elif not latin_src:
            name = tok if _latin_word_is_name(tok, lex, keep) else None
        elif looks_like_name(tok, src, lex, keep) or (bare(tok) in book.spell and not lex.is_known(bare(tok), src)):
            name = tok
        else:  # typed in lowercase: only names seen before
            name = book.lower.get(tok.lower()) if tok.islower() else None
        if name:
            spans.append((m.start(), end, name, ""))
    if not latin_src:
        rx = book.finder(src)
        if rx:
            for m in rx.finditer(text):
                if not any(s < m.end() and m.start() < e for s, e, _, _ in spans):
                    spans.append((m.start(), m.end(), book.back[src][m.group(2)], m.group(1)))
    if not spans:
        return text, []
    low = text.lower()
    free = [st for st in STANDINS if st.lower() not in low and not any(v in text for vs in STANDINS[st].values() for v in vs)]
    slots, out, last = [], [], 0
    for s, e, name, prefix in sorted(spans):
        idx = next((i for i, (n, _) in enumerate(slots) if n == name), None)
        if idx is None:
            if len(slots) == len(free):
                continue  # more names than stand-ins: the rest go to the model as typed
            slots.append((name, free[len(slots)]))
            idx = len(slots) - 1
        out.append(text[last:s])
        out.append(prefix + _standin_in(slots[idx][1], src))
        last = e
    out.append(text[last:])
    return "".join(out), slots


def spelling_jobs(names, targets, book):
    """Frame sentences still needed to write these names in zh/ar/hi -> [(target, name, sentence)]."""
    jobs = []
    for b in dict.fromkeys(bare(n) for n in names):
        for t in targets:
            if t in SCRIPT_LANGS and book.get(b, t) is None:
                jobs += [(t, b, f.format(b)) for f in FRAMES]
    return jobs


def take_spellings(jobs, outs, book):
    """Pick each name's spelling from its frame translations (most votes, then frame order)."""
    votes = {}
    for k, ((t, b, _), out) in enumerate(zip(jobs, outs)):
        v = votes.setdefault((b, t), Counter())
        m = next((m for rx in _CUT[t] for m in [rx.match(out.strip(_STRIP))] if m), None)
        if m:
            cand = m.group(1).strip(_STRIP)
            if cand and len(cand.split()) <= 3 and len(cand) <= 3 * len(b) + 4 and _is_script(cand, t):
                v[cand] += 1 - k * 1e-6  # ties go to the earlier frame
    for (b, t), v in votes.items():
        # No usable frame: keep it as typed rather than drop it.
        book.put(b, t, max(v, key=v.get) if v else b)


def render(name, t, book):
    b = bare(name)
    out = b if t not in SCRIPT_LANGS else book.get(b, t) or b
    return "@" + out if name.startswith("@") else out


_FIND = {}


def _finder(st, t):
    """Regex for a stand-in in a translation into t; brackets or quotes the model puts
    around it are dropped, an Arabic prefix letter (و، ب، ل...) is kept."""
    key = (st, t)
    if key not in _FIND:
        forms = sorted(STANDINS[st].get(t, [st]), key=len, reverse=True)
        v = "|".join(map(re.escape, forms))
        core = rf"(?:\((?:{v})\)|[\"«“](?:{v})[\"»”]|(?:{v}))"
        at = r"(?:[@＠]\s*)?"
        if t == "zh":
            rx = rf"{at}{core}"
        elif t in _BOUND:
            pre = f"(?:[{_AR_PREFIX}]ـ?)?" if t == "ar" else ""
            rx = rf"(?<![{_BOUND[t]}])({pre}){at}{core}(?![{_BOUND[t]}])"
        else:
            rx = rf"(?<![\w@]){at}{core}(?!\w)"
        _FIND[key] = re.compile(rx, re.I if t not in _BOUND and t != "zh" else 0)
    return _FIND[key]


def restore(text, slots, t, book):
    """Put names back in place of their stand-ins -> (text, every name found)."""
    found = 0
    for name, st in slots:
        r = render(name, t, book)
        rx = _finder(st, t)
        if t in _BOUND:
            text, n = rx.subn(lambda m: m.group(1) + r, text)
        else:
            text, n = rx.subn(lambda m: r, text)
        if t == "zh" and n:
            e = re.escape(r)
            cjk = r"[㐀-鿿　-〿＀-￯]"
            text = re.sub(rf"(?<={cjk})\s+(?={e})|(?<={e})\s+(?={cjk})", "", text)
        found += n > 0
    return text, found == len(slots)
