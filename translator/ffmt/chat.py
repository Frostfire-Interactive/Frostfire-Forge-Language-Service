"""Game chat shorthand, handled by rules before the model sees a sentence.

- Words every player knows in any language (gg, lol, afk, xD, laughter, emoji) at the
  start or end of a sentence are kept as typed; a sentence made only of them is not
  translated at all.
- Common abbreviations are written out per source language ("brb" -> "be right back",
  "vc" -> "você") so the model gets words it saw in training.
"""
import re
import unicodedata

# Kept verbatim (compared lowercase, without trailing punctuation).
UNIVERSAL = {
    "gg", "ggwp", "ggs", "wp", "gl", "glhf", "hf", "gz", "grats", "gratz", "afk", "ez", "kek",
    "lol", "lul", "lmao", "lmfao", "rofl", "xd", "xdd", "omg", "wtf", "ok", "okay", "k", "kk",
    "o7", "o/", "\\o", "\\o/", "+1", "-1", "owo", "uwu", "mdr", "ptdr", "jk",
}
_LAUGH = re.compile(
    r"^(?:a?(?:ha|he|hi){2,}h?|(?:ja|je|ji){2,}j?|k{3,}|(?:rs){2,}|(?:hue){2,}|x+d+|l+o+l+|l+m+a+o+"
    r"|哈{2,}|呵{2,}|ه{3,}|ह(?:ा|ाह)+|5{3,})$")
_EMOTICON = re.compile(r"^(?:[:;=8xX][-o^']?[()\[\]dDpP/\\|oO3*]+|<3+|\^+[_.-]?\^+|[tT]_[tT]|-_-|o\.o|O\.O)$")
_TRAIL = ".,!?~…。！？，؟"

ABBREV = {
    "en": {
        "brb": "be right back", "bbl": "be back later", "omw": "on my way", "idk": "I don't know",
        "idc": "I don't care", "imo": "in my opinion", "btw": "by the way", "nvm": "never mind",
        "np": "no problem", "ty": "thank you", "tyvm": "thank you very much", "thx": "thanks",
        "thnx": "thanks", "tnx": "thanks", "pls": "please", "plz": "please", "plox": "please",
        "u": "you", "ur": "your", "r": "are", "y": "why", "ppl": "people", "rn": "right now",
        "tmrw": "tomorrow", "tmr": "tomorrow", "2day": "today", "2nite": "tonight", "b4": "before",
        "sry": "sorry", "srry": "sorry", "mb": "my bad", "gj": "good job", "irl": "in real life",
        "ofc": "of course", "tho": "though", "cuz": "because", "cos": "because", "bc": "because",
        "wanna": "want to", "gonna": "going to", "gotta": "have to", "lemme": "let me", "gimme": "give me",
        "dunno": "don't know", "im": "I'm", "ive": "I've", "dont": "don't", "cant": "can't",
        "wont": "won't", "didnt": "didn't", "doesnt": "doesn't", "isnt": "isn't", "wasnt": "wasn't",
        "arent": "aren't", "havent": "haven't", "shouldnt": "shouldn't", "couldnt": "couldn't",
        "wouldnt": "wouldn't", "thats": "that's", "whats": "what's", "theres": "there's",
        "lf": "looking for", "lfg": "looking for group", "lfm": "looking for more",
        "wtb": "want to buy", "wts": "want to sell", "wtt": "want to trade", "inv": "invite",
        "lvl": "level", "dmg": "damage", "msg": "message", "pm": "message", "dm": "message",
        "gg": "good game", "wb": "welcome back", "smth": "something", "sth": "something",
        "abt": "about", "w/e": "whatever", "bf": "boyfriend", "gf": "girlfriend", "fr": "for real",
        "ngl": "not going to lie", "tbh": "to be honest", "asap": "as soon as possible",
    },
    "es": {
        "q": "que", "k": "que", "ke": "que", "xq": "porque", "pq": "porque", "porq": "porque",
        "tb": "también", "tmb": "también", "tmbn": "también", "d": "de", "x": "por", "pa": "para",
        "toy": "estoy", "bn": "bien", "nd": "nada", "ntp": "no te preocupes", "salu2": "saludos",
        "grax": "gracias", "grs": "gracias", "xfa": "por favor", "porfa": "por favor", "pf": "por favor",
        "tqm": "te quiero mucho", "msj": "mensaje", "dnd": "dónde", "aki": "aquí", "kiero": "quiero",
        "kien": "quien", "xa": "para", "sta": "está", "sty": "estoy", "hla": "hola", "ola": "hola",
        "bno": "bueno", "mñn": "mañana", "mñna": "mañana", "nse": "no sé", "nc": "no sé", "vdd": "verdad",
        "lpm": "la puta madre", "wey": "güey",
    },
    "pt": {
        "vc": "você", "vcs": "vocês", "tb": "também", "tbm": "também", "q": "que", "pq": "porque",
        "n": "não", "ñ": "não", "nao": "não", "ta": "está", "to": "estou", "tô": "estou", "blz": "beleza",
        "vlw": "valeu", "obg": "obrigado", "obgd": "obrigado", "mt": "muito", "mto": "muito",
        "msm": "mesmo", "cmg": "comigo", "dps": "depois", "hj": "hoje", "td": "tudo", "tds": "todos",
        "fds": "fim de semana", "pfv": "por favor", "pf": "por favor", "kd": "cadê", "cd": "cadê",
        "pra": "para", "tmj": "tamo junto", "flw": "falou", "sdds": "saudades", "agr": "agora",
        "qnd": "quando", "qdo": "quando", "oq": "o que", "eh": "é", "neh": "né",
        "tlgd": "tá ligado", "bjs": "beijos", "qm": "quem", "ctz": "certeza", "sla": "sei lá",
    },
    "fr": {
        "slt": "salut", "cc": "coucou", "stp": "s'il te plaît", "svp": "s'il vous plaît",
        "jsp": "je sais pas", "jsuis": "je suis", "chui": "je suis", "chuis": "je suis", "g": "j'ai",
        "c": "c'est", "ct": "c'était", "pk": "pourquoi", "pq": "pourquoi", "pcq": "parce que",
        "pck": "parce que", "bcp": "beaucoup", "mtn": "maintenant", "dac": "d'accord", "dak": "d'accord",
        "tkt": "t'inquiète", "qqn": "quelqu'un", "qqch": "quelque chose", "pr": "pour", "ds": "dans",
        "tt": "tout", "auj": "aujourd'hui", "dsl": "désolé", "bjr": "bonjour", "bsr": "bonsoir",
        "mrc": "merci", "mci": "merci", "jtm": "je t'aime", "vrmt": "vraiment", "jvais": "je vais",
        "koi": "quoi", "ki": "qui", "ke": "que", "kan": "quand", "keske": "qu'est-ce que",
        "jpp": "j'en peux plus", "osef": "on s'en fout", "tjrs": "toujours", "tjs": "toujours",
        "rdv": "rendez-vous", "dc": "donc", "bg": "beau gosse", "frr": "frère", "wsh": "wesh",
    },
}
_ABBREV_RE = {
    lang: re.compile(r"(?<![\w'/])(" + "|".join(re.escape(k) for k in sorted(d, key=len, reverse=True)) + r")(?![\w'/])", re.I)
    for lang, d in ABBREV.items()
}


def _bare(tok):
    return tok.rstrip(_TRAIL).lower()


def is_universal(tok):
    b = _bare(tok)
    if not b:
        return all(c in _TRAIL for c in tok)
    if b in UNIVERSAL or _LAUGH.match(b) or _EMOTICON.match(tok.rstrip(".,!?~")):
        return True
    return all(unicodedata.category(c) in ("So", "Sk") or c in "\u200d\ufe0f" for c in b)  # emoji


def expand(text, lang):
    rx = _ABBREV_RE.get(lang)
    if not rx:
        return text
    table = ABBREV[lang]

    def sub(m):
        w = m.group(0)
        out = table[w.lower()]
        return out[0].upper() + out[1:] if w[0].isupper() and not out[0].isupper() and len(w) > 1 else out
    return rx.sub(sub, text)


def peel(sentence):
    """-> (prefix, core, suffix): universal chat words at the edges, kept as typed, and the
    rest (empty when the whole sentence is universal chat words)."""
    toks = sentence.split()
    i, j = 0, len(toks)
    while i < j and is_universal(toks[i]):
        i += 1
    while j > i and is_universal(toks[j - 1]):
        j -= 1
    return " ".join(toks[:i]), " ".join(toks[i:j]), " ".join(toks[j:])


def prepare(sentence, lang):
    """peel() with the abbreviations in the core written out."""
    pre, core, suf = peel(sentence)
    return pre, expand(core, lang) if core else "", suf
