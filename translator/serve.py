"""Translator for other programs: one JSON request per stdin line, one JSON reply per stdout line.

Loads the offline model in model/ (exported by Frostfire Translate's scripts/export.py)
once and keeps it in memory. src/translator.ts starts this script as a child process.
ffmt/ is a copy of the inference modules from Frostfire Translate.

  request  {"id": 1, "text": "where is the blacksmith?", "targets": ["es", "fr"], "source": "en"}
           "source" is optional (detected when left out); "names" (list) is optional too.
  reply    {"id": 1, "source": "en", "translations": {"es": "...", "fr": "..."}}
           or {"id": 1, "error": "..."}
Requests that arrive together are answered together: the game asks once per language for
the same chat line, so requests for the same line are merged into one call, and each target
language runs as its own batch, all at the same time. Replies can come out of order (match by id).
The first line written is {"ready": true}, once the model has loaded.
Usage: python translator/serve.py [--model translator/model] [--threads 2] [--parallel 6]
"""
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ffmt.translator import LANGS, ChatTranslator  # noqa: E402

MAX_CHARS = 500
GAP_S = 0.002  # a request this soon after the previous one joins its group
WAIT_S = 0.005  # a group waits at most this long after its first request


def parse(req):
    """-> (text, targets, source, names); raises for a bad request."""
    text = str(req.get("text") or "").strip()[:MAX_CHARS]
    targets = [t for t in req.get("targets") or [] if t in LANGS]
    source = req.get("source")
    if source is not None and source not in LANGS:
        raise ValueError(f"unknown source language {source!r}")
    names = tuple(str(n) for n in req.get("names") or [])
    return text, targets, source, names


def read_requests(q):
    for line in sys.stdin:
        if line.strip():
            q.put(line)
    q.put(None)


def take_group(q):
    """The next request line and the ones that arrive right after it; None at the end of input."""
    first = q.get()
    if first is None:
        return None
    lines, end = [first], time.perf_counter() + WAIT_S
    while (left := end - time.perf_counter()) > 0:
        try:
            line = q.get(timeout=min(GAP_S, left))
        except queue.Empty:
            break
        if line is None:
            q.put(None)  # answer this group, then stop
            break
        lines.append(line)
    return lines


def answer(tr, lines):
    """One reply per request line; requests for the same line share one translate_for call."""
    replies, merged = [], {}
    for line in lines:
        reply = {"id": None}
        replies.append(reply)
        try:
            req = json.loads(line)
            reply["id"] = req.get("id")
            text, targets, source, names = parse(req)
        except Exception as e:  # noqa: BLE001 - one bad request must not stop the server
            reply["error"] = str(e)[:300]
            continue
        if not text or not targets:
            reply.update(source=source, translations={t: text for t in targets})
            continue
        merged.setdefault((text, source, names), []).append((reply, targets))
    for (text, source, names), asks in merged.items():
        targets = list(dict.fromkeys(t for _, ts in asks for t in ts))
        try:
            src, out = tr.translate_for(text, targets, source, list(names))
            for reply, ts in asks:
                reply.update(source=src, translations={t: out[t] for t in ts})
        except Exception as e:  # noqa: BLE001
            for reply, _ in asks:
                reply["error"] = str(e)[:300]
    return replies


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "model"))
    p.add_argument("--threads", type=int, default=2, help="CPU threads per language batch")
    p.add_argument("--parallel", type=int, default=6, help="language batches at the same time")
    args = p.parse_args()
    # Pipes on Windows default to the ANSI code page; requests and replies are UTF-8.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    tr = ChatTranslator(args.model, threads=args.threads, parallel=args.parallel)
    q = queue.Queue()
    threading.Thread(target=read_requests, args=(q,), daemon=True).start()
    print(json.dumps({"ready": True}), flush=True)
    while (lines := take_group(q)) is not None:
        for reply in answer(tr, lines):
            print(json.dumps(reply, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
