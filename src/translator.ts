// Runs the offline translation model (translator/serve.py) as a child process and talks to
// it over stdin/stdout, one JSON line per request. The model loads once and stays in memory.
// Nothing starts on import: the process starts with the first translation, or with
// start_translator(). If it stops or cannot start, requests in flight are rejected and the
// next translation starts it again, at most once every restart_delay_ms (until then requests
// fail at once). It also exits by itself when the program that started it exits (its stdin closes).
import { join } from "node:path";

const project_dir = join(import.meta.dir, "..");
// translator/ holds serve.py, the ffmt/ inference code and the model/ files.
const translator_dir = join(project_dir, "translator");
// The project's own Python environment (see README), with translator/requirements.txt installed.
const translator_python = process.platform === "win32"
    ? join(project_dir, ".venv", "Scripts", "python.exe")
    : join(project_dir, ".venv", "bin", "python");
const request_timeout_ms = 10_000;
const restart_delay_ms = 5_000;

export type TranslationResult = {
    source: string;
    translations: Record<string, string>;
};

export type Logger = {
    info: (message: string) => void;
    warn: (message: string) => void;
    error: (message: string) => void;
};

type Reply = {
    id?: number | null;
    ready?: boolean;
    source?: string;
    translations?: Record<string, string>;
    error?: string;
};

type Pending = {
    resolve: (result: TranslationResult) => void;
    reject: (error: Error) => void;
    timer: ReturnType<typeof setTimeout>;
};

type Child = ReturnType<typeof spawn_child>;

let logger: Logger = { info: console.log, warn: console.warn, error: console.error };
const pending = new Map<number, Pending>();
let next_id = 1;
let child: Child | null = null;
// The running (or loading) translator; null until something starts it, and again after it stops.
let ready: Promise<void> | null = null;
let stopped_at = -Infinity;

// A program that imports this module (the game engine) passes its own logger, so these
// lines go where its other log lines go.
export function set_logger(next: Logger) {
    logger = next;
}

export function log(): Logger {
    return logger;
}

function spawn_child() {
    return Bun.spawn([translator_python, "-u", "serve.py"], {
        cwd: translator_dir,
        stdin: "pipe",
        stdout: "pipe",
        stderr: "inherit",
    });
}

// Starts the translator unless it is running or loading. Resolves once the model has loaded;
// rejects if the process cannot start or stops first.
export function start_translator(): Promise<void> {
    if (ready) return ready;
    if (performance.now() - stopped_at < restart_delay_ms) {
        return failed("Translator stopped, starting it again in a few seconds");
    }
    let proc: Child;
    try {
        proc = spawn_child();
    } catch (e) {
        stopped_at = performance.now();
        logger.error(`Translator could not start (${translator_python}): ${e}. Set up its Python environment as the language service README says.`);
        return failed(e instanceof Error ? e.message : String(e));
    }
    ready = watch(proc);
    return ready;
}

function failed(message: string): Promise<void> {
    const failure = Promise.reject(new Error(message));
    failure.catch(() => {}); // callers see the error through translate()
    return failure;
}

// Resolves once the model has loaded; rejects if the process stops first.
function watch(proc: Child): Promise<void> {
    child = proc;
    const loading = new Promise<void>((resolve, reject) => {
        let loaded = false;
        read_lines(proc.stdout, (line) => {
            let reply: Reply;
            try {
                reply = JSON.parse(line);
            } catch {
                logger.error(`Translator wrote a line that is not JSON: ${line.slice(0, 200)}`);
                return;
            }
            if (reply.ready) {
                loaded = true;
                logger.info("Translator model loaded");
                resolve();
            } else {
                settle(reply);
            }
        }).catch((e) => logger.error(`Translator output could not be read: ${e}`));
        proc.exited.then((code) => {
            if (child === proc) {
                child = null;
                ready = null;
                stopped_at = performance.now();
            }
            for (const id of [...pending.keys()]) {
                settle({ id, error: `Translator stopped (exit code ${code})` });
            }
            if (!loaded) reject(new Error(`Translator stopped before the model loaded (exit code ${code})`));
            logger.error(`Translator stopped (exit code ${code}); the next translation starts it again`);
        });
    });
    loading.catch(() => {}); // callers see the error through translate()
    return loading;
}

function settle(reply: Reply) {
    if (reply.id == null) {
        logger.error(`Translator error without a request id: ${reply.error}`);
        return;
    }
    const p = pending.get(reply.id);
    if (!p) return;
    pending.delete(reply.id);
    clearTimeout(p.timer);
    if (reply.error) p.reject(new Error(reply.error));
    else p.resolve({ source: reply.source ?? "", translations: reply.translations ?? {} });
}

async function read_lines(stream: ReadableStream<Uint8Array>, on_line: (line: string) => void) {
    const decoder = new TextDecoder();
    let buffer = "";
    for await (const chunk of stream) {
        buffer += decoder.decode(chunk, { stream: true });
        let end: number;
        while ((end = buffer.indexOf("\n")) >= 0) {
            const line = buffer.slice(0, end).trim();
            buffer = buffer.slice(end + 1);
            if (line) on_line(line);
        }
    }
}

// Translate one chat line into each target language. Codes are the translator's
// (en, zh, hi, es, ar, pt, fr). Leave source out to detect the line's language.
export function translate(text: string, targets: string[], source?: string): Promise<TranslationResult> {
    const id = next_id++;
    return new Promise<TranslationResult>((resolve, reject) => {
        // The timeout also covers waiting for the model to load.
        const timer = setTimeout(() => {
            pending.delete(id);
            reject(new Error("Translator timed out"));
        }, request_timeout_ms);
        pending.set(id, { resolve, reject, timer });
        start_translator().then(
            () => {
                if (!pending.has(id)) return; // timed out while the model was loading
                if (!child) return settle({ id, error: "Translator is not running" });
                try {
                    child.stdin.write(JSON.stringify({ id, text, targets, source }) + "\n");
                    child.stdin.flush();
                } catch (e) {
                    settle({ id, error: `Translator could not be asked: ${e}` });
                }
            },
            (e) => settle({ id, error: e instanceof Error ? e.message : String(e) }),
        );
    });
}
