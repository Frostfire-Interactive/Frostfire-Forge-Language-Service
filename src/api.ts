// The language service. The game engine has this repository as a git submodule and calls
// translate_text in its own process (there is no HTTP server). Importing this starts nothing: the translator process starts with the first translation, or
// earlier with start_translator() (so the model is loaded before the first line comes in).
import { log, translate } from "./translator";

export { set_logger, start_translator, type Logger } from "./translator";

export const approved_languages = ["en", "fr", "es", "cn", "zh", "hi", "ar", "pt"];
// The translator calls Chinese "zh"; "cn" is accepted as well
const model_language: Record<string, string> = { cn: "zh" };
export const max_text_length = 500;

export type TranslationRequest = {
    // The language of the source text
    source_language: string;
    // The languages to translate into: "es", "es,fr" or ["es", "fr"]
    target_languages: string | string[];
    // The text to be translated
    source_text: string;
};

export type TranslationResponse = {
    source_language: string;
    target_languages: string[];
    source_text: string;
    // Text after it's translated, keyed by the requested target language: { es: "Hola." }
    translated_text: Record<string, string>;
    translation_time: string;
};

// A request the service refuses (unknown language, missing or too long text); any other error
// means the translator could not answer.
export class TranslationRequestError extends Error {
    override name = "TranslationRequestError";
}

export async function translate_text(request: TranslationRequest): Promise<TranslationResponse> {
    // Clean the language input to ensure it only contains alphabetic characters
    const source_language = clean_input(request.source_language || "");

    const target_list = Array.isArray(request.target_languages) ? request.target_languages : [request.target_languages || ""];
    const target_languages = target_list.join(",").split(",").map(clean_input);

    // Remove duplicate target languages and remove source language from target languages
    const unique_target_languages = [...new Set(target_languages)].filter((language) => language !== source_language);
    // Warn when duplicate target languages are removed
    if (unique_target_languages.length < target_languages.length) {
        log().warn(`Duplicate target languages removed: ${target_languages.filter((language, index) => target_languages.indexOf(language) !== index).join(", ")}`);
    }

    // source text before it's translated
    const source_text = request.source_text || "";

    // Not an authorized source language
    if (!source_language || !approved_languages.includes(source_language)) {
        throw new TranslationRequestError("Invalid language");
    }

    // Not an authorized target language
    if (!unique_target_languages.length || !unique_target_languages.every((language) => approved_languages.includes(language))) {
        throw new TranslationRequestError("Invalid target language");
    }

    // Missing source text
    if (!source_text.trim()) {
        throw new TranslationRequestError("Missing text");
    }

    // source text too long
    if (source_text.length > max_text_length) {
        throw new TranslationRequestError("Text too long");
    }

    const to_model = (language: string) => model_language[language] || language;
    const start_time = performance.now();
    const result = await translate(source_text, unique_target_languages.map(to_model), to_model(source_language));
    const translated_text = Object.fromEntries(
        unique_target_languages.map((language) => [language, result.translations[to_model(language)] ?? ""])
    );
    const translation_time = `${(performance.now() - start_time).toPrecision(3)}ms`;

    return {
        source_language,
        target_languages: unique_target_languages,
        source_text,
        translated_text,
        translation_time
    };
}

// Remove all non-alphabetic characters
function clean_input(input: string) {
    return input.replace(/[^a-zA-Z]/g, "");
}
