# frostfire-forge-language-service

To install dependencies:

```bash
bun install
```

## Use from the game engine

This is a module, not a server. The game engine has this repository as a git submodule at
`src/language-service` and calls it in its own process:

```ts
import { set_logger, start_translator, translate_text } from "../language-service/src/api";

set_logger({ info: log.info, warn: log.warn, error: log.error }); // optional, defaults to console
start_translator(); // optional: load the model now instead of with the first line

const { translated_text } = await translate_text({ source_language: "en", target_languages: ["es", "fr"], source_text: "where is the mine?" });
```

`translate_text` throws `TranslationRequestError` for a request it refuses (unknown language,
missing or too long text) and a plain `Error` when the translator could not answer. Importing
`src/api.ts` starts nothing. The Python environment below goes inside the submodule folder.

## Translator

Translations come from the offline model in `translator/`, run on the CPU by a Python
child process (`translator/serve.py`). It starts with the first translation (or with
`start_translator()`), loads in a few seconds, and starts again with the next translation
if it stops.

The game asks once per language for each chat line, all at the same time. Requests for the
same line that arrive within a few milliseconds of each other are merged, and each language
runs as its own batch in parallel (6 at once, 2 CPU threads each), so six languages take
about as long as one.

- `translator/model/`: the exported model (from Frostfire Translate's `scripts/export.py`)
- `translator/ffmt/`: a copy of Frostfire Translate's inference code (`ffmt/__init__.py`, `translator.py`, `chat.py`, `langid.py`, `names.py`)

It needs Python 3 in `.venv` in this folder. Create it once:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r translator/requirements.txt   # Linux/macOS: .venv/bin/python
```

To update the model, replace `translator/model/` with Frostfire Translate's `exports/latest/`.

`translate_text({ source_language: "en", target_languages: "es,fr,cn", source_text: "where is the mine?" })`
gives `translated_text` as `{"es": "...", "fr": "...", "cn": "..."}`. Chinese is accepted as `cn` or `zh`.

To type check:

```bash
bun check
```

## Model training data

The model in `translator/model/` was trained from scratch by Frostfire Interactive. Its training data keeps its own licences and credits:

- **OPUS** corpora, each under its own licence (see each corpus page on [opus.nlpl.eu](https://opus.nlpl.eu)).
- **MASSIVE 1.0** (train partition): FitzGerald et al., "MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages", 2022. Amazon, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), https://github.com/alexa/massive.
- **BPCC-H-Daily** (English-Hindi): Gala et al., "IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Languages of India", 2023. AI4Bharat, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), https://huggingface.co/datasets/ai4bharat/BPCC.
- **Game translations** made by the players of open-source games, each under its game's licence and its translators' copyright: [Battle for Wesnoth](https://github.com/wesnoth/wesnoth) (GPL-2.0+), [Shattered Pixel Dungeon](https://github.com/00-Evan/shattered-pixel-dungeon) (GPL-3.0), [OpenTTD](https://github.com/OpenTTD/OpenTTD) (GPL-2.0), [0 A.D.](https://github.com/0ad/0ad) (Wildfire Games, GPL-2.0+), [Mindustry](https://github.com/Anuken/Mindustry) (GPL-3.0), [Luanti](https://github.com/luanti-org/luanti) (LGPL-2.1+), [SuperTuxKart](https://github.com/supertuxkart/stk-code) (GPL-3.0).

## License

Free for noncommercial use under the [PolyForm Noncommercial License 1.0.0](LICENSE). Every commercial use needs a paid [commercial license](COMMERCIAL-LICENSE.md) from Frostfire Interactive. See [LICENSING.md](LICENSING.md) for which one applies to you.

Contributions are welcome and need a signed [Contributor License Agreement](CLA.md); see [CONTRIBUTING.md](CONTRIBUTING.md).