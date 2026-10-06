"""The generated long texts' one home: the deterministic generators the cases lanes used.

A case's ``text_ref`` names one of these generators with its parameters and the SHA-256 of the text it
produces; :func:`materialize` rebuilds the text (stdlib only -- ``random`` and the fixed word lists
here, no tokenizer and no network) and refuses a hash mismatch, so a mutated parameter, a changed
generator or a drifted wordlist fails the load instead of silently testing different bytes.

One generator per construction, each versioned (the ref names ``<name>@<version>``); a construction
change is a new version. Every generator was verified byte-for-byte against the texts the cases lanes
committed before this module existed (the conversion script asserted materialize == stored literal for
every converted case).

The generators are deliberately *word-level*: the lanes sized their texts against the recipes'
tokenizers, but reproducing a token-measured trim needs the tokenizer files, so the parameters record
the deterministic construction's word-level outcome (a stream offset, a word count, a repetition
count) and the hash pins the exact bytes. Re-deriving a new text for a new case belongs to the
lanes' build scripts (which run the product's fit); this module only materializes recorded ones.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from typing import Any

from .errors import CaseError

__all__ = ["GENERATORS", "materialize"]


def _sha256(text: str) -> str:
    """The SHA-256 of the text's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# qwen3-reranker-0.6b/4b/8b: sentence pools padded with single-token filler words
# ---------------------------------------------------------------------------

_RER_SENTENCES: dict[str, str] = {
    "A": (
        "The Bramfield harbour ledger records that the tide gauge was recalibrated on 14 October "
        "1873 by the assistant keeper."
    ),
    "B": ("The miller of Bramfield kept his grain accounts in a separate book bound in grey cardboard."),
    "C": ("Winter storms occasionally shifted the channel markers, and the pilots noted each change in the margin."),
    "BC": "",
    "BB": "",
    "CCC": "",
}
_RER_SENTENCES["BC"] = f"{_RER_SENTENCES['B']} {_RER_SENTENCES['C']}"
_RER_SENTENCES["BB"] = f"{_RER_SENTENCES['B']} {_RER_SENTENCES['B']}"
_RER_SENTENCES["CCC"] = " ".join([_RER_SENTENCES["C"]] * 3)
_RER_PADS = [
    "harbour",
    "ledger",
    "keeper",
    "gauge",
    "winter",
    "channel",
    "margin",
    "storm",
    "tide",
    "pilot",
    "accounts",
    "book",
    "grain",
    "paper",
    "wind",
    "rain",
]


def qwen3_rer_filler(params: dict[str, Any]) -> str:
    """The qwen3-reranker lanes' document: a named sentence pool repeated ``repeats`` times, then
    padded with single-token filler words (the recorded pad list, verbatim).

    Params: ``pool`` (``A``, ``B``, ``C``, ``BB``, ``BC`` or ``CCC`` -- the pool's sentences joined),
    ``repeats`` (how many times the pool's block repeats) and ``pads`` (the trailing filler words).
    The query texts are the lanes' own fixed sentences (never generated).
    """
    pool = str(params["pool"])
    repeats = int(params["repeats"])
    pads = [str(word) for word in params.get("pads", [])]
    block = _RER_SENTENCES[pool]
    parts = [block] * repeats + pads
    return " ".join(parts).rstrip() if parts and not pads else " ".join(parts).strip()


def qwen3_emb_corpus_text(params: dict[str, Any]) -> str:
    """One input of the qwen3-embedding-0.6b stream: the corpus's word slice at ``offset``, ``words``
    long (the lanes' builder trims word-by-word to the recorded token target; the slice of kept words
    is a prefix of the remaining stream, so the offset and the word count pin the text exactly)."""
    offset = int(params["offset"])
    count = int(params["words"])
    vocabulary = (
        "harbour lantern merchant anchorage tide current compass latitude longitude cargo "
        "expedition cartographer isthmus monsoon monsoon vessel provisioning latitude reef "
        "soundings azimuth sextant astrolabe keelBallast bowsprit mizzen topsail windward "
        "leeward fathom nautical almanac observatory chronometer"
    ).split()
    vocabulary = sorted({word.lower() for word in vocabulary})
    rng = random.Random(20251006)
    sentences = []
    for _ in range(6000):
        sentence_count = rng.randint(8, 14)
        sentence_words = [rng.choice(vocabulary) for _ in range(sentence_count)]
        sentences.append(" ".join(sentence_words).capitalize() + ".")
    stream = " ".join(sentences).split()
    return " ".join(stream[offset : offset + count])


# ---------------------------------------------------------------------------
# zembed-1-embedding: "Entry NNNNN of the ledger." blocks, two fixed sentences per block in rotation
# ---------------------------------------------------------------------------

_ZEMBED_SENTENCES = [
    "The harbour registrar recorded the arrival of forty crates of grain on the fourth of June.",
    "A quiet canal crossed the market district, carrying barges loaded with timber and slate.",
    "Every apprentice learned to mend the brass fittings before touching the clockwork itself.",
    "The committee agreed that the petition would be read aloud at the following assembly.",
    "Rain swept the valley overnight, and the surveyors delayed their climb until noon.",
    "Merchants traded salt, wool and dyestuffs along the old road beyond the northern ridge.",
    "The letter described a bridge of stone and timber spanning the river at its narrowest point.",
]


def zembed_ledger(params: dict[str, Any]) -> str:
    """The zembed lane's long text: ``blocks`` paragraphs, each ``Entry {i:05d} of the ledger.`` plus
    two of the seven fixed sentences in rotation, joined by newlines. Params: ``blocks``."""
    blocks = int(params["blocks"])
    return "\n".join(
        f"Entry {index:05d} of the ledger. {_ZEMBED_SENTENCES[index % 7]} {_ZEMBED_SENTENCES[(index + 3) % 7]}"
        for index in range(blocks)
    )


# ---------------------------------------------------------------------------
# topk-embed-v1-small: a template-bank sentence stream (random.Random(seed)), the first N words
# ---------------------------------------------------------------------------

_TOPK_ADJECTIVES = (
    "quarterly",
    "annual",
    "regional",
    "technical",
    "internal",
    "preliminary",
    "final",
    "operational",
    "financial",
    "independent",
    "statutory",
    "structural",
    "comparative",
    "environmental",
    "seasonal",
    "regional",
)


# The topk template bank, verbatim from the lane's committed generator (the one home for the bank;
# the case directory's make_long_inputs.py keeps its own build script for re-running the lane).
ADJECTIVES = (
    "quarterly",
    "annual",
    "regional",
    "technical",
    "internal",
    "preliminary",
    "final",
    "operational",
    "financial",
    "environmental",
    "structural",
    "seasonal",
    "statutory",
    "independent",
    "comparative",
    "consolidated",
)
NOUNS = (
    "report",
    "survey",
    "audit",
    "invoice",
    "contract",
    "estimate",
    "schedule",
    "manifest",
    "ledger",
    "archive",
    "catalogue",
    "inventory",
    "register",
    "abstract",
    "summary",
    "index",
)
VERBS = (
    "describes",
    "summarises",
    "documents",
    "records",
    "lists",
    "reviews",
    "examines",
    "measures",
    "compares",
    "tracks",
    "confirms",
    "revises",
    "estimates",
    "collects",
)
OBJECTS = (
    "the maintenance backlog",
    "regional revenue",
    "the inspection findings",
    "energy usage",
    "the shipping schedule",
    "vendor payments",
    "the staffing plan",
    "meter readings",
    "the training hours",
    "insurance claims",
    "warehouse capacity",
    "the repair costs",
    "customer complaints",
    "delivery delays",
    "the inspection budget",
    "fuel consumption",
)
QUALIFIERS = (
    "for the second half of the year",
    "across all field offices",
    "since the last audit",
    "during the winter period",
    "under the current framework",
    "for the northern region",
    "after the merger closed",
    "before the deadline passed",
    "since the policy changed",
    "over the trailing twelve months",
    "between March and June",
    "in the pilot programme",
)
NUMBERS = tuple(str(n) for n in (3, 7, 12, 18, 24, 39, 46, 58, 63, 71, 84, 92, 105, 118, 240, 512))
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

TEMPLATES = (
    "The {adj} {noun} {verb} {obj} {qual}.",
    "In {month}, the {noun} {verb} {obj} and noted a change of {num} percent {qual}.",
    "{Art} {adj} {noun} {verb} {obj}; the {adj2} figures appear in section {num}.",
    "The {noun} for {month} {verb} {obj}, and the {adj} totals differ by {num} percent.",
    "Appendix {num} {verb} {obj} {qual}, following the {adj} {noun}.",
    "Every {noun} in the {adj} series {verb} {obj} within {num} days {qual}.",
    "The {adj} {noun} was revised after {month}; the {noun} now {verb} {obj}.",
    "Each chart in the {noun} {verb} {obj} across {num} sites {qual}.",
    "The {adj} review {verb} {obj}, which the {month} {noun} had understated by {num} percent.",
    "The field team {verb} {obj} each {month}, and the {adj} {noun} reconciles the totals.",
    "The {noun} {verb} {obj} for {num} consecutive quarters {qual}.",
    "{Art} {adj} {noun} {verb} {obj} whenever the {adj2} threshold of {num} units is crossed.",
)


def _topk_sentence(rng: random.Random, index: int) -> str:
    """One sentence from the template bank; the index pins the sentence order (the lane's pattern)."""
    template = TEMPLATES[index % len(TEMPLATES)]
    adj = rng.choice(ADJECTIVES)
    return template.format(
        adj=adj,
        adj2=rng.choice(ADJECTIVES),
        Art="An" if adj[0].lower() in "aeiou" else "A",
        noun=rng.choice(NOUNS),
        verb=rng.choice(VERBS),
        obj=rng.choice(OBJECTS),
        qual=rng.choice(QUALIFIERS),
        num=rng.choice(NUMBERS),
        month=rng.choice(MONTHS),
    )


def topk_template(params: dict[str, Any]) -> str:
    """The topk lane's text: Random(seed) draws over the template bank, one sentence per call; the
    text is the stream's first ``words`` words. Params: ``seed`` and ``words``."""
    seed = int(params["seed"])
    count = int(params["words"])
    rng = random.Random(seed)
    words: list[str] = []
    index = 0
    while len(words) < count:
        words.extend(_topk_sentence(rng, index).split(" "))
        index += 1
    return " ".join(words[:count])


# ---------------------------------------------------------------------------
# qwen3-vl-embedding-2b: Random(seed) over a fixed word list, every 13th word capitalized + "."
# ---------------------------------------------------------------------------

_QWEN3VL_WORDS = (
    "time year people way day man thing woman life child world school state family "
    "student group country problem hand part place case week company system program "
    "question work government number night point home water room mother area money "
    "story fact month lot right study book eye job word business issue side kind "
    "head house service friend father power hour game line end member law car city "
    "community name president team minute idea kid body information back parent "
    "face others level office door health person art war history party result "
    "change morning reason research girl guy moment air teacher force education"
).split()


def qwen3vl_wordlist(params: dict[str, Any]) -> str:
    """The qwen3-vl-embedding lane's text: Random(seed) draws from the fixed word list, every 13th
    word capitalized and closed with a full stop. Params: ``seed`` and ``words``."""
    seed = int(params["seed"])
    count = int(params["words"])
    rng = random.Random(seed)
    words: list[str] = []
    while len(words) < count:
        word = rng.choice(_QWEN3VL_WORDS)
        if len(words) % 13 == 12:
            word = word.capitalize() + "."
        words.append(word)
    return " ".join(words)


# ---------------------------------------------------------------------------
# qwen3-vl-reranker-2b: a fixed word list shuffled once, cycled, behind a fixed intro text
# ---------------------------------------------------------------------------

_QWEN3VLR_WORDS = """the of and to in is was were a an for with on at by from up about into over after
beneath through between during before under above near across against along among around behind
below beyond plus except toward within without solar panel sunlight electricity photon silicon
semiconductor current inverter voltage battery energy turbine generator tidal ocean tide station
lighthouse coast rocky dusk beacon keeper lamp freighter glacier crevasse rope crampon harness
avalanche compass altitude weather rescue maintenance log inspector filament lens rotation
season axis tilt hemisphere equinox solstice atmosphere particle aurora oxygen nitrogen emission
green violet curtain polar magnetism vaccine immune antibody memory response dose trial virus
spectrum telescope galaxy nebula gravity orbit probe rocket engine combustion propellant alloy
concrete bridge tension compression mortar steel column arch beam foundation sediment erosion
basalt granite mineral crystal geology sedimentary igneous metamorphic fossil stratum
harvest wheat orchard ferment vineyard barley mill kiln brew distill filter copper kettle valve
pump pipeline reservoir aqueduct irrigation drainage canal levee dredge estuary marsh wetland
canopy forest understory mycorrhizae spore moss lichen fern conifer deciduous bark cambium
harbour vessel anchor rigging mast sail compass current nebula rite copper kettle
meadow canyon plateau reef dune fjord tundra savanna delta geyser basin summit valley""".split()
_QWEN3VLR_WORDS = list(dict.fromkeys(_QWEN3VLR_WORDS))

_QWEN3VLR_INTROS: dict[str, str] = {
    "seasons": (
        "The seasons on Earth are caused by the tilt of the planet's rotation axis relative to its "
        "orbital plane. As Earth travels around the sun, each hemisphere spends part of the year "
        "tilted toward the sun, receiving longer days and more direct sunlight, and part of the year "
        "tilted away, with shorter days and lower sun angles."
    ),
    "vaccine": (
        "Vaccines train the immune system by presenting a harmless version or fragment of a pathogen "
        "so that the body builds memory cells without suffering the disease. When the real pathogen "
        "later appears, those memory cells respond faster and more strongly than a naive immune "
        "system could."
    ),
    "aurora_short": (
        "The northern lights appear when charged particles from the sun collide with gases in Earth's "
        "atmosphere, producing glowing curtains of green, red and violet light near the poles."
    ),
    "aurora_mid": (
        "Charged particles from the solar wind are guided by Earth's magnetic field toward the polar "
        "regions. At altitudes between roughly one hundred and three hundred kilometres they excite "
        "oxygen and nitrogen atoms and molecules; the excited gases release the energy as light, with "
        "green from atomic oxygen at lower altitudes, red from oxygen higher up, and blue and violet "
        "from nitrogen. The shape and movement of the displays trace the field lines and the varying "
        "solar wind, which is why auroral activity follows the eleven-year solar cycle and "
        "intensifies during geomagnetic storms."
    ),
    "maintenance": (
        "The keeper's maintenance log for the lighthouse covers lamp replacement, lens cleaning, "
        "paint and rust treatment of the tower, battery bank checks for the light and fog signal, and "
        "the quarterly inspection of the mooring cleats and the winch used for supply landings."
    ),
}


def qwen3vl_rer_filler(params: dict[str, Any]) -> str:
    """The qwen3-vl-reranker lane's text: the named intro text, then the first ``filler_words`` of
    the fixed word list shuffled once with Random(seed), cycled. Params: ``seed``, ``intro`` and
    ``filler_words``."""
    seed = int(params["seed"])
    intro_id = str(params["intro"])
    if intro_id not in _QWEN3VLR_INTROS:
        raise CaseError(f"generator qwen3vl_rer_filler: unknown intro {intro_id!r}")
    filler_count = int(params["filler_words"])
    rng = random.Random(seed)
    order = list(_QWEN3VLR_WORDS)
    rng.shuffle(order)
    filler = " ".join(order[index % len(order)] for index in range(filler_count))
    return (_QWEN3VLR_INTROS[intro_id] + " " + filler).strip()


GENERATORS: dict[str, Callable[[dict[str, Any]], str]] = {
    "qwen3_rer_filler": qwen3_rer_filler,
    "qwen3_emb_corpus": qwen3_emb_corpus_text,
    "zembed_ledger": zembed_ledger,
    "topk_template": topk_template,
    "qwen3vl_wordlist": qwen3vl_wordlist,
    "qwen3vl_rer_filler": qwen3vl_rer_filler,
}

GENERATOR_VERSION = 1
"""The version stamped in every ``text_ref`` produced by this module; a construction change bumps it."""


def materialize(generator: str, params: dict[str, Any]) -> str:
    """The text a ``text_ref``'s generator and params produce.

    Inputs: ``generator`` as ``<name>@<version>`` (``1`` today) and the params mapping. Outputs: the
    text. The caller (the case loader) verifies it against the ref's ``sha256``.

    Raises:
        CaseError: the generator name or version is unknown, a param is missing or malformed, or the
            generator raises.
    """
    name, _, version = generator.partition("@")
    if version != str(GENERATOR_VERSION):
        raise CaseError(
            f"text_ref names generator {generator!r}, but this package implements version "
            f"{GENERATOR_VERSION}: re-record the text (the generator changed)"
        )
    if name not in GENERATORS:
        raise CaseError(
            f"text_ref names generator {name!r}, which the package does not implement "
            f"(implemented: {', '.join(sorted(GENERATORS))})"
        )
    try:
        return GENERATORS[name](params)
    except CaseError:
        raise
    except Exception as error:
        raise CaseError(f"generator {generator!r} failed on {params!r}: {error}") from error
