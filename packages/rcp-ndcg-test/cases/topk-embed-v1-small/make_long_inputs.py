#!/usr/bin/env python3
"""Deterministic generator for the ``generated`` cases of ``topk-embed-v1-small``.

Re-running this script rewrites the fifteen ``generated`` case files byte-identically
(Python's ``random.Random`` with fixed seeds is stable across runs and versions). The
``model_card`` case is hand-written against the card's README and is never touched here.

Usage:
    python make_long_inputs.py --tokenizer-json <pinned tokenizer.json> [--out DIR]

The tokenizer is the recipe's, at its pinned revision:
``topk-io/topk-embed-v1-small @ e54485ebab921f2c18c4d092b3f4c40dcca26781``,
``tokenizer.json`` sha256 ``e56427d66f44411c2dec1288b236f6d2c3eeafd611d1d0e2e92ad9301616e1e7``.
That file embeds a 1024-token right truncation (the recipe's gap G5), which would silently cap
every count; the generator resets it before counting so the recorded lengths are true rendered
prompt lengths -- the same reading transformers uses per call (it resets the embedded strategy on
every encode). No network at generation time: pass the pinned tokenizer.json explicitly. This
script is case provenance, never run by the tests.

Construction (recorded per case in ``notes``): sentences are drawn from the template bank below
with ``random.Random(seed)``; a document is grown sentence by sentence until its rendered prompt
("Document: " + text, the 3-token head included) reaches the target token count, then the last
sentence is trimmed word by word to approach the target from the required side (under: never
exceed; over: stay above). Lengths and keep-mask counts are measured, not asserted.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

REVISION = "e54485ebab921f2c18c4d092b3f4c40dcca26781"
TOKENIZER_SHA256 = "e56427d66f44411c2dec1288b236f6d2c3eeafd611d1d0e2e92ad9301616e1e7"
QUERY_PREFIX = "Query: "
DOCUMENT_PREFIX = "Document: "

MAX_TOKENS = 8192  # the recipe's document-side budget (the whole rendered prompt)
QUERY_CAP = 1024  # the model's own query cap (sentence_bert_config.json; recipe gap G2)

RECIPE = "topk-embed-v1-small"
TOLERANCE = {"abs": 0.02}
MEDIA = "media"

# --- the template bank ---------------------------------------------------------------

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

# Hand-written short documents and queries: natural prose, distinct retrieval targets.
SHORT_DOCS = {
    "short_a": "The quarterly report covers revenue, headcount, and regional growth for the second half of the year.",
    "short_b": "Encryption keys are rotated every ninety days under the current security policy.",
    "short_c": "The ferry crossing takes forty minutes in calm weather and roughly twice as long in a storm.",
    "short_d": "Photosynthesis converts carbon dioxide and water into glucose using energy absorbed from sunlight.",
    "short_e": "New employees complete safety training during their first week and refresher training every year.",
    "short_f": "The library closes early on public holidays, and the reading rooms reopen the following morning.",
    "short_g": "Cliff swallows nest under the bridge from April to September and raise two broods each season.",
    "short_h": "The warranty covers manufacturing defects for two years but excludes damage from improper storage.",
}
SHORT_QUERIES = {
    "rot": "How often are encryption keys rotated?",
    "rev": "What drove the regional revenue growth?",
    "fer": "How long does the ferry crossing take in calm weather?",
    "safe": "When do new employees receive safety training?",
}

# Punctuation-dense documents: prose whose punctuation renders as bare single-character
# tokens (word-attached commas and periods, abbreviations like "Fig."/"p."/"i.e.", "(a)",
# "%", "!", ":", "/"), which the checkpoint's scoring_skip_ids list drops on the document side
# (recipe gap G1; the query side keeps every token -- the asymmetry these cases pin).
PUNCT_DOCS = {
    "punct_a": (
        "Fig. 4.2 (a), on p. 10; cf. annex, ch. 3: the audits, i.e. Q1, Q2, and Q3, ran late, and "
        "costs rose 20% in 2024. Fees, page 33; claims no. 412, no. 587: unpaid! "
        "Meters failed 3 times (Nov., Dec., Jan.); the meter registry, v1.2.3, lists each fault. "
        "Next review: July; contact: records@example.com."
    ),
    "punct_b": (
        "Annex D (metering): readings for Q1, Q2, Q3, and Q4 arrived late, i.e. after the 15th; "
        "the penalty (clause 7.2) applies at 2% per week! Vendors must re-file forms no. 91, no. 92; "
        'terms "net 30" apply. Escalation: ops@example.org - subject: meters/late, urgent!'
    ),
}
PUNCT_QUERIES = {
    "punct_a": 'What were the audit "findings" in 2024?',
    "punct_b": "Which forms must vendors re-file?",
}

# --- the seeded long-text generator ---------------------------------------------------


def _sentence(rng: random.Random, index: int) -> str:
    """One sentence from the template bank; the index pins the sentence order."""
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


def _build_text(tokenizer, target: int, seed: int, side: str, prefix: str = DOCUMENT_PREFIX) -> str:
    """Grow a text whose rendered prompt reaches `target` tokens, approached from `side`.

    `under`: the final count never exceeds `target`; `over`: it stays above it. The last
    sentence is trimmed word by word so the landing point sits within a few tokens of `target`.
    """
    if side not in ("under", "over"):
        raise ValueError(side)
    rng = random.Random(seed)
    words: list[str] = []
    sentence_index = 0

    def rendered() -> int:
        return len(tokenizer.encode(prefix + " ".join(words), add_special_tokens=True).ids)

    if side == "under":
        while rendered() < target:
            words.extend(_sentence(rng, sentence_index).split(" "))
            sentence_index += 1
        _trim_to_target(tokenizer, prefix, words, target, "under")
        if rendered() > target:  # a single word cannot be split; the note records the truth
            raise RuntimeError("cannot land under target")
    else:
        while rendered() <= target:
            words.extend(_sentence(rng, sentence_index).split(" "))
            sentence_index += 1
        _trim_to_target(tokenizer, prefix, words, target, "over")
    return " ".join(words)


def _trim_to_target(tokenizer, prefix: str, words: list[str], target: int, side: str) -> None:
    """Pop trailing words until the prompt lands on `target`'s required side.

    under: pop while the prompt still exceeds the target (lands at or under it). over: pop
    while the prompt without the last word would still exceed it (lands just above it).
    """
    if side == "under":
        while len(words) > 1 and _count(tokenizer, prefix, " ".join(words)) > target:
            words.pop()
    else:
        while len(words) > 1 and _count(tokenizer, prefix, " ".join(words[:-1])) > target:
            words.pop()


def _count(tokenizer, prefix: str, text: str) -> int:
    """Rendered prompt tokens (the head included); the tokenizer's embedded cap must be reset."""
    return len(tokenizer.encode(prefix + text, add_special_tokens=True).ids)


# --- measurement and case assembly ----------------------------------------------------


def measure(tokenizer, text: str, prefix: str, skip: frozenset[int]) -> dict:
    """Rendered prompt token count and, for documents, the post-skip-mask kept count."""
    ids = tokenizer.encode(prefix + text, add_special_tokens=True).ids
    out = {"tokens": len(ids), "kept": None}
    if prefix is DOCUMENT_PREFIX:
        out["kept"] = sum(1 for t in ids if t not in skip)
    return out


def expected_block(kind: str) -> dict:
    """The pending expected block every generated case starts with (case-format.md)."""
    return {
        "kind": kind,
        "values": None,
        "tolerance": None if kind == "none" else dict(TOLERANCE),
        "origin": "reference",
        "status": "pending_gpu",
    }


def doc_line(tag: str, seed: int | None, target: int | None, m: dict) -> str:
    """One measured-length record; `kept` is the post-skip-mask vector count (documents)."""
    origin = f" (seed {seed}, target {target})" if seed is not None else ""
    line = f"{tag}{origin}: rendered prompt {m['tokens']} tokens"
    if m["kept"] is not None:
        line += f", {m['kept']} kept after the document-side scoring_skip_ids mask"
    return line + "."


RENDER_NOTE = (
    "Generated by make_long_inputs.py (alongside; template bank + random.Random(seed), "
    "word-for-word trim onto the target). Lengths measured with tokenizer.json @ "
    + REVISION
    + " (sha256 "
    + TOKENIZER_SHA256
    + "), the embedded 1024-token truncation reset before "
    'counting (recipe gap G5). Text prompts are raw, not chat: "Query: " + text.strip() and '
    '"Document: " + text, whole render stripped. The fixed head encodes to 3 tokens '
    "standalone (with its trailing space); in a rendered prompt the space merges into the first "
    "content token, so the head contributes 2 there. Queries keep every token, documents drop "
    "scoring_skip_ids positions (32 standalone-punctuation ids "
    "+ 9 specials at the revision)."
)
TOLERANCE_NOTE = (
    "expected.values stays null until the GPU wave fills it from the reference implementation "
    "and the engine (origin reference, status pending_gpu). tolerance abs 0.02 is the initial "
    "gate: the research R1/R6 equivalence bound (about 1e-2 at fp16 vector storage) with a 2x "
    "margin, for bf16 serving, fp16 transfer (recipe embed_dtype float16) and fp32 MaxSim; the "
    "wave must restate it against the measured fp32-storage agreement (R6)."
)
LONG_UNDER_NOTE = (
    "long_under: the rendered prompt sits within 5% under the 8192-token budget, so no cut "
    "fires and every anchor (all-token pooling; the 3-token head plus the content) survives."
)
LONG_OVER_NOTE_HEAD = (
    "long_over: the client must cut to the 8192-token budget (on_overflow cut; the content "
    'span only, the 3-token "Document: " head reserved and re-attached). Caveats the wave '
    "must check: under today's G5 counting ceiling the client sees 1024 tokens, so an uncut "
    "prompt reaches the engine --"
)
LONG_OVER_NOTE_TAIL = "-- until the product resets the embedded truncation when counting."


def long_over_note(rendered: dict[str, int]) -> str:
    """The long_over caveat with this case's measured engine behaviour filled in.

    `rendered` maps document id to its measured rendered prompt token count. Documents in
    (8192, 8448] still serve whole under the G5 ceiling (count mismatch, no 400); documents over
    8448 exceed max_model_len and are answered 400 (loud) unless the client cut fires.
    """
    parts = []
    within = [f"{k} ({v} tokens)" for k, v in sorted(rendered.items()) if 8192 < v <= 8448]
    above = [f"{k} ({v} tokens)" for k, v in sorted(rendered.items()) if v > 8448]
    if within:
        s = "s" if len(within) == 1 else ""
        parts.append(
            f"{' and '.join(within)} land{s} in (8192, 8448] and still serve{s} whole (count mismatch, no 400)"
        )
    else:
        parts.append("no document lands in (8192, 8448], the band that still serves whole under the bug")
    if above:
        parts.append(
            f"{' and '.join(above)} exceed{'s' if len(above) == 1 else ''} max_model_len 8448 and "
            "the engine answers 400 (loud)"
        )
    else:
        parts.append("no document exceeds max_model_len 8448")
    return f"{LONG_OVER_NOTE_HEAD} {', and '.join(parts)} {LONG_OVER_NOTE_TAIL}"


UNIFORM_NOTE = (
    "uniform batch: every document of the case sits in one length stratum (uniform does not mean "
    "identical lengths; mixed_length marks cases that straddle strata)."
)
PUNCT_NOTE = (
    "The document-side keep-mask is load-bearing here: the recorded kept counts are the vector "
    "counts the reference produces (document tokens whose id is not in scoring_skip_ids; recipe "
    "gap G1). Queries keep punctuation tokens (the reference's asymmetry, R10). Until the "
    "recipe's keep_mask field lands, a served document vector set is a superset of the "
    "reference's and the wave's per-token count check fails on this case by construction."
)
QUERY_CAP_NOTE = (
    f"The query renders to more tokens than the model's own query cap ({QUERY_CAP}, "
    "sentence_bert_config.json) while staying far under the 8192 budget: the reference "
    "right-truncates the query at 1024, but the recipe schema has no query-cap field (recipe "
    "gap G2), so the served query is uncut. expected.kind none: the case observes the G2 "
    "divergence (query vector counts served vs reference) and is not a score comparison. The "
    "wave must confirm the divergence and that a query of this length does not 400."
)
IMAGE_RENDER_NOTE = (
    "Image documents travel as an image-only user message through the checkpoint's chat "
    'template (no text part; the "Document: " head does NOT apply): prompt = [im_start, '
    "user, newline, vision_start] + N image-pad tokens + [vision_end, im_end, newline], "
    "N = ceil(H/32) * ceil(W/32) of the smart-resized image. Images are documents only (recipe "
    "gap G3): the pipeline must never send an image query."
)
IMAGE_BUDGET_NOTE = (
    "The input is 1280x1024 px = 1,310,720 px = the recipe's max_pixels exactly "
    "(mm_processor_kwargs), so the processor keeps it at the cap: N = 40 * 32 = 1280 patch "
    "tokens and the prompt is 4 + 1280 + 3 = 1287 tokens. R4 probe: usage.prompt_tokens must "
    "be exactly 1287 for this document (and never above 4 + 1280 + 3 for any image)."
)
IMAGE_SMALL_NOTE = (
    "The input is 256x256 px = the recipe's min_pixels (65,536 px) exactly, so the processor "
    "keeps it: N = 8 * 8 = 64 patch tokens expected (the T0 probe check pins the exact "
    "usage.prompt_tokens against the client count)."
)
MEDIA_NOTE = (
    "Media are synthetic (geometric page patterns, no third-party material), produced by "
    "media/make_media.py alongside; regenerate with `python media/make_media.py`."
)


def build_cases(tokenizer, skip: frozenset[int]) -> list[dict]:
    """Every generated case, measured from its own embedded texts."""
    q, d = SHORT_QUERIES, SHORT_DOCS
    long_under_single = _build_text(tokenizer, 7990, 1101, "under")
    long_under_a = _build_text(tokenizer, 7850, 1201, "under")
    long_under_b = _build_text(tokenizer, 8120, 1202, "under")
    long_over_single = _build_text(tokenizer, 8700, 1301, "over")
    long_over_a = _build_text(tokenizer, 8350, 1401, "over")
    long_over_b = _build_text(tokenizer, 9100, 1402, "over")
    mixed_under = _build_text(tokenizer, 7990, 1501, "under")
    mixed_over = _build_text(tokenizer, 8700, 1502, "over")
    final_under = _build_text(tokenizer, 7990, 2601, "under")
    final_over = _build_text(tokenizer, 8700, 2602, "over")
    capped_query = _build_text(tokenizer, 1090, 2101, "over", prefix=QUERY_PREFIX)

    def case(slug, modality, length, batch, queries, documents, static=(), kind="similarity_matrix", note_keys=()):
        measured = [
            doc_line(f"query {x['id']}", None, None, measure(tokenizer, x["text"], QUERY_PREFIX, skip)) for x in queries
        ]
        rendered_by_doc: dict[str, int] = {}
        for x in documents:
            if "text" in x:
                m = measure(tokenizer, x["text"], DOCUMENT_PREFIX, skip)
                rendered_by_doc[x["id"]] = m["tokens"]
                measured.append(doc_line(f"doc {x['id']}", x.get("seed"), x.get("target"), m))
        resolved = tuple(item(rendered_by_doc) if callable(item) else item for item in static)
        note = " ".join([RENDER_NOTE, *measured, *resolved, TOLERANCE_NOTE if kind != "none" else ""])
        return {
            "slug": slug,
            "modality": modality,
            "length": length,
            "batch": batch,
            "queries": queries,
            "documents": documents,
            "notes": note.strip(),
            "kind": kind,
        }

    def text_doc(doc_id, text, seed=None, target=None):
        return {"id": doc_id, "text": text, "seed": seed, "target": target}

    def image_doc(doc_id, path):
        return {"id": doc_id, "image": path}

    return [
        case(
            "text-short-single",
            "text",
            "short",
            "single",
            [{"id": "q1", "text": q["rot"]}],
            [text_doc("d1", d["short_b"])],
        ),
        case(
            "text-short-uniform",
            "text",
            "short",
            "uniform",
            [{"id": "q1", "text": q["rot"]}, {"id": "q2", "text": q["safe"]}],
            [
                text_doc("d1", d["short_b"]),
                text_doc("d2", d["short_e"]),
                text_doc("d3", d["short_f"]),
                text_doc("d4", d["short_h"]),
            ],
            static=(UNIFORM_NOTE,),
        ),
        case(
            "text-long-under-single",
            "text",
            "long_under",
            "single",
            [{"id": "q1", "text": q["rev"]}],
            [text_doc("d1", long_under_single, 1101, 7990)],
            static=(LONG_UNDER_NOTE,),
        ),
        case(
            "text-long-under-uniform",
            "text",
            "long_under",
            "uniform",
            [{"id": "q1", "text": q["rev"]}, {"id": "q2", "text": q["fer"]}],
            [text_doc("d1", long_under_a, 1201, 7850), text_doc("d2", long_under_b, 1202, 8120)],
            static=(LONG_UNDER_NOTE, UNIFORM_NOTE),
        ),
        case(
            "text-long-over-single",
            "text",
            "long_over",
            "single",
            [{"id": "q1", "text": q["rot"]}],
            [text_doc("d1", long_over_single, 1301, 8700)],
            static=(long_over_note,),
        ),
        case(
            "text-long-over-uniform",
            "text",
            "long_over",
            "uniform",
            [{"id": "q1", "text": q["rev"]}],
            [text_doc("d1", long_over_a, 1401, 8350), text_doc("d2", long_over_b, 1402, 9100)],
            static=(long_over_note, UNIFORM_NOTE),
        ),
        case(
            "text-mixed-length",
            "text",
            "mixed",
            "mixed_length",
            [{"id": "q1", "text": q["rot"]}, {"id": "q2", "text": q["rev"]}],
            [
                text_doc("d1", d["short_c"]),
                text_doc("d2", mixed_under, 1501, 7990),
                text_doc("d3", mixed_over, 1502, 8700),
            ],
            static=(LONG_UNDER_NOTE, long_over_note),
        ),
        case(
            "text-skip-mask-punctuation",
            "text",
            "short",
            "uniform",
            [{"id": "q1", "text": PUNCT_QUERIES["punct_a"]}, {"id": "q2", "text": PUNCT_QUERIES["punct_b"]}],
            [text_doc("d1", PUNCT_DOCS["punct_a"]), text_doc("d2", PUNCT_DOCS["punct_b"])],
            static=(PUNCT_NOTE, UNIFORM_NOTE),
        ),
        case(
            "query-cap-1024",
            "text",
            "short",
            "single",
            [{"id": "q1", "text": capped_query}],
            [text_doc("d1", d["short_a"])],
            static=(QUERY_CAP_NOTE,),
            kind="none",
        ),
        case(
            "image-short-single",
            "image",
            "short",
            "single",
            [{"id": "q1", "text": q["fer"]}],
            [image_doc("d1", "media/doc_small_1.png")],
            static=(IMAGE_RENDER_NOTE, IMAGE_SMALL_NOTE, MEDIA_NOTE),
        ),
        case(
            "image-budget-max",
            "image",
            "short",
            "single",
            [{"id": "q1", "text": q["rev"]}],
            [image_doc("d1", "media/doc_max_pixels.png")],
            static=(IMAGE_RENDER_NOTE, IMAGE_BUDGET_NOTE, MEDIA_NOTE),
        ),
        case(
            "image-mixed-sizes",
            "image",
            "mixed",
            "mixed_length",
            [{"id": "q1", "text": q["fer"]}],
            [image_doc("d1", "media/doc_small_1.png"), image_doc("d2", "media/doc_max_pixels.png")],
            static=(
                IMAGE_RENDER_NOTE,
                "One small and one cap-sized image in one batch: 64 vs 1280 patch tokens "
                "expected (71 vs 1287 prompt tokens), exercising the mixed-length image "
                "batch and the processor's per-image resize in one request. The T0 probe "
                "check pins the exact usage.prompt_tokens against the client count. "
                "Image prompts are always short of the 8192 budget: the 1280-patch cap "
                "bounds them at about 1287 tokens, so image x long_over is not an "
                "applicable cell.",
                MEDIA_NOTE,
            ),
        ),
        case(
            "image-uniform",
            "image",
            "short",
            "uniform",
            [{"id": "q1", "text": q["fer"]}, {"id": "q2", "text": q["rot"]}],
            [
                image_doc("d1", "media/doc_small_1.png"),
                image_doc("d2", "media/doc_small_2.png"),
                image_doc("d3", "media/doc_small_3.png"),
            ],
            static=(
                IMAGE_RENDER_NOTE,
                UNIFORM_NOTE,
                "doc_small_1 is 256x256 px (64 patch tokens expected), doc_small_2 is "
                "512x384 px (192 expected), doc_small_3 is 448x336 px (the processor "
                "resizes to 32-px multiples; the T0 probe pins the exact count). "
                "Image prompts are always short here: the 1280-patch budget caps them at "
                "about 1287 tokens, far under the 8192 budget, so image x long_over is "
                "not an applicable cell.",
                MEDIA_NOTE,
            ),
        ),
        case(
            "mixed-modality-short",
            "mixed",
            "short",
            "mixed_modality",
            [{"id": "q1", "text": q["rev"]}, {"id": "q2", "text": q["safe"]}],
            [
                text_doc("d1", d["short_a"]),
                text_doc("d2", d["short_d"]),
                image_doc("d3", "media/doc_small_2.png"),
                image_doc("d4", "media/doc_small_3.png"),
            ],
            static=(IMAGE_RENDER_NOTE, MEDIA_NOTE),
        ),
        case(
            "mixed-length-modality",
            "mixed",
            "mixed",
            "mixed_modality",
            [{"id": "q1", "text": q["rot"]}, {"id": "q2", "text": q["fer"]}],
            [
                text_doc("d1", d["short_b"]),
                text_doc("d2", final_under, 2601, 7990),
                text_doc("d3", final_over, 2602, 8700),
                image_doc("d4", "media/doc_max_pixels.png"),
            ],
            static=(LONG_UNDER_NOTE, long_over_note, IMAGE_RENDER_NOTE, MEDIA_NOTE),
        ),
    ]


# --- the deterministic YAML writer ----------------------------------------------------


def _str_representer(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    if len(data) > 80:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


class _Dumper(yaml.SafeDumper):  # noqa: N801 (yaml's own naming)
    pass


_Dumper.add_representer(str, _str_representer)


def case_document(case: dict) -> dict:
    """The case mapping in the case-format key order (documents lose the generator bookkeeping)."""
    documents = []
    for doc in case["documents"]:
        entry = {"id": doc["id"]}
        entry.update({k: doc[k] for k in ("text", "image", "video") if k in doc})
        documents.append(entry)
    return {
        "id": f"{RECIPE}/{case['slug']}",
        "recipe": RECIPE,
        "role": "multi_vector",
        "source": {"kind": "generated"},
        "strata": {"modality": case["modality"], "length": case["length"], "batch": case["batch"]},
        "inputs": {"instruction": None, "queries": case["queries"], "documents": documents},
        "expected": expected_block(case["kind"]),
        "notes": case["notes"],
    }


def write_case(out: Path, case: dict) -> None:
    """Dump one case file and prove the round trip (parse-back equals the built mapping)."""
    payload = case_document(case)
    text = yaml.dump(payload, Dumper=_Dumper, sort_keys=False, default_flow_style=False, allow_unicode=True, width=94)
    if yaml.safe_load(text) != payload:
        raise AssertionError(f"round-trip mismatch for {case['slug']}")
    (out / f"{case['slug']}.yaml").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="write the generated cases of topk-embed-v1-small")
    parser.add_argument(
        "--tokenizer-json",
        required=True,
        help="the pinned tokenizer.json (topk-io/topk-embed-v1-small @ " + REVISION + ")",
    )
    parser.add_argument("--out", default=Path(__file__).resolve().parent, type=Path)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer_json)
    tokenizer.no_truncation()  # the pinned file embeds {Right, 1024}; the recipe's gap G5

    config_json = Path(args.tokenizer_json).parent / "config.json"
    skip = frozenset(json.loads(config_json.read_text(encoding="utf-8"))["scoring_skip_ids"])

    cases = build_cases(tokenizer, skip)
    if len(cases) != 15:
        raise AssertionError(f"expected 15 generated cases, built {len(cases)}")
    for case in cases:
        write_case(args.out, case)
    print(f"wrote {len(cases)} generated cases to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
