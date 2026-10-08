# Report media-inputs: video and interleaved multimodal inputs for the GPU equivalence waves

**Status:** DONE, gated green on the merged head (`bin/gate lane/media-inputs`: PASS). Every fix is test-first;
two independent verifier rounds ran (round 1: two lenses; round 2: one fresh confirmation, because round 1 found
a blocker).

## What landed (per brief item)

1. **Video clips** (`observe/media_set.py`, `MEDIA_SET_VERSION` 3): a tiny MJPEG AVI (RIFF) per
   :data:`VIDEO_CLIPS` size (64x64 "icon", 224x224 "page") written on CPU from PIL-drawn frames (three scenes,
   a moving bar; no new dependencies, no container encoder). The frame count is the recipe's declared video
   policy's `num_frames` (the shipped one: 64), the fps 8.0; `video_entry` records the container's
   sha/mime/size/frames/duration, and the product's `probe_video_header` reads the generated headers back. The
   container mix is what vLLM v0.31.0's default video backend reads: OpenCV over in-memory bytes
   (`vllm/multimodal/video.py:202-249` `VideoBackend.load_bytes` with `backend: VideoDecoderBackend =
   "opencv"`; `vllm/multimodal/video_decoders/opencv.py:28-76` `decode_opencv`/`open_video_capture`
   `cv2.VideoCapture(BytesIO(data))`; MJPEG-in-AVI). The structural test parses the RIFF chunks back, decodes
   every JPEG frame with PIL and pins the BITMAPINFOHEADER's 40 bytes and `strh`'s rate and length. A recipe
   whose pairs video row's container is shorter than its declared sampling is refused by the client's own
   policy (the stage names the refusal).
2. **Interleaved and mixed rows** (per the recipe's `input`, `media_sides`, `max_images`): every media recipe
   plans a batch mixing a text-only and an image document; a query carrying an image where `media_sides`
   allows query media and the client can encode a media query (a declared query shape; a rerank pair carries
   both sides); and, where `max_images >= 2`, a text-image-text-image document (two images interleaved with
   text, in order -- the pairs media entries carry `text` segments) and a document with `max_images` images.
   Over the capacity stays the `edge:too_many_images` bare probe (kept). Video rows (alone and with text) for
   the recipes that accept them. The shipped media recipes all declare `max_images: 1`, so their manifests
   record `media:interleaved`/`media:several_images` absent with that reason and the machinery is exercised by
   the new `fixture-vl-video` fixture (max_images 2, media_sides default, query shape, a 4-frame pinned video
   sampling). The pairs media schema gains `text` entries (a part sequence's text segments).
3. **The media gate gates video and interleaved order** (`equivalence/media.py`): a video's declared frame
   count gates against the reference's `--mode media` (both sides declare the sampling); its container's token
   count gates against the engine -- the stage probes the sent container's header (the product's
   `probe_video_header`) and counts it exactly (`content_media_tokens` under the client's declared policies),
   so the engine's `usage.prompt_tokens` difference to the same request without its media must equal it (an
   engine not pinned to the declared sampling counts other frames and fails, as the image path does). An
   interleaved row gates the given part order: the client's fit joins a side's text parts into the first
   text part's position (`_with_text`), and the stage compares that placement with the card's. The test stub
   engine counts a video container the way vLLM v0.31.0's video path counts it (sampled to the engine's
   `--media-io-kwargs` frame count, else the default 32 at `vllm/multimodal/media/video.py:95`, patchified in
   time under the emulated family's video budget, through the product's own `content_media_tokens`) and
   refuses over-limit image/video counts like the per-prompt limits; the unpinned-stub control fails the video
   rows' engine check.
4. **Pairs regenerated** via the generator's own command (`python -m rcp_ndcg_vllm.observe.requests --out
   packages/rcp-ndcg-vllm/pairs --recipes qwen3-vl-embedding-2b,qwen3-vl-reranker-2b,topk-embed-v1-small
   --reference-python <venv> --suites nanobeir,bright`), `MEDIA_SET_VERSION` 3 (the media set's version bumps;
   `GENERATOR_VERSION` stays 1 so no text row re-sampled -- byte-identical text rows verified against the
   base). `media:video` present for `qwen3-vl-embedding-2b` (two 64-frame clips: 512 and 1952 tokens, exact,
   bounded 0), absent with the reason for the two image-only recipes; the new row strata recorded present or
   absent with the reason. The qwen3-vl-embedding file grows to ~504 KB (the clips), under the 2 MB cap.
5. **Docs and CHANGELOG**: the media-stage section of `docs/how-to/add-a-model.md` (the media rows' forms, the
   video and interleaved gating), the observations/pairs page
   (`packages/rcp-ndcg-vllm/schema/observation-corpus.md`: the media request set described as shipped -- the
   stale "not implemented yet (BLOCKED)" line and the stale "control (f) has no media gate" line are gone),
   and the CHANGELOG entry under `## Unreleased`.

Also: the generator's scratch tempdirs ignore cleanup errors (a written entry can turn visible after the
cleanup's scan on a network-backed tempdir and fail a recipe's validation with ENOTEMPTY; reproduced at ~1 run
in 4 against the pre-lane base, ten consecutive clean runs after), and the harness's scratch tempdirs in
`_validate_and_prune`/`_media_check`/`_reference_facts` carry `ignore_cleanup_errors=True`.

## Verification

- **Round 1, lens A (correctness, FAIL -> fixed)**: the one blocker was mine -- the AVI writer emitted a
  34-byte `strf` while `biSize` declared 40; the verifier reproduced, with the vLLM v0.31.0 backend code and
  two OpenCV builds from the local cache, that the shipped 64-frame containers decoded 63 of 64 frames (the
  demuxer dropped frame 0) and that the 40-byte header decodes 64 of 64. Fixed (40-byte
  BITMAPINFOHEADER, `MEDIA_SET_VERSION` 3, pairs regenerated, the structural test pins biSize/strf/strh).
  Its two minors (a stray `"""` in the observations page; the video+text stratum keyed on `len(clips)`) fixed.
- **Round 1, lens B (regressions and hygiene, PASS)**: four minors, all fixed -- the stray `"""`; a stale
  note in `qwen3-vl-embedding-2b/recipe.yaml` (said container tokens were not gated and the set carried no
  clip -- restated; recipe file, listed under files outside scope); the structural test not pinning `strh`
  (a dwRate mutation survived -- now asserted); the second MJPEG-AVI writer (the root suite's
  `tests/conftest.py::write_mjpeg_avi` predates the lane and is cross-referenced now; two writers by intent --
  cross-distribution). Its mutation checks: avih fps and frame-count mutations caught by the structural test;
  a wrong `--media-io-kwargs` pin fails the engine check on exactly the clip rows; a media-first placement
  mutation fails the interleaved row with `placement` only.
- **Round 2, one fresh confirmation verifier (lens A+B, PASS)**: rebuilt the pre-fix writer and ran the tag's
  actual decode path over both sizes: old header 63/64 (the blocker reproduces; the new assertions fail on
  it), fixed header 64/64 on both OpenCV builds; both shipped clips byte-identical to a fresh generation;
  every text row byte-identical to the base; the planner's strata match the committed manifest exactly; all
  suites green; CHANGELOG/docs truthful; round-1 minors gone. Its one minor (a one-size `VIDEO_CLIPS` would
  omit `media:video+text` from the strata instead of recording it absent) fixed test-first (`9c532327`).
- **Suites** (the gate's own runs on 9c532327): root 3261 passed / 82 skipped; contract+docs 273 passed /
  52 skipped; rcp-ndcg-test 124 passed; vllm-recipes job 342 passed / 209 skipped; ruff format+check and
  basedpyright clean; mkdocs --strict builds; run_all: leaderboards 1022 checks, 987 match, 35 known
  deviations, 0 failed; human study 67/67; external judges 82/82; worktree clean.

## Files outside scope

- `packages/rcp-ndcg-vllm/recipes/qwen3-vl-embedding-2b/recipe.yaml` -- its `notes` said the media stage
  reports a container's tokens "without gating them" and the media set "carries no clip"; both halves are now
  false, so the note was restated (gating + the clips + `probe_video_header`). No serving field changed.
- `tests/conftest.py` (root) -- a docstring cross-reference to the generator's writer only.

## Open questions

- **Interleaved rows are planned only where `max_images >= 2`; no shipped recipe declares that today.** The
  brief's list of row kinds is implemented for every media recipe *per its declared capacity* (its
  `media_sides`, `max_images`), and the manifests record `media:interleaved`/`media:several_images` absent
  with the max_images reason for all three shipped media recipes; the machinery is exercised end to end by
  `fixture-vl-video` (max_images 2). If the owner wants the shipped recipes to accept two images per request,
  that is a recipe change (serve `limit_mm_per_prompt` + client `max_images`) for the media wave, not a
  generator change.
- **The video row's clips carry exactly the declared `num_frames`** (the OBSERVATIONS-SPEC's "a short video at
  the pinned frame count"); an unpinned engine still samples its own default (32) and fails the count.
- **The clip sizes (64x64, 224x224) and fps (8) are the media set's own**; the recipe's per-clip pixel budget
  question (engine 25,165,824 px vs the card's 7,864,320 px per clip) stays open for the video wave -- the
  gate now compares the engine's count to the client's, so the wave will surface it either way.
- The two pre-existing manifest notes (round 2): the row-level `media:image` label is not a manifest stratum;
  `media:page_image` carries no reason for media-capable recipes. Identical at base; left alone.
