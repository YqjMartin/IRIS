# IRIS

IRIS synthesizes image editing pairs and evolves instruction-writing policies
from batch feedback: plan an edit, generate the image, judge the result, and
update policies for subsequent batches.

## Install

Python 3.10+. Synthesis uses remote model APIs and requires no local GPU.

```bash
python -m pip install -e .
cp .env.example .env
```

Set `apiurl` and `apikey` in `.env`, or use `NEWAPI_BASE_URL` and `NEWAPI_API_KEY`.
The endpoint must support Chat Completions with images and JSON output, and
multipart image editing. Use model IDs available from your provider.

## Inputs

Supply a scene pool containing `before/<scene_id>.jpg`, `scenes.jsonl`, and
`compatibility.jsonl`. Each JSONL line is one record:

```json
{"scene_id":"cup-001","prompt":"A ceramic cup on a desk.","category":"object","style":"photo"}
```

```json
{"compatibility_id":"material-scenes","edit_ref":"O2","allowed_categories":["object"],"allowed_scene_refs":["cup-001"]}
```

Provide at least `batch_size × max_batches` compatible images per edit type.
The supported types are O1–O9, T1–T7, H1–H3 and G1–G4.

## Run

The following command makes API requests:

```bash
python scripts/run_iris.py \
  --scene-pool /path/to/scene_pool --output out/run \
  --edits O2 --batch-size 10 --max-batches 5 --strict-batches \
  --text-model YOUR_VISION_MODEL --image-model YOUR_IMAGE_MODEL
```

Use `--prepare-only` to check inputs offline. To resume, repeat the command with
`--resume`, keeping inputs, models and batch settings unchanged.

```bash
python scripts/status.py out/run
python scripts/export_pairs.py --run out/run --output data/pairs
```

Acceptance requires a valid environment, instruction following ≥ 4, visual
consistency ≥ 3, visual quality ≥ 3, and an equal-weight mean ≥ 3.5. Export
includes complete accepted pairs, English instructions and image checksums.

Core code lives in `src/iris/`; `scripts/` contains run, status and export
entry points. Data, credentials, model weights and outputs are not bundled.
