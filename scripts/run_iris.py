"""Run the portable RSI harness against the user-configured compatible API."""
from __future__ import annotations

import argparse
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time

import requests
from PIL import Image
from api_config import load_newapi_credentials, read_env, runtime_settings

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
CONFIGS = ROOT / 'src/iris/configs'
SCENE_POOL = Path(os.environ.get('IRIS_SCENE_POOL', ROOT / 'data/scene_pool'))
BASELINE_MANIFEST = Path(os.environ['IRIS_BASELINE_MANIFEST']) if os.environ.get('IRIS_BASELINE_MANIFEST') else None
EDIT_CODES = [f'O{i}' for i in range(1, 10)] + [f'T{i}' for i in range(1, 8)] + ['H1', 'H2', 'H3'] + [f'G{i}' for i in range(1, 5)]
sys.path.insert(0, str(PROJECT / 'src'))
from iris.core.context import RunContext, RetryPolicy
from iris.core.imageops import parse_size, UnexpectedImageSize
from iris.core.parallel import FailureSink
from iris.core.schema import StepCost
from iris.core.storage import DirStorage
from iris.environment.schema import SceneDefinition, EditDefinition, CompatibilityDefinition, EnvironmentSnapshot
from iris.harness.campaign_spec import CampaignSpec
from iris.harness.loop import AgenticLoop
from iris.serving.base import LLMServingABC, VLMServingABC, ImageGenServingABC, ServingResult, ImageGenResult


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


class API:
    def __init__(self, output, profile='newapi', text_timeout=600, image_timeout=900, connect_timeout=120, stream_chat=True):
        config = ROOT / '.env'
        env = read_env(config)
        base_field, key_field = {'yuchat': ('YUCHAT_API_BASE', 'YUCHAT_API_KEY'),
            'newapi': ('newapi', 'newapikey'), 'sjapi': ('sjapi', 'sjapikey'),
            'yunapi': ('yunapi', 'yunapikey')}[profile]
        if profile == 'newapi':
            env[base_field], env[key_field] = load_newapi_credentials(config)
        if not env.get(base_field) or not env.get(key_field):
            raise ValueError('Missing API settings: fill .env or NEWAPI_BASE_URL / NEWAPI_API_KEY. No request sent.')
        self.base = env[base_field].rstrip('/')
        if profile == 'yunapi' and not self.base.endswith('/v1'):
            self.base += '/v1'
        self.key = env[key_field]
        self.output = output
        self.lock = threading.Lock()
        self.fatal_error = None
        self.text_timeout = text_timeout
        self.image_timeout = image_timeout
        self.connect_timeout = connect_timeout
        self.profile = profile
        self.stream_chat = stream_chat

    def request(self, route, *, payload=None, files=None, fields=None):
        if self.fatal_error:
            raise RuntimeError(self.fatal_error)
        start = time.monotonic()
        model = (payload or fields or {}).get('model')
        record = {'at': datetime.now(timezone.utc).isoformat(), 'route': route, 'requested_model': model,
                  'credential_profile': self.profile, 'api_base': self.base}
        try:
            streaming = route == '/chat/completions' and getattr(self, 'stream_chat', True)
            record['stream'] = streaming
            if streaming:
                payload = {**payload, 'stream': True, 'stream_options': {'include_usage': True}}
            response = requests.post(self.base + route, headers={'Authorization': 'Bearer ' + self.key},
                                     json=payload, data=fields, files=files, stream=streaming,
                                     timeout=(self.connect_timeout, self.image_timeout if route.startswith('/images/') else self.text_timeout))
            response.encoding = 'utf-8'
            record['status'] = response.status_code
            if not response.ok:
                message = response.text[:1500].replace(self.key, '[REDACTED]')
                try:
                    body = response.json()
                except ValueError:
                    body = None
                error = body.get('error') if isinstance(body, dict) else None
                if not isinstance(error, dict):
                    error = {}
                upstream = error.get('type') == 'upstream_error' or error.get('code') == 'upstream_error'
                if (response.status_code in {401, 403, 404} and not upstream) or any(marker in message.lower() for marker in ['model_not_found', 'no available channel', '\u65e0\u53ef\u7528\u6e20\u9053', 'insufficient_quota']):
                    self.fatal_error = f'HTTP {response.status_code}: {message}'
                if upstream:
                    raise RuntimeError(f'Upstream service failure (status {response.status_code}): {message}')
                raise RuntimeError(f'HTTP {response.status_code}: {message}')
            if streaming and 'text/event-stream' in response.headers.get('Content-Type', ''):
                text_parts = []
                result = {}
                try:
                    for line in response.iter_lines():
                        if not line or not line.startswith(b'data:'):
                            continue
                        data = line[5:].strip()
                        if data == b'[DONE]':
                            break
                        event = json.loads(data)
                        if event.get('error'):
                            raise RuntimeError(json.dumps(event['error'], ensure_ascii=False))
                        for field in ['id', 'model', 'usage']:
                            if event.get(field) is not None:
                                result[field] = event[field]
                        for choice in event.get('choices') or []:
                            content = (choice.get('delta') or {}).get('content')
                            if isinstance(content, str):
                                text_parts.append(content)
                    result['choices'] = [{'message': {'content': ''.join(text_parts)}}]
                finally:
                    response.close()
            else:
                result = response.json()
            record.update(response_model=result.get('model'), usage=result.get('usage'), response_id=result.get('id'))
            if result.get('error'):
                raise RuntimeError(json.dumps(result['error'], ensure_ascii=False).replace(self.key, '[REDACTED]'))
            if (payload or {}).get('response_format', {}).get('type') == 'json_object':
                try:
                    content = result['choices'][0]['message']['content']
                    parsed = json.loads(content)
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    raise RuntimeError(f'Invalid JSON model response: {type(exc).__name__}') from None
                if not isinstance(parsed, dict):
                    raise RuntimeError('Invalid JSON model response: expected an object')
            return result, time.monotonic() - start
        except Exception as exc:
            record['error'] = str(exc).replace(self.key, '[REDACTED]')[:1800]
            raise RuntimeError(record['error']) from None
        finally:
            record['latency_s'] = round(time.monotonic() - start, 3)
            with self.lock:
                with (self.output / 'api_calls.jsonl').open('a', encoding='utf8') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def cost(result, elapsed, model, step):
    usage = result.get('usage') or {}
    return StepCost(model=model, step=step, latency_s=elapsed,
                    prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'),
                    input_tokens=usage.get('input_tokens'), output_tokens=usage.get('output_tokens'),
                    total_tokens=usage.get('total_tokens'), input_tokens_details=usage.get('input_tokens_details'))


class TextVision(LLMServingABC, VLMServingABC):
    name = 'configured-api-gpt-5.5'
    model = 'gpt-5.5'

    def __init__(self, api):
        self.api = api

    def _call(self, content, system, json_output):
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': content})
        payload = {'model': self.model, 'messages': messages}
        if json_output:
            payload['response_format'] = {'type': 'json_object'}
        result, elapsed = self.api.request('/chat/completions', payload=payload)
        text = result['choices'][0]['message'].get('content')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Empty model response')
        return ServingResult(text, cost(result, elapsed, self.model, 'chat'),
                             {'response_id': result.get('id'), 'response_model': result.get('model'), 'usage': result.get('usage')})

    def generate(self, prompts, *, system=None, **kwargs):
        return [self._call(prompt, system, kwargs.get('response_format_json', False)) for prompt in prompts]

    def score(self, image_paths, *, prompt, rubric=None, **kwargs):
        content = [{'type': 'text', 'text': prompt}]
        for index, path in enumerate(image_paths):
            data = Path(path).read_bytes()
            with Image.open(BytesIO(data)) as im:
                mime = Image.MIME[im.format]
            content.extend([{'type': 'text', 'text': ('BEFORE image' if index == 0 else 'AFTER image')},
                            {'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + base64.b64encode(data).decode(), 'detail': 'high'}}])
        return [self._call(content, kwargs.get('system'), kwargs.get('response_format_json', False))]


class Images(ImageGenServingABC):
    name = 'configured-api-gpt-image-2.5'
    model = 'gpt-image-2.5'

    def __init__(self, api):
        self.api = api

    def _decode(self, result, elapsed):
        item = result['data'][0]
        if item.get('b64_json'):
            data = base64.b64decode(item['b64_json'], validate=True)
        elif item.get('url'):
            response = requests.get(item['url'], timeout=(self.api.connect_timeout, self.api.image_timeout))
            response.raise_for_status()
            data = response.content
        else:
            raise ValueError('Image response has neither b64_json nor url')
        with Image.open(BytesIO(data)) as im:
            width, height = im.size
            mime = Image.MIME[im.format]
            im.verify()
        return ImageGenResult(data, mime, width, height, cost(result, elapsed, self.model, 'image'),
                              {'usage': result.get('usage'), 'created': result.get('created')})

    def generate(self, prompt, *, size='1024x1024', seed=None, negative_prompt=None, **kwargs):
        result, elapsed = self.api.request('/images/generations', payload={'model': self.model, 'prompt': prompt, 'size': size, 'n': 1})
        return self._decode(result, elapsed)

    def edit(self, image_bytes, prompt, *, mask_bytes=None, size=None, **kwargs):
        fields = {'model': self.model, 'prompt': prompt, 'size': size or '1024x1024', 'n': '1'}
        files = {'image': ('before.png', image_bytes, 'image/png')}
        if mask_bytes:
            files['mask'] = ('mask.png', mask_bytes, 'image/png')
        result, elapsed = self.api.request('/images/edits', fields=fields, files=files)
        image = self._decode(result, elapsed)
        target = parse_size(fields['size'])
        if target and image.width * target[1] != image.height * target[0]:
            # Keep this validation inside the caller's request retry boundary.
            # Do not crop or stretch a nonconforming endpoint response.
            raise UnexpectedImageSize((image.width, image.height), target)
        return image


def prepare(output, edit_codes=None, *, batch_size=10, max_batches=5, include_reserve_scenes=False):
    if batch_size < 1 or max_batches < 1:
        raise ValueError('batch_size and max_batches must be positive')
    sample_budget = batch_size * max_batches
    edit_codes = edit_codes or ['H1', 'H2', 'H3']
    pool = SCENE_POOL
    env_dir = output / 'environment'
    env_dir.mkdir(parents=True, exist_ok=True)
    scenes = [SceneDefinition.model_validate({k: v for k, v in r.items() if k in SceneDefinition.model_fields}) for r in read_rows(pool / 'scenes.jsonl')]
    edits = [EditDefinition.model_validate(r) for r in read_rows(CONFIGS / 'environment/edits.jsonl')]
    compat = []
    for r in read_rows(pool / 'compatibility.jsonl'):
        code = r['edit_ref'].split('@')[0]
        if code not in edit_codes:
            continue
        compat.append(CompatibilityDefinition(compatibility_id=r['compatibility_id'], edit_ref=code,
                        allowed_categories=r['allowed_categories'], allowed_scene_refs=[s.split('@')[0] for s in r['allowed_scene_refs']]))
    for name, values in [('scenes', scenes), ('edits', edits), ('compatibility', compat)]:
        (env_dir / (name + '.jsonl')).write_text(''.join(v.model_dump_json() + '\n' for v in values), encoding='utf8')
    snapshot = EnvironmentSnapshot(snapshot_id='iris-scene-pool', description='Current scene pool and current portable edit definitions; normalized unversioned refs.',
        scene_registry='scenes.jsonl', edit_registry='edits.jsonl', compatibility_registry='compatibility.jsonl',
        scene_refs=[s.ref for s in scenes], edit_refs=[e.ref for e in edits], compatibility_refs=[c.ref for c in compat])
    save(env_dir / 'snapshot.json', snapshot.model_dump())
    old = read_rows(BASELINE_MANIFEST) if BASELINE_MANIFEST is not None else []
    selections = {}
    for code in edit_codes:
        # The original dataset stores extraction as O10; current registry uses O9.
        used = {r['scene_id'] for r in old if ('O9' if r['edit_type'] == 'O10' else r['edit_type']) == code}
        allowed = next(c.allowed_scene_refs for c in compat if c.edit_ref == code)
        available = {s for s in allowed if (pool / 'before' / (s + '.jpg')).exists()}
        original_scenes = {r['scene_id'] for r in old}
        ordered = (sorted(available & used) + sorted((available & original_scenes) - used)
                   + sorted(available - original_scenes)) if include_reserve_scenes else (
                   sorted(available & used) + sorted(available - used))
        if len(ordered) < sample_budget:
            raise ValueError(f'{code}: only {len(ordered)} compatible scenes with BEFORE images')
        chosen = ordered if include_reserve_scenes else ordered[:sample_budget]
        selections[code] = [{'scene_id': s, 'previously_used_for_type': s in used, 'batch': i // batch_size} for i, s in enumerate(chosen)]
    save(output / 'scene_selection.json', selections)
    inputs = ([BASELINE_MANIFEST] if BASELINE_MANIFEST is not None else []) + [pool / 'scenes.jsonl', pool / 'compatibility.jsonl',
              ROOT / 'scripts/api_config.py',
              CONFIGS / 'environment/edits.jsonl',
              PROJECT / 'src/iris/harness/evolve/render.py',
              PROJECT / 'src/iris/harness/evolve/prompts.py',
              PROJECT / 'src/iris/harness/loop.py', Path(__file__).resolve()]
    inputs += [pool / 'before' / (sid + '.jpg')
               for sid in sorted({row['scene_id'] for selected in selections.values() for row in selected})]
    config_dir = output / 'config_snapshot'
    config_dir.mkdir(exist_ok=True)
    for name, source in {
        'evolve_prompts.py': PROJECT / 'src/iris/harness/evolve/prompts.py',
        'if_vc_vq.json': CONFIGS / 'rubrics/if_vc_vq.json',
    }.items():
        destination = config_dir / name
        if not destination.exists():
            destination.write_bytes(source.read_bytes())
    save(output / 'run_inputs.json', {'models': {'text_vision': TextVision.model, 'image': Images.model},
        'scene_pool': str(SCENE_POOL.resolve()),
        'baseline_manifest': str(BASELINE_MANIFEST.resolve()) if BASELINE_MANIFEST else None,
        'edit_types': edit_codes,
        'batch_size': batch_size, 'batches_per_type': max_batches, 'type_workers': len(edit_codes), 'sample_workers_per_type': 10,
        'read_timeout_s': {'text': 600, 'image': 900},
        'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}})
    return env_dir / 'snapshot.json', selections


def make_retry_policy(max_retries):
    return RetryPolicy(max_retries=max_retries, backoff_s=2.0, backoff_factor=1.0,
                       backoff_cap_s=2.0, stochastic_backoff_s=2.0)


def require_complete_samples(storage, sample_ids):
    pending = []
    for sid in sample_ids:
        sample = storage.read_sample(sid)
        plan = sample.steps[0].raw.get('instruction_plan') or {}
        if sample.verification is None and plan.get('environment_valid') is not False:
            pending.append(sid)
    if pending:
        raise RuntimeError(f'Batch incomplete before feedback: {len(pending)} samples; first={pending[0]}')


def require_successful_feedback(run_dir, batch_id=None):
    path = run_dir / '_evolution.jsonl'
    rows = read_rows(path) if path.exists() else []
    if batch_id is not None:
        rows = [r for r in rows if r['batch_id'].rsplit(':', 1)[-1] == batch_id]
        if not rows:
            raise RuntimeError(f'Batch feedback failed: missing evolution record for {batch_id}')
    for row in rows:
        errors = {key: row[key] for key in ('analysis_error', 'commit_error', 'diversity_commit_error') if row.get(key)}
        if errors:
            raise RuntimeError('Batch feedback failed: ' + json.dumps({'batch_id': row['batch_id'], **errors}, ensure_ascii=False))


def run_one(code, output, snapshot, selection, api, resume=False, max_retries=3, image_api=None,
            batch_size=10, max_batches=5, strict_batches=False, replace_rejected_scenes=False):
    image_api = image_api or api
    run_dir = output / code
    spec = CampaignSpec(campaign_id=output.name + '-' + code, setting_id='policy-evolving',
        environment_snapshot=str(snapshot), restrict_edits=[code], restrict_scenes=[r['scene_id'] for r in selection],
        budget={'max_api_calls': 1000, 'batch_size': batch_size, 'max_batches': max_batches},
        generator_pool=[{'name': Images.name, 'model': Images.model}], verifier_stack=[{'name': TextVision.name, 'model': TextVision.model}],
        replace_rejected_scenes=replace_rejected_scenes,
        notes={'scene_priority': 'previously used for the same edit type, then compatible supplements', 'workers': 10})
    save(run_dir / 'campaign.json', spec.model_dump())
    text = TextVision(api)
    ctx = RunContext(seed=42, max_workers=10, retry_policy=make_retry_policy(max_retries), failure_sink=FailureSink(run_dir))
    class CheckedLoop(AgenticLoop):
        def _run_pipeline_on(self, sample_ids):
            result = super()._run_pipeline_on(sample_ids)
            if api.fatal_error or image_api.fatal_error:
                raise RuntimeError(api.fatal_error or image_api.fatal_error)
            if strict_batches:
                require_complete_samples(self.storage, sample_ids)
            return result

        def _append_curve(self, batch_id_str):
            if strict_batches:
                require_successful_feedback(self.storage.root, batch_id_str)
            return super()._append_curve(batch_id_str)

    if strict_batches:
        require_successful_feedback(run_dir)
    loop = CheckedLoop(spec, DirStorage(run_dir), ctx, llm_serving=text, image_serving=Images(image_api), vlm_serving=text,
        run_tag=output.name, before_bank=SCENE_POOL / 'before')
    order = {r['scene_id']: i for i, r in enumerate(selection)}
    loop.planner.schedule.sort(key=lambda p: order[p.scene_id])
    result = loop.run(resume=resume)
    save(run_dir / 'result.json', result)
    return result


def resolve_text_model(output, requested_model, resume):
    """New runs use the default; an existing checkpoint keeps its saved model."""
    inputs = Path(output) / 'run_inputs.json'
    if resume and inputs.exists():
        saved = json.loads(inputs.read_text(encoding='utf8'))['models']['text_vision']
        if requested_model is not None and requested_model != saved:
            raise ValueError('Resume must keep the original text/vision model; use a fresh output for a different model.')
        return saved
    return requested_model or TextVision.model


def main():
    global SCENE_POOL, BASELINE_MANIFEST
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene-pool', type=Path, default=SCENE_POOL, help='Directory with scenes.jsonl, compatibility.jsonl and before/*.jpg.')
    parser.add_argument('--baseline-manifest', type=Path, default=BASELINE_MANIFEST, help='Optional prior dataset manifest for scene ordering.')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--skip-preflight', action='store_true')
    parser.add_argument('--credential-profile', choices=['newapi', 'yuchat', 'sjapi'], default='newapi')
    parser.add_argument('--image-profile', choices=['newapi', 'yuchat', 'sjapi'])
    parser.add_argument('--text-model', default=None, help='New runs default to gpt-5.5; resume inherits the saved model.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--edits', nargs='+', choices=EDIT_CODES, default=['H1', 'H2', 'H3'])
    parser.add_argument('--image-model', default=Images.model)
    parser.add_argument('--text-timeout', type=int, default=360)
    parser.add_argument('--image-timeout', type=int, default=360)
    parser.add_argument('--connect-timeout', type=int, default=120)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--max-batches', type=int, default=5)
    parser.add_argument('--strict-batches', action='store_true', help='Stop before advancing past incomplete samples or failed feedback.')
    parser.add_argument('--non-streaming-text', action='store_true')
    parser.add_argument('--use-prepared-inputs', action='store_true', help='Use the frozen environment and scene schedule already prepared for this run.')
    parser.add_argument('--replace-rejected-scenes', action='store_true', help='Replace a rejected scene in its batch until the batch contains batch-size usable plans; no rejection retry.')
    args = parser.parse_args()
    SCENE_POOL = args.scene_pool.resolve()
    BASELINE_MANIFEST = args.baseline_manifest.resolve() if args.baseline_manifest else None
    if min(args.text_timeout, args.image_timeout, args.connect_timeout) <= 0 or args.max_retries < 0:
        parser.error('Timeouts must be positive and max-retries nonnegative.')
    if min(args.batch_size, args.max_batches) < 1:
        parser.error('Batch size and max batches must be positive.')
    output = args.output.resolve()
    if not args.resume and any(output.glob('*/_index.jsonl')):
        parser.error('Output already contains campaign samples; use --resume or a new output directory.')
    try:
        args.text_model = resolve_text_model(output, args.text_model, args.resume)
    except ValueError as exc:
        parser.error(str(exc))
    Images.model = args.image_model
    Images.name = 'configured-api-' + args.image_model
    TextVision.model = args.text_model
    TextVision.name = 'configured-api-' + args.text_model
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    output.mkdir(parents=True, exist_ok=True)
    overrides, overrides_path = runtime_settings(output)
    args.connect_timeout = overrides.get('connect_upload_timeout_s', args.connect_timeout)
    args.text_timeout = overrides.get('read_timeout_s', args.text_timeout)
    args.image_timeout = overrides.get('read_timeout_s', args.image_timeout)
    args.max_retries = overrides.get('max_retries', args.max_retries)
    previous_inputs = output / 'run_inputs.json'
    if (args.resume or args.use_prepared_inputs) and not previous_inputs.is_file():
        parser.error('No prepared run found in this output directory.')
    if args.resume and previous_inputs.exists():
        history = output / 'resume_configuration_history'
        history.mkdir(exist_ok=True)
        (history / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json')).write_bytes(previous_inputs.read_bytes())
    if args.resume and previous_inputs.exists():
        previous = json.loads(previous_inputs.read_text(encoding='utf8'))
        if previous['models']['image'] != args.image_model:
            parser.error('Resume must keep the original image model.')
        if previous.get('scene_pool') != str(SCENE_POOL) or previous.get('baseline_manifest') != (str(BASELINE_MANIFEST) if BASELINE_MANIFEST else None):
            parser.error('Resume must keep the original scene pool and baseline manifest.')
        for source, expected in previous['source_sha256'].items():
            if not Path(source).is_file() or hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
                parser.error('Run input changed; use a new output: ' + source)
        args.use_prepared_inputs = True
        if (previous.get('batch_size'), previous.get('batches_per_type')) != (args.batch_size, args.max_batches):
            parser.error('Resume must keep the original batch size and batch count.')
    if args.use_prepared_inputs:
        frozen = json.loads(previous_inputs.read_text(encoding='utf8'))
        if frozen.get('scene_pool') != str(SCENE_POOL) or frozen.get('baseline_manifest') != (str(BASELINE_MANIFEST) if BASELINE_MANIFEST else None):
            parser.error('Prepared inputs use a different scene pool or baseline manifest.')
        if not args.resume:
            for source, expected in frozen['source_sha256'].items():
                if not Path(source).is_file() or hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
                    parser.error('Run input changed; use a new output: ' + source)
        if frozen['models'] != {'text_vision': args.text_model, 'image': args.image_model}:
            parser.error('Prepared model configuration differs from requested models.')
        if (frozen['batch_size'], frozen['batches_per_type']) != (args.batch_size, args.max_batches):
            parser.error('Prepared batch configuration differs from requested configuration.')
        if frozen.get('replace_rejected_scenes', False) != args.replace_rejected_scenes:
            parser.error('Prepared scene-replacement mode differs from requested configuration.')
        selection = json.loads((output / 'scene_selection.json').read_text(encoding='utf8'))
        if set(selection) != set(args.edits):
            parser.error('Prepared edit types differ from requested edit types.')
        snapshot = output / 'environment/snapshot.json'
        if not snapshot.is_file():
            parser.error('Prepared environment snapshot is missing.')
    else:
        snapshot, selection = prepare(output, list(dict.fromkeys(args.edits)),
                                      batch_size=args.batch_size, max_batches=args.max_batches,
                                      include_reserve_scenes=args.replace_rejected_scenes)
    inputs = json.loads((output / 'run_inputs.json').read_text(encoding='utf8'))
    inputs.update(read_timeout_s={'text': args.text_timeout, 'image': args.image_timeout}, max_retries=args.max_retries, retry_interval_s=2,
                  connect_upload_timeout_s=args.connect_timeout, strict_batches=args.strict_batches,
                  runtime_settings_path=str(overrides_path) if overrides_path else None,
                  credential_profiles={'text_vision': args.credential_profile, 'image': args.image_profile or args.credential_profile})
    inputs['text_streaming'] = not args.non_streaming_text
    inputs['instruction_plan_rejection_retries'] = {}
    inputs['replace_rejected_scenes'] = args.replace_rejected_scenes
    save(output / 'run_inputs.json', inputs)
    os.environ['IRIS_EDIT_REGISTRY'] = str(output / 'environment/edits.jsonl')
    print(json.dumps({c: {'selected': len(v), 'reused': sum(r['previously_used_for_type'] for r in v)} for c, v in selection.items()}), flush=True)
    if args.prepare_only:
        return
    api = API(output, profile=args.credential_profile, text_timeout=args.text_timeout, image_timeout=args.image_timeout,
              connect_timeout=args.connect_timeout, stream_chat=not args.non_streaming_text)
    image_api = api if not args.image_profile or args.image_profile == args.credential_profile else API(
        output, profile=args.image_profile, text_timeout=args.text_timeout, image_timeout=args.image_timeout,
        connect_timeout=args.connect_timeout)
    image_api.lock = api.lock
    if not args.skip_preflight:
        probe = TextVision(api).generate(['Return a JSON object with ok=true.'], response_format_json=True)[0]
        save(output / 'text_preflight.json', {'model': TextVision.model, 'content': probe.content, 'raw': probe.raw})
        if json.loads(probe.content).get('ok') is not True:
            raise RuntimeError('Text preflight did not return ok=true; campaign was not started.')
        print(args.text_model + ' text preflight succeeded', flush=True)
    if args.probe_only:
        return
    for code in selection:
        if (output / code / '_index.jsonl').exists() and not args.resume:
            raise ValueError('Refusing to restart a nonempty campaign; use a new output or explicit recovery.')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(output / 'run.log', encoding='utf8'), logging.StreamHandler(sys.stdout)])
    results = {}
    with ThreadPoolExecutor(max_workers=len(selection)) as executor:
        futures = {executor.submit(run_one, c, output, snapshot, selection[c], api, args.resume, args.max_retries,
                                   image_api, args.batch_size, args.max_batches, args.strict_batches,
                                   args.replace_rejected_scenes): c for c in selection}
        for future in as_completed(futures):
            code = futures[future]
            try:
                results[code] = future.result()
            except Exception as exc:
                results[code] = {'error': str(exc).replace(api.key, '[REDACTED]').replace(image_api.key, '[REDACTED]')}
                logging.exception('%s campaign failed', code)
            save(output / 'results.json', results)
    print(json.dumps({c: r.get('summary', r) for c, r in results.items()}, ensure_ascii=False), flush=True)
    if any(r.get('error') for r in results.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
