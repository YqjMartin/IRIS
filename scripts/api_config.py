"""Read current API settings while supporting archived configuration names."""
from __future__ import annotations

import os
import json
from pathlib import Path
from urllib.parse import urlsplit

RUNTIME_FIELDS = ('connect_upload_timeout_s', 'read_timeout_s', 'max_retries')


def runtime_settings(output: Path) -> tuple[dict, Path | None]:
    """Only inherit overrides from this experiment's nearest queue directory."""
    for parent in (output.resolve(), *output.resolve().parents):
        if (parent / 'queue.json').is_file():
            path = parent / 'runtime_settings.json'
            if not path.exists():
                return {}, None
            values = json.loads(path.read_text(encoding='utf-8-sig'))
            bounds = {'connect_upload_timeout_s': (1, 240), 'read_timeout_s': (1, 900), 'max_retries': (0, 6)}
            for name, (low, high) in bounds.items():
                if name in values and (type(values[name]) is not int or not low <= values[name] <= high):
                    raise ValueError(f'Invalid experiment runtime setting: {name}')
            return {name: values[name] for name in RUNTIME_FIELDS if name in values}, path
    return {}, None


def read_env(path: Path) -> dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                name, value = line.split('=', 1)
                values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def load_newapi_credentials(path: Path) -> tuple[str, str]:
    values = read_env(path)
    base = (os.environ.get('NEWAPI_BASE_URL') or values.get('apiurl') or values.get('newapi') or '').strip().rstrip('/')
    key = (os.environ.get('NEWAPI_API_KEY') or values.get('apikey') or values.get('newapikey') or '').strip()
    if not base or not key:
        raise ValueError('Fill apiurl/apikey in .env, or set NEWAPI_BASE_URL / NEWAPI_API_KEY. No request sent.')
    parsed = urlsplit(base)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError('API base must be an HTTP(S) URL without a query or fragment. No request sent.')
    if not parsed.path:
        base += '/v1'
    return base, key
