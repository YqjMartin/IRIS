"""Export complete GoodCase pairs from one synthesis run without changing images."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from PIL import Image


def export(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    selected = json.loads((run / 'scene_selection.json').read_text())
    records = []
    sources = []
    seen = set()
    for code in selected:
        if not code or Path(code).name != code or code in {'.', '..'}:
            raise ValueError(f'Invalid edit code: {code}')
        for metadata in sorted((run / code).glob('*/metadata.json')):
            sample = json.loads(metadata.read_text())
            if (sample.get('verification') or {}).get('judgment') != 'GoodCase':
                continue
            sid = sample['sample_id']
            identity = code + '_' + metadata.parent.name
            if identity in seen:
                raise ValueError(f'Duplicate sample: {identity}')
            seen.add(identity)
            instruction = sample['steps'][0].get('instruction') or {}
            if not instruction.get('en', '').strip():
                raise ValueError(f'Missing English instruction: {sid}')
            pair = {}
            for key in ['before', 'after']:
                path = metadata.parent / (key + '.jpg')
                with Image.open(path) as image:
                    image.verify()
                pair[key] = path
            row = dict(id=identity, source_sample_id=sid, edit_type=code,
                       scene_id=sample['meta']['pair_meta']['scene_id'], judgment='GoodCase',
                       instruction_en=instruction['en'],
                       before=f'{identity}/before.jpg', after=f'{identity}/after.jpg')
            records.append(row)
            sources.append(pair)
    if not records:
        raise ValueError('No complete GoodCase pairs found')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.iris-export-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'dataset'
        (stage / 'provenance').mkdir(parents=True)
        checksums = []
        for row, pair in zip(records, sources):
            folder = stage / row['id']
            folder.mkdir()
            for key, source in pair.items():
                destination = stage / row[key]
                shutil.copy2(source, destination)
                checksums.append(dict(path=row[key], sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
            (folder / 'meta.json').write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
        for path, rows in [(stage / 'manifest.jsonl', records),
                           (stage / 'provenance/image_checksums.jsonl', checksums)]:
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        stage.rename(output)
    return dict(samples=len(records), output=str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.output)))
