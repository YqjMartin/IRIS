"""Print a compact summary of a synthesis run."""
import argparse
import json
from pathlib import Path


def summarize(root):
    root = Path(root)
    selected = json.loads((root / 'scene_selection.json').read_text())
    counts = dict(samples=0, judged=0, accepted=0, scene_rejected=0, pending=0)
    for code in selected:
        for path in (root / code).glob('*/metadata.json'):
            try:
                sample = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue  # A running worker may still be writing this file.
            verdict = (sample.get('verification') or {}).get('judgment')
            plan = sample['steps'][0].get('raw', {}).get('instruction_plan') or {}
            judged = verdict in {'GoodCase', 'BadCase'}
            rejected = plan.get('environment_valid') is False
            counts['samples'] += 1
            counts['judged'] += judged
            counts['accepted'] += verdict == 'GoodCase'
            counts['scene_rejected'] += rejected
            counts['pending'] += not (judged or rejected)
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    counts = summarize(parser.parse_args().output)
    print(' | '.join(f"{key}: {value}" for key, value in counts.items()))
