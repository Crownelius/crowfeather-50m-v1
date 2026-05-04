"""Diagnose dataset schemas before bulk download.

For each dataset in the registry, load a few samples and print the field
structure. Helps catch schema drift (e.g., the 'code' subset of
open-r1/Mixture-of-Thoughts having different fields than 'math'/'science')
before wasting time on a 30-min precache that produces empty files.

Usage:
    python diagnose_datasets.py
    python diagnose_datasets.py --dataset sonnet  # just one
"""
import argparse, json, sys


def _print_sample(name: str, sample: dict, max_chars: int = 200):
    print(f'\n  [{name}] keys: {list(sample.keys())}')
    for k, v in sample.items():
        if isinstance(v, str):
            preview = v[:max_chars].replace('\n', ' / ')
            ellip = '...' if len(v) > max_chars else ''
            print(f'    {k!s:20s} (str, {len(v)} chars)  {preview!r}{ellip}')
        elif isinstance(v, list):
            print(f'    {k!s:20s} (list, {len(v)} items)')
            if v and isinstance(v[0], dict):
                print(f'      [0] keys: {list(v[0].keys())}')
                for kk, vv in v[0].items():
                    if isinstance(vv, str):
                        preview = vv[:120].replace('\n', ' / ')
                        ellip = '...' if len(vv) > 120 else ''
                        print(f'        {kk!s:18s} ({len(vv)} chars)  {preview!r}{ellip}')
                    else:
                        print(f'        {kk!s:18s} ({type(vv).__name__})  {vv!r}')
        else:
            print(f'    {k!s:20s} ({type(v).__name__})  {v!r}')


def diagnose(name: str, hf_id: str, config: str = None, n_samples: int = 2):
    print(f'\n{"="*70}\n{name}: {hf_id}{f"  config={config!r}" if config else ""}\n{"="*70}')
    try:
        from datasets import load_dataset
        if config is not None:
            ds = load_dataset(hf_id, config, split='train', streaming=True)
        else:
            ds = load_dataset(hf_id, split='train', streaming=True)
        for i, sample in enumerate(ds):
            if i >= n_samples:
                break
            _print_sample(f'{name}[{i}]', sample)
    except Exception as e:
        print(f'  ERROR: {type(e).__name__}: {e}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='all',
                   choices=['all', 'numinamath', 'metamathqa', 'r1_math', 'r1_science',
                            'r1_code', 'r1_default', 'sonnet'])
    p.add_argument('--n-samples', type=int, default=2)
    args = p.parse_args()

    REGISTRY = {
        'numinamath':  ('AI-MO/NuminaMath-CoT', None),
        'metamathqa':  ('meta-math/MetaMathQA', None),
        'r1_default':  ('open-r1/Mixture-of-Thoughts', 'default'),
        'r1_math':     ('open-r1/Mixture-of-Thoughts', 'math'),
        'r1_science':  ('open-r1/Mixture-of-Thoughts', 'science'),
        'r1_code':     ('open-r1/Mixture-of-Thoughts', 'code'),
        'sonnet':      ('Roman1111111/claude-sonnet-4.6-120000x', None),
    }

    if args.dataset == 'all':
        for name, (hf_id, config) in REGISTRY.items():
            diagnose(name, hf_id, config, args.n_samples)
    else:
        hf_id, config = REGISTRY[args.dataset]
        diagnose(args.dataset, hf_id, config, args.n_samples)


if __name__ == '__main__':
    main()
