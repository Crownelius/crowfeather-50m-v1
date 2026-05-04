"""Download all Crowfeather-50M-v1 distillation datasets to a local directory.

Wraps precache_distill.py with sensible defaults for a local (non-Colab)
Windows or Linux machine. Default target on Windows: E:/crowfeather_data.
Default target on Linux/macOS: ~/crowfeather_data.

Usage:
    # Windows, full corpus, defaults to E:/crowfeather_data
    python scripts/download_local.py

    # Linux, custom target
    python scripts/download_local.py --target /mnt/data/crowfeather_data

    # Capped at 30 GB total
    python scripts/download_local.py --budget-mb 30000

The download takes ~60-90 min for full corpus over a 50 Mbps connection.
Output: per-source JSONL files + per-domain combined JSONL files in unified
schema (see docs/DATA_FORMAT.md). Both per-source and per-domain are kept
since they enable post-hoc rebalancing / Sonnet conversion of subsets.
"""
import argparse, os, platform, subprocess, sys


def default_target():
    return 'E:/crowfeather_data' if platform.system() == 'Windows' \
        else os.path.expanduser('~/crowfeather_data')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', default=default_target(),
                   help=f'output directory (default: {default_target()})')
    p.add_argument('--budget-mb', type=int, default=None,
                   help='cap each source proportionally; default is unlimited (full corpus)')
    p.add_argument('--no-force-refresh', action='store_true',
                   help='do NOT delete existing per-source JSONLs before download')
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    precache = os.path.join(here, 'precache_distill.py')
    if not os.path.exists(precache):
        print(f'ERROR: precache_distill.py not found at {precache}')
        sys.exit(1)

    os.makedirs(args.target, exist_ok=True)

    cmd = [
        sys.executable, precache,
        '--target-dir', args.target,
        '--drive-cache', args.target,  # Opus stream looks here for opus_4_6.jsonl
    ]
    if args.budget_mb:
        cmd += ['--budget-mb', str(args.budget_mb)]
    else:
        cmd.append('--unlimited')
    if not args.no_force_refresh:
        cmd.append('--force-refresh')

    print('=' * 70)
    print(f'Crowfeather-50M-v1 local download')
    print('=' * 70)
    print(f'Target dir:    {args.target}')
    print(f'Mode:          {"BUDGETED " + str(args.budget_mb) + " MB" if args.budget_mb else "UNLIMITED (full corpus)"}')
    print(f'Force-refresh: {not args.no_force_refresh}')
    print(f'Disk warning:  full corpus expects ~60-90 GB. Make sure {args.target.split(":")[0] if ":" in args.target else "destination"} has space.')
    print()

    if 'Roman1111111' in os.environ.get('CROWFEATHER_LEGACY_SONNET', ''):
        print('NOTE: legacy Sonnet env set; ignored — current pipeline uses Kimi K2.5.')

    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        print(f'\nprecache_distill.py exited with code {rc}')
        sys.exit(rc)

    print('\n' + '=' * 70)
    print('DOWNLOAD COMPLETE')
    print('=' * 70)
    total = 0
    for fn in sorted(os.listdir(args.target)):
        if not fn.endswith('.jsonl'):
            continue
        sz = os.path.getsize(os.path.join(args.target, fn))
        total += sz
        print(f'  {fn:25s} {sz/1e6:>10.1f} MB')
    print(f'  {"TOTAL":25s} {total/1e6:>10.1f} MB')
    print()
    print('Next step: convert highest-value rows through Sonnet 4.6 to enforce')
    print('  the optimal format. Start with a 1K sample to gauge quality:')
    print(f'  python scripts/sonnet_convert.py \\')
    print(f'      --input {args.target}/lang.jsonl \\')
    print(f'      --output {args.target}/lang.sonnet.jsonl \\')
    print(f'      --sample 1000')
    print('See docs/OPTIMAL_FORMAT.md for the format spec and Sonnet prompt rationale.')


if __name__ == '__main__':
    main()
