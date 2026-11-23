"""Fetch and verify six upstream H3 weights; standard-library only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib import request, error

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "runtime"))
from model_assets import selected_weights
MANIFEST = ROOT / 'runtime/models/manifest.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, item):
    return path.is_file() and path.stat().st_size == item['bytes'] and sha256(path) == item['sha256']


def download(item, folder):
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / item['name']
    if verify(dest, item):
        print('Verified existing:', dest.name, flush=True)
        return
    if dest.exists():
        raise RuntimeError(f'Refusing to overwrite invalid existing file: {dest}')
    part = dest.with_name(dest.name + '.part')
    if part.exists() and part.stat().st_size > item['bytes']:
        raise RuntimeError(f'Oversized partial file: {part}')
    url = item['source'].replace('/blob/', '/resolve/')
    for attempt in range(5):
        offset = part.stat().st_size if part.exists() else 0
        if offset >= item['bytes']:
            break
        headers = {'User-Agent': 'H3-Turbo-Runtime/1.0'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with request.urlopen(request.Request(url, headers=headers), timeout=90) as response:
                if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                    part.unlink()
                    raise RuntimeError('Server did not honor Range; restarting cleanly')
                with part.open('ab' if offset else 'wb') as out:
                    shutil.copyfileobj(response, out, 8 * 1024 * 1024)
        except (OSError, error.URLError, RuntimeError) as exc:
            if attempt == 4:
                print(f'Failed to retrieve {dest.name}', flush=True)
                raise
            print('Retrying', dest.name, exc, flush=True)
            time.sleep(2 * (attempt + 1))
    if part.stat().st_size != item['bytes'] or sha256(part) != item['sha256']:
        raise RuntimeError(f'Unverified download retained as {part}')
    part.replace(dest)
    print('Downloaded and verified:', dest.name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=ROOT / 'weights')
    parser.add_argument('--condition-models', type=Path, default=ROOT / 'workbench' / 'condition-models')
    parser.add_argument('--skip-conditioning', action='store_true', help='Skip text encoder and ClipProj weights; enough for CLI rendering with a pre-encoded condition')
    parser.add_argument('--variant', choices=('fl2va', 'ref2va'), default='fl2va')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='Offline SHA-256 verification')
    mode.add_argument('--fetch', action='store_true', help='Download missing and verify')
    args = parser.parse_args()
    data = json.loads(MANIFEST.read_text())
    items = [(x, args.models) for x in selected_weights(data, args.variant)]
    if not args.skip_conditioning:
        items += [(x, args.condition_models) for x in data['conditioning_weights']]
    print('Required weight files:', len(items), 'total GB:', round(sum(x['bytes'] for x, _ in items) / 1e9, 3), flush=True)
    failed = False
    for item, folder in items:
        if args.check:
            ok = verify(folder / item['name'], item)
            print('OK' if ok else 'MISSING_OR_MISMATCH', item['id'], flush=True)
            failed |= not ok
        else:
            download(item, folder)
    if failed:
        parser.exit(1, 'Weight validation failed\n')
    print('All required weight files verified for variant', args.variant, flush=True)


if __name__ == '__main__':
    main()
