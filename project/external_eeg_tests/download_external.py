"""Download original left/right imagery files from two official repositories."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def tasks(root):
    result = []
    for subject in range(1, 110):
        for run in [4, 8, 12]:
            relative = f'S{subject:03d}/S{subject:03d}R{run:02d}.edf'
            result.append(('PhysioNet', 'https://archive.physionet.org/pn4/eegmmidb/' + relative, root / 'data/raw/physionet' / relative))
    for subject in range(1, 10):
        for part in 'TE':
            name = f'A{subject:02d}{part}.mat'
            result.append(('BNCI2014_001', 'https://lampx.tugraz.at/~bci/database/001-2014/' + name, root / 'data/raw/bnci001' / name))
    return sorted(result, key=lambda item: item[0])


def fetch(task):
    dataset, url, dest = task
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        for attempt in range(5):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'EEG-research-downloader/1.0'})
                with urllib.request.urlopen(request, timeout=45) as response:
                    expected = response.headers.get('Content-Length')
                    with open(dest.with_suffix(dest.suffix + '.part'), 'wb') as output:
                        shutil.copyfileobj(response, output)
                partial = dest.with_suffix(dest.suffix + '.part')
                if expected and partial.stat().st_size != int(expected):
                    raise IOError('Incomplete HTTP response')
                partial.replace(dest)
                break
            except Exception as exc:
                print(f'Retry {dest.name}: {exc}', flush=True)
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))
    with open(dest, 'rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'dataset': dataset, 'url': url, 'relative_file': dest.relative_to(ROOT).as_posix(), 'bytes': dest.stat().st_size, 'sha256': digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    todo = tasks(ROOT)
    manifest = []
    failures = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch, task): task for task in todo}
        for future in as_completed(futures):
            try:
                item = future.result()
                manifest.append(item)
                if len(manifest) % 15 == 0 or len(manifest) <= 3:
                    print(f'{len(manifest)}/{len(todo)} files, {sum(r["bytes"] for r in manifest)/2**20:.1f} MiB, elapsed {time.time()-start:.0f}s', flush=True)
            except Exception as exc:
                failures.append({'url': futures[future][1], 'error': str(exc)})
    manifest.sort(key=lambda r: r['relative_file'])
    (ROOT / 'data/sources.json').write_text(json.dumps({'datasets': {'PhysioNet': {'page': 'https://physionet.org/content/eegmmidb/1.0.0/', 'license': 'Open Data Commons Attribution License v1.0', 'selection': 'All 109 subjects; only imagery runs 4,8,12. Original EDF+ includes event annotations.'}, 'BNCI2014_001': {'page': 'https://bnci-horizon-2020.eu/database/data-sets', 'license': 'CC BY-ND 4.0', 'selection': 'All 9 subjects, both T/E MAT files. Analysis retains classes 1 and 2 only.'}}, 'files': manifest, 'failures': failures}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Done: {len(manifest)} files, {len(failures)} failures.', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
