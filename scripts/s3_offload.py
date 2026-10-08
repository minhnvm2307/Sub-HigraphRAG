"""Offload slow-to-regenerate artefacts (embeddings, checkpoints, reasoning outputs) to S3 and restore them.
Policy (docs/05 §S3): only what is costly to rebuild goes to S3; re-downloadable sources are not stored.
Keys mirror the path under the project root: s3://banktransactiondata/subgraphrag_archive/<relative path>.
A local file is removed only with --delete and only after S3 reports the same size. Every upload is appended to
docs/S3_ARCHIVE.md. Credentials come from the project .env and are never printed.

  python s3_offload.py put [--delete] <path> [<path> ...]     # files or directories
  python s3_offload.py get <relative path or prefix>           # restore into the project root
  python s3_offload.py ls [prefix]
Needs boto3 (scratchpad vastenv); do not install it into the conda envs.
"""
import os, re, sys, time
import boto3
from boto3.s3.transfer import TransferConfig

ROOT = '/home/minhnv/Lab/SubgraphRAG'
BUCKET, PREFIX = 'banktransactiondata', 'subgraphrag_archive/'
MANIFEST = f'{ROOT}/docs/S3_ARCHIVE.md'
env = {}
for line in open(f'{ROOT}/.env'):
    m = re.match(r'\s*([A-Za-z_-]+)\s*=\s*"?([^",\n]*)"?', line)
    if m:
        env[m.group(1)] = m.group(2).strip()
s3 = boto3.client('s3', aws_access_key_id=env['AccessKey-Id'], aws_secret_access_key=env['SecretAccessKey'],
                  region_name=env.get('Region', 'us-east-1'))
CFG = TransferConfig(multipart_chunksize=64 * 2**20, max_concurrency=8)


def files_of(p):
    p = os.path.abspath(p)
    if os.path.isfile(p):
        return [p]
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs)


def put(paths, delete):
    if not os.path.exists(MANIFEST):
        open(MANIFEST, 'w').write('# S3 archive (s3://banktransactiondata/subgraphrag_archive/)\n\n'
                                  'Restore: `python scripts/s3_offload.py get <relative path>`.\n\n'
                                  '| Date | Relative path | Size (MB) | Local copy |\n|---|---|---|---|\n')
    for p in paths:
        for f in files_of(p):
            rel = os.path.relpath(f, ROOT); key = PREFIX + rel; size = os.path.getsize(f)
            t0 = time.time()
            s3.upload_file(f, BUCKET, key, Config=CFG)
            remote = s3.head_object(Bucket=BUCKET, Key=key)['ContentLength']
            ok = remote == size
            gone = False
            if ok and delete:
                os.remove(f); gone = True
            dt = time.time() - t0
            print(f'{"OK " if ok else "SIZE MISMATCH"} {rel} {size / 2**20:.0f} MB in {dt:.0f}s '
                  f'({size / 2**20 / max(dt, 1e-3):.1f} MB/s){" — local removed" if gone else ""}', flush=True)
            with open(MANIFEST, 'a') as fh:
                fh.write(f'| {time.strftime("%Y-%m-%d")} | `{rel}` | {size / 2**20:.0f} | '
                         f'{"removed" if gone else ("kept" if ok else "UPLOAD FAILED")} |\n')


def get(rel):
    keys = [o['Key'] for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=PREFIX + rel)
            for o in page.get('Contents', [])]
    for k in keys:
        dst = os.path.join(ROOT, k[len(PREFIX):]); os.makedirs(os.path.dirname(dst), exist_ok=True)
        s3.download_file(BUCKET, k, dst, Config=CFG); print('restored', dst, flush=True)


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'put':
        delete = '--delete' in args
        put([a for a in args if a != '--delete'], delete)
    elif cmd == 'get':
        get(args[0])
    elif cmd == 'ls':
        for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=PREFIX + (args[0] if args else '')):
            for o in page.get('Contents', []):
                print(o['Key'][len(PREFIX):], round(o['Size'] / 2**20), 'MB')
