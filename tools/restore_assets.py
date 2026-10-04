#!/usr/bin/env python3
"""Restore the original dataset/checkpoint paths from the published split archives."""
import argparse
import bisect
import collections
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'archives/assets_manifest.json'


def git_copy(path, output):
    proc = subprocess.Popen(['git', 'show', 'HEAD:' + path], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    shutil.copyfileobj(proc.stdout, output, 4 * 1024 * 1024)
    error = proc.stderr.read().decode(errors='replace')
    if proc.wait():
        raise RuntimeError(error.strip())


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return json.loads(subprocess.check_output(
        ['git', 'show', 'HEAD:archives/assets_manifest.json'], cwd=ROOT))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class PartsReader(io.RawIOBase):
    """Seek across numbered parts, with a bounded cache for sparse checkouts."""
    def __init__(self, parts):
        self.parts = parts
        self.offsets = [0]
        for part in parts:
            self.offsets.append(self.offsets[-1] + part['size'])
        self.position = 0
        self.handles = collections.OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.offsets[-1] + offset
        if position < 0:
            raise ValueError('Negative seek')
        self.position = position
        return position

    def handle(self, index):
        if index in self.handles:
            self.handles.move_to_end(index)
            return self.handles[index]
        part = self.parts[index]
        path = ROOT / part['path']
        if path.exists():
            handle = path.open('rb')
        else:
            handle = tempfile.TemporaryFile()
            git_copy(part['path'], handle)
            if handle.tell() != part['size']:
                handle.close()
                raise RuntimeError('Incorrect Git part size: ' + part['path'])
            handle.seek(0)
        self.handles[index] = handle
        while len(self.handles) > 2:
            _, old = self.handles.popitem(last=False)
            old.close()
        return handle

    def read(self, size=-1):
        remaining = max(0, self.offsets[-1] - self.position)
        size = remaining if size < 0 else min(size, remaining)
        result = bytearray()
        while size:
            index = bisect.bisect_right(self.offsets, self.position) - 1
            offset = self.position - self.offsets[index]
            count = min(size, self.parts[index]['size'] - offset)
            handle = self.handle(index)
            handle.seek(offset)
            data = handle.read(count)
            if len(data) != count:
                raise RuntimeError('Truncated archive part')
            result.extend(data)
            self.position += count
            size -= count
        return bytes(result)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        super().close()


class Restorer:
    def __init__(self, manifest, destination):
        self.manifest = manifest
        self.destination = destination.resolve()
        self.done = set()
        self.restored = 0
        self.active = set()

    def target(self, path):
        target = (self.destination / path).resolve()
        target.relative_to(self.destination)
        return target

    def present(self, path):
        record = self.manifest['files'][path]
        target = self.target(path)
        if not target.exists():
            return False
        if not target.is_file() or target.stat().st_size != record['size'] or digest(target) != record['sha256']:
            raise RuntimeError('Existing file differs; refusing to overwrite: ' + path)
        self.done.add(path)
        return True

    def write(self, path, stream):
        target = self.target(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        size = 0
        temporary = target.with_name(target.name + '.restoring')
        if temporary.exists():
            raise RuntimeError('Temporary restoration file already exists: ' + str(temporary))
        try:
            with temporary.open('xb') as output:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    output.write(block)
                    h.update(block)
                    size += len(block)
            expected = self.manifest['files'][path]
            if size != expected['size'] or h.hexdigest() != expected['sha256']:
                raise RuntimeError('Restoration checksum mismatch: ' + path)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        self.done.add(path)
        self.restored += 1

    def source(self, path):
        source = ROOT / path
        if source.exists():
            return source
        if path in self.manifest['files']:
            self.ensure(path)
            return self.target(path)
        # Small archive containers already tracked directly in Git.
        source = self.target(path)
        source.parent.mkdir(parents=True, exist_ok=True)
        with source.open('wb') as output:
            git_copy(path, output)
        return source

    def ensure(self, path):
        if path in self.done or self.present(path):
            return
        if path in self.active:
            raise RuntimeError('Cyclic restoration reference: ' + path)
        self.active.add(path)
        try:
            rep = self.manifest['files'][path]['storage']
            if 'copy' in rep:
                origin = rep['copy']
                with self.source(origin).open('rb') as stream:
                    self.write(path, stream)
            elif 'archive' in rep:
                source = self.source(rep['archive'])
                if rep['kind'] == 'zip':
                    with zipfile.ZipFile(source) as archive, archive.open(rep['member']) as stream:
                        self.write(path, stream)
                else:
                    with tarfile.open(source, 'r:gz') as archive:
                        with archive.extractfile(rep['member']) as stream:
                            self.write(path, stream)
            else:
                package = self.manifest['packages'][rep['package']]
                with PartsReader(package['parts']) as reader:
                    if package['format'] == 'raw':
                        self.write(path, reader)
                    else:
                        wanted = {name for name in package['files'] if name not in self.done}
                        with tarfile.open(fileobj=reader, mode='r|xz') as archive:
                            for member in archive:
                                if member.name in wanted:
                                    if not self.present(member.name):
                                        with archive.extractfile(member) as stream:
                                            self.write(member.name, stream)
                                    wanted.remove(member.name)
                        if wanted:
                            raise RuntimeError('Missing archive members: ' + str(wanted))
        finally:
            self.active.remove(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment', choices=['ca3', 'ca4', 'ca5', 'ca6', 'cae'])
    parser.add_argument('--list', action='store_true', help='show assignment sizes without restoring')
    parser.add_argument('--verify', action='store_true', help='verify existing working files only')
    parser.add_argument('--destination', type=Path, default=ROOT)
    args = parser.parse_args()
    manifest = load_manifest()
    files = {p: v for p, v in manifest['files'].items()
             if not args.assignment or p.startswith(('assignments/' + args.assignment + '/', 'archives/' + args.assignment + '/'))}
    if args.list:
        totals = collections.defaultdict(lambda: [0, 0])
        for path, record in files.items():
            key = path.split('/')[1]
            totals[key][0] += 1
            totals[key][1] += record['size']
        for key, (count, size) in sorted(totals.items()):
            print(f'{key}: {count:,} files, {size / 1e9:.2f} GB restored')
        return
    restorer = Restorer(manifest, args.destination)
    for index, path in enumerate(files, 1):
        if args.verify:
            if not restorer.present(path):
                raise RuntimeError('Missing working file: ' + path)
        else:
            restorer.ensure(path)
        if index % 1000 == 0:
            print(f'Checked {index:,}/{len(files):,} files', flush=True)
    print(f'Checked {len(files):,} files; restored {restorer.restored:,}. SHA-256 verification passed.')


if __name__ == '__main__':
    main()
