# Archives

Packaged assignments and datasets are kept here, along with notebook copies previously stored at assignment roots. Original ZIP contents have been left intact.

## Datasets and checkpoints

The working files are published through `assets_manifest.json` and numbered archive parts in `assets/`. Parts are at most 95 MB. The manifest records the original paths and SHA-256 hashes. Byte-identical copies and files already contained in existing archives reuse those contents.

From the repository root, list the available data and restore one assignment:

```bash
python3 tools/restore_assets.py --list
python3 tools/restore_assets.py --assignment ca6
```

Omit `--assignment` to restore everything. Allow enough space for the sizes shown by `--list`. Existing files are verified and left untouched; a differing file causes an error instead of being overwritten. Use `--verify` to check the working files without restoring anything.

Archive parts can be omitted from a local sparse checkout to save space. The restoration tool can read missing parts from the local Git object store. A regular clone includes them.

## Opening archives manually

Join numbered parts in order before opening a compressed file. For CamVid:

```bash
cat archives/ca3/camvid_main.zip.part* > /tmp/camvid_main.zip
unzip /tmp/camvid_main.zip -d assignments/ca3/q1/Data
```

Check the CamVid part hashes from their directory:

```bash
cd archives/ca3
sha256sum -c camvid_main.parts.sha256
```

The smaller assignment ZIPs open directly. Newly packaged working files use standard tar/XZ archives; after joining their parts, extract from the repository root with `tar -xJf archive.tar.xz` to recover their original paths.
