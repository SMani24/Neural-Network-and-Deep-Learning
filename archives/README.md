# Archives

These are the packaged assignments and datasets retained from the coursework. Notebook copies previously stored at assignment roots are also kept here.

Large compressed files use numbered parts, each under 100 MB. Join the parts in order before opening the archive. For CamVid:

```bash
cat archives/ca3/camvid_main.zip.part* > /tmp/camvid_main.zip
unzip /tmp/camvid_main.zip -d assignments/ca3/q1/Data
```

The archive contains a `CamVid-main/` directory, matching the notebook's relative data path. The working dataset has separate copies of some files; this extraction command restores the copy used by the notebook.

Check the part hashes from their directory:

```bash
cd archives/ca3
sha256sum -c camvid_main.parts.sha256
```

The smaller assignment ZIPs can be opened directly. Their original contents and filenames inside the archives have not been changed.
