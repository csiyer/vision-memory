#!/bin/bash
# Download the full THINGS object-images set (1854 concepts x ~12 photos, ~5 GB)
# from the THINGSplus OSF project (jum2f) and extract into the shared lab folder.
#
# License (from OSF): research / non-commercial use only, no redistribution.
# Re-runs are safe: curl resumes a partial download; unzip skips files that
# already exist on disk (-n).

set -euo pipefail

DEST="${THINGS_DEST:-/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/data/THINGS}"
ZIP_URL="https://osf.io/download/rdxy2/"
PASSWORD_URL="https://osf.io/download/j6a3m/"
ZIP_PATH="$DEST/images_THINGS.zip"
EXTRACT_DIR="$DEST/object_images_full"

mkdir -p "$DEST" "$EXTRACT_DIR"

echo "[1/3] Fetching extraction password from OSF..."
PW_FILE="$(mktemp)"
trap 'rm -f "$PW_FILE"' EXIT
curl -sSL "$PASSWORD_URL" -o "$PW_FILE"
PASSWORD="$(grep -oE 'things[A-Za-z0-9]+' "$PW_FILE" | tail -1)"
if [ -z "$PASSWORD" ]; then
    echo "ERROR: could not parse password from $PASSWORD_URL" >&2
    exit 1
fi

echo "[2/3] Downloading images_THINGS.zip (5.0 GB) to $ZIP_PATH ..."
curl -L --fail --retry 5 --retry-delay 10 -C - -o "$ZIP_PATH" "$ZIP_URL"

echo "[3/3] Extracting into $EXTRACT_DIR ..."
# System unzip 6.00 (2009) flags this archive's overlapping component layout as a
# zip bomb and aborts mid-extract, so use Python's zipfile, which has no such
# heuristic. Skip-if-exists makes re-runs safe.
ZIP_PATH="$ZIP_PATH" EXTRACT_DIR="$EXTRACT_DIR" PASSWORD="$PASSWORD" python3 - <<'PY'
import os, zipfile
zip_path = os.environ["ZIP_PATH"]
dest     = os.environ["EXTRACT_DIR"]
password = os.environ["PASSWORD"].encode()

extracted = skipped = 0
with zipfile.ZipFile(zip_path) as zf:
    zf.setpassword(password)
    for name in zf.namelist():
        target = os.path.join(dest, name)
        if name.endswith("/"):
            os.makedirs(target, exist_ok=True)
            continue
        if os.path.exists(target) and os.path.getsize(target) > 0:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with zf.open(name) as src, open(target, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        extracted += 1
        if extracted % 1000 == 0:
            print(f"  extracted={extracted} skipped={skipped}", flush=True)
print(f"  extracted={extracted} skipped={skipped}")
PY

echo "Done. Top-level entries in $EXTRACT_DIR:"
ls "$EXTRACT_DIR" | head -10
echo "Total subdirs: $(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 2 -type d | wc -l)"