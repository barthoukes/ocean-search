#!/usr/bin/sh

export TMPDIR="./tmp"
mkdir -p "$TMPDIR"
python3 -m venv backend/venv
. backend/venv/bin/activate  # source backend/venv/bin/activate
pip install -r requirements.txt
rm -rf tmp

