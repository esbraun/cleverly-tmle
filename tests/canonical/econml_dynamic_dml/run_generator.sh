#!/bin/sh
# Rebuild the pinned container and rewrite parity_fixture.json inside it.
# Run from any directory.  The library and the test suite never run this script.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
docker build -t cleverly-econml-parity "$here"
docker run --rm -v "$here":/work cleverly-econml-parity generate_parity_fixture.py
