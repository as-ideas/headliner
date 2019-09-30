#!/usr/bin/env bash

cp ../README.md docs/index.md
cp ../_readme/ docs/_readme/
cp ../CONTRIBUTING.md docs/CONTRIBUTING.md
cp ../LICENSE docs/LICENSE.md
python autogen.py
mkdir ../docs
mkdocs build -c -d ../docs/