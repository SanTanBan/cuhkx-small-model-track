"""
Convert the `# %%` cell-marked scripts in kaggle/ into .ipynb files that can be
imported straight into Kaggle (New Notebook -> File -> Import Notebook).

Keeps the source readable/diffable as plain Python while still shipping real
notebooks.

    python scripts/make_notebooks.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "kaggle")
OUT = os.path.join(ROOT, "notebooks")

CELL_RE = re.compile(r"^# %%(?:\s*\[(\w+)\])?\s*$")
# `#!writefile cuhkx.py` on its own line expands to a %%writefile cell holding
# that file's contents, so the notebooks stay self-contained on Kaggle while the
# library remains a single editable source of truth here.
INCLUDE_RE = re.compile(r"^#!writefile\s+(\S+)\s*$")


def expand_includes(lines):
    for i, line in enumerate(lines):
        m = INCLUDE_RE.match(line)
        if m:
            target = m.group(1)
            with open(os.path.join(SRC, target), "r", encoding="utf-8") as f:
                body = f.read().splitlines()
            return [f"%%writefile {target}"] + body
    return lines


def split_cells(text):
    cells, kind, buf = [], "code", []
    for line in text.splitlines():
        m = CELL_RE.match(line)
        if m:
            if buf:
                cells.append((kind, buf))
            kind = "markdown" if m.group(1) == "markdown" else "code"
            buf = []
        else:
            buf.append(line)
    if buf:
        cells.append((kind, buf))
    return cells


def to_source(lines):
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return [l + "\n" for l in lines[:-1]] + (lines[-1:] if lines else [])


def build(path, out_path):
    with open(path, "r", encoding="utf-8") as f:
        cells = split_cells(f.read())

    nb_cells = []
    for kind, lines in cells:
        if kind == "markdown":
            # strip the leading "# " comment prefix from markdown cells
            lines = [re.sub(r"^# ?", "", l) for l in lines]
        else:
            lines = expand_includes(lines)
        src = to_source(lines)
        if not src or not "".join(src).strip():
            continue
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {}, "source": src,
                             "execution_count": None, "outputs": []})

    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"{os.path.basename(path):<28} -> {os.path.relpath(out_path, ROOT)}  "
          f"({len(nb_cells)} cells)")


def main():
    srcs = sorted(f for f in os.listdir(SRC) if f.startswith("nb_") and f.endswith(".py"))
    if not srcs:
        print("no nb_*.py found in kaggle/")
        return
    for f in srcs:
        build(os.path.join(SRC, f),
              os.path.join(OUT, f[:-3].replace("nb_", "") + ".ipynb"))


if __name__ == "__main__":
    main()
