"""Repackage en_core_web_trf to work with spacy-transformers 1.4+ and spacy 3.8+.

Loads the installed en_core_web_trf model, converts serialized weights from
torch pickle to safetensors format, and writes a new spaCy model package
named en_core_web_hftrf.
"""
import json
import shutil
from pathlib import Path

import spacy
from spacy.util import get_package_path

SRC_VERSION = "3.6.1"
NEW_VERSION = "3.8.1"
PKG_NAME = "en_core_web_hftrf"
META_NAME = "core_web_hftrf"
SPACY_COMPAT = ">=3.8.0,<3.9.0"
SPACY_TRF_COMPAT = ">=1.4.0,<1.5.0"

src_path = get_package_path("en_core_web_trf")
model_dir = src_path / f"en_core_web_trf-{SRC_VERSION}"

# Layout: out_root/setup.cfg, out_root/en_core_web_hftrf/__init__.py, etc.
out_root = Path("dist") / f"{PKG_NAME}-{NEW_VERSION}"
pkg_dir = out_root / PKG_NAME
data_dir = pkg_dir / f"{PKG_NAME}-{NEW_VERSION}"

if out_root.exists():
    shutil.rmtree(out_root)

# Copy the full model directory into the package
shutil.copytree(model_dir, data_dir)

# --- Convert transformer weights to safetensors ---
print("Loading model to convert transformer weights...")
nlp = spacy.load(model_dir)
trf = nlp.get_pipe("transformer")
model_bytes = trf.model.to_bytes()
print(f"  Serialized transformer model: {len(model_bytes)} bytes")
(data_dir / "transformer" / "model").write_bytes(model_bytes)

# --- Update meta.json ---
meta = json.loads((data_dir / "meta.json").read_text())
meta["name"] = META_NAME
meta["version"] = NEW_VERSION
meta["spacy_version"] = SPACY_COMPAT
meta["requirements"] = [f"spacy-transformers{SPACY_TRF_COMPAT}"]
(data_dir / "meta.json").write_text(json.dumps(meta, indent=2))

# --- Write package __init__.py ---
init_py = f'''"""{PKG_NAME} model, repackaged for spacy-transformers {SPACY_TRF_COMPAT}."""
from pathlib import Path
from spacy.util import load_model_from_init_py, get_model_meta

__version__ = get_model_meta(Path(__file__).parent / "{PKG_NAME}-{NEW_VERSION}")["version"]

def load(**overrides):
    return load_model_from_init_py(__file__, **overrides)
'''
(pkg_dir / "__init__.py").write_text(init_py)

# spacy.load looks for meta.json next to __init__.py
shutil.copy(data_dir / "meta.json", pkg_dir / "meta.json")

# --- Write build files at project root ---
setup_cfg = f"""[metadata]
name = {PKG_NAME}
version = {NEW_VERSION}
description = English transformer pipeline (roberta-base), repackaged for spacy-transformers 1.4+
license = MIT

[options]
zip_safe = false
include_package_data = true
packages = {PKG_NAME}
python_requires = >=3.11
install_requires =
    spacy{SPACY_COMPAT}
    spacy-transformers{SPACY_TRF_COMPAT}
"""
(out_root / "setup.cfg").write_text(setup_cfg)

pyproject_toml = """[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
"""
(out_root / "pyproject.toml").write_text(pyproject_toml)

manifest = f"""recursive-include {PKG_NAME} *
"""
(out_root / "MANIFEST.in").write_text(manifest)

print(f"\nRepackaged model written to: {out_root}")
print(f"To build a wheel: cd {out_root} && python -m build --wheel")
print(f"To load: spacy.load('{PKG_NAME}')")
