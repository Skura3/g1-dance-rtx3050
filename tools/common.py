"""Paths shared by the local, staged video-to-G1 pipeline."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
GVHMR = ROOT / 'third_party/GVHMR'
GMR = ROOT / 'third_party/GMR'
SUPERMIMIC = ROOT / 'third_party/SuperMimic-Reproduced'
MJLAB = ROOT / 'third_party/mjlab'
CHECKPOINTS = GVHMR / 'inputs/checkpoints'

def add_sources():
    source=os.environ.get('G1_SUPERMIMIC_SOURCE','original')
    if source not in ('original','nju'):raise ValueError('G1_SUPERMIMIC_SOURCE must be original or nju')
    selected=SUPERMIMIC if source=='original' else ROOT/'third_party/SuperMimic-Repro-Sim2Real'
    if not (selected/'src/supermimic').is_dir():raise FileNotFoundError(selected)
    for p in (GVHMR, GMR, selected / 'src'):
        if p.exists():
            sys.path.insert(0, str(p))

def require_file(path, purpose='required input'):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'{purpose}: {path}. See docs/DOWNLOADS.md')
    return path

