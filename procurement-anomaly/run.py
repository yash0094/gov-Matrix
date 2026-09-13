#!/usr/bin/env python3
"""One-command start: build the cache if missing, then serve.

    python run.py
    open http://localhost:8000
"""
import subprocess, sys
from pathlib import Path

CACHE = Path(__file__).parent / "data" / "cache" / "metrics.json"

if not CACHE.exists():
    print("no cache found, running the pipeline on synthetic data...")
    subprocess.check_call([sys.executable, "-m", "src.engine.pipeline"])

import uvicorn
uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
