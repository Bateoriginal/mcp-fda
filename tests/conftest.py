"""Shared fixtures."""
import sys
from pathlib import Path

# Make repo root importable so tests can `import fda_mcp_server`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
