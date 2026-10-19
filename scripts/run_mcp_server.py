"""Launch the BankShield AI MCP server over stdio.

Exposes the five read-only investigation tools (see
src/bankshield/mcp_server.py). Requires the generated data and model:
run `python scripts/run_all.py` first.

Usage (normally launched by an MCP client, not by hand):
    python scripts/run_mcp_server.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankshield.mcp_server import main

if __name__ == "__main__":
    main()
