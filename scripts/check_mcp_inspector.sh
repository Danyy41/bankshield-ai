#!/usr/bin/env bash
# Exercise the BankShield MCP server with an independent MCP client: the
# official MCP Inspector CLI (TypeScript SDK), talking to
# scripts/run_mcp_server.py over a real stdio subprocess.
#
# Requires Node.js >= 22.19, jq, and the generated data/model
# (`python scripts/run_all.py`). Usage:
#     scripts/check_mcp_inspector.sh [python-executable]
set -euo pipefail

PYTHON="${1:-python}"
INSPECTOR_VERSION="2.9.0"
cd "$(dirname "$0")/.."

inspect() {
  # The Inspector exits non-zero when a call returns an error, but still
  # prints the result JSON; capture it and assert on the content instead.
  npx -y "@modelcontextprotocol/inspector@${INSPECTOR_VERSION}" --cli \
    "$PYTHON" scripts/run_mcp_server.py "$@" 2>/dev/null || true
}

fail() { echo "[FAIL] $1"; exit 1; }
ok() { echo "[ OK ] $1"; }

tools=$(inspect --method tools/list | jq -r '[.tools[].name] | sort | join(",")')
[ "$tools" = "get_auth_history,get_graph_neighbors,get_risk_score,get_transaction,search_policy" ] \
  || fail "tools/list returned: ${tools:-<nothing>}"
ok "tools/list -> exactly the 5 read-only tools"

inspect --method tools/list | jq -e 'all(.tools[]; .annotations.readOnlyHint == true and .inputSchema.additionalProperties == false)' >/dev/null \
  || fail "tools are not all annotated read-only with strict schemas"
ok "every tool is annotated readOnlyHint and advertises additionalProperties: false"

txn=$("$PYTHON" -c "import sys; sys.path.insert(0, 'src'); from bankshield.investigation import data_access as d; print(d.sample_transactions(n=1)[0]['transaction_id'])")

inspect --method tools/call --tool-name get_risk_score --tool-arg "transaction_id=$txn" \
  | jq -e '(.isError | not) and .structuredContent.risk_score >= 0 and .structuredContent.risk_score <= 1' >/dev/null \
  || fail "get_risk_score did not return a valid score"
ok "tools/call get_risk_score -> valid score"

inspect --method tools/call --tool-name search_policy --tool-arg "query=account takeover" --tool-arg top_k=2 \
  | jq -e '(.isError | not) and (.structuredContent.results | length) >= 1' >/dev/null \
  || fail "search_policy returned no passages"
ok "tools/call search_policy -> policy passages"

inspect --method tools/call --tool-name get_transaction --tool-arg "transaction_id=$txn" --tool-arg auto_approve=true \
  | jq -e '.isError == true and (.content[0].text | contains("Extra inputs are not permitted"))' >/dev/null \
  || fail "smuggled argument was not rejected"
ok "tools/call with a smuggled extra argument -> rejected by the validator"

# create_case isn't listed, so the Inspector refuses it client-side and
# reports the error on stderr; keep stderr for this one call. (The
# server-side refusal for clients that send it anyway is covered by
# tests/test_mcp_server.py.)
out=$(npx -y "@modelcontextprotocol/inspector@${INSPECTOR_VERSION}" --cli \
  "$PYTHON" scripts/run_mcp_server.py --method tools/call --tool-name create_case \
  --tool-arg "transaction_id=$txn" --tool-arg disposition=confirmed_fraud --tool-arg summary=x 2>&1 || true)
echo "$out" | grep -q '"tool_not_found"' || fail "create_case was not refused: $out"
ok "tools/call create_case -> refused (tool_not_found)"

echo
echo "MCP Inspector check passed."
