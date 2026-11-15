#!/usr/bin/env bash
# Pre-commit/pre-push secret scan. Never prints a secret value - only where it was found.
#
#   bash scripts/check_secrets.sh            # scan staged changes + all tracked files
#
# Two kinds of checks:
#   1. exact values: every secret-looking variable in .env (…KEY / …PASSWORD / …TOKEN / …SECRET) and the
#      token inside .secrets/*.kubeconfig are searched for literally in what git would publish
#   2. patterns: well-known credential shapes (Google API keys, JWTs, private keys, URL credentials)
# Exit code 1 if anything is found.
set -euo pipefail
cd "$(dirname "$0")/.."

PATTERNS=$(mktemp)
trap 'rm -f "$PATTERNS"' EXIT
found=0

# ---- 1. collect exact secret values (kept in a temp file, never echoed)
if [ -f .env ]; then
  grep -E '^[A-Z0-9_]*(KEY|PASSWORD|TOKEN|SECRET)[A-Z0-9_]*=' .env | while IFS='=' read -r name value; do
    value="${value%$'\r'}"; value="${value%\"}"; value="${value#\"}"
    [ ${#value} -ge 8 ] && printf '%s\n' "$value" >> "$PATTERNS"
  done || true
fi
for f in .secrets/*.kubeconfig; do
  [ -f "$f" ] && grep -oE 'token: [^ ]+' "$f" | cut -c8- >> "$PATTERNS" || true
done
n_values=$(wc -l < "$PATTERNS" | tr -d ' ')

scan_values() {  # $1 = description, stdin = text to scan
  if [ "$n_values" -gt 0 ] && grep -qF -f "$PATTERNS"; then
    echo "LEAK: a secret value from .env/.secrets appears in $1"; found=1
  fi
}

# ---- 2. scan what would be committed / pushed
git diff --cached | scan_values "the staged changes"
if [ "$n_values" -gt 0 ]; then
  hits=$(git grep -lF -f "$PATTERNS" -- . 2>/dev/null || true)
  [ -n "$hits" ] && { echo "LEAK: secret values found in tracked files:"; echo "$hits" | sed 's/^/  /'; found=1; }
fi

# generic credential shapes in tracked files (the redaction code that *defines* these patterns is excluded)
generic=$(git grep -lE 'AIza[0-9A-Za-z_-]{35}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.|-----BEGIN [A-Z ]*PRIVATE KEY-----|://[^/:@ ]+:[^/@ ]{6,}@' \
  -- . ':!agent/logging_utils.py' ':!scripts/check_secrets.sh' ':!tests/*' 2>/dev/null || true)
[ -n "$generic" ] && { echo "SUSPICIOUS: credential-shaped strings in:"; echo "$generic" | sed 's/^/  /'; found=1; }

# files that must never be tracked
tracked=$(git ls-files -- .env '.secrets/*' '*.kubeconfig' 'serving/model/*' || true)
[ -n "$tracked" ] && { echo "LEAK: sensitive files are tracked by git:"; echo "$tracked" | sed 's/^/  /'; found=1; }

if [ "$found" -eq 0 ]; then
  echo "OK: no secrets found (checked $n_values secret value(s) + credential patterns, staged diff + tracked files)"
fi
exit "$found"
