#!/usr/bin/env bash
# gen_cert.sh — generate a self-signed TLS certificate for local dev/testing.
# Run once before `docker compose up`.

set -euo pipefail

CERT_DIR="certificates"
mkdir -p "$CERT_DIR"

openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout "$CERT_DIR/private.key" \
  -out    "$CERT_DIR/certificate.crt" \
  -subj   "/C=EG/ST=Cairo/L=Cairo/O=AEyeGuard/CN=localhost"

echo "✅  Self-signed certificate written to $CERT_DIR/"
echo "    certificate.crt  ← public cert"
echo "    private.key      ← private key"
echo ""
echo "⚠️   For production, replace these with a cert from a trusted CA (e.g. Let's Encrypt)."
