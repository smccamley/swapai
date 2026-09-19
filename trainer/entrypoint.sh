#!/usr/bin/env bash
set -euo pipefail

mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh
public_key="${SSH_PUBLIC_KEY:-${PUBLIC_KEY:-}}"
if [[ -z "$public_key" ]]; then
  echo "SSH_PUBLIC_KEY is required" >&2
  exit 1
fi
printf '%s\n' "$public_key" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
exec /usr/sbin/sshd -D -e
