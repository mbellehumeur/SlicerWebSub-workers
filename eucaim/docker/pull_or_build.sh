#!/usr/bin/env bash
# Compatibility wrapper — Harbor pull is no longer used.
# Forwards to local cu128 builds.
exec "$(cd "$(dirname "$0")" && pwd)/build_cu128.sh" "$@"
