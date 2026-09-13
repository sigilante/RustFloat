#!/bin/sh
# The sdfloat crate must be a valid oracle-of-record for the VM, so no
# hardware float type may appear anywhere in it, tests included.
# Matches `f32`/`f64` as Rust types or literals/casts; the newtypes are `F32`/`F64`.
set -eu
cd "$(dirname "$0")/.."
if grep -rnE '(\b|[^A-Za-z_])f(32|64)\b|[0-9]\.[0-9]+(e[-+]?[0-9]+)?(f32|f64)?\b' --include='*.rs' crates/sdfloat/src crates/sdfloat/tests \
   | grep -vE 'oracle|sdfloat_oracle|conv::|//|"' ; then
  echo "hardware float type or literal found in sdfloat crate" >&2
  exit 1
fi
echo "ok: no hardware floats in crates/sdfloat/"
