#!/usr/bin/env python3
"""Generate Hoon ground-truth vectors for the sdfloat crate.

Evaluates the un-jetted `++ff` core (not the jetted `++rs` doors) through
`urbit eval` so the results are raw Hoon, per width and rounding mode, and
writes `crates/sdfloat/tests/vectors/hoon_f<W>.txt`. Requires a vere binary:
`URBIT=/path/to/urbit scripts/gen-hoon-vectors.py [pairs-per-mode]`.

Line format (all hex without dots; `?` fields are 1/0; `toi` is a decimal
integer with sign, `big+`/`big-` when |toi| > 2^127, or `none`):
  mode a b n add sub mul div sqt lth lte equ toi sun san
where sun = (sun:ff n) and san = (san:ff -n) for the integer n < 2^63.
"""
import os, random, re, subprocess, sys, pathlib

URBIT = os.environ.get("URBIT", "urbit")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates" / "sdfloat" / "tests" / "vectors"
OUT.mkdir(parents=True, exist_ok=True)

WIDTHS = {  # width: (exp bits, frac bits, bias)
    16: (5, 10, 15),
    32: (8, 23, 127),
    64: (11, 52, 1023),
    128: (15, 112, 16383),
}
MODES = ["n", "u", "d", "z", "a"]
rng = random.Random(20260912)


def sample(w):
    e, f, bias = WIDTHS[w]
    emax = (1 << e) - 1
    sign = rng.getrandbits(1) << (w - 1)
    frac_choice = rng.randrange(8)
    frac = [0, (1 << f) - 1, 1, 1 << (f - 1), ((1 << f) - 1) ^ 1,
            rng.getrandbits(f) & ~((1 << rng.randrange(f)) - 1),
            rng.getrandbits(f), rng.getrandbits(f)][frac_choice]
    field_choice = rng.randrange(16)
    if field_choice == 0: field = 0
    elif field_choice == 1: field = 1
    elif field_choice == 2: field = emax
    elif field_choice == 3: field = emax - 1
    elif field_choice in (4, 5): field = bias
    elif field_choice == 6: field = (bias + rng.randrange(8) - 4) % emax
    elif field_choice == 7: field = min(bias + f + rng.randrange(4), emax - 1)
    elif field_choice == 8: field = rng.randrange(4)
    elif field_choice == 9: field = emax - 1 - rng.randrange(4)
    else: field = rng.randrange(emax + 1)
    bits = (field << f) | frac
    special = rng.randrange(24)
    table = {0: 0, 1: emax << f, 2: (emax << f) | (1 << (f - 1)),
             3: (emax << f) | (1 << (f - 1)) | rng.getrandbits(f), 4: (emax << f) | 1,
             5: 1, 6: (1 << f) - 1, 7: 1 << f, 8: (emax << f) - 1,
             9: bias << f, 10: (bias + 1) << f}
    bits = table.get(special, bits)
    return sign | bits


def pair(w):
    a = sample(w)
    mask = (1 << w) - 1
    c = rng.randrange(8)
    if c == 0: b = a
    elif c == 1: b = a ^ (1 << (w - 1))
    elif c == 2: b = (a + 1) & mask
    elif c == 3: b = (a - 1) & mask
    elif c == 4:
        e, f, _ = WIDTHS[w]
        d = rng.randrange(2 * (f + 1) + 4)
        ef = (a >> f) & ((1 << e) - 1)
        ef2 = max(ef - d, 0)
        b = (a & ~(((1 << e) - 1) << f)) | (ef2 << f) | rng.getrandbits(f)
        b &= mask
    else: b = sample(w)
    return a, b


def integer(w):
    """A 63-bit magnitude, mostly within the width's exactly-representable
    range plus a few bits (where rounding happens), sometimes anything."""
    p = WIDTHS[w][1] + 1
    k = min(63, p + 6)
    c = rng.randrange(8)
    if c == 0: return rng.getrandbits(k) >> rng.randrange(k)
    if c == 1: return max(0, (1 << rng.randrange(k)) - rng.randrange(3))
    if c == 2: return (1 << rng.randrange(p - 1, k)) + rng.randrange(-2, 3)
    if c == 3: return rng.randrange(4096)
    if c == 4: return (1 << 63) - 1
    if c == 5: return rng.getrandbits(63)
    return rng.getrandbits(k)


def hx(v):
    """Hoon @ux literal: lowercase hex, dotted in groups of four from the right."""
    h = format(v, "x")
    groups = []
    while len(h) > 4:
        groups.insert(0, h[-4:])
        h = h[:-4]
    groups.insert(0, h)
    return "0x" + ".".join(groups)


HOON = """=/  fo  ~(. ff [[{e} {f} --{bias}] %{mode}])
=/  ps  ~[{pairs}]
%-  crip
%-  zing
%+  turn  ps
|=  [a=@ b=@ n=@]
^-  tape
;:  weld
  (trip (scot %ux a))  " "  (trip (scot %ux b))  " "  (trip (scot %ux n))  " "
  (trip (scot %ux (add:fo a b)))  " "
  (trip (scot %ux (sub:fo a b)))  " "
  (trip (scot %ux (mul:fo a b)))  " "
  (trip (scot %ux (div:fo a b)))  " "
  (trip (scot %ux (sqt:fo a)))  " "
  ?:((lth:fo a b) "1" "0")  " "
  ?:((lte:fo a b) "1" "0")  " "
  ?:((equ:fo a b) "1" "0")  " "
  =/  u  (toi:fo a)
  ?~  u  "none"
  ?:  (gth (abs:si u.u) (bex 127))
    ?:((syn:si u.u) "big+" "big-")
  (trip (scot %sd u.u))
  " "
  (trip (scot %ux (sun:fo n)))  " "
  (trip (scot %ux (san:fo (new:si | n))))
  ";"
==
"""

ANSI = re.compile(r"\x1b\[[0-9;]*m")

for w, (e, f, bias) in WIDTHS.items():
    lines = []
    for mode in MODES:
        cases = [(*pair(w), integer(w)) for _ in range(N)]
        pairs = " ".join(f"[{hx(a)} {hx(b)} {hx(n)}]" for a, b, n in cases)
        src = HOON.format(e=e, f=f, bias=f"{bias:,}".replace(",", "."), mode=mode, pairs=pairs)
        out = subprocess.run([URBIT, "eval"], input=src, capture_output=True, text=True, check=True).stdout
        out = ANSI.sub("", out)
        m = re.search(r"'(.*)'", out, re.S)
        if not m:
            sys.exit(f"unexpected eval output for f{w} %{mode}:\n{out[-500:]}")
        rows = [r.strip() for r in m.group(1).split(";") if r.strip()]
        if len(rows) != N:
            sys.exit(f"f{w} %{mode}: expected {N} rows, got {len(rows)}")
        for r in rows:
            fields = r.split()
            # strip Hoon's dotted grouping in hex and decimal atoms
            fields = [x.replace(".", "") for x in fields]
            lines.append(f"{mode} " + " ".join(fields))
        print(f"f{w} %{mode}: {len(rows)} rows", file=sys.stderr)
    path = OUT / f"hoon_f{w}.txt"
    path.write_text("# mode a b n add sub mul div sqt lth lte equ toi sun san  (generated by scripts/gen-hoon-vectors.py from un-jetted ++ff via `urbit eval`)\n" + "\n".join(lines) + "\n")
    print(f"wrote {path} ({len(lines)} lines)", file=sys.stderr)
