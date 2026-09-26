#!/usr/bin/env python3
"""Build the RP2 b12x tree plus EPI-PAR, a bit-identical speed change to the small-M FC1 epilogue.

Today the direct FC1 kernels (route_hoist_k5 for M1, route_hoist_direct32 for M2-16) quantize the
down input with one thread per row. The thread walks its row's owned_n/32 blocks one after another,
and with RP it runs the lo pass for each block too. Direct tiles hold one route (valid_rows=1), so a
single thread does all of that while the rest of the CTA waits at the closing barrier.

EPI-PAR gives each (row, 32-block) pair its own thread. The per-block code is the same text with
`tid` replaced by the row index: same shared-memory loads, same max order, same quantizer, same
stores. The outputs must therefore be bit-identical to the serial path, with or without RP/RP2;
the smoke test checks that before any timing is read.

Switch: B12X_P8_EPI_PAR=1, captured at P8NativeTPMoE construction and added to the compile spec.
Off (the default) leaves the traced kernels unchanged.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DST = Path(os.environ.get("EPIPAR_DST", str(HERE.parent / "b12x-rp2par")))
os.environ["DX2_DST"] = str(DST)
sys.path.insert(0, str(HERE))
import build_dx2_rp2_b12x as base  # noqa: E402  (the RP2 builder; named build_dx2_b12x.py in the testing workspace)

START = ("        elif tid < valid_rows:\n"
         "            scale_word = Uint32(0)\n"
         "            for local_block in cutlass.range_constexpr(self.owned_n // 32):\n"
         "                block = Int32(local_block) + subtile * Int32(self.owned_n // 32)\n")
END = "        cute.arch.sync_threads()\n"
HEAD = ("        elif cutlass.const_expr(getattr(self, \"p8_epi_par\", False)):\n"
        "            # EPI-PAR: one thread per (row, 32-block) instead of one thread per row.\n"
        "            # Per-block math is the serial path's text with tid -> epi_row, so outputs are bit-identical.\n"
        "            if tid < valid_rows * Int32(self.owned_n // 32):\n"
        "                epi_row = tid >> Int32((self.owned_n // 32).bit_length() - 1)\n"
        "                block = (tid & Int32(self.owned_n // 32 - 1)) + subtile * Int32(self.owned_n // 32)\n"
        "                scale_word = Uint32(0)\n")


def epipar(text: str, name: str) -> str:
    if text.count(START) != 1:
        sys.exit(f"{name}: serial epilogue anchor found {text.count(START)} times")
    i = text.index(START)
    j = text.index(END, i)
    body = text[i + len(START):j]
    for line in body.splitlines():
        if line.strip() and not line.startswith(" " * 16):
            sys.exit(f"{name}: unexpected indentation in the serial epilogue body: {line!r}")
    if "local_block" in body:
        sys.exit(f"{name}: serial epilogue body uses local_block")
    uses = len(re.findall(r"\btid\b", body))
    new_body = re.sub(r"\btid\b", "epi_row", body)
    print(f"{name}: serial body {len(body.splitlines())} lines, tid uses replaced {uses}")
    return text[:i] + HEAD + new_body + text[i:]


def main() -> None:
    base.DST = DST
    base.main()
    k = DST / base.K
    # EPIPAR_FILES selects the FC1 kernels that get the parallel branch (default: M1 and M2-16 direct kernels).
    for name in os.environ.get("EPIPAR_FILES", "route_hoist_k5.py,route_hoist_direct32.py").split(","):
        p = k / name
        p.write_text(epipar(p.read_text(), name))

    p = DST / base.T / "p8_native_kernel.py"
    s = p.read_text()
    s = base.rep(s, "        self.p8_input_rowpack = os.environ.get(\"B12X_P8_DOWN_REMAINDER\") == \"rp2\"\n",
                 "        self.p8_input_rowpack = os.environ.get(\"B12X_P8_DOWN_REMAINDER\") == \"rp2\"\n"
                 "        # EPI-PAR: bit-identical parallel small-M FC1 epilogue; joins the compile spec below.\n"
                 "        self.p8_epi_par = os.environ.get(\"B12X_P8_EPI_PAR\") == \"1\"\n"
                 "        if self.p8_epi_par:\n"
                 "            print(f\"P8_EPI_PAR_ACTIVE layer={layer} rank={tp_rank}\", flush=True)\n",
                 1, "p8_native_kernel.py")
    s = base.rep(s, "                fc1.p8_warp_quant = self.fc1_warp_quant\n",
                 "                fc1.p8_warp_quant = self.fc1_warp_quant\n"
                 "                fc1.p8_epi_par = self.p8_epi_par\n",
                 1, "p8_native_kernel.py")
    s = base.rep(s, "                (\"dx2_input_rowpack\", int(self.p8_input_rowpack and small_m)),\n",
                 "                (\"dx2_input_rowpack\", int(self.p8_input_rowpack and small_m)),\n"
                 "                (\"epi_par\", int(self.p8_epi_par and small_m)),\n",
                 1, "p8_native_kernel.py")
    p.write_text(s)
    print(f"built {DST} (RP2 + EPI-PAR switch)")


if __name__ == "__main__":
    main()
