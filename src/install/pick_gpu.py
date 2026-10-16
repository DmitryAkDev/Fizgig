# pick_gpu.py - which AMD GPU Fizgig should train on, for Launch Fizgig AMD.bat.
#
# On a PC with a processor's built-in graphics as well as a graphics card, ROCm can list the built-in one first
# (device 0), and torch then trains on it (#186: "AMD Radeon(TM) Graphics" ahead of an RX 7900 XT). This prints the
# index of the graphics card - not built in, the most VRAM of those - for the launcher to put in HIP_VISIBLE_DEVICES.
# It prints nothing when there is only one GPU, or when device 0 is already the one, so the usual case is untouched.
# Diagnostics go to stderr: the launcher's `for /f` reads stdout only.

import sys


def main():
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception as e:
        print(f"[AMD-ROCm] GPU pick skipped: {e}", file=sys.stderr)
        return
    if n < 2:
        return
    gpus = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        built_in = bool(getattr(p, "is_integrated", False)) or p.name.strip() == "AMD Radeon(TM) Graphics"
        gpus.append((i, p.name, p.total_memory, built_in))
        print(f"[AMD-ROCm] GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GB{' (built in)' if built_in else ''}",
              file=sys.stderr)
    cards = [g for g in gpus if not g[3]] or gpus
    best = max(cards, key=lambda g: g[2])
    if best[0] != 0:
        print(best[0])


if __name__ == "__main__":
    main()
