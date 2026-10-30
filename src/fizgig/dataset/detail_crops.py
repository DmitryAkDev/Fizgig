"""Detail crops, for a style LoRA: one square crop of each training picture at the picture's own scale, so brushwork,
canvas grain and texture train 1:1 instead of being shrunk away with the whole picture. They live in the training
folder's fizgig_detail_crops/<size>mp/ folder (one per Target Megapixels that has trained), each crop named as its
picture (its caption and its cropped mask beside it), and train alongside the pictures themselves, which keep the
composition. Switched on per training folder (fizgig_detail_crops.json, written by Image Prep's Detail crops); made at
Start for the size the run trains at (families/launch.py's Detail crops stage), so changing Target Megapixels or
Start at a smaller size never trains a crop shrunk.

The spot is random but fixed per picture (seeded from its name), so every size of one picture's crop shares a centre,
and a re-run makes the same crops (their captions and caches are kept). A spot that falls on a nearly flat area (a plain
sky) moves to the picture's most detailed one. No torch: Pillow and numpy only."""

import hashlib
import json
import os
import random

DIR = "fizgig_detail_crops"            # the folder in a training folder; Fizgig's own, so sub-folders skip it
SETTINGS = "fizgig_detail_crops.json"   # its switch, in the training folder (fizgig_*.json travels with a move)
MANIFEST = ".crops.json"               # per size folder: each crop's source picture and box
PICS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
RULE = 3                                # the spot rule's version: a crop made by an older one is made again
FLAT = 0.4                              # a spot with under 40% of the detail of the picture's busiest spot counts as flat


def is_on(folder) -> bool:
    try:
        with open(os.path.join(folder, SETTINGS), encoding="utf-8") as f:
            return bool(json.load(f).get("on"))
    except (OSError, ValueError):
        return False


def switch(folder, on=True):
    with open(os.path.join(folder, SETTINGS), "w", encoding="utf-8") as f:
        json.dump({"on": bool(on)}, f)


def side(mp) -> int:
    """The crop's side at a Target Megapixels: the dataset TOML's resolution (a square of that area on the 16 grid)."""
    return int((float(mp) * 1_000_000) ** 0.5) // 16 * 16


def size_name(mp) -> str:
    return f"{float(mp):g}mp"


def size_dir(folder, mp) -> str:
    return os.path.join(folder, DIR, size_name(mp))


def sizes(folder):
    """[(label, path)] of the crop sizes made so far, smallest first."""
    root = os.path.join(folder, DIR)
    try:
        names = [n for n in os.listdir(root) if n.endswith("mp") and os.path.isdir(os.path.join(root, n))]
    except OSError:
        return []
    def mp(n):
        try:
            return float(n[:-2])
        except ValueError:
            return 0.0
    return [(f"{mp(n):g} MP", os.path.join(root, n)) for n in sorted(names, key=mp) if mp(n) > 0]


def pictures(folder):
    from fizgig.dataset.masks import is_mask_file
    try:
        return sorted(n for n in os.listdir(folder) if n.lower().endswith(PICS) and not is_mask_file(n)
                      and os.path.isfile(os.path.join(folder, n)))
    except OSError:
        return []


def source(folder, name):
    """The picture to crop from: its full-size original when Image Prep kept one (originals/, originals_2/... - the
    biggest), else the training picture itself."""
    stem = os.path.splitext(name)[0].lower()
    best, area = os.path.join(folder, name), -1
    try:
        dirs = [d for d in os.listdir(folder) if d.lower() == "originals" or d.lower().startswith("originals_")]
    except OSError:
        dirs = []
    for d in dirs:
        try:
            for f in os.listdir(os.path.join(folder, d)):
                if os.path.splitext(f)[0].lower() == stem and f.lower().endswith(PICS):
                    from PIL import Image
                    with Image.open(os.path.join(folder, d, f)) as im:
                        a = im.size[0] * im.size[1]
                    if a > area:
                        best, area = os.path.join(folder, d, f), a
        except OSError:
            continue
    return best


def _detail(gray):
    """A detail map: the gradient size of a small grey copy (texture, strokes, edges are high; a flat sky low)."""
    import numpy as np
    g = np.asarray(gray, dtype=np.float32)
    return np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1, :]


def centre(name, w, h, gray=None):
    """The crop's centre in a picture of w x h, the same for every crop size: random, seeded from the file name; a
    nearly flat spot (judged in a window the size of the smallest crop) moves to the most detailed one (gray: the
    picture's grey copy, for that test)."""
    rnd = random.Random(int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:12], 16))
    cx, cy = rnd.random() * w, rnd.random() * h
    if gray is None:
        return cx, cy
    k = gray.width / w                                     # the grey copy's scale
    d = _detail(gray)
    r = min(side(0.25), w, h)                              # the window the test looks through
    win = max(2, int(r * k))

    def at(x, y):                                          # the detail in the window around (x, y)
        a = int(min(max(0, y - r / 2), h - r) * k)
        b = int(min(max(0, x - r / 2), w - r) * k)
        return float(d[a:a + win, b:b + win].mean()) or 1e-6
    best, bx, by = -1.0, cx, cy
    for i in range(7):                                     # the most detailed of a 7 x 7 grid of spots
        for j in range(7):
            x, y = r / 2 + i / 6 * (w - r), r / 2 + j / 6 * (h - r)
            v = at(x, y)
            if v > best:
                best, bx, by = v, x, y
    # flat next to the picture's busiest part (not its average: a dark background can be most of a portrait)
    return (cx, cy) if at(cx, cy) >= FLAT * best else (bx, by)


def spot(cx, cy, w, h, s):
    """The top-left corner of a crop of side s around (cx, cy); near an edge it slides in to fit."""
    return min(max(0, round(cx - s / 2)), w - s), min(max(0, round(cy - s / 2)), h - s)


def uncaptioned(d, ext=".txt"):
    """The crops in one size folder with no caption yet."""
    from fizgig.dataset.masks import is_mask_file
    try:
        names = os.listdir(d)
    except OSError:
        return []
    have = {os.path.splitext(n)[0] for n in names if n.endswith(ext)}
    return [os.path.join(d, n) for n in sorted(names) if n.lower().endswith(".png") and not is_mask_file(n)
            and os.path.splitext(n)[0] not in have]


def share_captions(folder, ext=".txt"):
    """A crop with no caption gets the same picture's caption from another size, the largest that has one: every
    size is cut around the same spot, so one caption fits them all. Returns how many were copied."""
    import shutil
    dirs = [d for _label, d in sizes(folder)][::-1]       # largest first
    n = 0
    for d in dirs:
        for c in uncaptioned(d, ext):
            stem = os.path.splitext(os.path.basename(c))[0]
            src = next((os.path.join(o, stem + ext) for o in dirs if os.path.isfile(os.path.join(o, stem + ext))), "")
            if src:
                shutil.copyfile(src, os.path.join(d, stem + ext))
                n += 1
    return n


def to_caption(folder, ext=".txt"):
    """The crops a captioner should write after share_captions: each picture with no caption at any size, once, at
    the largest size it has a crop (share_captions then copies it to the rest)."""
    out, seen = [], set()
    for _label, d in sizes(folder)[::-1]:                 # largest first
        for c in uncaptioned(d, ext):
            stem = os.path.splitext(os.path.basename(c))[0]
            if stem not in seen:
                seen.add(stem)
                out.append(c)
    return out


def make(folder, mp, log=print, ext=".txt"):
    """The crops for one size, made where missing or out of date (the picture changed). A picture smaller than the
    crop has none (it trains whole). A crop whose picture left the folder goes, with its caption. Returns the crops with
    no caption yet."""
    from PIL import Image
    from fizgig.dataset.masks import mask_for_stem
    from fizgig.utils.upright import open_upright
    s = side(mp)
    out = size_dir(folder, mp)
    os.makedirs(out, exist_ok=True)
    mpath = os.path.join(out, MANIFEST)
    try:
        with open(mpath, encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError):
        man = {}
    pics = pictures(folder)
    stems = {os.path.splitext(n)[0] for n in pics}
    for stem in [k for k in man if k not in stems]:        # its picture is gone: so is its crop
        for n in os.listdir(out):
            if os.path.splitext(n)[0] in (stem, stem + "-masklabel"):
                try:
                    os.remove(os.path.join(out, n))
                except OSError:
                    pass
        man.pop(stem, None)
    made = skipped = 0
    for n in pics:
        stem = os.path.splitext(n)[0]
        src = source(folder, n)                            # its full-size original when Image Prep kept one
        mt = os.path.getmtime(src)
        dst = os.path.join(out, stem + ".png")
        if man.get(stem, {}).get("mtime") == mt and man[stem].get("rule") == RULE and os.path.isfile(dst):
            continue
        try:
            im = open_upright(src).convert("RGB")
        except Exception as e:
            log(f"[detail crops] {n}: can't read it ({e})")
            continue
        w, h = im.size
        if min(w, h) < s:                                  # smaller than the crop: it trains whole, no crop
            skipped += 1
            man.pop(stem, None)
            continue
        gray = im.convert("L")
        gray.thumbnail((256, 256))
        x, y = spot(*centre(stem, w, h, gray), w, h, s)
        im.crop((x, y, x + s, y + s)).save(dst)
        mk = mask_for_stem(folder, stem)
        if mk:                                             # masked training: the mask cut at the same spot
            try:
                m = Image.open(mk)
                if m.size != (w, h):
                    m = m.resize((w, h))
                m.crop((x, y, x + s, y + s)).save(os.path.join(out, stem + "-masklabel.png"))
            except Exception:
                pass
        if man.get(stem) and man[stem].get("box") != [x, y, s]:   # a new crop: its old caption no longer fits
            try:
                os.remove(os.path.join(out, stem + ext))
            except OSError:
                pass
        man[stem] = {"mtime": mt, "box": [x, y, s], "source": n, "rule": RULE}
        made += 1
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(man, f)
    if made or skipped:
        log(f"[detail crops] {size_name(mp)}: {made} made" + (f", {skipped} picture(s) smaller than "
            f"{s}x{s} train whole" if skipped else ""))
    return [os.path.join(out, stem + ".png") for stem in sorted(man)
            if not os.path.isfile(os.path.join(out, stem + ext))]
