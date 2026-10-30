"""Turns holes into screw sites. Plain Python (no numpy), so it runs inside Fusion too.

A hole is a dict: p0 (one end on the axis, m), axis (unit), d (m), depth (m), link, body, and
optionally csk_end ('lo' = at p0, 'hi' = at the other end) and csk_d (m) for a countersink.
``run(holes, cast)`` stacks coaxial touching holes, probes each stack's two mouths with
``cast(origin_m, direction) -> (distance_m, (link, body)) | None`` and classifies the stacks.
"""

import math
from collections import defaultdict

STACK_ANG = math.cos(math.radians(2))
STACK_OFF = 0.0003   # max offset between coaxial hole axes [m] (more for larger holes, see below)
STACK_GAP = 0.0012   # max axial gap between consecutive holes [m]
OPEN = 3e-3          # a mouth with nothing within this is open
TOUCH = 8e-4         # gaps below this count as seated (some interfaces have modeled shims)

# ISO 10642 countersunk socket screws: head Ø dk, head height k (mm)
COUNTERSUNK = {3: (6.72, 1.86), 4: (8.96, 2.48), 5: (11.2, 3.1), 6: (13.44, 3.72), 8: (17.92, 4.96)}
# ISO 273 clearance (fine, medium, coarse), tap drill, ISO 4762 lengths; mm
CLEARANCE = {2: (2.2, 2.4, 2.6), 2.5: (2.7, 2.9, 3.1), 3: (3.2, 3.4, 3.6), 4: (4.3, 4.5, 4.8),
             5: (5.3, 5.5, 5.8), 6: (6.4, 6.6, 7.0), 8: (8.4, 9.0, 10.0)}
TAP = {2: 1.6, 2.5: 2.05, 3: 2.5, 4: 3.3, 5: 4.2, 6: 5.0, 8: 6.8}
LENGTHS = [3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100]

# Mating interfaces of bought-in parts whose threads the CAD leaves out. maxon datasheets
# (global.maxongroup.com/high-efficiency-joints): HEJ 90-48-140 rev 2.3, HEJ 70-48-60 rev 1.2,
# HEJ 50-48-30 rev 1.1. Fields: bolt circle Ø, thread, count, thread depth (None: not specified).
INTERFACES = [
    ("HEJ 90 output", 50.0, 6, 12, 14.0),
    ("HEJ 90 housing", 97.0, 6, 8, None),
    ("HEJ 70 output", 39.0, 4, 10, 11.0),
    ("HEJ 70 housing", 79.0, 4, 6, 16.0),
    ("HEJ 50 output", 31.0, 4, 8, 9.0),
    ("HEJ 50 housing", 64.0, 4, 6, None),
]
# Dowel pins on the actuator outputs: (pin circle Ø, pin Ø)
PINS = [(52.0, 3.0), (40.0, 3.0), (32.5, 2.5)]


# --- vectors ----------------------------------------------------------------------------------

def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def mul(a, k):
    return tuple(x * k for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    return math.sqrt(dot(a, a))


def radial(v, n):
    """Distance of offset ``v`` from the line along unit ``n``."""
    return norm(sub(v, mul(n, dot(v, n))))


def perpendicular(a):
    ref = (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)
    x = cross(ref, a)
    return mul(x, 1.0 / norm(x))


def fit_circle(points, n):
    """Circle through points lying in planes normal to unit ``n``: (center, radius, max residual)."""
    u = perpendicular(n)
    v = cross(n, u)
    c0 = mul(points[0], 1.0)
    xy = [(dot(sub(p, c0), u), dot(sub(p, c0), v)) for p in points]
    # least squares for x^2 + y^2 = 2ax + 2by + c via the 3x3 normal equations
    rows = [(2 * x, 2 * y, 1.0, x * x + y * y) for x, y in xy]
    m = [[sum(r[i] * r[j] for r in rows) for j in range(3)] + [sum(r[i] * r[3] for r in rows)] for i in range(3)]
    for i in range(3):  # Gauss-Jordan with partial pivoting
        p = max(range(i, 3), key=lambda k: abs(m[k][i]))
        m[i], m[p] = m[p], m[i]
        if abs(m[i][i]) < 1e-30:
            return c0, 0.0, float("inf")
        for k in range(3):
            if k != i:
                f = m[k][i] / m[i][i]
                m[k] = [a - f * b for a, b in zip(m[k], m[i])]
    a, b, c = (m[i][3] / m[i][i] for i in range(3))
    r = math.sqrt(max(c + a * a + b * b, 0.0))
    center = add(c0, add(mul(u, a), mul(v, b)))
    res = max(abs(radial(sub(p, center), n) - r) for p in points)
    return center, r, res


# --- stacks -----------------------------------------------------------------------------------

def build_stacks(holes):
    parent = list(range(len(holes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # only parallel holes can stack: grid by link and |axis| components, compare neighboring cells
    cells = defaultdict(list)
    for i, h in enumerate(holes):
        cells[(h["link"],) + tuple(int(abs(x) / 0.1) for x in h["axis"])].append(i)
    near = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for key, idx in cells.items():
        for a in idx:
            ha, n = holes[a], holes[a]["axis"]
            others = [b for o in near for b in cells.get((key[0],) + tuple(c + d for c, d in zip(key[1:], o)), ())]
            for b in others:
                if b <= a:
                    continue
                hb = holes[b]
                if abs(dot(n, hb["axis"])) < STACK_ANG:
                    continue
                d = sub(hb["p0"], ha["p0"])
                # misaligned holes still form a joint (and get flagged): up to 60 % of the smaller radius
                if radial(d, n) > max(STACK_OFF, 0.3 * min(ha["d"], hb["d"])):
                    continue
                s = dot(d, n)
                tb = sorted((s, s + dot(hb["axis"], n) * hb["depth"]))
                if tb[0] - ha["depth"] <= STACK_GAP and -tb[1] <= STACK_GAP:
                    parent[find(a)] = find(b)

    groups = defaultdict(list)
    for i in range(len(holes)):
        groups[find(i)].append(i)
    stacks = []
    for members in groups.values():
        ref = max((holes[i] for i in members), key=lambda h: h["d"])  # head side: the widest hole
        n, o = ref["axis"], ref["p0"]
        segs, misalign = [], 0.0
        for i in members:
            h = holes[i]
            d = sub(h["p0"], o)
            misalign = max(misalign, radial(d, n))
            s = dot(d, n)
            e = s + dot(h["axis"], n) * h["depth"]
            seg = dict(body=h["body"], d=h["d"] * 1e3, t0=min(s, e) * 1e3, t1=max(s, e) * 1e3)
            if "csk_end" in h:  # stored as a position along the stack axis, so flipping keeps it
                seg.update(csk_t=(s if h["csk_end"] == "lo" else e) * 1e3, csk_d=h["csk_d"] * 1e3)
            segs.append(seg)
        segs.sort(key=lambda x: x["t0"])
        stacks.append(dict(link=ref["link"], origin=o, axis=n, segs=segs, misalign_mm=misalign * 1e3,
                           bodies=sorted({x["body"] for x in segs})))
    return stacks


# --- classification ---------------------------------------------------------------------------

def clearance_size(d):
    err, m = min((abs(d - c), m) for m, cs in CLEARANCE.items() for c in cs)
    return m if err <= 0.25 else None


def tap_size(d):
    err, m = min((abs(d - t), m) for m, t in TAP.items())
    return m if err <= 0.12 else None


def std_length(lo, hi, ideal=None):
    """Standard length in [lo, hi] closest to ``ideal`` (default: hi), else the shortest one >= lo."""
    lo, hi = round(lo, 1) - 0.05, round(hi, 1) + 0.05
    ideal = hi if ideal is None else round(ideal, 1)  # ties go to the longer screw
    fit = [L for L in LENGTHS if lo <= L <= hi]
    if fit:
        return min(fit, key=lambda L: (abs(L - ideal), -L))
    return next((L for L in LENGTHS if L >= lo), LENGTHS[-1])


def tapped_from_lo(segs):
    """True if the far end is a tap drill narrower than the entry hole."""
    return len(segs) > 1 and segs[-1]["d"] < segs[0]["d"] - 0.4 and tap_size(segs[-1]["d"]) is not None


def mouth(stack, hit):
    if hit is None or hit[0] >= OPEN:
        return "air"
    t, (link, body) = hit
    if link == stack["link"] and body in stack["bodies"]:
        return "self"
    return "face" if t < TOUCH else "near"


def flip(seg):
    """The segment seen from the other end of the stack."""
    out = dict(seg, t0=-seg["t1"], t1=-seg["t0"])
    if "csk_t" in seg:
        out["csk_t"] = -seg["csk_t"]
    return out


def head_type(seg):
    """'countersunk' if the hole the head goes into opens into a countersink, else 'socket'."""
    if "csk_t" in seg and abs(seg["csk_t"] - seg["t0"]) < 0.3:
        return "countersunk"
    return "socket"


def rounded(v):
    return [round(float(x), 6) for x in v]


def one_body_blind(st, lo):
    """A clearance-size blind hole ending inside its own body: parts merged into one solid (gusset
    screwed into a plate edge), drawn as one hole. The screw fills the hole."""
    segs = st["segs"]
    d, depth = segs[0]["d"], segs[-1]["t1"] - segs[0]["t0"]
    note = "blind hole ends inside the same body (merged parts?); length fills the hole"
    size = clearance_size(d)
    if tap_size(d) is not None:
        # a deep tap drill from the surface of a merged part: its clamped plate was drawn without
        # a clearance hole (shallow ones are open threads for parts not in the CAD)
        size = tap_size(d) if depth >= 3 * tap_size(d) else None
        note = (f"Ø{d:.1f} tap drill runs through the clamped plate: it needs a "
                f"Ø{CLEARANCE[size][1] if size else 0:g} clearance hole there, fix the CAD")
    if len(segs) > 1 or size is None or depth < 2 * size:
        return None
    axis = st["axis"] if lo == "air" else mul(st["axis"], -1)
    head = add(st["origin"], mul(st["axis"], (segs[0]["t0"] if lo == "air" else segs[-1]["t1"]) * 1e-3))
    return dict(link=st["link"], bodies=st["bodies"][0], head_body=st["bodies"][0], mates="",
                size=f"M{size:g}", head=head_type(segs[0] if lo == "air" else flip(segs[-1])),
                length=std_length(depth * 0.6, depth - 0.5), len_max_mm=round(depth - 0.5, 2),
                kind="one-body-blind", confidence="medium",
                clamp_mm=round(depth, 2), hole_d_mm=round(d, 2), pattern=f"{d:.1f}x{depth:.1f}",
                note=note, head_m=rounded(head), axis=rounded(axis))


def classify(st):
    """A screw site for the stack, or None if it does not look like one."""
    segs = st["segs"]
    lo, hi = mouth(st, st["hit_lo"]), mouth(st, st["hit_hi"])
    multi = len(st["bodies"]) > 1
    if min(x["t1"] - x["t0"] for x in segs) < 0.8 or min(x["d"] for x in segs) > 9.5:
        return None
    rev = [flip(x) for x in reversed(segs)]
    merged_tapped = tapped_from_lo(segs) if lo == "air" else tapped_from_lo(rev)
    if not multi and sorted((lo, hi)) == ["air", "self"] and not merged_tapped:
        return one_body_blind(st, lo)
    if not multi and not merged_tapped and not ("face" in (lo, hi) and ({"air", "near"} & {lo, hi})):
        return None  # lone hole that does not sit on another part
    if multi and lo in ("self", "face") and hi in ("self", "face"):
        return None  # enclosed on both sides

    # head goes on the clearance side of a tapped stack, else on the open end, else on the wider end
    if tapped_from_lo(segs) != tapped_from_lo(rev):
        head_lo = tapped_from_lo(segs)
    elif lo == hi or "air" not in (lo, hi):
        head_lo = segs[0]["d"] >= segs[-1]["d"]
    else:
        head_lo = lo == "air"
    axis = st["axis"] if head_lo else mul(st["axis"], -1)
    if not head_lo:
        segs = rev
    far = st["hit_hi"] if head_lo else st["hit_lo"]
    blank_end = (hi if head_lo else lo) == "face"
    start = segs[0]["t0"]
    notes = []
    head_kind = head_type(segs[0])
    play = (segs[0]["d"] - (clearance_size(segs[0]["d"]) or segs[0]["d"])) / 2
    if st.get("misalign_mm", 0) > max(play, 0.05):
        notes.append(f"holes misaligned {st['misalign_mm']:.2f} mm (clearance play {play:.2f} mm): "
                     "screw will not pass, fix the CAD")
    if blank_end and far[0] > 3e-4:
        notes.append(f"{far[0] * 1e3:.2f} mm gap to mating face")

    # counterbore: wide entry of the same body directly followed by a narrower hole
    if len(segs) > 1 and segs[0]["d"] > 1.5 * segs[1]["d"] and segs[0]["body"] == segs[1]["body"]:
        start = segs[0]["t1"]
        notes.append(f"counterbore Ø{segs[0]['d']:.1f}")
        segs = segs[1:]

    last = segs[-1]
    if tapped_from_lo(segs):
        size, kind = tap_size(last["d"]), "tapped"
        conf = "high" if segs[0]["d"] >= size else "low"
        clamp = last["t0"] - start
        engage = last["t1"] - last["t0"] - (0 if blank_end else 0.5)
        length, len_max = std_length(clamp + min(size, engage), clamp + engage), clamp + engage
        if engage < size:
            notes.append(f"short thread engagement {engage:.1f} mm")
    else:
        size = clearance_size(segs[0]["d"])
        conf = "medium"
        if size is None:
            size, conf = round(segs[0]["d"]), "low"
            notes.append(f"Ø{segs[0]['d']:.1f} is not an ISO 273 clearance; size guessed")
        clamp = segs[-1]["t1"] - start
        if blank_end:
            kind = "thread-not-modeled"
            length, len_max = std_length(clamp + size, clamp + 2 * size, clamp + 1.5 * size), clamp + 2 * size
        else:
            kind, conf = "through", "low"
            notes.append("no thread found: nut, heat-set insert or dowel pin?")
            length, len_max = std_length(clamp + 0.8 * size + 1.0, clamp + 0.8 * size + 3.0), clamp + 0.8 * size + 3
    if any(h and h[1][0] != st["link"] for h in (st["hit_lo"], st["hit_hi"])):
        notes.append("mating part is in another link")
    head = add(st["origin"], mul(axis, start * 1e-3))
    mates = "/".join(far[1]) if blank_end else ""
    return dict(link=st["link"], bodies=" ".join(st["bodies"]), head_body=segs[0]["body"], mates=mates,
                size=f"M{size:g}", length=length, head=head_kind, len_max_mm=round(len_max, 2),
                kind=kind, confidence=conf, clamp_mm=round(clamp, 2), hole_d_mm=round(segs[0]["d"], 2),
                pattern=" | ".join(f"{x['d']:.1f}x{x['t1'] - x['t0']:.1f}" for x in segs),
                note="; ".join(notes), head_m=rounded(head), axis=rounded(axis))


def apply_interfaces(sites):
    """Size sites that screw into a known actuator interface from its datasheet; drop pin holes and
    holes that no actuator thread lines up with."""
    by_body = defaultdict(list)
    for s in sites:
        if not s["mates"]:
            continue
        axis = tuple(s["axis"])
        s["_face"] = add(s["head_m"], mul(axis, s["clamp_mm"] * 1e-3))
        line = mul(axis, 1.0 if axis[max(range(3), key=lambda k: abs(axis[k]))] > 0 else -1.0)
        by_body[(s["mates"], tuple(round(x, 2) for x in line))].append(s)
    groups = []  # sites on one mating face: same body, same axis, face planes within 0.5 mm
    for (_, line), members in by_body.items():
        members.sort(key=lambda s: dot(s["_face"], line))
        groups.append([members[0]])
        for prev, s in zip(members, members[1:]):
            if dot(sub(s["_face"], prev["_face"]), line) > 5e-4:
                groups.append([])
            groups[-1].append(s)
    for group in groups:
        n = tuple(group[0]["axis"])
        matched = None
        for name, circle, thread, count, depth in INTERFACES:
            pts = [s["_face"] for s in group if s["size"] == f"M{thread:g}"]
            while len(pts) >= max(3, count // 2):
                center, r, res = fit_circle(pts, n)
                if res < 2e-4:
                    break
                pts.pop(max(range(len(pts)), key=lambda i: abs(radial(sub(pts[i], center), n) - r)))
            else:
                continue
            if abs(2e3 * r - circle) < 0.3:
                matched = (name, circle, thread, count, depth, center)
                break
        if matched is None:
            continue
        name, circle, thread, count, depth, center = matched
        on_ring = 0
        for s in group:
            dia = 2e3 * radial(sub(s["_face"], center), n)
            pin = next((p for c, p in PINS if abs(dia - c) < 0.3 and p <= s["hole_d_mm"] <= p + 1.2), None)
            if pin:
                s.update(kind="dowel-pin", confidence="excluded", note=f"{name}: Ø{pin:g} h6 dowel pin hole")
            elif abs(dia - circle) > 0.3 or s["size"] != f"M{thread:g}":
                s.update(kind="no-thread", confidence="excluded",
                         note=f"faces {name} but no M{thread:g} thread there per datasheet")
            else:
                on_ring += 1
                clamp = s["clamp_mm"]
                if depth is not None:
                    s["length"] = std_length(clamp + min(thread, depth - 0.5), clamp + depth - 0.5)
                    s["len_max_mm"] = round(clamp + depth - 0.5, 2)
                    spec = f"{name}: M{thread:g}, {depth:g} deep"
                else:
                    spec = f"{name}: M{thread:g}, depth not specified"
                s.update(kind="actuator-thread", confidence="high",
                         note="; ".join([spec] + [x for x in s["note"].split("; ") if "gap" in x]))
        if on_ring > count:
            for s in group:
                if s["kind"] == "actuator-thread":
                    s["confidence"] = "medium"
                    s["note"] += f"; {on_ring} holes face this interface but it has only {count} threads"
    for s in sites:
        s.pop("_face", None)


def countersunk_length(site):
    """Countersunk screws are ordered by overall length: shank under the head plus the head."""
    if site["head"] != "countersunk":
        return
    size = float(site["size"][1:])
    if size not in COUNTERSUNK:
        site["head"] = "socket"
        site["note"] = "; ".join(filter(None, [site["note"], "countersunk hole, no ISO 10642 size: socket head"]))
        return
    k, clamp, hi = COUNTERSUNK[size][1], site["clamp_mm"], site["len_max_mm"]
    site["length"] = std_length(k + clamp + min(size, hi - clamp), k + hi, k + min(clamp + 1.5 * size, hi))
    site["note"] = "; ".join(filter(None, ["countersunk (ISO 10642)", site["note"]]))


def run(holes, cast, progress=None):
    """Screw sites (with ids) for the holes; ``cast`` probes what each stack's mouths open onto."""
    holes = [h for h in holes if h["depth"] > 1e-4]  # zero-length cylinders are artifacts
    stacks = build_stacks(holes)
    for i, st in enumerate(stacks):
        o, n = st["origin"], st["axis"]
        a, b = st["segs"][0]["t0"] * 1e-3, max(x["t1"] for x in st["segs"]) * 1e-3
        st["hit_lo"] = cast(add(o, mul(n, a + 1e-5)), mul(n, -1))
        st["hit_hi"] = cast(add(o, mul(n, b - 1e-5)), n)
        if progress:
            progress(i + 1, len(stacks))
    sites = [s for s in map(classify, stacks) if s]
    apply_interfaces(sites)
    for s in sites:
        countersunk_length(s)
    sites.sort(key=lambda s: (s["link"], float(s["size"][1:]), s["length"], s["head_m"]))
    for i, s in enumerate(sites, 1):
        s["id"] = i
    return stacks, sites
