"""
Shared helpers for OptiTrack (Motive) C3D exports of ball-tracking takes.

Motive exports differ from the venue Avatar-Pro exports in ways that break
extract_canonical.py, so this family of tools handles them separately:
  - POINT:UNITS is often missing; units may be meters OR millimeters
  - one labeled rigid-body asset ("<Asset>:Marker_NNN") plus thousands of
    transient "Unlabeled_NNNN" trajectory columns (one column per re-acquisition)
  - files can be huge (25 min @ 360 Hz x 20k columns ~ 180 GB) and are often
    TRUNCATED (export interrupted): header frame count > frames actually stored
  - a fully mocap-coated ball has NO rigid marker template: the asset markers
    are reconstruction points wandering on the ball surface (no usable quats,
    -> no spin measurement from these takes)

Nothing here loads the whole file: probe() reads the header, iter_chunks()
streams fixed-stride frames with retry (external exFAT drives dropping mid-read
otherwise SIGBUS a memmap).
"""
import os
import time
import warnings

import numpy as np

R_BALL = 0.020


def probe(path):
    """Header/geometry probe. Returns a dict; never reads the data section.

    Keys: rate, header_nf, data_nf (from file size -- the trustworthy one),
    truncated, npts, data_offset, frame_stride_bytes, point_bytes, labels,
    analog_per_frame_floats.
    """
    import c3d  # only probe() needs the c3d package

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "rb") as f:
            r = c3d.Reader(f)
            labels = [l.split("\x00")[0].strip() for l in r.point_labels]
            hdr = r.header
            if float(r.point_scale) >= 0:
                raise NotImplementedError(
                    f"{path}: integer-format C3D (POINT:SCALE={r.point_scale} >= 0); "
                    "only float exports are supported")
            npts = int(r.point_used)
            analog_floats = int(r.analog_used) * int(hdr.analog_per_frame) \
                if r.analog_used else 0
            offset = (int(hdr.data_block) - 1) * 512
            stride = 4 * (4 * npts + analog_floats)
            header_nf = int(r.last_frame) - int(r.first_frame) + 1
    fsize = os.path.getsize(path)
    data_nf = (fsize - offset) // stride
    return dict(
        path=path, rate=float(r.point_rate),
        header_nf=header_nf, data_nf=int(data_nf),
        truncated=bool(data_nf < header_nf),
        npts=npts, data_offset=offset, frame_stride_bytes=stride,
        point_bytes=4 * 4 * npts, analog_per_frame_floats=analog_floats,
        labels=labels, file_bytes=fsize,
    )


def iter_chunks(geom, chunk=1024, retries=5, progress=None):
    """Stream (start_frame, points (n, npts, 4) float32) over the data section.

    Plain buffered reads with reopen-and-retry: survives transient I/O errors
    from external drives (a memmap would die with SIGBUS instead).
    """
    path, off = geom["path"], geom["data_offset"]
    npts, stride, nf = geom["npts"], geom["frame_stride_bytes"], geom["data_nf"]
    pb = geom["point_bytes"]
    f = open(path, "rb")
    try:
        for s in range(0, nf, chunk):
            e = min(s + chunk, nf)
            nb = (e - s) * stride
            for attempt in range(retries):
                try:
                    f.seek(off + s * stride)
                    buf = f.read(nb)
                    if len(buf) == nb:
                        break
                except OSError as err:
                    print(f"read error at frame {s} (attempt {attempt + 1}): {err}",
                          flush=True)
                time.sleep(2)
                try:
                    f.close()
                except OSError:
                    pass
                f = open(path, "rb")
            else:
                raise RuntimeError(f"unreadable frames {s}:{e} after {retries} tries")
            raw = np.frombuffer(buf, np.float32).reshape(e - s, stride // 4)
            pts = raw[:, :pb // 4].reshape(e - s, npts, 4)
            if progress and (s // chunk) % 20 == 0:
                progress(e, nf)
            yield s, pts
    finally:
        f.close()


def detect_unit_scale(geom, n_sample_chunks=3):
    """Meters-vs-millimeters from coordinate magnitudes (POINT:UNITS is often
    absent in Motive exports). Returns multiplier to METERS (1.0 or 1e-3)."""
    nf = geom["data_nf"]
    mags = []
    with open(geom["path"], "rb") as f:
        for k in range(n_sample_chunks):
            s = (nf * (2 * k + 1)) // (2 * n_sample_chunks)
            f.seek(geom["data_offset"] + s * geom["frame_stride_bytes"])
            buf = f.read(64 * geom["frame_stride_bytes"])
            n = len(buf) // geom["frame_stride_bytes"]
            if n == 0:
                continue
            raw = np.frombuffer(buf[:n * geom["frame_stride_bytes"]], np.float32)
            raw = raw.reshape(n, geom["frame_stride_bytes"] // 4)
            pts = raw[:, :geom["point_bytes"] // 4].reshape(n, geom["npts"], 4)
            ok = pts[:, :, 3] >= 0
            if ok.any():
                mags.append(np.abs(pts[:, :, :3][ok]).ravel())
    if not mags:
        raise RuntimeError("no valid points found while sampling for unit detection")
    p95 = float(np.percentile(np.concatenate(mags), 95))
    if p95 > 50.0:
        return 1e-3, p95   # millimeters
    return 1.0, p95        # meters


def group_labeled_columns(labels):
    """Split label list into asset-marker groups and unlabeled columns.

    Returns dict asset_name -> ordered column indices (sorted by trailing
    marker number), plus 'unlabeled' -> indices. Columns beyond len(labels)
    (LABELS params exhausted) are treated as unlabeled by the callers."""
    groups = {}
    unlabeled = []
    for i, l in enumerate(labels):
        if l.startswith("Unlabeled") or not l:
            unlabeled.append(i)
            continue
        asset = l.split(":")[0] if ":" in l else l.rsplit("_", 1)[0]
        groups.setdefault(asset, []).append(i)

    def marker_no(i):
        tail = labels[i].rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else i

    return {a: sorted(ix, key=marker_no) for a, ix in groups.items()}, unlabeled
