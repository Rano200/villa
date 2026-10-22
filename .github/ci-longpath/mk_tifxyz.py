"""Write a tiny tifxyz (12x12 plane, float32 x/y/z.tif + meta.json) with the stdlib only."""
import json
import os
import struct
import sys

out = sys.argv[1]
n = 12
os.makedirs(out, exist_ok=True)


def write_tif(path, values):
    data = struct.pack("<%df" % (n * n), *values)
    entries = [
        (256, 4, 1, n),  # ImageWidth
        (257, 4, 1, n),  # ImageLength
        (258, 3, 1, 32),  # BitsPerSample
        (259, 3, 1, 1),  # Compression none
        (262, 3, 1, 1),  # Photometric min-is-black
        (273, 4, 1, 8),  # StripOffsets
        (277, 3, 1, 1),  # SamplesPerPixel
        (278, 4, 1, n),  # RowsPerStrip
        (279, 4, 1, len(data)),  # StripByteCounts
        (339, 3, 1, 3),  # SampleFormat IEEE float
    ]
    ifd_off = 8 + len(data)
    ifd = struct.pack("<H", len(entries))
    for tag, typ, cnt, val in entries:
        ifd += struct.pack("<HHII", tag, typ, cnt, val)
    ifd += struct.pack("<I", 0)
    with open(path, "wb") as f:
        f.write(b"II*\x00" + struct.pack("<I", ifd_off) + data + ifd)


xs = [100.0 + c for r in range(n) for c in range(n)]
ys = [200.0 + r for r in range(n) for c in range(n)]
zs = [300.0 for r in range(n) for c in range(n)]
write_tif(os.path.join(out, "x.tif"), xs)
write_tif(os.path.join(out, "y.tif"), ys)
write_tif(os.path.join(out, "z.tif"), zs)
with open(os.path.join(out, "meta.json"), "w") as f:
    json.dump({"format": "tifxyz", "type": "seg", "uuid": "longpath-ci",
               "scale": [1.0, 1.0], "bbox": [[100, 200, 300], [111, 211, 300]]}, f)
