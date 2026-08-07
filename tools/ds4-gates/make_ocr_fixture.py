#!/usr/bin/env python3
"""Generate the deterministic OCR test image used by arch_serve_gate.sh in image mode.

⚠ GENERATED, NOT DOWNLOADED, AND THAT IS THE POINT. A fixture pulled off the internet can change,
disappear, or carry text nobody read. This one is 268 bytes, reproduced exactly by this script, and
every glyph in it is drawn here in the open. When the arbiter decides whether a paged run is correct,
the arbiter has to be the most trusted object in the room.

Deterministic: no randomness, no timestamps, no compression-level ambiguity. Byte-identical on rerun.
"""
import struct
import sys
import zlib

W, H = 320, 96
FONT = {  # 5x7, only the glyphs this fixture needs
    'P': ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    'A': ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    'R': ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    'I': ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    'S': ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    ' ': ["00000"] * 7,
}


def render(text="PARIS", ox=40, oy=24, scale=8):
    px = [[255] * W for _ in range(H)]
    for ci, ch in enumerate(text):
        glyph = FONT.get(ch, FONT[' '])
        for r in range(7):
            for c in range(5):
                if glyph[r][c] == '1':
                    for dy in range(scale):
                        for dx in range(scale):
                            y, x = oy + r * scale + dy, ox + (ci * 6) * scale + c * scale + dx
                            if 0 <= y < H and 0 <= x < W:
                                px[y][x] = 0
    return px


def png_bytes(px):
    raw = b''.join(b'\x00' + bytes(row) for row in px)

    def chunk(tag, data):
        body = tag + data
        return struct.pack('>I', len(data)) + body + struct.pack('>I', zlib.crc32(body) & 0xffffffff)

    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 0, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 9))
            + chunk(b'IEND', b''))


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'paris.png'
    data = png_bytes(render())
    open(out, 'wb').write(data)
    print("wrote %s: %dx%d, %d bytes, the word PARIS" % (out, W, H, len(data)))
