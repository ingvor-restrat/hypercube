#!/usr/bin/env python3
"""Render the deterministic terminal examples into committed GIFs."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RECORDINGS = [
    (
        "etf",
        Path(__file__).with_name("etf.gif"),
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "hypercube-engine",
            "--example",
            "etf",
            "--",
            "--record",
            "--ticks",
            "28",
            "--entities",
            "160",
            "--funds",
            "12",
            "--top",
            "5",
            "--interval-ms",
            "0",
            "--seed",
            "335342",
        ],
    ),
    (
        "pairs",
        Path(__file__).with_name("pairs.gif"),
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "hypercube-engine",
            "--example",
            "pairs",
            "--",
            "--record",
            "--ticks",
            "28",
            "--pairs",
            "24",
            "--top",
            "10",
            "--interval-ms",
            "0",
            "--seed",
            "335341",
        ],
    ),
    (
        "circuit",
        Path(__file__).with_name("circuit-replay.gif"),
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "hypercube-circuit",
            "--example",
            "circuit",
            "--",
            "--record",
            "--ticks",
            "28",
            "--top",
            "8",
            "--interval-ms",
            "0",
        ],
    ),
]

ANSI = re.compile(r"\x1b\[([0-9;]*)m")
REGULAR_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
BACKGROUND = "#07111d"
PANEL = "#0b1725"
CHROME = "#111f30"
DEFAULT = "#d8e2ec"


def xterm(index: int) -> tuple[int, int, int]:
    basic = [
        (0, 0, 0),
        (205, 49, 49),
        (13, 188, 121),
        (229, 229, 16),
        (36, 114, 200),
        (188, 63, 188),
        (17, 168, 205),
        (229, 229, 229),
        (102, 102, 102),
        (241, 76, 76),
        (35, 209, 139),
        (245, 245, 67),
        (59, 142, 234),
        (214, 112, 214),
        (41, 184, 219),
        (255, 255, 255),
    ]
    if index < 16:
        return basic[index]
    if index < 232:
        index -= 16
        red, green, blue = index // 36, index // 6 % 6, index % 6
        channel = lambda value: 0 if value == 0 else 55 + value * 40
        return channel(red), channel(green), channel(blue)
    shade = 8 + (index - 232) * 10
    return shade, shade, shade


def spans(line: str):
    position = 0
    color: tuple[int, int, int] | str = DEFAULT
    bold = False
    dim = False
    for match in ANSI.finditer(line):
        if match.start() > position:
            yield line[position : match.start()], color, bold, dim
        codes = [int(code) for code in match.group(1).split(";") if code] or [0]
        cursor = 0
        while cursor < len(codes):
            code = codes[cursor]
            if code == 0:
                color, bold, dim = DEFAULT, False, False
            elif code == 1:
                bold = True
            elif code == 2:
                dim = True
            elif code == 22:
                bold, dim = False, False
            elif 30 <= code <= 37:
                color = xterm(code - 30)
            elif 90 <= code <= 97:
                color = xterm(code - 90 + 8)
            elif code == 38 and cursor + 2 < len(codes) and codes[cursor + 1] == 5:
                color = xterm(codes[cursor + 2])
                cursor += 2
            elif code == 39:
                color = DEFAULT
            cursor += 1
        position = match.end()
    if position < len(line):
        yield line[position:], color, bold, dim


def extent(frame: str) -> tuple[int, int]:
    clean_lines = frame.strip("\n").splitlines()
    visible = [ANSI.sub("", line) for line in clean_lines]
    return max(map(len, visible)), len(visible)


def render(
    frame: str,
    regular,
    bold_font,
    label: str,
    columns: int,
    rows: int,
) -> Image.Image:
    clean_lines = frame.strip("\n").splitlines()
    probe = Image.new("RGB", (1, 1))
    probe_draw = ImageDraw.Draw(probe)
    char_width = probe_draw.textlength("M", font=regular)
    line_height = 23
    left, top, chrome = 30, 47, 30
    width = int(left * 2 + columns * char_width)
    height = top + rows * line_height + 27
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, width - 8, height - 8), 14, fill=PANEL)
    draw.rounded_rectangle((8, 8, width - 8, 8 + chrome), 14, fill=CHROME)
    draw.rectangle((8, 8 + chrome - 12, width - 8, 8 + chrome), fill=CHROME)
    for offset, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        x = 27 + offset * 20
        draw.ellipse((x, 18, x + 10, 28), fill=color)
    draw.text((width - 172, 15), f"{label} · synthetic", font=regular, fill="#71849a")

    for row, line in enumerate(clean_lines):
        x = left
        y = top + row * line_height
        for text, color, is_bold, is_dim in spans(line):
            chosen = bold_font if is_bold else regular
            if is_dim and isinstance(color, tuple):
                color = tuple(int(channel * 0.68) for channel in color)
            draw.text((x, y), text, font=chosen, fill=color)
            x += char_width * len(text)
    return image


def record(
    label: str,
    output: Path,
    command: list[str],
    regular,
    bold_font,
) -> None:
    process = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    frames = [
        frame.strip("\n")
        for frame in process.stdout.decode("utf-8").split("\x0c")
        if frame.strip()
    ]
    if not frames:
        raise RuntimeError(f"{label} produced no recording frames")
    extents = [extent(frame) for frame in frames]
    columns = max(item[0] for item in extents)
    rows = max(item[1] for item in extents)
    images = [
        render(frame, regular, bold_font, label, columns, rows) for frame in frames
    ]
    images.extend([images[-1]] * 6)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=130,
        loop=0,
        disposal=2,
        optimize=True,
    )
    print(f"wrote {output.relative_to(ROOT)} ({len(frames)} source frames)")


def main() -> None:
    selected = set(sys.argv[1:])
    known = {label for label, _, _ in RECORDINGS}
    unknown = selected - known
    if unknown:
        raise SystemExit(f"unknown recording: {', '.join(sorted(unknown))}")
    regular = ImageFont.truetype(REGULAR_FONT, 15)
    bold_font = ImageFont.truetype(BOLD_FONT, 15)
    for label, output, command in RECORDINGS:
        if selected and label not in selected:
            continue
        record(label, output, command, regular, bold_font)


if __name__ == "__main__":
    main()
