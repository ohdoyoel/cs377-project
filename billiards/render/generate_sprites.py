"""Generate ball sprite PNGs deterministically.

Run via: ``uv run python billiards/render/generate_sprites.py``

Produces 128x128 RGBA PNGs with anti-aliased filled circles, a subtle
dark rim, and a soft off-center highlight (light from the upper-left).

Color palette follows competitive Korean 4-ball / 3-cushion convention:
white + yellow as the two cue balls, two reds as objects.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SPRITE_SIZE = 128       # display-time size
SUPERSAMPLE = 4         # render at 4x and downscale for AA

# Korean carom set, calibrated against typical aragonite/phenolic ball look.
BALL_COLORS: dict[str, tuple[int, int, int]] = {
    "white":  (240, 232, 215),   # ivory white
    "yellow": (242, 198,  46),   # carom yellow
    "red":    (196,  46,  44),   # carom red
}


def _draw_ball(rgb: tuple[int, int, int]) -> Image.Image:
    big = SPRITE_SIZE * SUPERSAMPLE
    pad = 2

    # 1. solid colored disk on transparent canvas
    base = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(base).ellipse(
        (pad, pad, big - pad - 1, big - pad - 1),
        fill=(*rgb, 255),
    )

    # 2. soft highlight: a small white ellipse, blurred, masked to disk.
    hl = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    cx, cy = int(big * 0.36), int(big * 0.32)
    r = int(big * 0.20)
    ImageDraw.Draw(hl).ellipse(
        (cx - r, cy - r, cx + r, cy + r),
        fill=(255, 255, 255, 170),
    )
    hl = hl.filter(ImageFilter.GaussianBlur(radius=big * 0.05))

    # mask highlight by the ball silhouette
    silhouette = Image.new("L", (big, big), 0)
    ImageDraw.Draw(silhouette).ellipse(
        (pad, pad, big - pad - 1, big - pad - 1), fill=255
    )
    hl_a = hl.split()[3]
    new_a = Image.eval(silhouette, lambda v: v)  # copy
    # multiply hl alpha by silhouette alpha
    new_a = Image.merge("L", [Image.eval(
        hl_a, lambda v: v
    )]).point(lambda v: v)
    # explicit pixel-wise multiply:
    sa = silhouette.load()
    ha = hl_a.load()
    out_a = Image.new("L", (big, big), 0)
    oa = out_a.load()
    for y in range(big):
        for x in range(big):
            oa[x, y] = (sa[x, y] * ha[x, y]) // 255
    hl.putalpha(out_a)

    # 3. rim shadow: thin dark outline blurred slightly
    rim = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    rim_w = max(3, big // 80)
    ImageDraw.Draw(rim).ellipse(
        (pad, pad, big - pad - 1, big - pad - 1),
        outline=(0, 0, 0, 100),
        width=rim_w,
    )
    rim = rim.filter(ImageFilter.GaussianBlur(radius=big * 0.008))

    # composite: colored disk → rim → highlight (highlight should be above rim)
    out = Image.alpha_composite(base, rim)
    out = Image.alpha_composite(out, hl)

    return out.resize((SPRITE_SIZE, SPRITE_SIZE), Image.LANCZOS)


def _draw_yellow_with_dot() -> Image.Image:
    """Yellow cue ball with a small red dot — disambiguates the second
    cue ball, mirroring real carom sets where one of the white balls
    has a dot. Optional; not used in current viewer."""
    img = _draw_ball(BALL_COLORS["yellow"])
    # add red dot
    big = SPRITE_SIZE
    d = ImageDraw.Draw(img)
    rdot = max(4, big // 16)
    cx, cy = int(big * 0.62), int(big * 0.62)
    d.ellipse((cx - rdot, cy - rdot, cx + rdot, cy + rdot),
              fill=(196, 46, 44, 255))
    return img


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "sprites"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rgb in BALL_COLORS.items():
        sprite = _draw_ball(rgb)
        sprite.save(out_dir / f"ball_{name}.png", format="PNG")
        # Sanity print: center pixel RGB
        cx = cy = sprite.size[0] // 2
        print(
            f"wrote {out_dir / f'ball_{name}.png'}  "
            f"size={sprite.size}  center={sprite.getpixel((cx, cy))}"
        )


if __name__ == "__main__":
    main()
