#!/usr/bin/env python3
"""KRONA — rocket launch from Earth to the Moon and flag deployment.

Run:                    python3 krona_moon.py
Banded colors?          python3 krona_moon.py --256
Jump to finale:         python3 krona_moon.py --end
Exit: Ctrl+C. Works best in Windows Terminal, window size >= 100x30.
"""
import colorsys
import math
import os
import random
import shutil
import sys
import time

FPS = 24
USE_256 = "--256" in sys.argv or os.environ.get("TERM_PROGRAM") == "Apple_Terminal"

# --- Timeline (seconds) ---
T_FADE = 2.0        # Dawn fade-in on the launchpad
T_IGN = 1.8         # Main engine ignition
T_LIFT = 3.5        # Liftoff from Earth
T_MAXQ = 11.8       # Max-Q shockwaves and vapor cone
T_SEP = 17.0        # Side booster separation in orbit
T_TLI = 18.2        # Trans-lunar injection (Earth recedes)
T_MOON_IN = 21.5    # Moon rises into view
T_BRAKE = 25.0      # Pitch-up and landing burn
T_LAND = 29.8       # Soft touchdown on the lunar surface
T_EVA = 31.0        # Hatch opens, astronaut descends ladder
T_FLAG = 34.5       # Flagpole planted in lunar regolith
T_END = 38.0        # Flag unfurled, KRONA illuminated

WORD = "KRONA"
FONT = {
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "N": ["#...#", "##..#", "#.#.#", "#.#.#", "#.#.#", "#..##", "#...#"],
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
}

if os.name == "nt":
    os.system("")  # Enable ANSI escape sequences in Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_rgb(c1, c2, t):
    t = clamp(t)
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def add_rgb(c1, c2, k=1.0):
    return (
        min(255, max(0, int(c1[0] + c2[0] * k))),
        min(255, max(0, int(c1[1] + c2[1] * k))),
        min(255, max(0, int(c1[2] + c2[2] * k))),
    )


def hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, clamp(s), clamp(v))
    return (int(r * 255), int(g * 255), int(b * 255))


def esc_fg(rgb):
    r, g, b = rgb
    if USE_256:
        return f"\x1b[38;5;{16 + 36 * round(r / 51) + 6 * round(g / 51) + round(b / 51)}m"
    return f"\x1b[38;2;{r};{g};{b}m"


def esc_bg(rgb):
    r, g, b = rgb
    if USE_256:
        return f"\x1b[48;5;{16 + 36 * round(r / 51) + 6 * round(g / 51) + round(b / 51)}m"
    return f"\x1b[48;2;{r};{g};{b}m"


def term_size():
    s = shutil.get_terminal_size((100, 30))
    return max(50, s.columns), max(18, s.lines)


class Scene:
    def __init__(self):
        self.size = None

    def setup(self, w, rows):
        self.size = (w, rows)
        self.H = rows * 2  # Half-block vertical resolution
        R = random.Random(88)

        # Stars with varying brightness and color temperature
        self.stars = [
            (R.random(), R.random(), R.uniform(0.6, 2.5), R.uniform(0, 6.28),
             R.choice([0.08, 0.14, 0.58, 0.63]), R.random() > 0.82)
            for _ in range(max(45, w * rows // 34))
        ]

        # Cloud layers (x_norm, y_norm, radius_x, radius_y, speed)
        self.clouds = [
            (R.uniform(0.0, 1.0), R.uniform(0.15, 0.72), R.uniform(7, 16), R.uniform(2.0, 4.2), R.uniform(0.3, 0.9))
            for _ in range(14)
        ]

        # Distant spaceport horizon lights
        self.city_lights = [
            (R.uniform(0.02, 0.98), R.uniform(0.0, 1.0), R.choice([0.08, 0.12, 0.55]))
            for _ in range(max(20, w // 4))
        ]

        # Ignition exhaust smoke puffs
        self.smoke_puffs = [
            (R.uniform(-1.0, 1.0), R.uniform(0.2, 1.0), R.uniform(0.5, 1.4), R.uniform(0, 6.28))
            for _ in range(40)
        ]

        # Birds scattering at engine ignition
        self.scared_birds = [
            (R.uniform(0.15, 0.35), R.uniform(0.55, 0.75), R.uniform(-1.0, 1.0), R.uniform(0, 6.28))
            for _ in range(9)
        ]

        # Lunar craters and surface rocks
        self.craters = [
            (0.10, 0.35, 5.5, 1.6),
            (0.28, 0.58, 4.0, 1.2),
            (0.46, 0.42, 7.5, 2.0),
            (0.66, 0.65, 5.0, 1.5),
            (0.84, 0.38, 6.5, 1.8),
            (0.93, 0.68, 3.5, 1.1),
        ]
        self.rocks = [
            (R.uniform(0.04, 0.96), R.uniform(0.15, 0.85), R.uniform(1.0, 2.2))
            for _ in range(16)
        ]
        self.moon_dust = [
            (R.uniform(-1.0, 1.0), R.uniform(0.4, 1.3), R.uniform(0.6, 1.5), R.uniform(0, 6.28))
            for _ in range(36)
        ]

        # Flag pixel scale fitted to terminal dimensions
        if w >= 120 and self.H >= 64:
            self.fw, self.fh = 2, 2
        elif w >= 72 and self.H >= 36:
            self.fw, self.fh = 2, 1
        else:
            self.fw, self.fh = 1, 1

    # ------------------------------------------------------------------
    def frame(self, t):
        w, rows = term_size()
        if (w, rows) != self.size:
            self.setup(w, rows)
        H = self.H

        pix = [[(0, 0, 0)] * w for _ in range(H)]
        ov = [[None] * w for _ in range(rows)]

        def pset(x, y, rgb, alpha=1.0):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < H:
                if alpha >= 0.99:
                    pix[iy][ix] = rgb
                elif alpha > 0.01:
                    pix[iy][ix] = lerp_rgb(pix[iy][ix], rgb, alpha)

        def padd(x, y, rgb, k=1.0):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < H:
                pix[iy][ix] = add_rgb(pix[iy][ix], rgb, k)

        def put_ov(x, y, ch, rgb):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < rows:
                ov[iy][ix] = (ch, rgb)

        def puts_ov(x, y, s, rgb):
            for i, ch in enumerate(s):
                if ch != " ":
                    put_ov(x + i, y, ch, rgb)

        fi = smooth(t / T_FADE)
        alt = smooth((t - T_LIFT) / 14.0)
        space_mix = smooth((alt - 0.35) / 0.45)

        # Continuous trans-lunar cruise progress (0 = Earth orbit, 1 = Moon)
        tli_q = smooth((t - T_TLI) / (T_BRAKE - T_TLI + 1.5))
        moon_rise = smooth((t - T_MOON_IN) / (T_BRAKE - T_MOON_IN + 2.8))

        shake_ign = smooth((t - T_IGN) / 0.8) * (1.0 - smooth((t - 7.5) / 3.0))
        shake_maxq = math.exp(-((t - T_MAXQ) / 1.4) ** 2) * 0.75
        shake_land = math.exp(-((t - T_LAND) / 0.35) ** 2) * 0.85 if t >= T_LAND - 0.5 else 0.0
        shake = shake_ign * 0.9 + shake_maxq + shake_land
        sx_off = math.sin(t * 47.0) * shake * 0.65
        sy_off = math.cos(t * 53.0) * shake * 0.55

        # 1. Sky gradient: dawn -> stratosphere -> deep space
        for y in range(H):
            f = y / max(1, H - 1)
            dawn_top = (10, 16, 48)
            dawn_mid = (28, 72, 145)
            dawn_bot = (225, 115, 58)
            c_dawn = lerp_rgb(dawn_top, dawn_mid, smooth(f * 1.4)) if f < 0.72 else lerp_rgb(dawn_mid, dawn_bot, (f - 0.72) / 0.28)
            c_space = lerp_rgb((2, 3, 10), (7, 13, 34), f ** 1.6)
            c_sky = lerp_rgb(c_dawn, c_space, space_mix)
            for x in range(w):
                pix[y][x] = c_sky

        # 2. Milky Way band
        if space_mix > 0.05:
            mw_shift = tli_q * 18.0
            for x in range(0, w, 2):
                ym = H * (0.68 - 0.50 * (x / w)) + math.sin((x + mw_shift) * 0.12) * 3.0
                for dy in range(-5, 6):
                    band = math.exp(-(dy * dy) / 14.0) * (0.5 + 0.5 * math.sin((x + mw_shift) * 0.3 + dy * 0.7))
                    if band > 0.25:
                        padd(x, ym + dy, (16, 25, 56), band * space_mix * 0.45)

        # 3. Stars (additive blending, gentle twinkle)
        if space_mix > 0.02:
            cruise_speed = math.sin(tli_q * math.pi) if 0.0 < tli_q < 1.0 else 0.0
            star_y_off = alt * H * 0.25 * (1.0 - tli_q)
            star_x_off = -tli_q * w * 0.45
            for xn, yn, sp, phs, hue, is_bright in self.stars:
                tw = 0.72 + 0.28 * math.sin(t * sp * 1.4 + phs)
                sx = (xn * w + star_x_off * (0.4 + sp * 0.25)) % w
                sy = (yn * H + star_y_off) % H
                v = space_mix * tw * (0.90 if is_bright else 0.45)
                col = hsv(hue, 0.22, v)
                if cruise_speed > 0.30 and is_bright:
                    for k in range(int(1 + cruise_speed * 3)):
                        padd((sx + k * 0.9) % w, sy, col, 0.65 * (1.0 - k * 0.25))
                elif is_bright and v > 0.58:
                    put_ov(sx, sy / 2, "*" if tw > 0.90 else "+", col)
                else:
                    padd(sx, sy, col, 0.80)

        # 4. Continuous Earth: recedes smoothly from low orbit into the lunar sky
        if alt > 0.30:
            earth_vis = smooth((alt - 0.30) / 0.35)
            er_big = max(w * 0.95, H * 1.45)
            er_small = clamp(min(w, H) * 0.09, 3.0, 5.2)
            er = er_big * ((er_small / er_big) ** (tli_q ** 0.72))
            ecx = lerp(w * 0.5 + sx_off, w * 0.82, tli_q ** 0.85)
            ecy_big = H * (0.85 + 0.15 * (1.0 - alt)) + er_big + sy_off
            ecy = lerp(ecy_big, H * 0.18, tli_q ** 0.85)

            y0 = max(0, int(ecy - er - 4))
            y1 = min(H - 1, int(ecy + er + 4))
            for y in range(y0, y1 + 1):
                dy = y - ecy
                under = er * er - dy * dy
                if under < 0:
                    continue
                half_span = math.sqrt(under)
                x0 = max(0, int(ecx - half_span - 3))
                x1 = min(w - 1, int(ecx + half_span + 3))
                for x in range(x0, x1 + 1):
                    dx = x - ecx
                    dist = math.hypot(dx, dy)
                    if dist <= er:
                        lon = dx / max(1.0, er) * 3.5 + t * 0.03
                        lat = (dist - er) / max(1.0, er) * 4.0
                        land_n = math.sin(lon * 2.1 + math.cos(lat * 2.7)) + 0.6 * math.cos(lon * 4.3 - lat * 3.1)
                        cloud_n = math.sin(lon * 3.8 + t * 0.08) * math.cos(lat * 5.2 + 0.7)
                        sun_bias = lerp(0.5, 0.25, tli_q)
                        sun = clamp(0.18 + 0.85 * (sun_bias - dx / max(1.0, er * 0.85)))
                        if tli_q > 0.6 and sun < 0.08:
                            continue
                        land_mix = smooth((land_n - 0.20) / 0.35)
                        base = lerp_rgb((16, 68, 158), (52, 130, 74), land_mix)
                        col = (int(base[0] * sun), int(base[1] * sun), int(base[2] * sun))
                        if sun < 0.42 and land_n > 0.40 and er > 15 and (int(x * 7 + y * 13) % 19 == 0):
                            col = lerp_rgb(col, (255, 200, 95), 0.75)
                        if cloud_n > 0.25:
                            col = lerp_rgb(col, (int(235 * sun), int(242 * sun), int(255 * sun)), clamp((cloud_n - 0.25) * 1.2))
                        limb = smooth((dist / max(1.0, er) - 0.90) / 0.10)
                        col = lerp_rgb(col, (85, 185, 255), limb * 0.85)
                        edge_aa = clamp(er + 0.6 - dist)
                        pset(x, y, col, earth_vis * edge_aa)
                    elif dist <= er + 3.5:
                        glow = (1.0 - (dist - er) / 3.5) ** 1.6
                        aur = math.exp(-((dx + er * 0.22) / max(1.0, er * 0.14)) ** 2) * (0.5 + 0.5 * math.sin(dx * 0.2 + t * 2.0)) * (1.0 - tli_q)
                        c_limb = lerp_rgb((65, 175, 255), (65, 255, 165), aur * 0.85)
                        padd(x, y, c_limb, glow * earth_vis * (0.85 + 0.45 * aur) * (1.0 - 0.5 * tli_q))

        # 5. Lower Earth layers (clouds, plane, launchpad)
        if t < T_TLI + 1.0:
            self._draw_earth_lower(pset, padd, put_ov, puts_ov, t, w, H, alt, sx_off, sy_off)

        # 6. Lunar surface approaching smoothly from below
        moon_y = H * 0.78 + (1.0 - moon_rise) * H * 0.55 + (sy_off if t < T_LAND else 0.0)
        if moon_rise > 0.01:
            curve = (1.0 - moon_rise) * 0.018
            for x in range(w):
                dx_c = x - w * 0.35
                hill = moon_y + curve * (dx_c * dx_c) * 0.18 - 2.0 * math.sin(x * 0.08 + 0.4) - 1.2 * math.cos(x * 0.21)
                for y in range(max(0, int(hill)), H):
                    d = (y - hill) / max(1.0, H - hill)
                    slope = math.cos(x * 0.08 + 0.4) * 0.12
                    shade = clamp(0.58 - 0.28 * d + slope + 0.04 * math.sin(x * 0.35 + y * 0.5))
                    col = lerp_rgb((68, 72, 82), (175, 180, 192), shade)
                    edge_m = clamp((y - hill) + 0.8)
                    pset(x, y, col, smooth(moon_rise) * edge_m)

            if moon_rise > 0.45:
                c_vis = smooth((moon_rise - 0.45) / 0.55)
                for cx_n, cy_n, crx, cry in self.craters:
                    cx_c = cx_n * w
                    cy_c = moon_y + 2 + cy_n * max(1.0, H - moon_y - 4)
                    if cy_c >= H:
                        continue
                    for dy in range(-int(cry) - 1, int(cry) + 2):
                        for dx in range(-int(crx) - 1, int(crx) + 2):
                            d2 = (dx / crx) ** 2 + (dy / cry) ** 2
                            if d2 <= 1.0:
                                rim = dx / crx
                                c_cr = lerp_rgb((42, 46, 54), (195, 200, 212), clamp(0.45 + rim * 0.45 + dy / cry * 0.2))
                                pset(cx_c + dx, cy_c + dy, c_cr, c_vis * 0.85)
                for rx_n, ry_n, _ in self.rocks:
                    rx_k = rx_n * w
                    ry_k = moon_y + 1 + ry_n * max(1.0, H - moon_y - 3)
                    if ry_k < H - 1:
                        pset(rx_k, ry_k, (185, 190, 200), c_vis)
                        pset(rx_k + 1, ry_k, (75, 80, 90), c_vis)

        # 7. Continuous rocket: launch -> trans-lunar pitch -> braking -> landing
        self._draw_continuous_rocket(pset, padd, put_ov, t, w, H, alt, tli_q, moon_y, sx_off, sy_off)

        # 8. Astronaut EVA and KRONA flag deployment
        if t >= T_EVA:
            r_scale = clamp(min(w / 95.0, H / 56.0), 0.72, 1.25)
            body_h = int(23 * r_scale)
            body_r = max(2.6, 3.4 * r_scale)
            rx_land = w * 0.22
            nozzle_land = moon_y - 4.0

            eva_dt = t - T_EVA
            ladder_q = smooth(eva_dt / 0.8)
            lx0, ly0 = rx_land + body_r + 0.5, nozzle_land - body_h * 0.25
            lx1, ly1 = rx_land + body_r + 3.2, moon_y + 0.5
            for k in range(int(ladder_q * 8) + 1):
                q = k / 8.0
                pset(lerp(lx0, lx1, q), lerp(ly0, ly1, q), (210, 218, 230))

            astro_start_x = lx1 + 1.0
            flag_x = rx_land + body_r + clamp(w * 0.12, 8.0, 14.0)
            walk_q = smooth((t - (T_EVA + 0.6)) / (T_FLAG - (T_EVA + 0.6)))
            ax = lerp(astro_start_x, flag_x - 3.5, walk_q)
            hop = abs(math.sin(walk_q * math.pi * 3.0)) * 3.0 * (1.0 if walk_q < 0.99 else 0.0)
            ay = moon_y - hop

            for s in range(int(walk_q * 6)):
                fx_s = lerp(astro_start_x, flag_x - 3.5, (s + 0.5) / 6.0)
                pset(fx_s, moon_y + 1, (55, 58, 68), 0.7)

            salute = smooth((t - (T_FLAG + 1.2)) / 0.8)
            self._draw_astronaut(pset, ax, ay, salute, walk_q < 0.99)

            if t < T_FLAG:
                for dy in range(0, 7):
                    pset(ax + 2, ay - 2 - dy, (220, 230, 245))
            else:
                self._draw_krona_flag(pset, t, w, flag_x, moon_y)

        # 9. Flight telemetry HUD
        if w >= 60:
            self._draw_hud(puts_ov, t)

        # 10. Exit hint
        hint = smooth((t - (T_END - 0.5)) / 1.5)
        if hint > 0.02:
            for i, ch in enumerate("Ctrl+C to exit"):
                put_ov(int(w * 0.5 - 7) + i, rows - 1, ch, hsv(0.55, 0.20, 0.70 * hint))

        # Render frame buffer to ANSI
        out = ["\x1b[H"]
        last_fg = None
        last_bg = None
        for r in range(rows):
            y_top = r * 2
            y_bot = y_top + 1
            for c in range(w):
                top = pix[y_top][c]
                bot = pix[y_bot][c]
                if fi < 0.999:
                    top = (int(top[0] * fi), int(top[1] * fi), int(top[2] * fi))
                    bot = (int(bot[0] * fi), int(bot[1] * fi), int(bot[2] * fi))

                cell_ov = ov[r][c]
                if cell_ov is not None:
                    ch, fg = cell_ov
                    if fi < 0.999:
                        fg = (int(fg[0] * fi), int(fg[1] * fi), int(fg[2] * fi))
                    bg = ((top[0] + bot[0]) // 2, (top[1] + bot[1]) // 2, (top[2] + bot[2]) // 2)
                elif top == bot:
                    ch = " "
                    fg = last_fg
                    bg = top
                else:
                    ch = "▀"
                    fg = top
                    bg = bot

                if bg != last_bg:
                    out.append(esc_bg(bg))
                    last_bg = bg
                if ch != " " and fg != last_fg:
                    out.append(esc_fg(fg))
                    last_fg = fg
                out.append(ch)
            if r < rows - 1:
                out.append("\n")
        out.append("\x1b[0m")
        return "".join(out)

    # ------------------------------------------------------------------
    # Flight telemetry HUD
    # ------------------------------------------------------------------
    def _draw_hud(self, puts_ov, t):
        if t < T_LIFT:
            met = f"T-{T_LIFT - t:04.1f}s"
            phase = "IGNITION" if t >= T_IGN else "STANDBY"
            info = "EARTH PAD"
        elif t < T_TLI:
            dt = t - T_LIFT
            met = f"T+{dt:04.1f}s"
            phase = "STAGE 2" if t >= T_SEP else "MAX-Q" if abs(t - T_MAXQ) < 2.2 else "ASCENT"
            info = f"ALT {(dt ** 1.8) * 1.4:5.1f}km"
        elif t < T_BRAKE:
            met = f"T+{t - T_LIFT:04.1f}s"
            phase = "TRANS-LUNAR"
            dist_k = max(12, int(384 * (1.0 - smooth((t - T_TLI) / (T_BRAKE - T_TLI)))))
            info = f"DIST {dist_k:3d}k km"
        elif t < T_LAND:
            met = f"T+{t - T_LIFT:04.1f}s"
            phase = "LANDING BURN"
            rem = max(0.0, (T_LAND - t) * 14.5)
            info = f"LUNA {rem:4.0f}m"
        elif t < T_FLAG:
            met = f"T+{t - T_LIFT:04.1f}s"
            phase = "EVA WALK" if t >= T_EVA else "TOUCHDOWN"
            info = "SEA OF TRANQUILITY"
        else:
            met = f"T+{t - T_LIFT:04.1f}s"
            phase = "FLAG DEPLOYED"
            info = "KRONA BASE"

        puts_ov(2, 1, f"KRONA-1  {met}", (120, 200, 240))
        puts_ov(2, 2, info, (235, 245, 255))
        puts_ov(2, 3, f"[{phase}]", (90, 245, 175) if t >= T_LAND else (255, 190, 85))

    # ------------------------------------------------------------------
    # Lower Earth atmosphere, clouds, plane, and launchpad
    # ------------------------------------------------------------------
    def _draw_earth_lower(self, pset, padd, put_ov, puts_ov, t, w, H, alt, sx_off, sy_off):
        if alt < 0.72:
            cloud_fade = (1.0 - smooth((alt - 0.45) / 0.25)) * smooth((alt + 0.1) / 0.2)
            if cloud_fade > 0.02:
                for xn, yn, rx, ry, sp in self.clouds:
                    cx0 = (xn * w + t * sp * 2.5) % (w + rx * 2) - rx
                    cy0 = (yn * H + alt * H * 2.6) % (H + ry * 4) - ry * 2
                    for dy in range(-int(ry), int(ry) + 1):
                        for dx in range(-int(rx), int(rx) + 1):
                            d2 = (dx / rx) ** 2 + (dy / ry) ** 2
                            if d2 <= 1.0:
                                sh = clamp(1.0 - d2)
                                lit = 0.65 + 0.35 * (-dy / ry)
                                pset(cx0 + dx, cy0 + dy, lerp_rgb((110, 125, 165), (245, 225, 210), lit), sh * 0.55 * cloud_fade)

        if 6.5 <= t <= 11.2:
            pq = (t - 6.5) / 4.7
            plx = w * (0.05 + pq * 0.85)
            ply = H * 0.32 + (alt * H * 0.5)
            if ply < H:
                for k in range(1, 14):
                    padd(plx - k * 1.1, ply, (225, 235, 245), (1.0 - k / 14) * 0.55)
                puts_ov(plx, ply / 2, ">═>", (240, 245, 255))
                if int(t * 5) % 2 == 0:
                    padd(plx + 1, ply, (255, 60, 50), 0.9)

        pad_drop = smooth((t - T_LIFT) / 4.2) * (H * 1.2)
        ground_y = H * 0.84 + pad_drop + sy_off
        px0 = w * 0.5 + sx_off
        if ground_y < H + 20:
            if t < T_LIFT + 2.5:
                beam_a = 1.0 - smooth((t - T_LIFT) / 2.5)
                for side in (-1, 1):
                    bx0 = px0 + side * 22
                    ang = -1.57 - side * (0.35 + 0.12 * math.sin(t * 1.4 + side))
                    for dist in range(2, int(H * 0.65)):
                        lx = bx0 + math.cos(ang) * dist
                        ly = ground_y + math.sin(ang) * dist
                        bw = 0.8 + dist * 0.08
                        fade = (1.0 - dist / (H * 0.65)) * 0.28 * beam_a
                        for dx in range(-int(bw), int(bw) + 1):
                            padd(lx + dx, ly, (190, 225, 255), fade * (1.0 - abs(dx) / (bw + 1)))

            for x in range(w):
                mh = ground_y - 3 - 4.0 * math.sin(x * 0.07) - 2.2 * math.cos(x * 0.19)
                for y in range(max(0, int(mh)), min(H, int(ground_y))):
                    pset(x, y, (36, 40, 68), 0.85)
            for y in range(max(0, int(ground_y)), H):
                for x in range(w):
                    pset(x, y, lerp_rgb((46, 52, 62), (20, 22, 28), clamp((y - ground_y) / 10.0)))
            for xn, ph, hue in self.city_lights:
                lx, ly = xn * w, ground_y - 0.5
                if 0 <= ly < H:
                    padd(lx, ly, hsv(hue, 0.8, 0.6 + 0.4 * math.sin(t * 4 + ph * 6)), 0.9)
            for y in range(max(0, int(ground_y - 2)), min(H, int(ground_y + 3))):
                for x in range(int(px0 - 18), int(px0 + 19)):
                    pset(x, y, (88, 94, 104))
            for side in (-1, 1):
                mx_l = int(px0 + side * 24)
                mh_l = int(clamp(H * 0.42, 16, 26))
                for y in range(max(0, int(ground_y - mh_l)), min(H, int(ground_y))):
                    put_ov(mx_l, y / 2, "│", (110, 118, 132))
                if int(t * 4 + side) % 2 == 0:
                    padd(mx_l, ground_y - mh_l - 1, (255, 60, 45), 0.85)
            tower_x = int(px0 - 13)
            tower_h = int(clamp(H * 0.52, 20, 34))
            for y in range(max(0, int(ground_y - tower_h)), min(H, int(ground_y))):
                put_ov(tower_x, y / 2, "║", (155, 162, 175))
                put_ov(tower_x + 2, y / 2, "║", (125, 132, 145))
                if y % 3 == 0:
                    put_ov(tower_x + 1, y / 2, "╳", (140, 145, 158))
            padd(tower_x + 1, ground_y - tower_h - 1, (255, 45, 35), 0.4 + 0.6 * (0.5 + 0.5 * math.sin(t * 6.0)))
            if smooth((t - T_LIFT + 0.3) / 1.2) < 0.8:
                puts_ov(tower_x + 3, int((ground_y - tower_h * 0.68) / 2), "━━━━━", (170, 175, 185))
            if T_IGN <= t < T_LIFT + 3.0:
                bt = t - T_IGN
                for bx_n, by_n, dir_, phs in self.scared_birds:
                    bx = (bx_n if dir_ < 0 else 1.0 - bx_n) * w + dir_ * bt * 12.0
                    by = by_n * H - bt * 5.0 + math.sin(t * 8 + phs) * 1.2 + pad_drop * 0.5
                    put_ov(bx, by / 2, "v" if math.sin(t * 14 + phs) > 0 else "^", (40, 45, 65))

            if T_IGN <= t < T_LIFT + 4.5:
                puff_vis = smooth((t - T_IGN) / 0.8) * (1.0 - smooth((t - T_LIFT - 2.0) / 2.5))
                if puff_vis > 0.02:
                    for dx_n, sp, rad_k, phs in self.smoke_puffs:
                        age = ((t - T_IGN) * 0.6 + phs * 0.1) % 1.0
                        px = px0 + dx_n * (4 + age * 28 * sp)
                        py = ground_y - 1 - age * 5 * sp
                        pr = (2.0 + age * 4.5) * rad_k
                        for dy in range(-int(pr), int(pr) + 1):
                            for dx in range(-int(pr * 1.4), int(pr * 1.4) + 1):
                                d2 = (dx / (pr * 1.4)) ** 2 + (dy / pr) ** 2
                                if d2 <= 1.0:
                                    sh = (1.0 - d2) * (1.0 - age * 0.6) * puff_vis
                                    fire_lit = math.exp(-abs(px - px0) / 10.0) * (1.0 - age)
                                    pset(px + dx, py + dy, lerp_rgb((175, 180, 190), (255, 165, 70), fire_lit), sh * 0.75)

    # ------------------------------------------------------------------
    # Continuous rocket: launch -> pitch -> cruise -> braking -> touchdown
    # ------------------------------------------------------------------
    def _draw_continuous_rocket(self, pset, padd, put_ov, t, w, H, alt, tli_q, moon_y, sx_off, sy_off):
        r_scale = clamp(min(w / 95.0, H / 56.0), 0.75, 1.32)
        body_h = int(lerp(26 * r_scale, 23 * r_scale, tli_q))
        body_r = lerp(max(2.8, 3.7 * r_scale), max(2.6, 3.4 * r_scale), tli_q)

        pad_drop = smooth((t - T_LIFT) / 4.2) * (H * 1.2)
        ground_y = H * 0.84 + pad_drop + sy_off

        if t < T_LIFT:
            rx = w * 0.5 + sx_off
            nozzle_y = ground_y - 3
            tilt = 0.0
        elif t < T_TLI:
            climb = smooth((t - T_LIFT) / 5.5)
            rx = w * 0.5 + sx_off
            nozzle_y = lerp(ground_y - 3, H * 0.58, climb) + sy_off
            tilt = 0.0
        elif t < T_BRAKE:
            q_fl = smooth((t - T_TLI) / (T_BRAKE - T_TLI))
            rx = lerp(w * 0.5, w * 0.28, q_fl) + sx_off * 0.3
            nozzle_y = lerp(H * 0.58, H * 0.44, math.sin(q_fl * math.pi * 0.65))
            tilt = math.sin(q_fl * math.pi) * 0.72
        else:
            q_br = smooth((t - T_BRAKE) / (T_LAND - T_BRAKE))
            rx = lerp(w * 0.28, w * 0.22, q_br) + (sx_off if t < T_LAND else 0.0)
            nozzle_y = lerp(H * 0.44, moon_y - 4.0, q_br ** 0.85) + (sy_off if t < T_LAND else 0.0)
            tilt = 0.0

        piv_y = nozzle_y - body_h * 0.45
        ca, sa = math.cos(tilt), math.sin(tilt)

        def rot(lx, ly):
            dy_p = (nozzle_y + ly) - piv_y
            return rx + lx * ca - dy_p * sa, piv_y + lx * sa + dy_p * ca

        # Supersonic bow shock and compression rings
        if 9.8 <= t <= 16.8:
            shock_env = smooth((t - 9.8) / 1.4) * (1.0 - smooth((t - 15.4) / 1.4))
            nose_y = nozzle_y - body_h - 3
            bow_h = int(body_h * 2.1)
            for dy in range(0, bow_h):
                w_arc = math.sqrt(dy + 0.4) * 3.1
                wave_pulse = 0.55 + 0.45 * math.sin(dy * 0.55 - t * 18.0)
                fade = (1.0 - (dy / bow_h) ** 1.4) * shock_env * wave_pulse
                for thick in range(3):
                    alpha = fade * (0.75 - thick * 0.24)
                    col_arc = lerp_rgb((120, 215, 255), (240, 250, 255), 1.0 - thick / 3.0)
                    padd(rx - w_arc + thick * 0.7, nose_y + dy, col_arc, alpha)
                    padd(rx + w_arc - thick * 0.7, nose_y + dy, col_arc, alpha)
            for k in range(2):
                cyc = ((t - 9.8) * 0.85 + k * 0.5) % 1.0
                wy = nose_y + cyc * (body_h * 1.75)
                rad_x = body_r * (1.3 + cyc * 4.2)
                ring_a = math.sin(cyc * math.pi) * shock_env * 0.65
                if ring_a > 0.04:
                    n_pts = int(rad_x * 12) + 36
                    for i in range(n_pts):
                        ang = i / n_pts * 6.283
                        padd(rx + rad_x * math.cos(ang), wy + rad_x * 0.30 * math.sin(ang), (200, 238, 255), ring_a * 0.45)

        # Exhaust plume
        if T_IGN <= t < T_LAND + 0.15:
            thrust = smooth((t - T_IGN) / 0.9) * (1.0 - smooth((t - (T_LAND - 0.35)) / 0.45))
            if T_TLI + 1.5 < t < T_BRAKE - 0.5:
                thrust *= 0.65
            is_vac = smooth((t - T_SEP) / 1.5)
            plume_len = (11 + 17 * smooth((t - T_LIFT) / 3.0)) * r_scale * thrust
            plume_w = body_r * (0.95 + 1.35 * is_vac)
            for dy in range(0, int(plume_len) + 1):
                q = dy / max(1.0, plume_len)
                pulse = 1.0 + 0.18 * math.sin(dy * 1.15 - t * 28.0) * (1.0 - is_vac)
                hw = plume_w * (0.55 + 0.70 * q * is_vac + 0.35 * (1.0 - q) * (1.0 - is_vac)) * pulse
                for dx in range(-int(hw * 1.3) - 1, int(hw * 1.3) + 2):
                    nd = abs(dx) / max(0.5, hw)
                    if nd <= 1.15:
                        core = clamp(1.0 - nd) * (1.0 - q * 0.85)
                        diamond = max(0.0, math.sin(dy * 1.35 - t * 18.0)) * (1.0 - nd) * (1.0 - q) * (1.0 - is_vac)
                        if is_vac > 0.3:
                            c_pl = lerp_rgb((95, 130, 255), (240, 250, 255), clamp(core + 0.3 * (1 - q)))
                            c_pl = lerp_rgb((165, 85, 255), c_pl, clamp(core * 1.3))
                        else:
                            c_pl = lerp_rgb((255, 85, 25), (255, 215, 80), clamp(core * 1.2))
                            c_pl = lerp_rgb(c_pl, (255, 255, 240), clamp(core ** 1.8 + diamond * 0.7))
                        px_f, py_f = rot(dx, dy)
                        padd(px_f, py_f, c_pl, clamp((1.0 - nd * 0.8) * (1.0 - q * 0.75) * thrust))

        # Lunar dust kicked up during landing burn + shadow under lander
        if T_LAND - 2.2 <= t < T_LAND + 2.2:
            dust_env = smooth((t - (T_LAND - 2.2)) / 0.8) * (1.0 - smooth((t - (T_LAND + 0.4)) / 1.8))
            for dx_n, sp, rk, phs in self.moon_dust:
                age = ((t - (T_LAND - 2.2)) * 0.8 + phs * 0.1) % 1.0
                px = rx + dx_n * (2.5 + age * 16 * sp)
                py = moon_y - 0.5 - math.sin(age * math.pi) * 2.5 * rk
                pset(px, py, (200, 205, 215), (1.0 - age) * dust_env * 0.75)
        if t >= T_BRAKE:
            sh_a = smooth((t - T_BRAKE) / (T_LAND - T_BRAKE)) * 0.65
            for dx in range(-int(body_r * 2.2), int(body_r * 2.2) + 1):
                pset(rx + dx, moon_y + 1, (22, 24, 30), sh_a * (1.0 - abs(dx) / (body_r * 2.4)))

        # Side boosters (visible until separation & coast)
        if t < T_TLI + 2.5:
            sep_dt = max(0.0, t - T_SEP)
            sep_drop = (sep_dt ** 1.6) * 6.0 if t >= T_SEP else 0.0
            sep_spread = sep_dt * 2.4 if t >= T_SEP else 0.0
            b_h = int(body_h * 0.56)
            b_r = max(1.3, body_r * 0.44)
            for side in (-1, 1):
                bx = rx + side * (body_r + b_r + 0.7 + sep_spread)
                by = nozzle_y + sep_drop
                if by - b_h < H + 8:
                    for dy in range(0, 4):
                        pset(bx + side * (b_r + (3 - dy) * 0.55), by - dy, (175, 185, 200))
                    for dy in range(b_h):
                        y = by - dy
                        q = dy / max(1, b_h)
                        w_loc = b_r * (1.0 if q < 0.76 else (1.0 - q) / 0.24)
                        for dx in range(-int(math.ceil(w_loc)), int(math.ceil(w_loc)) + 1):
                            if abs(dx) <= w_loc:
                                lit = clamp(0.55 - 0.45 * (dx / max(0.5, w_loc)))
                                col = lerp_rgb((88, 98, 118), (235, 242, 250), lit)
                                if 0.60 < q < 0.70:
                                    col = lerp_rgb((175, 60, 18), (255, 138, 45), lit)
                                pset(bx + dx, y, col)
                    if 0.4 < sep_dt < 2.2:
                        for k in range(3):
                            padd(bx - side * (b_r + k), by - b_h * 0.75, (220, 240, 255), (1.0 - k / 3) * 0.7)

        # Deployable landing legs
        legs_deploy = smooth((t - (T_BRAKE - 0.5)) / 1.8)
        if legs_deploy > 0.05:
            leg_span = body_r * (1.0 + 1.35 * legs_deploy)
            leg_drop = 3.8 * legs_deploy
            for side in (-1, 1):
                x0, y0 = rot(side * (body_r * 0.8), -3.0)
                x1, y1 = rot(side * leg_span, leg_drop)
                for i in range(7):
                    q = i / 6.0
                    pset(lerp(x0, x1, q), lerp(y0, y1, q), (195, 205, 220))
                pset(x1 - 1, y1, (220, 225, 235))
                pset(x1, y1, (220, 225, 235))
                pset(x1 + 1, y1, (220, 225, 235))

        # Main rocket body (sub-pixel sampling when tilted to avoid gaps)
        heat_on = 10.0 < t < 16.0
        for side in (-1, 1):
            for dy in range(5):
                px_s, py_s = rot(side * (body_r + (4 - dy) * 0.45), -dy)
                pset(px_s, py_s, (155, 165, 182))
            for off in (1.2, 2.0):
                px_f, py_f = rot(side * (body_r + off), -int(body_h * 0.62))
                pset(px_f, py_f, (85, 92, 108))

        step = 0.5 if abs(tilt) > 0.04 else 1.0
        n_dy = int(body_h / step)
        for idy in range(n_dy):
            dy = idy * step
            q = dy / max(1, body_h)
            w_loc = body_r * ((1.0 - q) / 0.24) ** 0.72 if q > 0.76 else body_r
            n_dx = int(math.ceil(w_loc / step))
            for idx in range(-n_dx, n_dx + 1):
                dx = idx * step
                if abs(dx) <= w_loc:
                    lit = clamp(0.62 - 0.52 * (dx / max(0.5, w_loc)))
                    if q > 0.76:
                        col = lerp_rgb((105, 115, 138), (252, 253, 255), lit)
                        if heat_on and q > 0.88:
                            col = lerp_rgb(col, (255, 145, 60), 0.70)
                    elif 0.58 < q < 0.70:
                        rib = 0.85 if int(round(dx)) % 2 == 0 else 1.15
                        col = lerp_rgb((22, 26, 36), (78, 86, 104), clamp(lit * rib))
                    elif 0.50 < q <= 0.58:
                        col = lerp_rgb((180, 75, 18), (255, 170, 50), lit)
                    elif 0.46 < q <= 0.50:
                        col = lerp_rgb((20, 115, 130), (65, 230, 225), lit)
                    else:
                        col = lerp_rgb((92, 102, 122), (242, 247, 255), lit)
                        if abs(dx) < 0.6:
                            col = lerp_rgb(col, (140, 150, 168), 0.55)
                        elif int(round(dy)) % 6 == 0:
                            col = lerp_rgb(col, (60, 68, 82), 0.35)
                    px_b, py_b = rot(dx, -dy)
                    pset(px_b, py_b, col)

        px_w, py_w = rot(-body_r * 0.25, -body_h * 0.73)
        pset(px_w, py_w, (80, 210, 255))
        for k, c_tip in ((0, (220, 230, 242)), (-1, (200, 210, 225)),
                         (-2, (255, 90, 70) if int(t * 6) % 2 == 0 else (220, 230, 245))):
            px_t, py_t = rot(0, -body_h + k)
            pset(px_t, py_t, c_tip)
        for nx_off in (-body_r * 0.55, 0.0, body_r * 0.55):
            px1, py1 = rot(nx_off, 1)
            px2, py2 = rot(nx_off, 2)
            pset(px1, py1, (55, 60, 72))
            pset(px2, py2, (255, 160, 75) if T_IGN <= t < T_LAND else (45, 48, 58))

    # ------------------------------------------------------------------
    # Astronaut and KRONA flag
    # ------------------------------------------------------------------
    def _draw_astronaut(self, pset, ax, ay, salute, walking):
        white = (242, 246, 252)
        shade = (175, 185, 200)
        visor = (255, 195, 55)
        pack = (150, 160, 178)
        pset(ax - 1, ay, shade)
        pset(ax + 1, ay, white)
        pset(ax - 1, ay - 1, white)
        pset(ax + 1, ay - 1, white)
        pset(ax - 2, ay - 2, pack)
        pset(ax - 2, ay - 3, pack)
        for dy in (2, 3, 4):
            pset(ax - 1, ay - dy, shade)
            pset(ax, ay - dy, white)
            pset(ax + 1, ay - dy, white)
        pset(ax, ay - 3, (255, 145, 50))
        for dx in (-1, 0, 1):
            pset(ax + dx, ay - 6, white)
            pset(ax + dx, ay - 5, white)
        pset(ax, ay - 5, visor)
        pset(ax + 1, ay - 5, visor)
        if salute > 0.3:
            pset(ax + 2, ay - 4, white)
            pset(ax + 2, ay - 5, white)
        elif not walking:
            pset(ax + 2, ay - 3, white)

    def _draw_krona_flag(self, pset, t, w, flag_x, moon_y):
        dt = t - T_FLAG
        plant_q = smooth(dt / 0.7)
        unfurl_q = smooth((dt - 0.5) / 1.8)
        logo_q = smooth((dt - 1.6) / 1.4)

        fw, fh = self.fw, self.fh
        text_w = 29 * fw
        text_h = 7 * fh * 2
        pad_x, pad_y = 3, 3
        flag_w = text_w + pad_x * 2
        flag_h = text_h + pad_y * 2

        pole_h = flag_h + 7
        pole_top_y = moon_y - pole_h * plant_q
        for y in range(int(pole_top_y), int(moon_y + 1)):
            pset(flag_x, y, (220, 228, 240))
        pset(flag_x, pole_top_y - 1, (255, 215, 95))

        if unfurl_q <= 0.02:
            return

        cur_w = max(2, int(round(flag_w * unfurl_q)))
        fx0 = flag_x + 1
        fy0 = pole_top_y + 1

        for dx in range(cur_w):
            pset(fx0 + dx, fy0 - 1, (200, 210, 225))

        sweep = (t * 22.0) % (w * 1.5) - w * 0.25
        bw = max(4.0, w * 0.05)

        for dy in range(flag_h):
            for dx in range(cur_w):
                wave = math.sin(dx * 0.22 - t * 2.2 + dy * 0.15) * 0.5 + 0.5
                wy = fy0 + dy + math.sin(dx * 0.20 - t * 2.0) * 0.45 * (dx / max(1, flag_w))
                wx = fx0 + dx
                if dy == 0 or dy == flag_h - 1 or dx == 0 or dx == cur_w - 1:
                    col = lerp_rgb((185, 135, 45), (255, 215, 105), wave)
                else:
                    c_bg = lerp_rgb((14, 28, 64), (18, 68, 96), dx / max(1, flag_w))
                    col = lerp_rgb(c_bg, (28, 92, 128), wave * 0.35)
                pset(wx, wy, col)

        if logo_q > 0.02 and cur_w >= text_w + pad_x:
            tx0 = fx0 + pad_x
            ty0 = fy0 + pad_y
            for li, letter in enumerate(WORD):
                for py, row in enumerate(FONT[letter]):
                    for px, c in enumerate(row):
                        if c != "#":
                            continue
                        for sy in range(fh * 2):
                            for sx in range(fw):
                                tx = tx0 + (li * 6 + px) * fw + sx
                                ty = ty0 + py * fh * 2 + sy
                                wy = ty + math.sin((tx - fx0) * 0.20 - t * 2.0) * 0.45 * ((tx - fx0) / max(1, flag_w))
                                band = math.exp(-((tx + ty * 0.7 - sweep) ** 2) / (2 * bw * bw))
                                v_frac = (py * fh * 2 + sy) / max(1, text_h - 1)
                                hue = 0.50 + 0.12 * (li / 4.0)
                                val = clamp(0.92 - 0.16 * v_frac + 0.25 * band)
                                col = hsv(hue - 0.10 * band, 0.72 - 0.65 * band, val)
                                pset(tx + 1, wy + 1, (6, 12, 28), 0.70 * logo_q)
                                pset(tx, wy, col, logo_q)


def main():
    scene = Scene()
    sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[2J")
    start = time.time() - (39.5 if "--end" in sys.argv else 0.0)
    try:
        while True:
            t0 = time.time()
            sys.stdout.write(scene.frame(t0 - start))
            sys.stdout.flush()
            time.sleep(max(0.0, 1 / FPS - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()


if __name__ == "__main__":
    main()