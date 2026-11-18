"""Static validation of static/style.css (no browser). Covers syntax sanity, class coverage against
app.js/index.html, custom-property hygiene, theme parity, style policy and WCAG contrast."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CSS_PATH = ROOT / "static" / "style.css"
RAW = CSS_PATH.read_bytes()
CSS = RAW.decode("utf-8")
JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8", errors="replace")
HTML = (ROOT / "static" / "index.html").read_text(encoding="utf-8", errors="replace")


def strip_comments(s):
    return re.sub(r"/\*.*?\*/", "", s, flags=re.S)


NOCOMMENT = strip_comments(CSS)


# ---------------------------------------------------------------- tiny CSS parser
def parse_blocks(text):
    """Return list of (prelude, body, depth) for every { } block, nested included."""
    out, stack = [], []
    last = 0
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append((text[last:i].strip(), i + 1))
            last = i + 1
        elif ch == "}":
            prelude, start = stack.pop()
            out.append((prelude, text[start:i], len(stack)))
            last = i + 1
        elif ch == ";":
            last = i + 1
    return out


BLOCKS = parse_blocks(NOCOMMENT)


def decl_blocks():
    """Blocks that hold declarations (not @media wrappers)."""
    return [(p, b, d) for p, b, d in BLOCKS if not p.startswith(("@media", "@supports")) and "{" not in b]


def all_selectors():
    sels = []
    for p, _b, _d in decl_blocks():
        if p.startswith("@"):
            continue
        sels.extend(s.strip() for s in p.split(","))
    return sels


def decls(body):
    return [d.strip() for d in body.split(";") if d.strip()]


# ---------------------------------------------------------------- syntax / file hygiene
def test_valid_utf8_no_raw_b7_no_bom():
    RAW.decode("utf-8")
    assert b"\xb7" not in RAW.replace(b"\xc2\xb7", b""), "raw 0xB7 byte (latin-1 middle dot) found"
    assert not RAW.startswith(b"\xef\xbb\xbf"), "BOM present"


def test_line_endings_consistent():
    crlf = RAW.count(b"\r\n")
    lone_lf = RAW.count(b"\n") - crlf
    lone_cr = RAW.count(b"\r") - crlf
    assert lone_cr == 0
    assert crlf == 0 or lone_lf == 0, f"mixed endings: {crlf} CRLF vs {lone_lf} LF"


def test_braces_and_parens_balanced():
    for open_c, close_c in ("{}", "()", "[]"):
        depth = 0
        for i, ch in enumerate(NOCOMMENT):
            if ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
            assert depth >= 0, f"unmatched {close_c!r} at offset {i}"
        assert depth == 0, f"unbalanced {open_c}{close_c}: {depth}"


def test_no_unterminated_comment_or_string():
    assert CSS.count("/*") == CSS.count("*/")
    for line in NOCOMMENT.splitlines():
        assert line.count('"') % 2 == 0, f"unbalanced quote: {line!r}"


def test_no_empty_or_truncated_rules():
    assert BLOCKS, "no rules parsed"
    for prelude, body, _d in BLOCKS:
        assert prelude, f"rule with empty selector: {{{body[:40]}}}"
        assert body.strip(), f"empty rule body for {prelude!r}"
        if not prelude.startswith(("@media", "@supports")):
            for d in decls(body):
                if "{" in d:
                    continue
                assert ":" in d, f"declaration without colon in {prelude!r}: {d!r}"
                prop, _, val = d.partition(":")
                assert prop.strip() and val.strip(), f"empty prop/value in {prelude!r}: {d!r}"
    assert NOCOMMENT.rstrip().endswith("}"), "file does not end with a closed rule"


# ---------------------------------------------------------------- class coverage
# Purely semantic hooks used by JS (querySelector / closest) but intentionally unstyled.
SEMANTIC_CLASS_EXCEPTIONS = {
    "copy",  # JS hook: button.copy click delegation; visual styling comes from .ghost
}

# Values that reach className only through variables (bubble(cls), toast(kind), upload tone).
DYNAMIC_VALUES = {"user", "bot", "err", "error", "ok", "warn", "bad", "wait", "done"}


def classes_used_by_markup():
    found = set()

    def add(s):
        for t in s.split():
            if re.fullmatch(r"-?[A-Za-z_][\w-]*", t):
                found.add(t)

    for m in re.finditer(r"""\bclass\s*:\s*(['"])(.*?)\1""", JS):
        add(m.group(2))
    for m in re.finditer(r"""className\s*=\s*(['"])(.*?)\1""", JS):
        add(m.group(2))
    for m in re.finditer(r"""classList\.(?:add|remove|toggle)\(\s*(['"])(.*?)\1""", JS):
        add(m.group(2))
    for m in re.finditer(r"""class=\\?"([^"\\]*)\\?"|class='([^']*)'""", JS):
        add(m.group(1) or m.group(2) or "")
    for m in re.finditer(r"""class="([^"]*)\"""", HTML):
        add(m.group(1))
    for line in JS.splitlines():  # ternary class fragments, e.g. class: 'item' + (x ? ' active' : '')
        if "class:" in line or "className" in line:
            line = re.sub(r"\brole\s*:[^,}]*", "", line)  # role: x ? 'alert' : 'status' is not a class
            for m in re.finditer(r"""\?\s*(['"])\s*([\w -]+?)\1\s*:\s*(['"])([\w -]*)\3""", line):
                add(m.group(2))
                add(m.group(4))
    return found | DYNAMIC_VALUES


def css_classes():
    return set(re.findall(r"\.(-?[A-Za-z_][\w-]*)", "\n".join(all_selectors())))


def test_every_dynamic_class_has_a_rule():
    used = classes_used_by_markup() - SEMANTIC_CLASS_EXCEPTIONS
    assert len(used) > 25, "class extraction looks broken"
    missing = sorted(used - css_classes())
    assert not missing, f"classes used by JS/HTML but never styled: {missing}"


@pytest.mark.parametrize("sel", [
    ".msg.user", ".msg.bot", ".msg.err", ".toast.error", ".toast.ok", ".badge.ok", ".badge.warn",
    ".badge.bad", ".badge.wait", ".badge.done", ".item.active", ".drop.over", ".logo.big", ".line.wrap",
])
def test_compound_state_selectors_exist(sel):
    flat = [re.sub(r"\s+", "", s) for s in all_selectors()]
    assert any(sel in s for s in flat), f"no rule for {sel}"


def test_exception_list_is_honest():
    for c in SEMANTIC_CLASS_EXCEPTIONS:
        assert c in JS, f"exception {c!r} no longer used by app.js; remove it"
        assert c not in css_classes(), f"exception {c!r} is styled now; remove it from the list"


def test_id_selectors_exist_in_html():
    ids = set(re.findall(r'\bid="([^"]+)"', HTML))
    css_ids = set(re.findall(r"#([A-Za-z_][\w-]*)", "\n".join(all_selectors())))
    assert css_ids - ids == set(), f"CSS ids not in index.html: {sorted(css_ids - ids)}"


# ---------------------------------------------------------------- custom properties
def defined_props():
    props = set()
    for _p, b, _d in decl_blocks():
        for d in decls(b):
            m = re.match(r"(--[\w-]+)\s*:", d)
            if m:
                props.add(m.group(1))
    return props


def used_props():
    return set(re.findall(r"var\(\s*(--[\w-]+)", NOCOMMENT))


def test_all_vars_used_are_defined():
    undefined = sorted(used_props() - defined_props())
    assert not undefined, f"var() of undefined custom property: {undefined}"


def test_all_defined_custom_properties_are_used():
    unused = sorted(defined_props() - used_props())
    assert not unused, f"custom properties defined but never used: {unused}"


def test_brand_tokens_present():
    assert re.search(r"--page\s*:\s*#F4F0E8\b", NOCOMMENT, re.I)  # beige light page
    assert re.search(r"--accent\s*:\s*#B8532F\b", NOCOMMENT, re.I)  # Claude terracotta (AA-darkened)
    assert re.search(r"--accent\s*:\s*#D97757\b", NOCOMMENT, re.I)  # lightened for dark mode
    assert re.search(r"--page\s*:\s*#0B0B0B\b", NOCOMMENT, re.I)  # near-black dark page
    assert "var(--font)" in NOCOMMENT


# ---------------------------------------------------------------- dark mode parity
def _block(pred):
    for p, b, _d in BLOCKS:
        if pred(re.sub(r"\s+", "", p)):
            return b
    return None


def _tokens(body):
    return {m.group(1) for d in decls(body) if (m := re.match(r"(--[\w-]+|color-scheme)\s*:", d))}


GUARDED = lambda p: p == ":root:not([data-theme=light])"  # noqa: E731
EXPLICIT = lambda p: p == ":root[data-theme=dark]"  # noqa: E731


def test_dark_overrides_in_both_places_with_same_tokens():
    assert re.search(r"@media\(prefers-color-scheme:dark\)\{\s*:root:not\(\[data-theme=light\]\)\{",
                     NOCOMMENT), "prefers-color-scheme:dark with :root:not([data-theme=light]) guard missing"
    guarded, explicit = _block(GUARDED), _block(EXPLICIT)
    assert guarded and explicit, "one of the dark override blocks is missing"
    g, e = _tokens(guarded), _tokens(explicit)
    assert g == e, f"token sets differ: {sorted(g ^ e)}"
    assert {"--page", "--ink", "color-scheme"} <= e


def test_dark_override_values_identical():
    norm = lambda b: sorted(re.sub(r"\s+", "", d) for d in decls(b))  # noqa: E731
    assert norm(_block(GUARDED)) == norm(_block(EXPLICIT))


def test_dark_overrides_only_redefine_known_tokens():
    root = _tokens(_block(lambda p: p == ":root"))
    dark = _tokens(_block(EXPLICIT))
    assert dark - root == set(), f"dark overrides tokens not defined on :root: {sorted(dark - root)}"


# ---------------------------------------------------------------- style policy
def test_no_box_shadow_except_none():
    for m in re.finditer(r"box-shadow\s*:\s*([^;}]*)", NOCOMMENT):
        assert m.group(1).strip().lower() == "none", m.group(0)


def test_border_radius_policy():
    allowed = {"0", "4px", "50%", "inherit", "var(--radius)"}
    for m in re.finditer(r"border(?:-[a-z]+)*-radius\s*:\s*([^;}]*)", NOCOMMENT):
        assert m.group(1).strip().lower() in allowed, m.group(0)
    assert re.search(r"--radius\s*:\s*4px", NOCOMMENT)


def test_no_external_resources_or_webfonts():
    low = NOCOMMENT.lower()
    assert "@import" not in low
    assert "@font-face" not in low
    assert not re.search(r"url\(\s*['\"]?(https?:)?//", low)
    assert not re.search(r"https?://", low)
    for bad in ("googleapis", "gstatic", "typekit"):
        assert bad not in low


SYSTEM_FONTS = {"system-ui", "-apple-system", "segoe ui", "roboto", "helvetica neue", "helvetica", "arial",
                "sans-serif", "ui-monospace", "sfmono-regular", "menlo", "consolas", "liberation mono", "monospace",
                "inherit"}
# Netflix Sans is never loaded; naming it in the dark --font token only matters if the user has it installed.
DOCUMENTED_EXCEPTION = "netflix sans"


def _families(val):
    return [f.strip().strip("\"'").lower() for f in val.split(",")]


def test_font_family_only_system_stack():
    seen = 0
    for m in re.finditer(r"font-family\s*:\s*([^;}]*)", NOCOMMENT):
        seen += 1
        val = m.group(1).strip()
        if val.startswith("var("):
            continue
        for fam in _families(val):
            assert fam in SYSTEM_FONTS, f"non-system font {fam!r}"
    assert seen
    assert re.search(r"body\s*\{[^}]*font-family\s*:\s*var\(--font\)", NOCOMMENT)
    for fam in _families(re.search(r"--mono\s*:\s*([^;]*)", NOCOMMENT).group(1)):
        assert fam in SYSTEM_FONTS, fam
    # `font:` shorthand may only be `font:inherit` (a shorthand could smuggle in a family)
    for m in re.finditer(r"(?<![\w-])font\s*:\s*([^;}]*)", NOCOMMENT):
        assert m.group(1).strip() == "inherit", m.group(0)


def test_font_token_netflix_only_in_dark():
    light = _families(re.search(r"--font\s*:\s*([^;]*)", _block(lambda p: p == ":root")).group(1))
    assert set(light) <= SYSTEM_FONTS and DOCUMENTED_EXCEPTION not in light
    for blk in (_block(GUARDED), _block(EXPLICIT)):
        fams = _families(re.search(r"--font\s*:\s*([^;]*)", blk).group(1))
        assert fams[0] == DOCUMENTED_EXCEPTION
        assert set(fams) - {DOCUMENTED_EXCEPTION} <= SYSTEM_FONTS
        assert "helvetica" in fams
    assert NOCOMMENT.lower().count("netflix sans") == 2


def test_focus_visible_outline_rule():
    assert re.search(r":focus-visible\s*\{[^}]*outline\s*:\s*[^;}]*\bsolid\b", NOCOMMENT)
    assert not re.search(r"outline\s*:\s*(none|0)\b", NOCOMMENT), "outline removed somewhere"


# ---------------------------------------------------------------- responsive / drawer
def test_768_media_query_exists():
    assert re.search(r"@media\s*\(\s*min-width\s*:\s*768px\s*\)", NOCOMMENT)


def test_mobile_drawer_rules():
    assert re.search(r"body\.nav-open\s+aside\s*\{[^}]*visibility\s*:\s*visible", NOCOMMENT)
    assert re.search(r"aside\s*\{[^}]*visibility\s*:\s*hidden[^}]*\}", NOCOMMENT)
    assert re.search(r"aside\s*\{[^}]*transform\s*:\s*translateX\(-", NOCOMMENT)
    assert re.search(r"body\.nav-open\s+#scrim\s*\{[^}]*display\s*:\s*block", NOCOMMENT)
    assert re.search(r"#scrim\s*\{\s*display\s*:\s*none", NOCOMMENT)
    assert re.search(r"#menuBtn\s*\{\s*display\s*:\s*none", NOCOMMENT)
    assert re.search(r"max-width\s*:\s*767px", NOCOMMENT)  # pairs with min-width:768px, no gap


def test_hidden_attribute_and_reduced_motion():
    assert re.search(r"\[hidden\]\s*\{\s*display\s*:\s*none\s*!important", NOCOMMENT)
    assert "prefers-reduced-motion" in NOCOMMENT


# ---------------------------------------------------------------- contrast
def hex_rgb(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def lum(rgb):
    def f(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (f(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def mix(a, b, pct):
    return tuple(round(a[i] * pct / 100 + b[i] * (100 - pct) / 100) for i in range(3))


def _decl_map(body):
    return {k.strip(): v.strip() for k, _, v in (d.partition(":") for d in decls(body))}


SHARED = ":root,aside,header,.toast"
ROOT_D = _decl_map(_block(lambda p: p == ":root"))
DARK_D = _decl_map(_block(EXPLICIT))
FALLBACK_D = _decl_map(next(b for p, b, d in BLOCKS if d == 0 and re.sub(r"\s+", "", p) == SHARED))
SUPPORTS_D = _decl_map(next(b for p, b, d in BLOCKS if d == 1 and re.sub(r"\s+", "", p) == SHARED))
LIGHT = {**ROOT_D, **FALLBACK_D, **SUPPORTS_D}
DARK = {**LIGHT, **DARK_D}
BAR_PAGE = {"--page": "var(--bar-bg)"}  # what aside/header/.toast re-point


def resolve(name, scope, depth=0):
    assert depth < 10, "token cycle"
    return resolve_value(scope[name], scope, depth)


def resolve_value(v, scope, depth=0):
    """#hex, var(), rgba(var(--ink-rgb),a) blended over --page, and two-colour color-mix(in srgb, A x%, B)."""
    v = v.strip()
    if v.startswith("#"):
        return hex_rgb(v)
    m = re.fullmatch(r"var\((--[\w-]+)\)", v)
    if m:
        return resolve(m.group(1), scope, depth + 1)
    m = re.fullmatch(r"rgba\(var\(--ink-rgb\),\s*([\d.]+)\)", v)
    if m:
        ink = tuple(int(x) for x in scope["--ink-rgb"].split(","))
        return mix(ink, resolve("--page", scope, depth + 1), round(float(m.group(1)) * 100))
    m = re.fullmatch(r"color-mix\(in srgb,\s*(.+?)\s+(\d+)%,\s*(.+)\)", v)
    if m and "transparent" not in m.group(3):
        return mix(resolve_value(m.group(1), scope, depth + 1),
                   resolve_value(m.group(3), scope, depth + 1), int(m.group(2)))
    return None


def bars(scope):
    return {**scope, **BAR_PAGE}


MODES = [("light", LIGHT), ("dark", DARK)]
ALL_SCOPES = [("light", LIGHT), ("dark", DARK), ("light-bar", bars(LIGHT)), ("dark-bar", bars(DARK))]


def test_hex_tokens_resolve():
    assert resolve("--page", LIGHT) == (0xF4, 0xF0, 0xE8)
    assert resolve("--accent", LIGHT) == (0xB8, 0x53, 0x2F)
    assert resolve("--page", DARK) == (0x0B, 0x0B, 0x0B)
    assert resolve("--muted", LIGHT) is not None


def test_derived_tokens_declared_where_page_is_repointed():
    flat = re.sub(r"\s+", "", NOCOMMENT)
    m = re.search(r"([^{}]+)\{--page:var\(--bar-bg\)\}", flat)
    assert m, "bar scope that re-points --page missing"
    assert set(m.group(1).split(",")) == {"aside", "header", ".toast"}
    for tok in ("--muted", "--line", "--tint", "--code-bg"):
        assert tok in FALLBACK_D, f"{tok} not declared on the shared scope list"
        assert tok not in ROOT_D and tok not in DARK_D, f"{tok} declared on :root only; would not re-evaluate in bars"


def test_color_mix_has_plain_fallback_first():
    assert re.search(r"@supports\s*\(\s*color\s*:\s*color-mix\(", NOCOMMENT)
    outside = re.sub(r"@supports\s*\(color:color-mix\(in srgb,red,blue\)\)", "", NOCOMMENT.split("@supports")[0])
    assert "color-mix(" not in outside, "color-mix used before/outside @supports"
    assert NOCOMMENT.count("color-mix(") == len(SUPPORTS_D) + 1  # +1 for the @supports condition itself
    for tok, val in SUPPORTS_D.items():
        assert "color-mix(" in val
        assert tok in FALLBACK_D and "color-mix" not in FALLBACK_D[tok], f"no plain fallback for {tok}"


def test_scrim_is_plain_rgba():
    assert re.search(r"--scrim\s*:\s*rgba\(", NOCOMMENT)
    assert re.search(r"::backdrop\s*\{[^}]*var\(--scrim\)", NOCOMMENT)
    assert re.search(r"body\.nav-open\s+#scrim\s*\{[^}]*var\(--scrim\)", NOCOMMENT)


def test_disabled_buttons_do_not_react_to_hover():
    assert "button:not(:disabled):hover" in NOCOMMENT
    assert "button:not(:disabled):active" in NOCOMMENT
    for m in re.finditer(r"(?<![\w.#-])button(?:\.[\w-]+)*(:hover|:active)", NOCOMMENT):
        raise AssertionError(f"bare button{m.group(1)} would react while disabled")


def test_coarse_pointer_touch_targets():
    m = re.search(r"@media\s*\(\s*pointer\s*:\s*coarse\s*\)\s*\{([^@]*?\})\s*\}", NOCOMMENT)
    assert m, "pointer:coarse rule missing"
    body = m.group(1)
    for sel in ("button.icon", "button.small", ".item .del", ".line button", ".code-h button"):
        assert sel in body, sel
    assert re.search(r"min-height\s*:\s*40px", body) and re.search(r"min-width\s*:\s*40px", body)


def test_active_item_focus_ring_contrasts_with_pill():
    m = re.search(r"\.item\.active \.open:focus-visible\s*\{[^}]*outline\s*:\s*2px solid var\((--[\w-]+)\)", NOCOMMENT)
    assert m, "active item needs its own focus ring"
    for mode, scope in MODES:
        assert ratio(resolve(m.group(1), scope), resolve("--accent", scope)) >= 3.0, mode


@pytest.mark.parametrize("mode,scope", ALL_SCOPES)
def test_ink_on_page_contrast(mode, scope):
    r = ratio(resolve("--ink", scope), resolve("--page", scope))
    assert r >= 4.5, f"{mode}: ink on page {r:.2f}"


@pytest.mark.parametrize("mode,scope", MODES)
def test_ink_on_other_surfaces(mode, scope):
    for surf in ("--bot", "--tint", "--code-bg", "--bar-bg"):
        r = ratio(resolve("--ink", scope), resolve(surf, scope))
        assert r >= 4.5, f"{mode}: ink on {surf} {r:.2f}"


@pytest.mark.parametrize("mode,scope", MODES)
def test_accent_text_contrast(mode, scope):
    for bg in ("--accent", "--accent-hover"):  # filled buttons, user bubble, active item, dialog header, drop zone
        r = ratio(resolve("--accent-fg", scope), resolve(bg, scope))
        assert r >= 4.5, f"{mode}: accent-fg on {bg} {r:.2f}"
    for surf in ("--page", "--bot", "--tint", "--code-bg", "--bar-bg"):  # links and focus rings
        r = ratio(resolve("--link", scope), resolve(surf, scope))
        assert r >= 4.5, f"{mode}: link on {surf} {r:.2f}"


@pytest.mark.parametrize("mode,scope", ALL_SCOPES)
def test_muted_text_contrast(mode, scope):
    surfaces = ("--page", "--tint", "--code-bg") + (() if "bar" in mode else ("--bot",))
    for surf in surfaces:
        r = ratio(resolve("--muted", scope), resolve(surf, scope))
        assert r >= 4.5, f"{mode}: muted on {surf} {r:.2f}"


@pytest.mark.parametrize("mode,scope", ALL_SCOPES)
def test_muted_plain_fallback_contrast(mode, scope):
    plain = {**scope, **FALLBACK_D}
    for surf in ("--page", "--tint"):
        r = ratio(resolve("--muted", plain), resolve(surf, scope))
        assert r >= 4.5, f"{mode}: fallback muted on {surf} {r:.2f}"


@pytest.mark.parametrize("mode,scope", ALL_SCOPES)
def test_status_colours_on_surfaces(mode, scope):
    surfaces = ("--page", "--tint") + (() if "bar" in mode else ("--bot",))
    for tok in ("--ok", "--warn", "--err"):
        for surf in surfaces:
            r = ratio(resolve(tok, scope), resolve(surf, scope))
            assert r >= 4.5, f"{mode}: {tok} on {surf} {r:.2f}"
