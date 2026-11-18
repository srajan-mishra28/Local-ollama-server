"""Static checks of the web UI (no browser, and no JS execution: node is not available)."""
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app as server  # noqa: E402

HTML = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
JS = (ROOT / "static" / "app.js").read_bytes().decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def page():
    r = TestClient(server.app).get("/")
    assert r.status_code == 200
    return r


def test_index_served_as_html(page):
    assert page.headers["content-type"].startswith("text/html")


def test_no_inline_script_style_or_handlers(page):
    body = page.text
    for m in re.finditer(r"<script\b([^>]*)>(.*?)</script>", body, re.S | re.I):
        assert re.search(r"\bsrc\s*=", m.group(1)), f"inline script: {m.group(0)[:60]}"
        assert not m.group(2).strip()
    assert not re.search(r"<style\b", body, re.I)
    assert not re.search(r"\sstyle\s*=", body, re.I)
    assert not re.search(r"\son[a-z]+\s*=", body, re.I)
    assert not re.search(r"javascript:", body, re.I)


def test_references_assets_that_are_served(page):
    for ref in ("/static/theme.js", "/static/app.js", "/static/style.css"):
        assert ref in page.text
        assert TestClient(server.app).get(ref).status_code == 200


def test_theme_script_loads_before_stylesheet_and_app_script_is_last(page):
    t = page.text
    assert t.index("theme.js") < t.index("style.css") < t.index("app.js")
    assert t.index("app.js") > t.index("</main>")


def html_ids():
    ids = re.findall(r'\bid="([^"]+)"', HTML)
    return ids


def test_html_ids_unique():
    ids = html_ids()
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})


def test_all_ids_looked_up_by_app_js_exist_in_html():
    ids = set(html_ids())
    wanted = set(re.findall(r"\$\(\s*['\"]([\w-]+)['\"]\s*\)", JS))
    wanted |= set(re.findall(r"getElementById\(\s*['\"]([\w-]+)['\"]\s*\)", JS))
    wanted |= set(re.findall(r"querySelector(?:All)?\(\s*['\"]#([\w-]+)", JS))
    assert len(wanted) > 20  # the regex really found the lookups
    assert wanted - ids == set(), f"app.js looks up ids missing from index.html: {sorted(wanted - ids)}"


def test_html_cross_references_resolve():
    ids = set(html_ids())
    refs = set(re.findall(r'\b(?:for|aria-controls|aria-labelledby)="([^"]+)"', HTML))
    assert refs - ids == set()


def _routes():
    """[(path template, {METHOD: required query param names})] from the OpenAPI schema.

    (app.routes can hold opaque included-router objects in recent FastAPI.)"""
    out = []
    for path, ops in server.app.openapi()["paths"].items():
        methods = {
            m.upper(): {p["name"] for p in op.get("parameters", []) if p["in"] == "query" and p.get("required")}
            for m, op in ops.items()
        }
        out.append((path, methods))
    return out


def _js_calls():
    """[(method, path pattern with {} for dynamic parts, query keys, line no)] for /api/ URLs in app.js."""
    calls = []
    for n, line in enumerate(JS.splitlines(), 1):
        for m in re.finditer(r"""(?:'(/api/[^']*)'(\s*\+)?|`(/api/[^`]*)`)""", line):
            if m.group(3) is not None:
                url = re.sub(r"\$\{[^}]*\}", "{}", m.group(3))
            else:
                url = m.group(1) + ("{}" if m.group(2) else "")
            path, _, query = url.partition("?")
            qkeys = set(re.findall(r"(?:^|&)(\w+)=", query))
            rest = line[m.end():]
            mm = re.search(r"method:\s*'(\w+)'", rest) or re.search(r"\bjson\(\s*'(\w+)'", rest)
            method = mm.group(1).upper() if mm else "GET"
            calls.append((method, path, qkeys, n))
    return calls


def _match(route_path: str, path: str) -> bool:
    pattern = re.sub(r"\{[^}]+\}", r"[^/]+", route_path)
    return re.fullmatch(pattern, path.replace("{}", "1")) is not None


def test_js_calls_found():
    assert len(_js_calls()) >= 12


def test_every_api_url_in_app_js_has_a_registered_route_with_that_method():
    problems = []
    for method, path, _q, n in _js_calls():
        cands = [(rp, ms) for rp, ms in _routes() if _match(rp, path)]
        if not cands:
            problems.append(f"app.js:{n} {method} {path}: no route with that path")
        elif not any(method in ms for _, ms in cands):
            have = sorted({m for _, ms in cands for m in ms})
            problems.append(f"app.js:{n} {method} {path}: path exists but only for {have}")
    assert not problems, "; ".join(problems)


def test_required_query_params_are_sent_by_app_js():
    problems = []
    for method, path, qkeys, n in _js_calls():
        for rp, ms in _routes():
            if _match(rp, path) and method in ms and ms[method] - qkeys:
                problems.append(f"app.js:{n} {method} {path} lacks required query {sorted(ms[method] - qkeys)}")
    assert not problems, "; ".join(problems)


def test_upload_forms_send_the_right_field_names():
    assert re.search(r"append\(\s*'file'", JS)  # general file transfer: POST /api/upload
    assert re.search(r"append\(\s*'files'", JS)  # project upload: POST /api/projects/{id}/files


def test_js_uses_the_project_workspace_endpoints():
    calls = {(m, p) for m, p, _q, _n in _js_calls()}
    for want in [
        ("GET", "/api/projects"), ("POST", "/api/projects"), ("GET", "/api/projects/{}"),
        ("DELETE", "/api/projects/{}"), ("POST", "/api/projects/{}/sessions"),
        ("POST", "/api/projects/{}/files"), ("DELETE", "/api/projects/{}/files"),
        ("GET", "/api/sessions"), ("POST", "/api/sessions"), ("POST", "/api/chat"),
        ("PUT", "/api/sessions/{}"), ("GET", "/api/health"),
    ]:
        assert want in calls, want


@pytest.mark.parametrize("gone", ["/context", "/grep", "/index", "/upload'", "ctxDlg", "ctxSelected", "idxDlg"])
def test_folder_and_pinned_file_ui_is_gone(gone):
    assert gone not in JS.replace("'/api/upload'", "") and gone not in HTML


def test_sidebar_has_chats_and_projects_sections():
    for needle in ('id="sessions"', 'id="projects"', 'id="newBtn"', 'id="newProjBtn"', "New chat", "New project"):
        assert needle in HTML, needle


def test_project_view_has_upload_and_file_list_controls():
    for i in ("projectView", "projNewChat", "dropZone", "projUpload", "upResult", "projFiles", "projDelBtn"):
        assert f'id="{i}"' in HTML, i
    assert re.search(r'id="projUpload"[^>]*multiple', HTML)


def test_project_delete_requires_confirm():
    m = re.search(r"\$\('projDelBtn'\)\.onclick = guard\(async \(\) => \{(.*?)\n\}\);", JS, re.S)
    assert m and m.group(1).index("confirm(") < m.group(1).index("method: 'DELETE'")


def test_sources_header_name_matches_the_server():
    from backend.routes.chat import SOURCES_HEADER
    assert f"'{SOURCES_HEADER}'" in JS


def test_innerhtml_only_gets_escaped_markdown_or_a_fixed_literal():
    uses = re.findall(r"\.innerHTML\s*=\s*([^;\n]*)", JS)
    assert uses  # the regex found them
    for u in uses:
        assert u.startswith("markdown(") or u.startswith("'<span class=\"typing\""), u


@pytest.mark.parametrize("name", ["index.html", "app.js", "style.css", "theme.js"])
def test_static_files_are_valid_utf8(name):
    (ROOT / "static" / name).read_bytes().decode("utf-8")


# ---- user-feedback fixes: stop, retry, rename, notes, health banner, counter ----------------

def test_new_controls_exist_in_the_html():
    for i in ("stop", "renameBtn", "health", "count"):
        assert f'id="{i}"' in HTML, i
    assert re.search(r'id="stop"[^>]*aria-label="[^"]+"', HTML) and re.search(r'id="renameBtn"[^>]*aria-label="[^"]+"', HTML)
    assert re.search(r'id="health"[^>]*aria-live="polite"', HTML)


def test_response_header_names_match_the_server():
    from backend.routes.chat import STATUS_HEADER, TRIMMED_HEADER
    assert f"'{STATUS_HEADER}'" in JS and f"'{TRIMMED_HEADER}'" in JS


def test_every_rag_status_the_server_can_send_has_a_note_in_the_ui():
    from backend.routes.chat import RAG_HINTS
    for status in RAG_HINTS:  # none, empty, failed
        assert re.search(rf"\b{status}:\s*'", JS), status
    assert "'Used project files: '" in JS
    assert "No project files matched this question" in JS and "Could not search project files" in JS
    assert "Earlier messages are not sent to the model" in JS


def test_stop_uses_an_abort_controller_on_the_chat_fetch():
    assert "new AbortController()" in JS and "signal: ctl.signal" in JS and ".abort()" in JS
    assert re.search(r"\$\('stop'\)\.onclick", JS) and "AbortError" in JS


def test_retry_button_resends_the_same_text():
    assert re.search(r"class: 'ghost retry'", JS) and re.search(r"send\(text, false\)", JS)


def test_rename_has_button_enter_escape_and_double_click():
    assert "json('PUT'" in JS and "'Rename ' + s.title" in JS
    assert re.search(r"e\.key === 'Enter'.*?finish\(true", JS, re.S) and re.search(r"e\.key === 'Escape'.*?finish\(false", JS, re.S)
    assert "addEventListener('dblclick'" in JS
    assert "oncontextmenu" in JS  # the old shortcut still works, but delete and rename also have buttons
    assert "'aria-label': 'Delete ' + s.title" in JS


def test_rename_input_is_built_with_the_dom_api_not_innerhtml():
    assert re.search(r"el\('input', \{class: 'rename'", JS)


def test_health_is_polled_every_30_seconds_and_cannot_throw():
    assert re.search(r"setInterval\(checkHealth,\s*30000\)", JS)
    body = re.search(r"async function checkHealth\(\) \{(.*?)\n\}", JS, re.S).group(1)
    assert "try {" in body and "catch" in body
    for text in ("Ollama not reachable", "ollama pull"):
        assert text in JS


def test_counter_shows_near_the_limit_and_the_limit_comes_from_the_server():
    assert "maxChars * 0.9" in JS and "max_message_chars" in JS
    from backend.routes.health import router  # noqa: F401  (the field is part of /api/health)
    assert "max_message_chars" in (ROOT / "backend" / "routes" / "health.py").read_text(encoding="utf-8")


def test_error_text_shows_the_servers_detail_without_a_duplicate_prefix():
    assert "Ollama did not answer properly. Is it running" not in JS
    assert re.search(r"return detail \|\| HINTS\[r\.status\]", JS)


def test_failed_message_is_put_back_in_the_input_when_refused():
    assert "REFUSED" in JS and re.search(r"q\.value = text", JS)
