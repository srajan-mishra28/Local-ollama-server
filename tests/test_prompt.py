"""Pure prompt helpers: the retrieval query and the session history window."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag  # noqa: E402
from backend.prompt import build_retrieval_query, fit_history  # noqa: E402


def u(text):
    return {"role": "user", "content": text}


def a(text):
    return {"role": "assistant", "content": text}


# ---- build_retrieval_query ----------------------------------------------------------

def test_first_message_is_the_query_alone():
    assert build_retrieval_query([], "what is in the report?") == "what is in the report?"


def test_follow_up_gets_the_previous_user_turn():
    history = [u("list the invoices in march"), a("There are three: ...")]
    assert build_retrieval_query(history, "and the second one?") == "list the invoices in march\nand the second one?"


def test_assistant_replies_are_left_out():
    history = [u("first question"), a("a very long answer " * 50), u("second question"), a("another answer")]
    assert build_retrieval_query(history, "third") == "first question\nsecond question\nthird"


def test_newest_turns_win_the_history_budget_and_oldest_is_cut_to_its_tail():
    history = [u("a" * 100), u("b" * 100), u("c" * 100)]
    q = build_retrieval_query(history, "now", history_chars=150)
    parts = q.split("\n")
    assert parts[-1] == "now" and parts[-2] == "c" * 100
    assert parts[0] == "b" * 49  # 150 - 101 left for the next older turn, keeping its tail
    assert len(parts) == 3


def test_history_part_is_bounded_by_default_constant():
    history = [u("x" * 5000)] * 5
    q = build_retrieval_query(history, "q")
    assert len(q) <= 400 + 1 + 1  # budget + newline + latest


def test_latest_message_is_capped_and_crowds_out_history():
    q = build_retrieval_query([u("earlier")], "y" * 5000, history_chars=400, max_chars=1000)
    assert q == "y" * 1000  # no room left for history
    assert len(build_retrieval_query([u("e" * 300)], "y" * 900, history_chars=400, max_chars=1000)) <= 1001


def test_blank_turns_are_ignored():
    assert build_retrieval_query([u("   "), u("real")], "now") == "real\nnow"


# ---- fit_history -----------------------------------------------------------------------------

def msgs(*sizes):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"{i}" * n} for i, n in enumerate(sizes)]


def test_everything_fits_returns_all_in_order():
    m = msgs(10, 10, 10)
    assert fit_history(m, budget=100) == m


def test_oldest_messages_drop_first_and_order_is_preserved():
    m = msgs(40, 40, 40, 40, 10)
    kept = fit_history(m, budget=85)
    assert kept == [m[3], m[4]]  # 10 + 40 fit; a third 40 would exceed 85


def test_latest_message_is_always_kept_even_alone_over_budget():
    m = msgs(10, 10, 5000)
    assert fit_history(m, budget=100) == [m[-1]]


def test_window_stops_at_first_message_that_does_not_fit():
    # a small old message must not be kept when a bigger one between it and the latest was dropped
    m = msgs(5, 500, 5, 5)
    assert fit_history(m, budget=50) == [m[2], m[3]]


def test_budget_boundary_is_inclusive():
    m = msgs(30, 30, 40)
    assert fit_history(m, budget=100) == m
    assert fit_history(m, budget=99) == [m[1], m[2]]


def test_single_and_empty():
    assert fit_history([]) == []
    one = msgs(5)
    assert fit_history(one, budget=1) == one


@pytest.mark.parametrize("budget", [0, 1, 10, 55, 1000])
def test_result_is_a_suffix_and_within_budget_unless_only_latest(budget):
    m = msgs(20, 7, 33, 12, 25)
    kept = fit_history(m, budget=budget)
    assert kept == m[len(m) - len(kept):] and kept[-1] is m[-1]
    assert len(kept) == 1 or sum(len(x["content"]) for x in kept) <= budget


# ---- rag.fit_hits (excerpt size bound) -----------------------------------------------------------------

def hit(n, path="a.md"):
    return {"path": path, "chunk": 0, "text": "t" * n, "score": 1.0}


def test_fit_hits_keeps_best_first_until_the_budget_is_spent():
    hits = [hit(100, "1"), hit(100, "2"), hit(100, "3")]
    assert [h["path"] for h in rag.fit_hits(hits, 250)] == ["1", "2"]
    assert [h["path"] for h in rag.fit_hits(hits, 300)] == ["1", "2", "3"]


def test_fit_hits_always_keeps_the_best_hit_cut_to_size():
    out = rag.fit_hits([hit(1000), hit(10, "b")], 100)
    assert len(out) == 1 and len(out[0]["text"]) == 100


def test_fit_hits_stops_at_first_that_does_not_fit_and_empty_is_empty():
    assert [h["path"] for h in rag.fit_hits([hit(60, "1"), hit(60, "2"), hit(5, "3")], 100)] == ["1"]
    assert rag.fit_hits([], 100) == []
