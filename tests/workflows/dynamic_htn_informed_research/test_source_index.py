from __future__ import annotations

from workflows.dynamic_htn_informed_research.source_index import (
    build_cross_host_title_collision_warning,
    build_observed_source_index,
)


def _card(url: str, title: str) -> dict:
    return {
        "claim": "captured snippet",
        "source": {"url": url, "title": title},
        "supporting_quotes": ["captured snippet"],
    }


def test_source_index_matches_exact_and_canonical_urls() -> None:
    report = """
Evidence:
- https://example.com/news/item — exact
- http://www.example.org/story/?utm_source=test#details — canonical
"""
    rendered = build_observed_source_index(
        report,
        [
            _card("https://example.com/news/item", "Exact title"),
            _card("https://example.org/story", "Canonical title"),
        ],
    )

    assert rendered is not None
    assert '[exact_observed] "Exact title"' in rendered
    assert '[canonical_observed] "Canonical title"' in rendered
    assert rendered.count("https://example.com/news/item") == 1


def test_source_index_unwraps_known_jina_proxy() -> None:
    rendered = build_observed_source_index(
        "Evidence: https://example.com/archive/story",
        [_card("https://r.jina.ai/https://example.com/archive/story", "Archived story")],
    )

    assert rendered is not None
    assert '[canonical_observed] "Archived story"' in rendered


def test_source_index_uses_unique_same_site_fuzzy_match() -> None:
    rendered = build_observed_source_index(
        "Evidence: https://example.com/news/restaurant-accident-four-waiters",
        [
            _card(
                "https://example.com/news/restaurant-accident-four-waiter",
                "Restaurant accident",
            )
        ],
    )

    assert rendered is not None
    assert '[fuzzy_observed score=' in rendered
    assert '"Restaurant accident"' in rendered


def test_source_index_does_not_fuzzy_match_across_sites() -> None:
    rendered = build_observed_source_index(
        "Evidence: https://reported.example/news/restaurant-accident-four-waiters",
        [
            _card(
                "https://observed.example/news/restaurant-accident-four-waiters",
                "Wrong-site title",
            )
        ],
    )

    assert rendered is not None
    assert "[unobserved]" in rendered
    assert "Wrong-site title" not in rendered


def test_source_index_rejects_ambiguous_fuzzy_match() -> None:
    rendered = build_observed_source_index(
        "Evidence: https://example.com/news/restaurant-accident-four-waiters",
        [
            _card(
                "https://example.com/news/restaurant-accident-four-waiter",
                "Candidate one",
            ),
            _card(
                "https://example.com/news/restaurant-accident-five-waiters",
                "Candidate two",
            ),
        ],
    )

    assert rendered is not None
    assert "[ambiguous]" in rendered
    assert "Candidate one" not in rendered
    assert "Candidate two" not in rendered


def test_source_index_marks_report_only_url_unobserved() -> None:
    rendered = build_observed_source_index(
        "Evidence: [claim](https://example.com/not-in-tool-trace).",
        [],
    )

    assert rendered is not None
    assert "[unobserved] https://example.com/not-in-tool-trace" in rendered


def test_collision_warning_flags_only_exact_normalized_titles_across_hosts() -> None:
    warning = build_cross_host_title_collision_warning(
        [
            '- [exact_observed] "Shared: Headline!" — https://news.example/story',
            '- [canonical_observed] "shared headline" — observed: '
            "https://mirror.example/copy | reported: https://mirror.example/amp/copy",
            '- [exact_observed] "Different headline" — https://third.example/story',
        ]
    )

    assert warning is not None
    assert '"Shared: Headline!" appears on distinct hosts: mirror.example, news.example' in warning
    assert "Different headline" not in warning
    assert "not a source-family inference" in warning
    assert "title equality alone is not a reason to discard it" in warning


def test_collision_warning_ignores_same_host_and_unobserved_urls() -> None:
    warning = build_cross_host_title_collision_warning(
        [
            '- [exact_observed] "Same title" — https://example.com/one',
            '- [exact_observed] "Same title" — https://example.com/two',
            "- [unobserved] https://other.example/three",
        ]
    )

    assert warning is None


def test_collision_warning_ignores_generic_extraction_titles() -> None:
    warning = build_cross_host_title_collision_warning(
        [
            '- [exact_observed] "**Extracted Information:**" — '
            "https://first.example/story",
            '- [canonical_observed] "### EXTRACTED INFORMATION" — observed: '
            "https://second.example/copy | reported: https://second.example/amp/copy",
            '- [exact_observed] "Extracted Information about Tom Hauge" — '
            "https://third.example/profile",
            '- [exact_observed] "**Extracted Information about Tom Hauge**" — '
            "https://fourth.example/profile",
        ]
    )

    assert warning is None


def test_collision_warning_keeps_real_collision_alongside_generic_title() -> None:
    warning = build_cross_host_title_collision_warning(
        [
            '- [exact_observed] "Extracted Information" — '
            "https://first.example/story",
            '- [exact_observed] "**EXTRACTED INFORMATION**" — '
            "https://second.example/copy",
            '- [exact_observed] "Real Shared Headline" — '
            "https://news.example/story",
            '- [exact_observed] "real shared headline" — '
            "https://mirror.example/copy",
        ]
    )

    assert warning is not None
    assert "Extracted Information" not in warning
    assert '"Real Shared Headline" appears on distinct hosts' in warning
