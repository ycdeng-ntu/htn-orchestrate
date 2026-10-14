"""Avoid adding usable research pages to the failed-fetch registry."""
from __future__ import annotations

import unittest

from workflows.dynamic_htn_informed_research.tools._pure import is_garbage_content


class FetchContentQualityTests(unittest.TestCase):
    def test_ordinary_articles_about_robots_and_web_security_are_usable(self):
        articles = (
            "Robot navigation research\nA robot uses several sensors to map its surroundings.",
            "Cloudflare announces new services\nCloudflare provides network and security products.",
            "CAPTCHA accessibility\nCAPTCHA challenges can exclude people with disabilities.",
            "A history of access control\nThe server displayed access denied during the experiment.",
            "Browser check technologies have improved over time. This article compares their designs.",
            "Security verification in distributed systems\nWe study several verification protocols.",
        )
        for article in articles:
            with self.subTest(article=article):
                self.assertFalse(is_garbage_content(article))

    def test_clear_challenge_and_access_walls_are_rejected(self):
        blocked = (
            "Access denied",
            "# Access denied\nYou do not have permission to access this resource.",
            "Title: Access denied\nURL Source: https://example.test/blocked",
            "Access denied. Please contact the administrator.",
            "Cloudflare\nPlease verify that you are human before continuing.",
            "Verify you are human to continue.",
            "Browser check\nChecking your browser before accessing the site.",
            "Checking if the site connection is secure. Please wait.",
            "Enable JavaScript and cookies to continue",
            "Please enable cookies to continue.",
            "Sign in to access this content.",
            "Subscribe to read this article.",
            "Purchase this article",
        )
        for page in blocked:
            with self.subTest(page=page):
                self.assertTrue(is_garbage_content(page))

    def test_empty_pages_fail_but_short_valid_text_and_isolated_names_do_not(self):
        for empty in ("", " ", "\n\t\r "):
            with self.subTest(empty=empty):
                self.assertTrue(is_garbage_content(empty))
        for text in ("OK", "42", "Robot", "Cloudflare", "CAPTCHA", "Institutional login"):
            with self.subTest(text=text):
                self.assertFalse(is_garbage_content(text))

    def test_substantial_article_with_challenge_text_is_preserved(self):
        article = (
            "Browser check\n"
            "This research article explains the history of browser verification. " * 40
        )
        self.assertGreater(len(article), 2000)
        self.assertFalse(is_garbage_content(article))

    def test_body_discussion_and_long_opening_paragraph_are_not_block_pages(self):
        late_instruction = (
            "Research on accessible verification\n"
            + "The study measured user experience and successful completion. " * 8
            + "\nVerify you are human to continue.\nThat was the text shown in the experiment."
        )
        long_paragraph = (
            "Verify you are human to continue. "
            + "This quoted interface message is the subject of the article. " * 5
        )
        self.assertFalse(is_garbage_content(late_instruction))
        self.assertFalse(is_garbage_content(long_paragraph))


if __name__ == "__main__":
    unittest.main()
