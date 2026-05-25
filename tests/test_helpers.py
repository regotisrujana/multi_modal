from __future__ import annotations

import unittest

from utils.helpers import (
    detect_file_type,
    parse_json_response,
    get_ffmpeg_executable,
    require_extractable_text,
    safe_filename,
)


class HelperTests(unittest.TestCase):
    def test_detect_supported_and_legacy_file_types(self):
        self.assertEqual(detect_file_type("resume.PDF"), "resume_pdf")
        self.assertEqual(detect_file_type("portfolio.pptx"), "portfolio_ppt")
        self.assertEqual(detect_file_type("old_resume.doc"), "unsupported_legacy_word")
        self.assertEqual(
            detect_file_type("old_deck.ppt"),
            "unsupported_legacy_presentation",
        )

    def test_parse_json_response_from_fenced_text(self):
        parsed = parse_json_response('```json\n{"score": 92}\n```')
        self.assertEqual(parsed, {"score": 92})

    def test_safe_filename_removes_windows_reserved_chars(self):
        self.assertEqual(safe_filename('Jane:Doe<>Report?.txt'), "Jane_Doe__Report_.txt")

    def test_require_extractable_text_rejects_blank_result(self):
        with self.assertRaises(ValueError):
            require_extractable_text({"text": "   "}, "blank.pdf")

    def test_ffmpeg_lookup_returns_none_or_existing_path(self):
        ffmpeg = get_ffmpeg_executable()
        self.assertTrue(ffmpeg is None or len(ffmpeg) > 0)


if __name__ == "__main__":
    unittest.main()
