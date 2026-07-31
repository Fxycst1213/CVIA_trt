from __future__ import annotations

import sys
import unittest
from pathlib import Path


WEB_MONITOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_MONITOR))

from server import tracker_selector  # noqa: E402


class MocapTargetSelectorTests(unittest.TestCase):
    def test_name_selector_preserves_spaces(self) -> None:
        self.assertEqual(
            tracker_selector("name", "  Tracker 4  "),
            "name:Tracker 4",
        )

    def test_id_selector_accepts_integer_or_canonical_text(self) -> None:
        self.assertEqual(tracker_selector("id", 0), "id:0")
        self.assertEqual(tracker_selector("id", "17"), "id:17")

    def test_invalid_target_values_are_rejected(self) -> None:
        for mode, value in (
            ("name", ""),
            ("name", "Tracker\n4"),
            ("id", -1),
            ("id", "1.5"),
            ("id", True),
            ("other", "Tracker4"),
        ):
            with self.subTest(mode=mode, value=value):
                with self.assertRaises(ValueError):
                    tracker_selector(mode, value)


if __name__ == "__main__":
    unittest.main()
