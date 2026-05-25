from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from vectorstore import chroma_db


class ChromaDbTests(unittest.TestCase):
    def test_embed_texts_falls_back_when_transformer_unavailable(self):
        with patch("vectorstore.chroma_db._get_embedder", side_effect=RuntimeError("boom")):
            vectors = chroma_db.embed_texts(["python streamlit resume analysis"])

        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 384)
        self.assertTrue(any(value != 0.0 for value in vectors[0]))
        norm = math.sqrt(sum(value * value for value in vectors[0]))
        self.assertAlmostEqual(norm, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
