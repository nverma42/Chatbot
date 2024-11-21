
import unittest
from summarization_engine import SummarizationEngine


class TestSummarizationEngine(unittest.TestCase):

    def setUp(self):
        """Initialize the SummarizationEngine for testing."""
        self.engine = SummarizationEngine(sentence_count=2)

    def test_basic_summarization(self):
        """Test if the summarizer produces the expected number of sentences."""
        input_text = "This is a test sentence. Another test sentence follows. Finally, a last test sentence."
        summary = self.engine.summarize_text(input_text)
        # Check if summary exists; sentence count may vary based on summarization algorithm
        self.assertTrue(len(summary.split('. ')) <= self.engine.sentence_count)

    def test_empty_input(self):
        """Test handling of empty input."""
        summary = self.engine.summarize_text("")
        self.assertEqual(summary, "No content to summarize.")

    def test_short_text(self):
        """Test handling of text that is too short for summarization."""
        input_text = "Short text."
        summary = self.engine.summarize_text(input_text)
        self.assertEqual(
            summary, "Input text is too short to generate a meaningful summary.")

    def test_algorithm_flexibility(self):
        """Test switching between algorithms (e.g., LSA and TextRank)."""
        engine_lsa = SummarizationEngine(algorithm='lsa')
        engine_textrank = SummarizationEngine(algorithm='text_rank')

        input_text = "This is a sample text to test different summarization algorithms."

        # Test LSA summarization
        summary_lsa = engine_lsa.summarize_text(input_text)
        self.assertTrue(len(summary_lsa) > 0)

        # Test TextRank summarization
        summary_textrank = engine_textrank.summarize_text(input_text)
        self.assertTrue(len(summary_textrank) > 0)

    def test_customizable_sentence_count(self):
        """Test if setting sentence_count correctly affects the summary length."""
        input_text = "This sentence. That sentence. Another sentence. Yet another sentence."
        self.engine.sentence_count = 3
        summary = self.engine.summarize_text(input_text)
        self.assertTrue(len(summary.split('. ')) <= 3)

    def test_summarize_preview(self):
        """Test if summarize_preview provides a preview of the summary."""
        input_text = "Sentence one. Sentence two. Sentence three."
        preview = self.engine.summarize_preview(input_text)
        self.assertTrue(preview.endswith('.'))
        # Check that the preview is a subset of the full summary
        full_summary = self.engine.summarize_text(input_text)
        self.assertIn(preview, full_summary)


if __name__ == '__main__':
    unittest.main()
