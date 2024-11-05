import logging
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationEngine:
    def __init__(self, language='english', sentence_count=2, algorithm='lsa', **kwargs):
        self.language = language
        self.sentence_count = sentence_count
        self.summarizer = self._select_algorithm(algorithm)
        # Pass kwargs to algorithm if applicable (e.g., topic number for LSA).
        for param, value in kwargs.items():
            if hasattr(self.summarizer, param):
                setattr(self.summarizer, param, value)

    def _select_algorithm(self, algorithm):
        if algorithm == 'text_rank':
            return TextRankSummarizer()
        elif algorithm == 'lex_rank':
            return LexRankSummarizer()
        else:
            return LsaSummarizer()  # Default to LSA

    def preprocess_text(text):
        # Remove stopwords or unwanted characters if needed.
        stop_words = set(stopwords.words('english'))
        return ' '.join([word for word in text.split() if word.lower() not in stop_words])

    def summarize_preview(self, text):
        """
        Provides a preview of the summary (e.g., first sentence only).
        """
        summary = self.summarize_text(text)
        return summary.split('. ')[0] + '.' if summary else "No preview available."

    def set_sentence_count(self, count):
        """
        Sets the number of sentences to be used in the summary.

        Args:
            count (int): The new number of sentences.
        """
        self.sentence_count = count

    def summarize_text(self, text):
        """
        Summarizes the input text into a specified number of sentences.

        Args:
            text (str): The text to summarize.

        Returns:
            str: The summarized text or an error message if input is insufficient.
        """
        # Check for empty or whitespace-only input
        if not text.strip():
            return "No content to summarize."

        try:
            # Parse the input text
            parser = PlaintextParser.from_string(
                text, Tokenizer(self.language))

            # Check if text has enough sentences for summarization
            if len(text.split('. ')) < self.sentence_count:
                return "Input text is too short to generate a meaningful summary."

            # Perform summarization
            summary = self.summarizer(parser.document, self.sentence_count)

            # Join sentences to form the summary
            return ' '.join(str(sentence) for sentence in summary)

        except Exception as e:
            return f"An error occurred during summarization: {e}"
