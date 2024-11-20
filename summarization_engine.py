import logging
from telnetlib import DO
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import numpy as np
import sys

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
    
    def best_cosine_similarity(self, v, v_list):
        best_score = 0
        for i in (range(len(v_list))):
            score = np.dot(v, v_list[i]) /(norm(v) * norm(v_list[i]))
            if (score > best_score):
                best_score = score
        return best_score

    def summarize_text_MMR(self, question, answer):
        """
        Summarizes the input text into a specified number of sentences using maximum
        marginal relevance method.

        Args:
            text (str): The text to summarize.

        Returns:
            str: The summarized text or an error message if input is insufficient.
        """
        # Check for empty or whitespace-only input
        if not answer.strip():
            return "No content to summarize."

        # Parse the text in individual sentences
        answer_sentences = answer.split('. ')

        # Load pretrained Sentence-BERT model.
        encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        A = encoder.encode(answer_sentences)
        q = encoder.encode(question)

        # Run the MMR loop
        w = 0.7
        count = 0
        S = []
        S_v = []
        while (count < self.sentence_count):
            best_score = -sys.float_info.max
            best_index = -1
            for i in (range(len(A))):
                score = w * np.dot(q, A[i]) /(norm(q) * norm(A[i]))
                score -= (1-w) * self.best_cosine_similarity(A[i], S_v)
                if (score > best_score and i not in S):
                    best_score = score
                    best_index = i

            S.append(best_index)
            S_v.append(A[best_index])
            count += 1

        # Sort the array
        sorted_S = np.sort(S)

        summary = ''
        for k in sorted_S:
            summary += answer_sentences[k] + '.'

        return summary

        

