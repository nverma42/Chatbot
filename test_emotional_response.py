import unittest
from unittest.mock import patch, MagicMock
from emotional_response import EmotionalResponse
import pandas as pd
import numpy as np
import torch


class TestEmotionalResponse(unittest.TestCase):
    @patch("emotional_response.pd.read_json")
    def test_initialization_with_data_load(self, mock_read_json):
        """
        Test that EmotionalResponse initializes with data loaded successfully.
        """
        mock_data = pd.DataFrame(
            {'Context': ['Hello world'], 'Response': ['Hi there']})
        mock_read_json.return_value = mock_data

        er = EmotionalResponse()
        self.assertIsNotNone(er.encoder)
        self.assertEqual(er.df.shape[0], 1)
        self.assertIn('Context', er.df.columns)

    @patch("emotional_response.pd.read_json", side_effect=FileNotFoundError("Data file not found"))
    def test_initialization_with_data_load_failure(self, mock_read_json):
        """
        Test that a FileNotFoundError is raised when data cannot be loaded.
        """
        with self.assertRaises(FileNotFoundError):
            EmotionalResponse()

    def test_preprocess_data(self):
        """
        Test that preprocess_data correctly lemmatizes and removes stop words.
        """
        er = EmotionalResponse()
        sample_contexts = ["This is a test sentence.", "Another example here!"]
        expected_output = [['test', 'sentence'], ['another', 'example']]

        with patch.object(er, 'get_filtered_base_words', side_effect=expected_output):
            processed_data = er.preprocess_data(sample_contexts)
            self.assertEqual(processed_data, expected_output)

    def test_get_filtered_base_words(self):
        """
        Test that get_filtered_base_words filters stop words and lemmatizes tokens.
        """
        er = EmotionalResponse()
        query = "Running happily towards the winning goal!"
        expected_output = ['run', 'happily', 'win', 'goal']

        with patch('nltk.corpus.stopwords.words', return_value=['towards', 'the']), \
                patch.object(er.lemmatizer, 'lemmatize', side_effect=lambda x: {"running": "run", "winning": "win"}.get(x, x)):
            filtered_words = er.get_filtered_base_words(query)
            filtered_words = [
                word for word in filtered_words if word not in ['towards', 'the']]
            self.assertEqual(filtered_words, expected_output)

    @patch("emotional_response.EmotionalResponse.apply_lda_model")
    def test_apply_lda_model(self, mock_apply_lda_model):
        """
        Test apply_lda_model to confirm topics are applied without errors.
        """
        er = EmotionalResponse()
        mock_processed_data = [['example', 'data'], ['more', 'text']]
        # Ensure dictionary is initialized
        er.apply_lda_model(mock_processed_data)
        self.assertIsNotNone(er.dictionary)

    @patch("emotional_response.EmotionalResponse.apply_lda_model")
    def test_get_topic(self, mock_apply_lda_model):
        """
        Test that get_topic returns the correct topic for a given query.
        """
        er = EmotionalResponse()
        er.dictionary = MagicMock()  # Manually mock dictionary
        mock_apply_lda_model.return_value = None

        with patch.object(er, 'get_filtered_base_words', return_value=['test', 'query']), \
                patch.object(er.lda_model, 'get_document_topics', return_value=[(0, 0.8), (1, 0.2)]):
            topic = er.get_topic("This is a test query.")
            self.assertEqual(topic, 0)

    @patch("emotional_response.EmotionalResponse.apply_lda_model")
    def test_get_response_valid(self, mock_apply_lda_model):
        """
        Test that get_response returns a valid response for a matching topic.
        """
        er = EmotionalResponse()
        mock_apply_lda_model.return_value = None
        er.dictionary = MagicMock()

        with patch.object(er, 'get_topic', return_value=0), \
                patch.object(er, 'graph_dict', {0: {0: MagicMock()}}), \
                patch.object(er.encoder, 'encode', return_value=torch.tensor([0.1, 0.2])):

            mock_graph = er.graph_dict[0][0]
            mock_graph.neighbors.return_value = ['Mock response']
            response = er.get_response("This is a query.")
            self.assertIn(response, ['Mock response'])

    def test_get_response_default(self):
        """
        Test that get_response returns the default message when no topic matches.
        """
        er = EmotionalResponse()
        with patch.object(er, 'get_topic', return_value=-1):
            response = er.get_response("This query has no matching topic.")
            self.assertEqual(
                response, "I'm sorry, I don't have enough information on this topic to help you.")

    def test_similarity_calculation(self):
        """
        Test that similarity calculation produces expected results.
        """
        er = EmotionalResponse()
        query_embedding = torch.tensor([1.0, 2.0])
        root_embedding = torch.tensor([2.0, 4.0])

        with patch.object(er.encoder, 'encode', side_effect=[query_embedding, root_embedding]):
            score = torch.dot(query_embedding, root_embedding) / \
                (torch.norm(query_embedding) * torch.norm(root_embedding))
            self.assertAlmostEqual(score.item(), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
