import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedSummarizationEngine:
    """
    Enhanced summarization engine using sentence transformers and MMR algorithm
    for query-focused summarization with better semantic understanding.
    """

    def __init__(
        self,
        model_name: str = 'paraphrase-MiniLM-L6-v2',
        lambda_param: float = 0.7,
        top_k: int = 3,
        device: str = None
    ):
        """
        Initialize the summarization engine.

        Args:
            model_name: Name of the sentence transformer model to use
            lambda_param: Weight parameter for MMR algorithm (0 to 1)
            top_k: Number of sentences to include in summary
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.encoder = SentenceTransformer(model_name, device=device)
        self.lambda_param = lambda_param
        self.top_k = top_k

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using period as delimiter while handling
        common abbreviations and edge cases.
        """
        # Handle common abbreviations to avoid incorrect splitting
        text = text.replace("Mr.", "Mr")
        text = text.replace("Mrs.", "Mrs")
        text = text.replace("Dr.", "Dr")
        text = text.replace("Ph.D.", "PhD")
        text = text.replace("e.g.", "eg")
        text = text.replace("i.e.", "ie")

        # Split by period and restore them
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
        return sentences

    def _vectorize_sentences(
        self,
        sentences: List[str],
        query: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode sentences and query using the sentence transformer.

        Args:
            sentences: List of sentences to encode
            query: Optional query to encode

        Returns:
            Tuple of sentence embeddings and query embedding (if provided)
        """
        sentence_embeddings = self.encoder.encode(
            sentences,
            convert_to_tensor=True,
            show_progress_bar=False
        ).cpu().numpy()

        if query:
            query_embedding = self.encoder.encode(
                query,
                convert_to_tensor=True,
                show_progress_bar=False
            ).cpu().numpy()
            return sentence_embeddings, query_embedding

        return sentence_embeddings, None

    def _compute_similarities(
        self,
        query_embedding: np.ndarray,
        sentence_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarities between query and sentences.
        """
        # Compute dot product
        dot_product = np.dot(query_embedding, sentence_embeddings.T)

        # Compute norms
        query_norm = np.linalg.norm(query_embedding)
        sentence_norms = np.linalg.norm(sentence_embeddings, axis=1)

        # Compute cosine similarity
        similarities = dot_product / (query_norm * sentence_norms)
        return similarities

    def _select_mmr_sentences(
        self,
        similarities: np.ndarray,
        sentence_embeddings: np.ndarray,
        n_sentences: int
    ) -> List[int]:
        """
        Select sentences using Maximal Marginal Relevance algorithm.

        Args:
            similarities: Cosine similarities between query and sentences
            sentence_embeddings: Encoded sentences
            n_sentences: Number of sentences to select

        Returns:
            List of selected sentence indices
        """
        selected_indices = []
        unselected_indices = list(range(len(similarities)))

        # Select the first sentence with highest similarity to query
        first_idx = np.argmax(similarities)
        selected_indices.append(first_idx)
        unselected_indices.remove(first_idx)

        # Select remaining sentences using MMR
        while len(selected_indices) < n_sentences and unselected_indices:
            mmr_scores = []

            for idx in unselected_indices:
                # Compute similarity to query
                query_sim = similarities[idx]

                # Compute similarities to selected sentences
                selected_embeddings = sentence_embeddings[selected_indices]
                current_embedding = sentence_embeddings[idx].reshape(1, -1)
                redundancy_sims = self._compute_similarities(
                    current_embedding,
                    selected_embeddings
                )
                max_redundancy = np.max(redundancy_sims)

                # Compute MMR score
                mmr = self.lambda_param * query_sim - \
                    (1 - self.lambda_param) * max_redundancy
                mmr_scores.append(mmr)

            # Select sentence with highest MMR score
            next_idx = unselected_indices[np.argmax(mmr_scores)]
            selected_indices.append(next_idx)
            unselected_indices.remove(next_idx)

        return selected_indices

    def summarize(
        self,
        text: str,
        query: str = None,
        custom_k: int = None
    ) -> Dict[str, any]:
        """
        Generate a query-focused summary of the input text.

        Args:
            text: Input text to summarize
            query: Optional query to focus the summary
            custom_k: Optional override for number of sentences

        Returns:
            Dictionary containing summary and metadata
        """
        try:
            # Split text into sentences
            sentences = self._split_into_sentences(text)
            if len(sentences) == 0:
                return {
                    "summary": "",
                    "error": "No valid sentences found in input text."
                }

            # Use default query if none provided
            if not query:
                query = "What is the main point of this text?"

            # Vectorize sentences and query
            sentence_embeddings, query_embedding = self._vectorize_sentences(
                sentences,
                query
            )

            # Compute similarities
            similarities = self._compute_similarities(
                query_embedding,
                sentence_embeddings
            )

            # Select top sentences using MMR
            k = custom_k if custom_k is not None else self.top_k
            # Ensure k doesn't exceed sentence count
            k = min(k, len(sentences))
            selected_indices = self._select_mmr_sentences(
                similarities,
                sentence_embeddings,
                k
            )

            # Order sentences by original position
            selected_indices.sort()
            summary_sentences = [sentences[idx] for idx in selected_indices]

            # Combine sentences into final summary
            summary = " ".join(summary_sentences)

            return {
                "summary": summary,
                "original_sentences": len(sentences),
                "selected_sentences": len(summary_sentences),
                "selected_indices": selected_indices,
                "similarity_scores": similarities[selected_indices].tolist()
            }

        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            return {
                "summary": "",
                "error": f"Summarization failed: {str(e)}"
            }

    def update_parameters(
        self,
        lambda_param: float = None,
        top_k: int = None
    ) -> None:
        """
        Update the engine's parameters.

        Args:
            lambda_param: New lambda parameter for MMR
            top_k: New number of sentences for summary
        """
        if lambda_param is not None:
            if 0 <= lambda_param <= 1:
                self.lambda_param = lambda_param
            else:
                raise ValueError("lambda_param must be between 0 and 1")

        if top_k is not None:
            if top_k > 0:
                self.top_k = top_k
            else:
                raise ValueError("top_k must be greater than 0")
