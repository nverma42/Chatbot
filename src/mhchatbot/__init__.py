"""
Mental Health Chatbot Package

A comprehensive mental health support chatbot that provides both informational
and emotional support through machine learning-based query classification.
"""

__version__ = "1.0.0"
__author__ = "MHChatBot Team"

from .chatbot import MentalHealthChatbot
from .emotional_response import EmotionalResponse
from .summarization import EnhancedSummarizationEngine

__all__ = [
    "MentalHealthChatbot",
    "EmotionalResponse",
    "EnhancedSummarizationEngine"
]
