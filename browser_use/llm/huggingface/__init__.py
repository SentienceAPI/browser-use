"""
Hugging Face transformers integration for browser-use.
"""

from browser_use.llm.huggingface.chat import ChatHuggingFace
from browser_use.llm.huggingface.serializer import HuggingFaceMessageSerializer

__all__ = ['ChatHuggingFace', 'HuggingFaceMessageSerializer']
