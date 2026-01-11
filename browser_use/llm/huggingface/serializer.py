"""
Serializer for converting browser-use messages to Hugging Face transformers format.
"""

from typing import Any

from browser_use.llm.messages import (
    AssistantMessage,
    BaseMessage,
    SystemMessage,
    UserMessage,
)


class HuggingFaceMessageSerializer:
    """Serializer for converting between browser-use messages and Hugging Face chat format."""

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        """Extract text content from message content, ignoring images."""
        if content is None:
            return ''
        if isinstance(content, str):
            return content

        text_parts: list[str] = []
        for part in content:
            if hasattr(part, 'type'):
                if part.type == 'text':
                    text_parts.append(part.text)
                elif part.type == 'refusal':
                    text_parts.append(f'[Refusal] {part.refusal}')
            # Skip image parts (transformers may not support images in all models)

        return '\n'.join(text_parts)

    @staticmethod
    def serialize(message: BaseMessage) -> dict[str, str]:
        """Serialize a browser-use message to Hugging Face chat format.
        
        Returns:
            Dict with 'role' and 'content' keys compatible with transformers chat templates.
        """
        if isinstance(message, SystemMessage):
            return {
                'role': 'system',
                'content': HuggingFaceMessageSerializer._extract_text_content(message.content),
            }
        elif isinstance(message, UserMessage):
            return {
                'role': 'user',
                'content': HuggingFaceMessageSerializer._extract_text_content(message.content),
            }
        elif isinstance(message, AssistantMessage):
            return {
                'role': 'assistant',
                'content': HuggingFaceMessageSerializer._extract_text_content(message.content) or '',
            }
        else:
            raise ValueError(f'Unknown message type: {type(message)}')

    @staticmethod
    def serialize_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
        """Serialize a list of browser-use messages to Hugging Face chat format.
        
        Returns:
            List of dicts with 'role' and 'content' keys.
        """
        return [HuggingFaceMessageSerializer.serialize(m) for m in messages]
