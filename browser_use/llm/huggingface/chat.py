"""
ChatHuggingFace - Wrapper for Hugging Face transformers models.

This allows using local Hugging Face models directly without Ollama.
Supports models like Qwen 2.5 3B, BitNet, and other transformer models.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
    # Try to enable progress bars via huggingface_hub
    try:
        import os
        # Enable verbose output for transformers (shows progress bars)
        if 'TRANSFORMERS_VERBOSITY' not in os.environ:
            os.environ['TRANSFORMERS_VERBOSITY'] = 'info'
        # Ensure huggingface_hub shows progress
        if 'HF_HUB_DISABLE_PROGRESS_BARS' not in os.environ:
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'  # 0 = show progress bars
    except Exception:
        pass
except ImportError:
    TRANSFORMERS_AVAILABLE = False

T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger(__name__)


@dataclass
class ChatHuggingFace(BaseChatModel):
    """
    Wrapper for Hugging Face transformers models.
    
    Usage:
        from browser_use.llm.huggingface import ChatHuggingFace
        
        llm = ChatHuggingFace(
            model="Qwen/Qwen2.5-3B-Instruct",
            device_map="auto",  # or "cpu", "cuda", etc.
        )
    """

    model: str
    """Model name or path (e.g., "Qwen/Qwen2.5-3B-Instruct")"""
    
    device_map: str = "auto"
    """Device to load model on: "auto", "cpu", "cuda", "cuda:0", etc."""
    
    torch_dtype: str | None = None
    """Torch dtype: "float16", "bfloat16", "float32", or None for auto"""
    
    load_in_8bit: bool = False
    """Load model in 8-bit mode (requires bitsandbytes)"""
    
    load_in_4bit: bool = False
    """Load model in 4-bit mode (requires bitsandbytes)"""
    
    max_new_tokens: int = 2048
    """Maximum number of new tokens to generate"""
    
    temperature: float = 0.7
    """Sampling temperature"""
    
    top_p: float = 0.9
    """Top-p sampling"""
    
    do_sample: bool = True
    """Whether to use sampling"""
    
    trust_remote_code: bool = False
    """Trust remote code when loading model"""
    
    # Internal state
    _tokenizer: Any = None
    _model: Any = None
    _model_loaded: bool = False

    def __post_init__(self):
        """Validate transformers is available."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for ChatHuggingFace. "
                "Install with: pip install transformers torch"
            )

    @property
    def provider(self) -> str:
        return 'huggingface'

    @property
    def name(self) -> str:
        return self.model

    def _load_model(self) -> None:
        """Lazy load the model and tokenizer."""
        if self._model_loaded:
            return

        print(f"\n🔄 Loading Hugging Face model: {self.model}", flush=True)
        print("   This may take a few minutes on first run (downloading ~6GB)...", flush=True)
        
        try:
            # Ensure progress bars are enabled for huggingface_hub
            import os
            # Enable verbose output (shows progress bars)
            os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'info')
            # Explicitly enable progress bars (0 = show, 1 = hide)
            os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '0')
            # Use regular download (not hf_transfer) to show progress
            os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
            
            # Check if model is already cached
            try:
                from pathlib import Path
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
                model_cache_path = cache_dir / f"models--{self.model.replace('/', '--')}"
                if model_cache_path.exists():
                    size = sum(f.stat().st_size for f in model_cache_path.rglob('*') if f.is_file()) / (1024**3)
                    print(f"   ✅ Model found in cache: {model_cache_path}", flush=True)
                    print(f"   📦 Cache size: {size:.2f} GB", flush=True)
                else:
                    print(f"   📥 Model not in cache, will download from Hugging Face...", flush=True)
                    print(f"   ⏳ Download size: ~6GB (Qwen 2.5 3B)", flush=True)
            except Exception:
                pass
            
            # Load tokenizer (transformers will show progress bar automatically if tqdm is installed)
            print("   📥 Loading tokenizer...", flush=True)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model,
                trust_remote_code=self.trust_remote_code,
            )
            print("   ✅ Tokenizer loaded", flush=True)
            
            # Set pad token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Prepare model loading kwargs
            model_kwargs: dict[str, Any] = {
                'trust_remote_code': self.trust_remote_code,
            }
            
            # Handle quantization
            if self.load_in_8bit or self.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=self.load_in_8bit,
                        load_in_4bit=self.load_in_4bit,
                    )
                    model_kwargs['quantization_config'] = quantization_config
                except ImportError:
                    logger.warning("bitsandbytes not available, ignoring quantization settings")
            
            # Handle device and dtype
            if self.device_map == "auto":
                # Check if accelerate is available (required for device_map="auto")
                try:
                    import accelerate
                    # Ensure accelerate is imported (transformers checks for it)
                    model_kwargs['device_map'] = "auto"
                    print(f"   ✅ Using device_map='auto' (accelerate {accelerate.__version__} available)", flush=True)
                except ImportError:
                    print("   ⚠️  accelerate not installed, falling back to CPU", flush=True)
                    print("   💡 Install with: pip install accelerate", flush=True)
                    model_kwargs['device_map'] = "cpu"
            else:
                model_kwargs['device_map'] = self.device_map
            
            if self.torch_dtype:
                dtype_map = {
                    'float16': torch.float16,
                    'bfloat16': torch.bfloat16,
                    'float32': torch.float32,
                }
                if self.torch_dtype in dtype_map:
                    model_kwargs['torch_dtype'] = dtype_map[self.torch_dtype]
            
            # Load model (transformers/huggingface_hub will show progress bars automatically)
            print("   📥 Loading model weights...", flush=True)
            print("   ⏳ This may take 5-15 minutes on first download (~6GB)", flush=True)
            print("   💡 Progress bars should appear below (if tqdm is installed)", flush=True)
            print("   💡 Tip: Model will be cached locally after first download", flush=True)
            print("   💡 Monitor progress: watch -n 2 'du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/ 2>/dev/null || echo Not started'", flush=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model,
                **model_kwargs,
            )
            print("   🔧 Setting model to evaluation mode...", flush=True)
            
            # Set to eval mode
            self._model.eval()
            
            self._model_loaded = True
            print(f"✅ Model fully loaded: {self.model}\n", flush=True)
            
        except Exception as e:
            raise ModelProviderError(
                message=f"Failed to load Hugging Face model {self.model}: {str(e)}",
                model=self.model,
            ) from e

    def _format_messages_for_chat(self, messages: list[BaseMessage]) -> str:
        """Format messages using the model's chat template."""
        from browser_use.llm.huggingface import HuggingFaceMessageSerializer
        
        # Convert to chat format
        chat_messages = HuggingFaceMessageSerializer.serialize_messages(messages)
        
        # Apply chat template if available
        if hasattr(self._tokenizer, 'apply_chat_template') and self._tokenizer.chat_template:
            try:
                formatted = self._tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return formatted
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}, using simple format")
        
        # Fallback: simple format
        formatted_parts = []
        for msg in chat_messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted_parts.append(f"System: {content}")
            elif role == 'user':
                formatted_parts.append(f"User: {content}")
            elif role == 'assistant':
                formatted_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(formatted_parts) + "\n\nAssistant:"

    @overload
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: None = None, **kwargs: Any
    ) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T], **kwargs: Any
    ) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """Invoke the model asynchronously."""
        # Load model if not already loaded (this may download from Hugging Face)
        if not self._model_loaded:
            print("🔄 Model loading triggered (this may download from Hugging Face)...", flush=True)
        try:
            self._load_model()
        except Exception as e:
            print(f"❌ Model loading failed: {e}", flush=True)
            raise
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        try:
            if output_format is None:
                # Simple text generation
                completion, usage = await loop.run_in_executor(
                    None,
                    self._generate_text,
                    messages,
                )
                return ChatInvokeCompletion(completion=completion, usage=usage)
            else:
                # Structured output - use JSON schema in prompt
                schema = output_format.model_json_schema()
                completion, usage = await loop.run_in_executor(
                    None,
                    self._generate_structured,
                    messages,
                    schema,
                )
                # Parse JSON response
                try:
                    parsed = output_format.model_validate_json(completion)
                    return ChatInvokeCompletion(completion=parsed, usage=usage)
                except Exception as e:
                    logger.warning(f"Failed to parse structured output: {e}, returning raw text")
                    return ChatInvokeCompletion(completion=completion, usage=usage)
                    
        except Exception as e:
            raise ModelProviderError(
                message=f"Failed to generate text: {str(e)}",
                model=self.name,
            ) from e

    def _generate_text(self, messages: list[BaseMessage]) -> tuple[str, ChatInvokeUsage]:
        """Generate text synchronously (runs in thread pool).
        
        Returns:
            Tuple of (completion_text, usage_info)
        """
        # Format messages
        prompt = self._format_messages_for_chat(messages)
        
        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt")
        prompt_tokens = inputs['input_ids'].shape[1]
        
        # Move to same device as model
        if hasattr(self._model, 'device'):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        elif hasattr(self._model, 'hf_device_map'):
            # Multi-device model, use first device
            first_device = list(self._model.hf_device_map.values())[0]
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                # Prevent early stopping to ensure complete JSON generation
                # Don't stop on EOS token until we have complete JSON
                # Note: This might generate extra tokens, but ensures JSON completeness
            )
        
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        completion_tokens = len(generated_tokens)
        completion = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate usage
        total_tokens = prompt_tokens + completion_tokens
        usage = ChatInvokeUsage(
            prompt_tokens=prompt_tokens,
            prompt_cached_tokens=None,
            prompt_cache_creation_tokens=None,
            prompt_image_tokens=None,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        
        return completion.strip(), usage

    def _generate_structured(self, messages: list[BaseMessage], schema: dict[str, Any]) -> tuple[str, ChatInvokeUsage]:
        """Generate structured output with JSON schema.
        
        Returns:
            Tuple of (completion_text, usage_info)
        """
        # Add explicit, strict JSON format instruction (optimized for small local LLMs)
        # Following Sentience SDK playground pattern: very explicit, no reasoning
        required_fields = schema.get('required', [])
        properties = schema.get('properties', {})
        
        # Build explicit format example
        example_fields = []
        for field in required_fields:
            if field in properties:
                prop = properties[field]
                prop_type = prop.get('type', 'string')
                if prop_type == 'array':
                    example_fields.append(f'  "{field}": []')
                elif prop_type == 'string':
                    example_fields.append(f'  "{field}": ""')
                elif prop_type == 'object':
                    example_fields.append(f'  "{field}": {{}}')
                else:
                    example_fields.append(f'  "{field}": null')
        
        example_json = "{\n" + ",\n".join(example_fields) + "\n}"
        
        # Build minimal instruction (optimized for small local LLMs)
        # Keep it very short to avoid confusing the model
        schema_instruction = f"\n\nJSON only:\n{example_json}"
        
        # Create modified messages
        modified_messages = list(messages)
        if modified_messages and hasattr(modified_messages[-1], 'content'):
            last_msg = modified_messages[-1]
            if isinstance(last_msg.content, str):
                modified_messages[-1] = type(last_msg)(
                    content=last_msg.content + schema_instruction
                )
        
        # Generate with schema instruction
        completion, usage = self._generate_text(modified_messages)
        
        # Try to extract JSON from response
        completion = completion.strip()
        
        # Try to find JSON in the response (in case model adds extra text)
        if completion.startswith('```json'):
            # Extract from code block
            completion = completion.replace('```json', '').replace('```', '').strip()
        elif completion.startswith('```'):
            completion = completion.replace('```', '').strip()
        
        # Try to parse to validate JSON
        try:
            json.loads(completion)
        except json.JSONDecodeError:
            logger.warning(f"Generated text is not valid JSON: {completion[:200]}")
        
        return completion, usage
