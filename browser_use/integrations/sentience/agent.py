"""
SentienceAgent: Custom agent with full control over prompt construction.

This agent uses Sentience SDK snapshots as the primary, compact prompt format,
with automatic fallback to vision mode when snapshots fail.

Features:
- Sentience snapshot as compact prompt (~3K tokens vs ~40K for vision)
- Vision fallback when snapshot fails
- Native AgentRuntime integration for verification assertions
- Machine-verifiable task completion via assert_done()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

from pydantic import BaseModel, Field

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import SystemMessage, UserMessage
from browser_use.tokens.service import TokenCost
from browser_use.tokens.views import UsageSummary

if TYPE_CHECKING:
    from browser_use.browser.session import BrowserSession
    from browser_use.tools.service import Tools
    from sentience.agent_runtime import AgentRuntime
    from sentience.tracing import Tracer
    from sentience.verification import Predicate

logger = logging.getLogger(__name__)


@dataclass
class SentienceAgentConfig:
    """Configuration for Sentience snapshot behavior."""

    sentience_api_key: str | None = None
    """Sentience API key for gateway mode."""

    sentience_use_api: bool | None = None
    """Force API vs extension mode (auto-detected if None)."""

    sentience_max_elements: int = 60
    """Maximum elements to fetch from snapshot."""

    sentience_show_overlay: bool = False
    """Show visual overlay highlighting elements in browser."""

    sentience_wait_for_extension_ms: int = 5000
    """Maximum time to wait for extension injection (milliseconds)."""

    sentience_retries: int = 2
    """Number of retry attempts on snapshot failure."""

    sentience_retry_delay_s: float = 1.0
    """Delay between retries in seconds."""


@dataclass
class VisionFallbackConfig:
    """Configuration for vision fallback behavior."""

    enabled: bool = True
    """Whether to fall back to vision mode when Sentience fails."""

    detail_level: Literal['auto', 'low', 'high'] = 'auto'
    """Vision detail level for screenshots."""

    include_screenshots: bool = True
    """Whether to include screenshots in vision fallback."""


@dataclass
class VerificationConfig:
    """Configuration for Sentience SDK verification (AgentRuntime integration).

    This enables machine-verifiable assertions during agent execution,
    providing observability into agent behavior and task completion status.
    """

    enabled: bool = False
    """Whether to enable verification via AgentRuntime."""

    step_assertions: list[dict[str, Any]] = field(default_factory=list)
    """Per-step assertions to run after each action.

    Each assertion dict should have:
    - predicate: A Predicate callable (e.g., url_contains("example.com"))
    - label: String label for the assertion
    - required: Optional bool, if True failing this assertion marks step as failed

    Example:
        step_assertions=[
            {"predicate": url_contains("news.ycombinator.com"), "label": "on_hackernews", "required": True},
            {"predicate": exists("role=link[text*='Show HN']"), "label": "show_hn_visible"},
        ]
    """

    done_assertion: Any | None = None
    """Predicate for machine-verifiable task completion.

    When set, this predicate is evaluated after each step. If it returns True,
    the task is considered complete (independent of LLM's done action).

    Example:
        done_assertion=all_of(
            url_contains("news.ycombinator.com/show"),
            exists("role=link[text*='Show HN']"),
        )
    """

    trace_dir: str = "traces"
    """Directory for trace output files."""


class SentienceAgentSettings(BaseModel):
    """Settings for SentienceAgent."""

    task: str = Field(..., description="The task for the agent to complete")
    max_steps: int = Field(default=100, description="Maximum number of steps")
    max_failures: int = Field(default=3, description="Maximum consecutive failures before stopping")
    calculate_cost: bool = Field(default=True, description="Track token usage and costs")
    llm_timeout: int = Field(default=60, description="Timeout for LLM calls in seconds")
    step_timeout: int = Field(default=120, description="Timeout for each step in seconds")

    # Sentience configuration
    sentience_config: SentienceAgentConfig = Field(
        default_factory=SentienceAgentConfig,
        description="Configuration for Sentience snapshot behavior"
    )

    # Vision fallback configuration
    vision_fallback: VisionFallbackConfig = Field(
        default_factory=VisionFallbackConfig,
        description="Configuration for vision fallback behavior"
    )

    # Verification configuration (AgentRuntime integration)
    verification: VerificationConfig = Field(
        default_factory=VerificationConfig,
        description="Configuration for Sentience SDK verification assertions"
    )


class SentienceAgent:
    """
    Custom agent with full control over prompt construction.

    Features:
    - Primary: Sentience snapshot as compact prompt (~1.3K tokens)
    - Fallback: Vision mode when snapshot fails (~4K tokens)
    - Token usage tracking via browser-use utilities
    - Clear isolation from built-in vision model
    """

    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        browser_session: BrowserSession,
        tools: Tools | None = None,
        *,
        # Sentience configuration
        sentience_config: SentienceAgentConfig,
        # Vision fallback configuration
        vision_fallback_enabled: bool = True,
        vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
        vision_include_screenshots: bool = True,
        vision_llm: BaseChatModel | None = None,
        # Token tracking
        calculate_cost: bool = True,
        # Agent settings
        max_steps: int = 100,
        max_failures: int = 3,
        llm_timeout: int = 60,
        step_timeout: int = 120,
        # Verification configuration (Sentience SDK AgentRuntime)
        enable_verification: bool = False,
        step_assertions: list[dict[str, Any]] | None = None,
        done_assertion: Any | None = None,
        tracer: Any | None = None,
        trace_dir: str = "traces",
        **kwargs,
    ):
        """
        Initialize SentienceAgent.

        Args:
            task: The task for the agent to complete
            llm: Language model to use (primary model for Sentience snapshots)
            browser_session: Browser session instance
            tools: Tools registry (optional)
            sentience_config: SentienceAgentConfig object with all Sentience configuration
            vision_fallback_enabled: Enable vision fallback
            vision_detail_level: Vision detail level
            vision_include_screenshots: Include screenshots in fallback
            vision_llm: Optional vision-capable LLM for vision fallback mode.
                      If None, uses the primary `llm` for vision fallback too.
            calculate_cost: Track token usage
            max_steps: Maximum steps
            max_failures: Maximum failures
            llm_timeout: LLM timeout
            step_timeout: Step timeout
            enable_verification: Enable Sentience SDK verification via AgentRuntime
            step_assertions: Per-step assertions (list of dicts with predicate, label, required)
            done_assertion: Predicate for machine-verifiable task completion
            tracer: Optional Tracer instance (auto-created if None and verification enabled)
            trace_dir: Directory for trace output files
        """
        self.task = task
        self.llm = llm
        self.vision_llm = vision_llm  # Optional vision-capable model for fallback
        self.browser_session = browser_session
        
        # Initialize tools if not provided
        if tools is None:
            from browser_use.tools.service import Tools
            self.tools = Tools()
        else:
            self.tools = tools

        # Initialize file system for actions that require it (e.g., done action)
        from browser_use.filesystem.file_system import FileSystem
        import tempfile
        self.file_system = FileSystem(base_dir=tempfile.mkdtemp(prefix="sentience_agent_"))

        # Build settings
        vision_fallback = VisionFallbackConfig(
            enabled=vision_fallback_enabled,
            detail_level=vision_detail_level,
            include_screenshots=vision_include_screenshots,
        )
        verification_config = VerificationConfig(
            enabled=enable_verification,
            step_assertions=step_assertions or [],
            done_assertion=done_assertion,
            trace_dir=trace_dir,
        )
        self.settings = SentienceAgentSettings(
            task=task,
            max_steps=max_steps,
            max_failures=max_failures,
            calculate_cost=calculate_cost,
            llm_timeout=llm_timeout,
            step_timeout=step_timeout,
            sentience_config=sentience_config,
            vision_fallback=vision_fallback,
            verification=verification_config,
        )

        # Initialize SentienceContext (lazy import to avoid hard dependency)
        self._sentience_context: Any | None = None

        # Initialize AgentRuntime for verification (if enabled)
        self._runtime: Any | None = None
        self._tracer: Any | None = tracer
        self._verification_initialized = False

        # Initialize token cost service
        self.token_cost_service = TokenCost(include_cost=calculate_cost)
        self.token_cost_service.register_llm(llm)

        # Initialize message manager for history tracking
        from browser_use.integrations.sentience.message_manager import CustomMessageManager

        system_message = self._get_system_message()
        self.message_manager = CustomMessageManager(
            task=task,
            system_message=system_message,
            max_history_items=4,  # Keep recent history for context (0 may cause issues with some LLMs)
        )

        # Track state
        self._current_step = 0
        self._consecutive_failures = 0
        self._sentience_used_in_last_step = False
        self._current_sentience_state: Any | None = None  # Store current Sentience snapshot for element lookup

        logger.info(
            f"Initialized SentienceAgent: task='{task}', "
            f"sentience_max_elements={sentience_config.sentience_max_elements}, "
            f"vision_fallback={'enabled' if vision_fallback_enabled else 'disabled'}"
        )

    @property
    def runtime(self) -> Any | None:
        """Access the AgentRuntime instance (if verification is enabled)."""
        return self._runtime

    async def _initialize_verification(self) -> None:
        """Initialize AgentRuntime for verification assertions.

        Creates a BrowserBackend from the browser_session and sets up
        the AgentRuntime with tracer for verification events.
        """
        if self._verification_initialized or not self.settings.verification.enabled:
            return

        try:
            from sentience.agent_runtime import AgentRuntime
            from sentience.backends import BrowserUseAdapter
            from sentience.tracing import JsonlTraceSink, Tracer

            # Create backend from browser_session
            adapter = BrowserUseAdapter(self.browser_session)
            backend = await adapter.create_backend()

            # Create tracer if not provided
            if self._tracer is None:
                import os
                import time
                os.makedirs(self.settings.verification.trace_dir, exist_ok=True)
                run_id = f"sentience-agent-{int(time.time())}"
                sink = JsonlTraceSink(
                    f"{self.settings.verification.trace_dir}/{run_id}.jsonl"
                )
                self._tracer = Tracer(run_id=run_id, sink=sink)
                logger.info(f"📝 Verification trace: {self.settings.verification.trace_dir}/{run_id}.jsonl")

            # Create AgentRuntime
            self._runtime = AgentRuntime(
                backend=backend,
                tracer=self._tracer,
                sentience_api_key=self.settings.sentience_config.sentience_api_key,
            )

            self._verification_initialized = True
            logger.info("✅ Verification enabled via Sentience AgentRuntime")

        except ImportError as e:
            logger.warning(
                f"⚠️  Verification requested but Sentience SDK not fully installed: {e}. "
                "Install with: pip install sentienceapi"
            )
            self.settings.verification.enabled = False

    def _get_sentience_context(self) -> Any:
        """Get or create SentienceContext instance."""
        if self._sentience_context is None:
            try:
                from sentience.backends import SentienceContext

                self._sentience_context = SentienceContext(
                    sentience_api_key=self.settings.sentience_config.sentience_api_key,
                    use_api=self.settings.sentience_config.sentience_use_api,
                    max_elements=self.settings.sentience_config.sentience_max_elements,
                    show_overlay=self.settings.sentience_config.sentience_show_overlay,
                )
            except ImportError as e:
                logger.info(f"Sentience SDK not available: {e}")
                raise ImportError(
                    "Sentience SDK is required for SentienceAgent. "
                    "Install it with: pip install sentienceapi"
                ) from e
        return self._sentience_context

    async def _prepare_context(self) -> tuple[UserMessage, bool]:
        """
        Prepare context with Sentience-first, vision-fallback strategy.

        Returns:
            (user_message, sentience_used): Tuple of message and whether Sentience was used
        """
        # Try Sentience first
        sentience_state = await self._try_sentience_snapshot()

        if sentience_state:
            # Store current Sentience state for element lookup during action execution
            self._current_sentience_state = sentience_state
            # Use Sentience prompt block
            user_message = await self._build_sentience_message(sentience_state)
            self._sentience_used_in_last_step = True
            logger.info("✅ Using Sentience snapshot for prompt")
            return user_message, True
        else:
            # Clear Sentience state if snapshot failed
            self._current_sentience_state = None
            # Fall back to vision
            if self.settings.vision_fallback.enabled:
                user_message = await self._build_vision_message()
                self._sentience_used_in_last_step = False
                logger.info("⚠️ Sentience failed, falling back to vision mode")
                return user_message, False
            else:
                # No fallback: return minimal message
                user_message = self._build_minimal_message()
                self._sentience_used_in_last_step = False
                logger.info("⚠️ Sentience failed and vision fallback disabled, using minimal message")
                return user_message, False

    async def _try_sentience_snapshot(self) -> Any | None:
        """
        Attempt to get Sentience snapshot.

        Returns:
            SentienceContextState if successful, None otherwise
        """
        try:
            # CRITICAL: Check if we're on about:blank - Sentience extension doesn't inject there
            # The extension's content scripts only inject on actual URLs (<all_urls> doesn't include about:blank)
            current_url = await self.browser_session.get_current_page_url()
            if current_url == 'about:blank' or not current_url or current_url.startswith('about:'):
                logger.info(
                    f"⚠️  Current page is '{current_url}' - Sentience extension doesn't inject on about:blank. "
                    f"Extracting URL from task or navigating to default page..."
                )
                
                # Try to extract URL from task
                import re
                url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                urls = re.findall(url_pattern, self.task)
                
                if urls:
                    target_url = urls[0]
                    logger.info(f"📍 Found URL in task: {target_url} - navigating...")
                else:
                    # Default to a simple page if no URL in task
                    # The agent will navigate to the actual target page in the next step
                    target_url = "https://www.google.com"
                    logger.info(f"📍 No URL in task - navigating to default page: {target_url}")
                
                # Navigate to a real URL so extension can inject
                await self.browser_session.navigate_to(target_url)
                
                # Wait a moment for navigation and extension injection
                await asyncio.sleep(1.0)
                
                # Verify we're no longer on about:blank
                new_url = await self.browser_session.get_current_page_url()
                if new_url == 'about:blank' or new_url.startswith('about:'):
                    logger.warning(f"⚠️  Navigation may have failed, still on: {new_url}")
                else:
                    logger.info(f"✅ Navigated to: {new_url}")
            
            sentience_context = self._get_sentience_context()
            logger.info(
                f"Attempting Sentience snapshot on URL: {await self.browser_session.get_current_page_url()}, "
                f"wait_for_extension_ms={self.settings.sentience_config.sentience_wait_for_extension_ms}, "
                f"retries={self.settings.sentience_config.sentience_retries}, "
                f"use_api={self.settings.sentience_config.sentience_use_api}"
            )
            sentience_state = await sentience_context.build(
                self.browser_session,
                goal=self.task,
                wait_for_extension_ms=self.settings.sentience_config.sentience_wait_for_extension_ms,
                retries=self.settings.sentience_config.sentience_retries,
                retry_delay_s=self.settings.sentience_config.sentience_retry_delay_s,
            )
            if sentience_state:
                num_elements = len(sentience_state.snapshot.elements) if hasattr(sentience_state, 'snapshot') else 'unknown'
                logger.info(f"✅ Sentience snapshot successful: {num_elements} elements")
                
                # Log overlay status (SDK handles overlay display during snapshot if show_overlay=True)
                if self.settings.sentience_config.sentience_show_overlay:
                    logger.info(
                        f"🎨 Overlay should be visible in browser (auto-clears after 5 seconds). "
                        f"Elements highlighted: {num_elements}"
                    )
                else:
                    logger.debug("Overlay disabled (sentience_show_overlay=False)")
            return sentience_state
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.info(
                f"❌ Sentience snapshot failed: {error_type}: {error_msg}\n"
                f"   This usually means:\n"
                f"   - Extension not injected (check if extension is loaded in browser)\n"
                f"   - Extension injection timeout (increase wait_for_extension_ms)\n"
                f"   - Snapshot API call failed (check network/API key)\n"
                f"   - Page not ready (wait for page load to complete)"
            )
            import traceback
            logger.debug(f"Sentience snapshot failure traceback:\n{traceback.format_exc()}")
            return None

    def _find_element_in_snapshot(self, snapshot: Any, element_id: int | None = None, text: str | None = None) -> Any | None:
        """
        Find an element in Sentience snapshot using SDK's find() function.
        
        Args:
            snapshot: Sentience Snapshot object
            element_id: Element ID to find (backend_node_id)
            text: Text to search for (uses SDK's text matching)
        
        Returns:
            Element if found, None otherwise
        """
        if not hasattr(snapshot, 'elements'):
            return None
        
        # If searching by ID, iterate directly (most efficient)
        if element_id is not None:
            for el in snapshot.elements:
                if hasattr(el, 'id') and el.id == element_id:
                    return el
        
        # If searching by text, use SDK's find() function
        if text:
            try:
                from sentience.query import find
                # Try exact match first
                element = find(snapshot, f"text='{text}'")
                if element:
                    return element
                # Fallback to contains match (case-insensitive)
                element = find(snapshot, f"text~'{text[:50]}'")  # Limit to 50 chars for contains
                if element:
                    return element
            except ImportError:
                logger.debug("SDK query module not available, using direct iteration for text search")
                # Fallback: iterate and match text manually
                text_lower = text.lower()
                for el in snapshot.elements:
                    if hasattr(el, 'text') and el.text and text_lower in el.text.lower():
                        return el
        
        return None

    async def _build_sentience_message(self, sentience_state: Any) -> UserMessage:
        """
        Build user message using Sentience prompt block.

        Args:
            sentience_state: SentienceContextState from SDK

        Returns:
            UserMessage with Sentience prompt block
        """
        # Get agent history from message manager
        history_text = self._get_agent_history_description()

        # Get read_state if available
        read_state_text = ""
        if self.message_manager.state.read_state_description:
            read_state_text = (
                f"\n<read_state>\n{self.message_manager.state.read_state_description}\n</read_state>\n"
            )

        # Include task in agent_state (required for LLM to know what to do)
        agent_state_text = f"<user_request>\n{self.task}\n</user_request>"

        # Extract and validate Sentience element IDs against browser-use selector_map
        available_ids = []
        if hasattr(sentience_state, 'snapshot') and hasattr(sentience_state.snapshot, 'elements'):
            available_ids = [el.id for el in sentience_state.snapshot.elements if hasattr(el, 'id')]
            
            # Get browser-use selector_map to check overlap
            selector_map = await self.browser_session.get_selector_map()
            if not selector_map:
                # Trigger DOM build if selector_map is empty
                from browser_use.browser.events import BrowserStateRequestEvent
                event = self.browser_session.event_bus.dispatch(
                    BrowserStateRequestEvent(include_screenshot=False)
                )
                await event
                await event.event_result(raise_if_any=True, raise_if_none=False)
                selector_map = await self.browser_session.get_selector_map()
            
            # Check which Sentience IDs exist in selector_map
            selector_map_keys = set(selector_map.keys()) if selector_map else set()
            sentience_ids_set = set(available_ids)
            matching_ids = sentience_ids_set & selector_map_keys
            missing_ids = sentience_ids_set - selector_map_keys
            
            logger.info(
                f"📋 Sentience snapshot: {len(available_ids)} elements, "
                f"{len(matching_ids)} match selector_map, {len(missing_ids)} missing from selector_map"
            )
            if missing_ids:
                missing_list = sorted(list(missing_ids))[:10]
                logger.info(
                    f"  ⚠️  Sentience IDs not in selector_map (first 10): {missing_list}"
                    f"{'...' if len(missing_ids) > 10 else ''} "
                    f"(These elements may not be interactive by browser-use's criteria)"
                )
        
        # Log the FULL Sentience prompt block for debugging
        logger.info(
            f"📋 Sentience prompt block ({len(sentience_state.prompt_block)} chars, "
            f"~{len(sentience_state.prompt_block) // 4} tokens):\n"
            f"{sentience_state.prompt_block}"
        )

        # Combine agent history + agent state + Sentience prompt block + read_state
        # Note: We explicitly avoid screenshots here for clear isolation
        content = (
            f"<agent_history>\n{history_text}\n</agent_history>\n\n"
            f"<agent_state>\n{agent_state_text}\n</agent_state>\n\n"
            f"<browser_state>\n{sentience_state.prompt_block}\n</browser_state>"
            f"{read_state_text}"
        )

        return UserMessage(content=content, cache=True)

    async def _build_vision_message(self) -> UserMessage:
        """
        Build user message using vision (screenshots + DOM).

        This is the fallback when Sentience fails. It uses browser-use's
        built-in browser state summary with screenshots and full DOM tree.

        Returns:
            UserMessage with screenshots and comprehensive DOM state
        """
        # Get browser state summary with screenshots (only in fallback mode)
        browser_state = await self.browser_session.get_browser_state_summary(
            include_screenshot=self.settings.vision_fallback.include_screenshots
        )

        # Build comprehensive DOM state description (Phase 2: full DOM extraction)
        dom_state = self._build_dom_state(browser_state)

        # Get agent history from message manager
        history_text = self._get_agent_history_description()

        # Include task in agent_state (required for LLM to know what to do)
        agent_state_text = f"<user_request>\n{self.task}\n</user_request>"

        # Get read_state if available
        read_state_text = ""
        if self.message_manager.state.read_state_description:
            read_state_text = (
                f"\n<read_state>\n{self.message_manager.state.read_state_description}\n</read_state>\n"
            )

        # Combine into message
        content = (
            f"<agent_history>\n{history_text}\n</agent_history>\n\n"
            f"<agent_state>\n{agent_state_text}\n</agent_state>\n\n"
            f"<browser_state>\n{dom_state}\n</browser_state>"
            f"{read_state_text}"
        )

        # If screenshots are enabled, add them to the message
        if (
            self.settings.vision_fallback.include_screenshots
            and browser_state.screenshot
        ):
            from browser_use.llm.messages import (
                ContentPartImageParam,
                ContentPartTextParam,
                ImageURL,
            )

            # Resize screenshot if needed (similar to AgentMessagePrompt)
            screenshot = self._resize_screenshot_if_needed(browser_state.screenshot)

            content_parts = [
                ContentPartTextParam(text=content),
                ContentPartTextParam(text="Current screenshot:"),
                ContentPartImageParam(
                    image_url=ImageURL(
                        url=f"data:image/png;base64,{screenshot}",
                        media_type="image/png",
                        detail=self.settings.vision_fallback.detail_level,
                    )
                ),
            ]
            return UserMessage(content=content_parts, cache=True)

        return UserMessage(content=content, cache=True)

    def _resize_screenshot_if_needed(self, screenshot_b64: str) -> str:
        """
        Resize screenshot if it's too large for the LLM.

        Args:
            screenshot_b64: Base64-encoded screenshot

        Returns:
            Resized screenshot as base64 string (or original if no resize needed)
        """
        # For Phase 2, we'll use a simple approach - return as-is
        # In future phases, we can add actual resizing logic similar to AgentMessagePrompt
        # For now, LLMs can handle reasonable screenshot sizes
        return screenshot_b64

    def _build_minimal_message(self) -> UserMessage:
        """
        Build minimal message when both Sentience and vision fallback are disabled.

        Returns:
            UserMessage with minimal state
        """
        history_text = self._get_agent_history_description()
        
        # Include task in agent_state (required for LLM to know what to do)
        agent_state_text = f"<user_request>\n{self.task}\n</user_request>"
        
        read_state_text = ""
        if self.message_manager.state.read_state_description:
            read_state_text = (
                f"\n<read_state>\n{self.message_manager.state.read_state_description}\n</read_state>\n"
            )
        content = (
            f"<agent_history>\n{history_text}\n</agent_history>\n\n"
            f"<agent_state>\n{agent_state_text}\n</agent_state>"
            f"{read_state_text}"
        )
        return UserMessage(content=content, cache=True)

    def _get_agent_history_description(self) -> str:
        """
        Get agent history description from message manager.

        Returns:
            History description string
        """
        return self.message_manager.agent_history_description

    def _build_dom_state(self, browser_state: Any) -> str:
        """
        Build comprehensive DOM state description from browser state.

        This is used in vision fallback mode to provide full DOM context
        when Sentience snapshot is not available.

        Args:
            browser_state: BrowserStateSummary

        Returns:
            Complete DOM state description string with page info, stats, and DOM tree
        """
        from browser_use.dom.views import DEFAULT_INCLUDE_ATTRIBUTES, NodeType, SimplifiedNode

        # Extract page information
        url = getattr(browser_state, "url", None) or "unknown"
        title = getattr(browser_state, "title", None) or "unknown"
        page_info = getattr(browser_state, "page_info", None)
        dom_state = getattr(browser_state, "dom_state", None)

        # Build page statistics (similar to AgentMessagePrompt._extract_page_statistics)
        page_stats = self._extract_page_statistics(browser_state)

        # Format statistics for LLM
        stats_text = "<page_stats>"
        if page_stats["total_elements"] < 10:
            stats_text += "Page appears empty (SPA not loaded?) - "
        stats_text += (
            f'{page_stats["links"]} links, {page_stats["interactive_elements"]} interactive, '
            f'{page_stats["iframes"]} iframes, {page_stats["scroll_containers"]} scroll containers'
        )
        if page_stats["shadow_open"] > 0 or page_stats["shadow_closed"] > 0:
            stats_text += (
                f', {page_stats["shadow_open"]} shadow(open), '
                f'{page_stats["shadow_closed"]} shadow(closed)'
            )
        if page_stats["images"] > 0:
            stats_text += f', {page_stats["images"]} images'
        stats_text += f', {page_stats["total_elements"]} total elements'
        stats_text += "</page_stats>\n"

        # Get DOM tree representation
        elements_text = ""
        if dom_state:
            # Use the same method as AgentMessagePrompt to get LLM representation
            try:
                elements_text = dom_state.llm_representation(
                    include_attributes=DEFAULT_INCLUDE_ATTRIBUTES
                )
            except Exception as e:
                logger.info(f"Error getting DOM representation: {e}")
                elements_text = "Error extracting DOM tree"

        # Truncate DOM if too long (default max for vision fallback: 40000 chars)
        max_dom_length = 40000
        if len(elements_text) > max_dom_length:
            elements_text = elements_text[:max_dom_length]
            truncated_text = f" (truncated to {max_dom_length} characters)"
        else:
            truncated_text = ""

        # Build page info text
        page_info_text = ""
        has_content_above = False
        has_content_below = False

        if page_info:
            pi = page_info
            pages_above = pi.pixels_above / pi.viewport_height if pi.viewport_height > 0 else 0
            pages_below = pi.pixels_below / pi.viewport_height if pi.viewport_height > 0 else 0
            has_content_above = pages_above > 0
            has_content_below = pages_below > 0
            total_pages = pi.page_height / pi.viewport_height if pi.viewport_height > 0 else 0

            page_info_text = "<page_info>"
            page_info_text += f"{pages_above:.1f} pages above, "
            page_info_text += f"{pages_below:.1f} pages below, "
            page_info_text += f"{total_pages:.1f} total pages"
            page_info_text += "</page_info>\n"

        # Format elements text with page position indicators
        if elements_text:
            if has_content_above:
                if page_info:
                    pages_above = (
                        page_info.pixels_above / page_info.viewport_height
                        if page_info.viewport_height > 0
                        else 0
                    )
                    elements_text = f"... {pages_above:.1f} pages above ...\n{elements_text}"
            else:
                elements_text = f"[Start of page]\n{elements_text}"
            if not has_content_below:
                elements_text = f"{elements_text}\n[End of page]"
        else:
            elements_text = "empty page"

        # Build tabs information
        tabs_text = ""
        tabs = getattr(browser_state, "tabs", [])
        if tabs:
            tabs_text = "<tabs>\n"
            for tab in tabs:
                tab_id = getattr(tab, "target_id", "unknown")
                tab_url = getattr(tab, "url", "unknown")
                tab_title = getattr(tab, "title", "unknown")
                # Use last 4 chars of target_id for display
                tab_id_short = tab_id[-4:] if isinstance(tab_id, str) and len(tab_id) >= 4 else str(tab_id)
                tabs_text += f"Tab {tab_id_short}: {tab_url} - {tab_title[:30]}\n"
            tabs_text += "</tabs>\n"

        # Combine all parts
        dom_state_text = (
            f"URL: {url}\n"
            f"Title: {title}\n"
            f"{stats_text}"
            f"{page_info_text}"
            f"{tabs_text}"
            f"<dom_tree>\n{elements_text}{truncated_text}\n</dom_tree>"
        )

        return dom_state_text

    def _extract_page_statistics(self, browser_state: Any) -> dict[str, int]:
        """
        Extract high-level page statistics from DOM tree.

        Args:
            browser_state: BrowserStateSummary

        Returns:
            Dictionary with page statistics
        """
        from browser_use.dom.views import NodeType, SimplifiedNode

        stats = {
            "links": 0,
            "iframes": 0,
            "shadow_open": 0,
            "shadow_closed": 0,
            "scroll_containers": 0,
            "images": 0,
            "interactive_elements": 0,
            "total_elements": 0,
        }

        dom_state = getattr(browser_state, "dom_state", None)
        if not dom_state or not hasattr(dom_state, "_root") or not dom_state._root:
            return stats

        def traverse_node(node: SimplifiedNode) -> None:
            """Recursively traverse simplified DOM tree to count elements"""
            if not node or not hasattr(node, "original_node") or not node.original_node:
                return

            original = node.original_node
            stats["total_elements"] += 1

            # Count by node type and tag
            if original.node_type == NodeType.ELEMENT_NODE:
                tag = original.tag_name.lower() if hasattr(original, "tag_name") and original.tag_name else ""

                if tag == "a":
                    stats["links"] += 1
                elif tag in ("iframe", "frame"):
                    stats["iframes"] += 1
                elif tag == "img":
                    stats["images"] += 1

                # Check if scrollable
                if hasattr(original, "is_actually_scrollable") and original.is_actually_scrollable:
                    stats["scroll_containers"] += 1

                # Check if interactive
                if hasattr(node, "is_interactive") and node.is_interactive:
                    stats["interactive_elements"] += 1

                # Check if this element hosts shadow DOM
                if hasattr(node, "is_shadow_host") and node.is_shadow_host:
                    # Check if any shadow children are closed
                    has_closed_shadow = False
                    if hasattr(node, "children"):
                        for child in node.children:
                            if (
                                hasattr(child, "original_node")
                                and child.original_node
                                and child.original_node.node_type == NodeType.DOCUMENT_FRAGMENT_NODE
                                and hasattr(child.original_node, "shadow_root_type")
                                and child.original_node.shadow_root_type
                                and child.original_node.shadow_root_type.lower() == "closed"
                            ):
                                has_closed_shadow = True
                                break
                    if has_closed_shadow:
                        stats["shadow_closed"] += 1
                    else:
                        stats["shadow_open"] += 1

            # Traverse children
            if hasattr(node, "children"):
                for child in node.children:
                    traverse_node(child)

        traverse_node(dom_state._root)
        return stats

    async def _run_verification_assertions(
        self,
        step: int,
        sentience_state: Any | None,
        model_output: Any | None,
    ) -> tuple[bool, bool]:
        """Run verification assertions for the current step.

        Args:
            step: Current step number
            sentience_state: Current Sentience snapshot state (if available)
            model_output: Model output with next_goal (for step labeling)

        Returns:
            Tuple of (all_step_assertions_passed, task_done_by_assertion)
        """
        if not self._runtime or not self.settings.verification.enabled:
            return True, False

        # Begin step in AgentRuntime
        goal = ""
        if model_output and hasattr(model_output, "next_goal"):
            goal = model_output.next_goal or ""
        self._runtime.begin_step(goal=f"Step {step + 1}: {goal[:50]}")

        # Inject current snapshot into runtime (avoid double-snapshot)
        if sentience_state and hasattr(sentience_state, "snapshot"):
            self._runtime.last_snapshot = sentience_state.snapshot
            self._runtime._cached_url = sentience_state.snapshot.url if hasattr(sentience_state.snapshot, "url") else None

        # Run step assertions
        all_passed = True
        for assertion_config in self.settings.verification.step_assertions:
            predicate = assertion_config.get("predicate")
            label = assertion_config.get("label", "unnamed")
            required = assertion_config.get("required", False)

            if predicate:
                passed = self._runtime.assert_(predicate, label=label, required=required)
                if required and not passed:
                    all_passed = False
                logger.info(f"  🔍 Assertion '{label}': {'✅ PASS' if passed else '❌ FAIL'}")

        # Check done assertion
        task_done = False
        done_assertion = self.settings.verification.done_assertion
        if done_assertion:
            task_done = self._runtime.assert_done(done_assertion, label="task_complete")
            if task_done:
                logger.info("  🎯 Task verified complete by assertion!")

        return all_passed, task_done

    async def run(self) -> Any:
        """
        Run the agent loop with full action execution and history tracking.

        Returns:
            Dictionary with execution results (will return AgentHistoryList in future phases)
        """
        from browser_use.agent.views import AgentOutput, AgentStepInfo, ActionResult

        logger.info(f"Starting SentienceAgent: task='{self.task}'")

        # Initialize browser session if needed (start() is idempotent)
        await self.browser_session.start()

        # Initialize verification if enabled
        if self.settings.verification.enabled:
            await self._initialize_verification()

        # Get AgentOutput type from tools registry
        # Create action model from registered actions
        action_model = self.tools.registry.create_action_model()
        # Create AgentOutput type with custom actions
        from browser_use.agent.views import AgentOutput
        AgentOutputType = AgentOutput.type_with_custom_actions(action_model)

        # Track execution history
        execution_history: list[dict[str, Any]] = []
        sentience_used_in_any_step = False  # Track if Sentience was used in ANY step
        verification_task_done = False  # Track if task completed by assertion

        # Main agent loop
        for step in range(self.settings.max_steps):
            self._current_step = step
            step_info = AgentStepInfo(step_number=step, max_steps=self.settings.max_steps)
            logger.info(f"📍 Step {step + 1}/{self.settings.max_steps}")

            # Prepare context
            try:
                user_message, sentience_used = await self._prepare_context()
                # Log token usage breakdown
                message_content = str(user_message.content)
                history_text = self.message_manager.agent_history_description
                logger.info(
                    f"Context prepared: sentience_used={sentience_used}, "
                    f"message_length={len(message_content)} chars (~{len(message_content) // 4} tokens), "
                    f"history_length={len(history_text)} chars (~{len(history_text) // 4} tokens)"
                )

                # Get messages from message manager
                messages = self.message_manager.get_messages(user_message=user_message)

                # Select LLM: use vision_llm for vision fallback, primary llm for Sentience
                active_llm = self.vision_llm if (not sentience_used and self.vision_llm is not None) else self.llm
                if not sentience_used and self.vision_llm is not None:
                    logger.info("👁️ Using vision LLM for vision fallback mode")
                elif sentience_used:
                    logger.info("📊 Using primary LLM for Sentience snapshot mode")

                # Call LLM with structured output
                # NOTE: For Hugging Face models, this is where model loading/downloading happens
                logger.info("🤖 Calling LLM (this may trigger model download/loading for Hugging Face models)...")
                kwargs: dict = {"output_format": AgentOutputType, "session_id": self.browser_session.id}
                response = await asyncio.wait_for(
                    active_llm.ainvoke(messages, **kwargs),
                    timeout=self.settings.llm_timeout,
                )
                logger.info("✅ LLM response received")

                # Parse AgentOutput from response
                # Handle case where LLM returns string instead of structured output
                if isinstance(response.completion, str):
                    logger.warning(
                        f"⚠️  LLM returned raw text instead of structured output. "
                        f"This may happen with smaller local models. Response: {response.completion[:200]}..."
                    )
                    # Try to parse as JSON manually with improved repair logic
                    try:
                        import json
                        import re
                        
                        # Try to extract JSON from response (might be wrapped in markdown or have extra text)
                        json_text = response.completion.strip()
                        
                        # Log the full response for debugging (truncated JSON issues)
                        logger.debug(f"Full LLM response ({len(json_text)} chars): {json_text[:1000]}...")
                        
                        # Remove markdown code blocks if present
                        if json_text.startswith('```json'):
                            json_text = re.sub(r'^```json\s*', '', json_text, flags=re.MULTILINE)
                            json_text = re.sub(r'```\s*$', '', json_text, flags=re.MULTILINE)
                        elif json_text.startswith('```'):
                            json_text = re.sub(r'^```\s*', '', json_text, flags=re.MULTILINE)
                            json_text = re.sub(r'```\s*$', '', json_text, flags=re.MULTILINE)
                        
                        # Try to find JSON object in the text (from first { to last })
                        json_match = re.search(r'\{.*', json_text, re.DOTALL)
                        if json_match:
                            json_text = json_match.group(0)
                        
                        # Try to fix incomplete JSON (common with truncated responses)
                        # Count braces and brackets to see what's missing
                        open_braces = json_text.count('{')
                        close_braces = json_text.count('}')
                        open_brackets = json_text.count('[')
                        close_brackets = json_text.count(']')
                        
                        # Find the last complete structure and close everything after it
                        # Strategy: Find the last complete key-value pair or array element, then close everything
                        if open_braces > close_braces or open_brackets > close_brackets:
                            logger.debug(
                                f"JSON appears incomplete: braces {open_braces}/{close_braces}, "
                                f"brackets {open_brackets}/{close_brackets}. Attempting repair..."
                            )
                            
                            # Try to find where the JSON was cut off
                            # Look for incomplete strings, incomplete objects, etc.
                            
                            # Close missing brackets first (they're usually nested inside objects)
                            if open_brackets > close_brackets:
                                missing_brackets = open_brackets - close_brackets
                                json_text += ']' * missing_brackets
                            
                            # Close missing braces
                            if open_braces > close_braces:
                                missing_braces = open_braces - close_braces
                                json_text += '\n' + '}' * missing_braces
                            
                            # Try to fix incomplete strings (if JSON was cut off mid-string)
                            # Count unescaped quotes
                            unescaped_quotes = len(re.findall(r'(?<!\\)"', json_text))
                            if unescaped_quotes % 2 != 0:
                                # Odd number of quotes means incomplete string
                                # Find the last unescaped quote and close the string
                                last_quote_pos = json_text.rfind('"')
                                if last_quote_pos > 0 and json_text[last_quote_pos - 1] != '\\':
                                    # Check if we're in a string context
                                    before_quote = json_text[:last_quote_pos]
                                    # If the last quote is opening a string (not closing), add closing quote
                                    if before_quote.count('"') % 2 == 0:
                                        json_text = json_text[:last_quote_pos + 1] + '"' + json_text[last_quote_pos + 1:]
                        
                        logger.debug(f"Repaired JSON ({len(json_text)} chars): {json_text[:500]}...")
                        parsed = json.loads(json_text)
                        model_output = AgentOutputType.model_validate(parsed)
                    except (json.JSONDecodeError, Exception) as e:
                        logger.error(f"Failed to parse LLM response as JSON: {e}")
                        # Log the problematic JSON for debugging
                        logger.error(f"Problematic JSON (first 800 chars): {json_text[:800]}")
                        logger.error(f"Full raw response length: {len(response.completion)} chars")
                        
                        # Try one more aggressive repair: if JSON is clearly truncated, try to salvage what we can
                        try:
                            # Find the last complete field and create minimal valid JSON
                            # Look for the last complete key-value pair
                            last_comma = json_text.rfind(',')
                            last_colon = json_text.rfind(':')
                            
                            if last_comma > 0 and last_colon > last_comma:
                                # We have at least one complete field
                                # Try to extract up to the last complete field and close it
                                # Find the last complete field by looking for pattern: "key": value,
                                field_pattern = r'"\w+":\s*[^,}]+,'
                                matches = list(re.finditer(field_pattern, json_text))
                                if matches:
                                    last_match = matches[-1]
                                    # Extract up to and including the last complete field
                                    salvage_text = json_text[:last_match.end()]
                                    # Close any open structures
                                    salvage_text = salvage_text.rstrip(', \n')
                                    if salvage_text.count('{') > salvage_text.count('}'):
                                        salvage_text += '\n' + '}' * (salvage_text.count('{') - salvage_text.count('}'))
                                    if salvage_text.count('[') > salvage_text.count(']'):
                                        salvage_text += ']' * (salvage_text.count('[') - salvage_text.count(']'))
                                    
                                    logger.debug(f"Attempting salvage repair on: {salvage_text[:300]}...")
                                    parsed = json.loads(salvage_text)
                                    model_output = AgentOutputType.model_validate(parsed)
                                    logger.info("✅ Successfully salvaged incomplete JSON")
                                else:
                                    raise  # Re-raise original error
                            else:
                                raise  # Re-raise original error
                        except Exception:
                            # Salvage failed, use error fallback
                            logger.debug(f"Raw response (first 500 chars): {response.completion[:500]}")
                        # Create a minimal AgentOutput with error (using required fields only)
                        model_output = AgentOutputType(
                            evaluation_previous_goal="Failed to parse LLM output",
                            memory=f"LLM returned invalid JSON: {str(e)[:100]}",
                            next_goal="Retry with simpler request",
                            action=[],  # Empty action list to indicate failure
                        )
                        # Add error to history
                        self.message_manager.update_history(
                            model_output=None,
                            result=[ActionResult(error=f"LLM failed to generate valid structured output: {str(e)[:200]}")],
                            step_info=step_info,
                        )
                        self._consecutive_failures += 1
                        continue
                else:
                    model_output: AgentOutput = response.completion  # type: ignore[assignment]

                logger.info(
                    f"LLM response received: {len(model_output.action) if model_output.action else 0} actions"
                )

                # Execute actions
                action_results: list[ActionResult] = []
                if model_output.action:
                    action_results = await self._execute_actions(model_output.action)

                # Run verification assertions (if enabled)
                assertions_passed, verification_task_done = await self._run_verification_assertions(
                    step=step,
                    sentience_state=self._current_sentience_state,
                    model_output=model_output,
                )

                # Update history with model output and action results
                self.message_manager.update_history(
                    model_output=model_output,
                    result=action_results,
                    step_info=step_info,
                )

                # Track Sentience usage across all steps
                if sentience_used:
                    sentience_used_in_any_step = True

                # Track in execution history (include verification results)
                step_entry = {
                    "step": step + 1,
                    "model_output": model_output,
                    "action_results": action_results,
                    "sentience_used": sentience_used,
                }
                if self.settings.verification.enabled:
                    step_entry["assertions_passed"] = assertions_passed
                    step_entry["verification_task_done"] = verification_task_done
                execution_history.append(step_entry)

                # Check if done (by LLM action OR by verification assertion)
                is_done = any(result.is_done for result in action_results if result.is_done)
                if is_done:
                    logger.info("✅ Task completed (LLM done action)")
                    break
                if verification_task_done:
                    logger.info("✅ Task completed (verified by assertion)")
                    break

                # Check for errors
                has_errors = any(result.error for result in action_results if result.error)
                if has_errors:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self.settings.max_failures:
                        logger.info("Max failures reached, stopping")
                        break
                else:
                    self._consecutive_failures = 0  # Reset on success

            except asyncio.TimeoutError:
                logger.info(f"Step {step + 1} timed out after {self.settings.llm_timeout}s")
                self._consecutive_failures += 1
                # Update history with error
                self.message_manager.update_history(
                    model_output=None,
                    result=None,
                    step_info=step_info,
                )
                if self._consecutive_failures >= self.settings.max_failures:
                    logger.info("Max failures reached, stopping")
                    break
            except Exception as e:
                logger.info(f"Step {step + 1} failed: {e}", exc_info=True)
                self._consecutive_failures += 1
                # Update history with error
                self.message_manager.update_history(
                    model_output=None,
                    result=None,
                    step_info=step_info,
                )
                if self._consecutive_failures >= self.settings.max_failures:
                    logger.info("Max failures reached, stopping")
                    break

        # Return usage summary and execution history
        usage_summary = await self.token_cost_service.get_usage_summary()
        logger.info(f"Agent completed: {usage_summary}")

        # Count how many steps used Sentience
        steps_using_sentience = sum(1 for entry in execution_history if entry.get("sentience_used", False))
        total_steps = len(execution_history)

        # Build verification summary (if enabled)
        verification_summary = None
        if self.settings.verification.enabled and self._runtime:
            verification_summary = {
                "enabled": True,
                "all_assertions_passed": self._runtime.all_assertions_passed(),
                "required_assertions_passed": self._runtime.required_assertions_passed(),
                "task_verified_complete": self._runtime.is_task_done,
                "assertions": self._runtime.get_assertions_for_step_end().get("assertions", []),
            }
            logger.info(
                f"📊 Verification Summary: "
                f"all_passed={verification_summary['all_assertions_passed']}, "
                f"task_done={verification_summary['task_verified_complete']}"
            )

            # Close tracer if we created it
            if self._tracer and hasattr(self._tracer, "close"):
                self._tracer.close()
                logger.info(f"📝 Trace saved to: {self.settings.verification.trace_dir}/")

        # Return execution summary (will return AgentHistoryList in future phases)
        result = {
            "steps": self._current_step + 1,
            "sentience_used": sentience_used_in_any_step,
            "sentience_usage_stats": {
                "steps_using_sentience": steps_using_sentience,
                "total_steps": total_steps,
                "sentience_percentage": (steps_using_sentience / total_steps * 100) if total_steps > 0 else 0,
            },
            "usage": usage_summary,
            "execution_history": execution_history,
        }

        # Add verification results if enabled
        if verification_summary:
            result["verification"] = verification_summary

        return result

    async def _execute_actions(self, actions: list[Any]) -> list[Any]:
        """
        Execute a list of actions.

        Args:
            actions: List of ActionModel instances

        Returns:
            List of ActionResult instances
        """
        from browser_use.agent.views import ActionResult
        from browser_use.browser.events import BrowserStateRequestEvent

        results: list[ActionResult] = []
        total_actions = len(actions)

        # Ensure selector_map is built before executing actions
        # This is needed because Sentience uses backend_node_ids that must exist in selector_map
        selector_map = await self.browser_session.get_selector_map()
        if not selector_map:
            logger.info("  🔄 Selector map is empty, triggering DOM build...")
            # Trigger browser state request to build DOM and selector_map
            event = self.browser_session.event_bus.dispatch(
                BrowserStateRequestEvent(include_screenshot=False)
            )
            await event
            await event.event_result(raise_if_any=True, raise_if_none=False)
            selector_map = await self.browser_session.get_selector_map()
            logger.info(f"  ✅ Selector map built: {len(selector_map)} elements available")

        for i, action in enumerate(actions):
            # Wait between actions (except first)
            if i > 0:
                wait_time = getattr(
                    self.browser_session.browser_profile, "wait_between_actions", 0.5
                )
                await asyncio.sleep(wait_time)

            try:
                # Get action name for logging
                action_data = action.model_dump(exclude_unset=True)
                action_name = next(iter(action_data.keys())) if action_data else "unknown"
                action_params = action_data.get(action_name, {})
                
                # Check if action uses an index and validate it exists in selector_map
                action_index = action_params.get('index')
                if action_index is not None and action_name in ('click', 'input', 'input_text'):
                    selector_map = await self.browser_session.get_selector_map()
                    if action_index not in selector_map:
                        # Try to find element in Sentience snapshot using SDK's find() function
                        sentience_element = None
                        if self._current_sentience_state and hasattr(self._current_sentience_state, 'snapshot'):
                            snapshot = self._current_sentience_state.snapshot
                            
                            # First, try to find by ID
                            sentience_element = self._find_element_in_snapshot(snapshot, element_id=action_index)
                            
                            # If not found by ID and this is an input action, try to find by text
                            if not sentience_element and action_name == 'input' and 'text' in action_params:
                                text_to_find = action_params.get('text', '')
                                if text_to_find:
                                    sentience_element = self._find_element_in_snapshot(snapshot, text=text_to_find)
                                    if sentience_element:
                                        logger.info(
                                            f"  🔍 Element {action_index} not found by ID, but found by text '{text_to_find[:30]}...' "
                                            f"in Sentience snapshot. Using element ID {sentience_element.id}."
                                        )
                                        # Update action_index to use the found element's ID
                                        action_index = sentience_element.id
                                        action_params['index'] = action_index
                        
                        if sentience_element:
                            logger.info(
                                f"  🔍 Element {action_index} not in selector_map, but found in Sentience snapshot. "
                                f"Validating backend_node_id exists in CDP before adding to selector_map."
                            )
                            
                            # Get current target_id for the element - use agent_focus_target_id which is the active tab
                            target_id = self.browser_session.agent_focus_target_id
                            if not target_id:
                                # Fallback: get first available target
                                targets = await self.browser_session.session_manager.get_all_targets()
                                if targets:
                                    target_id = list(targets.keys())[0]
                            
                            # Validate that the backend_node_id actually exists in CDP before adding to selector_map
                            # This prevents "No node with given id found" errors
                            backend_node_id = action_index
                            node_exists = False
                            try:
                                cdp_session = await self.browser_session.get_or_create_cdp_session(
                                    target_id=target_id, focus=False
                                )
                                # Try to resolve the node to verify it exists
                                result = await cdp_session.cdp_client.send.DOM.resolveNode(
                                    params={'backendNodeId': backend_node_id},
                                    session_id=cdp_session.session_id,
                                )
                                if result.get('object') and result['object'].get('objectId'):
                                    node_exists = True
                                    logger.info(f"  ✅ Validated backend_node_id {backend_node_id} exists in CDP")
                            except Exception as e:
                                logger.warning(
                                    f"  ⚠️  backend_node_id {backend_node_id} not found in CDP (node may be stale): {e}. "
                                    f"Skipping adding to selector_map to avoid fallback typing."
                                )
                            
                            if not node_exists:
                                # Node doesn't exist - don't add to selector_map, let the action fail naturally
                                logger.info(
                                    f"  ⚠️  Cannot add element {action_index} to selector_map - backend_node_id is stale. "
                                    f"Action will fail and agent should retry with a fresh snapshot."
                                )
                            else:
                                # Node exists - create minimal EnhancedDOMTreeNode and add to selector_map
                                from browser_use.dom.views import EnhancedDOMTreeNode, NodeType
                                
                                # Extract role and other info from Sentience element
                                role = getattr(sentience_element, 'role', 'div') or 'div'
                                
                                # For input actions, prefer textbox/searchbox over combobox if the element text suggests it's a search box
                                if action_name == 'input' and role.lower() == 'combobox':
                                    element_text = getattr(sentience_element, 'text', '') or ''
                                    if any(keyword in element_text.lower() for keyword in ['search', 'query', 'find']):
                                        logger.info(f"  🔄 Overriding role from 'combobox' to 'searchbox' based on element text")
                                        role = 'searchbox'
                                
                                # Map common roles to HTML tag names
                                role_to_tag = {
                                    'textbox': 'input',
                                    'searchbox': 'input',
                                    'button': 'button',
                                    'link': 'a',
                                    'combobox': 'select',
                                }
                                tag_name = role_to_tag.get(role.lower(), 'div')
                                
                                # Create minimal EnhancedDOMTreeNode with proper target_id
                                # Don't set session_id - let cdp_client_for_node use target_id strategy (more reliable)
                                minimal_node = EnhancedDOMTreeNode(
                                    node_id=0,  # Will be resolved when needed via CDP using backend_node_id
                                    backend_node_id=backend_node_id,  # This is the key - matches Sentience element.id
                                    node_type=NodeType.ELEMENT_NODE,
                                    node_name=tag_name,
                                    node_value='',
                                    attributes={'role': role, 'type': 'text'} if role in ('textbox', 'searchbox') else {'role': role} if role else {},
                                    is_visible=True,  # Sentience elements are visible
                                    target_id=target_id or '',  # type: ignore
                                    session_id=None,  # Let cdp_client_for_node use target_id strategy instead
                                    frame_id=None,
                                    content_document=None,
                                    shadow_root_type=None,
                                    shadow_roots=None,
                                    parent_node=None,
                                    children_nodes=None,
                                    ax_node=None,
                                    snapshot_node=None,
                                    is_scrollable=None,
                                    absolute_position=None,
                                )
                                
                                # Add to selector_map temporarily
                                selector_map[backend_node_id] = minimal_node
                                # Also update cached selector_map
                                self.browser_session.update_cached_selector_map(selector_map)
                                logger.info(f"  ✅ Added element {backend_node_id} (role={role}, tag={tag_name}) to selector_map temporarily")
                        else:
                            available_indices = sorted(list(selector_map.keys()))[:20]
                            logger.info(
                                f"  ⚠️  Action {action_name} uses index {action_index}, but it's not in selector_map or Sentience snapshot. "
                                f"Available indices: {available_indices}{'...' if len(selector_map) > 20 else ''} "
                                f"(total: {len(selector_map)})"
                            )
                
                logger.info(f"  ▶️  {action_name}: {action_params}")
                
                # Warn about multiple scroll actions (potential jittery behavior)
                if action_name == "scroll" and i > 0:
                    prev_action_data = actions[i - 1].model_dump(exclude_unset=True)
                    prev_action_name = next(iter(prev_action_data.keys())) if prev_action_data else "unknown"
                    if prev_action_name == "scroll":
                        logger.info(f"  ⚠️  Multiple scroll actions detected - may cause jittery behavior")

                # Execute action
                result = await self.tools.act(
                    action=action,
                    browser_session=self.browser_session,
                    file_system=self.file_system,
                    page_extraction_llm=self.llm,  # Use the same LLM for extraction
                    sensitive_data=None,  # TODO: Add sensitive data support
                    available_file_paths=None,  # TODO: Add file paths support
                )

                results.append(result)

                # Log result
                if result.error:
                    logger.info(f"  ❌ Action failed: {result.error}")
                elif result.is_done:
                    logger.info(f"  ✅ Task done: {result.long_term_memory or result.extracted_content}")

                # Stop if done or error (for now, continue on error)
                if result.is_done:
                    break

            except Exception as e:
                logger.info(f"  ❌ Action execution error: {e}", exc_info=True)
                # Create error result
                error_result = ActionResult(
                    error=f"Action execution failed: {str(e)}",
                    is_done=False,
                )
                results.append(error_result)

        return results

    def _get_system_message(self) -> SystemMessage:
        """
        Get system message for the agent.

        Uses the standard browser-use system prompt to ensure consistency.

        Returns:
            SystemMessage
        """
        from browser_use.agent.prompts import SystemPrompt

        # Use standard system prompt with Sentience-specific extensions
        system_prompt = SystemPrompt(
            max_actions_per_step=3,  # Default
            use_thinking=True,
            flash_mode=False,
            is_anthropic=False,  # Will be auto-detected if needed
            is_browser_use_model=False,  # Will be auto-detected if needed
            extend_system_message=(
                "\n<sentience_format>\n"
                "CRITICAL: When browser_state contains elements in Sentience format, "
                "the first column is labeled 'ID' but browser-use actions use a parameter called 'index'.\n"
                "You MUST use the ID value (first column) as the 'index' parameter value for ALL interactions.\n"
                "\n"
                "Format: ID|role|text|imp|is_primary|docYq|ord|DG|href\n"
                "- The first column is the ID (e.g., in '21|link|Some text|...', the ID is 21)\n"
                "- This ID is a backend_node_id from Chrome DevTools Protocol\n"
                "- Browser-use actions use a parameter called 'index' (not 'id')\n"
                "- Use the ID value as the index parameter value: ID → index parameter\n"
                "\n"
                "Usage Rules:\n"
                "- For '21|link|Some text|...', use: click with index: 21 (the ID value becomes the index value)\n"
                "- For '48|textbox|Search...', use: input with index: 48, text: \"your text\"\n"
                "- The Sentience ID value IS the browser-use index value - use it directly\n"
                "\n"
                "Examples:\n"
                "- Sentience format: '21|link|Click here|100|1|0|1|1|https://...'\n"
                "  → Action: click with index: 21 (use the ID value 21 as the index parameter)\n"
                "- Sentience format: '48|textbox|Search...|95|0|0|-|0|'\n"
                "  → Action: input with index: 48, text: \"your text\"\n"
                "\n"
                "Terminology Note:\n"
                "- Sentience format column name: 'ID' (first column)\n"
                "- Browser-use action parameter name: 'index'\n"
                "- The ID value from Sentience becomes the index value for browser-use actions\n"
                "\n"
                "IMPORTANT WARNINGS:\n"
                "- ONLY use ID values that appear in the Sentience format list\n"
                "- Some Sentience IDs may not be available if the element is not interactive by browser-use's criteria\n"
                "- If an action fails with 'Element index X not available', that ID doesn't exist in the selector_map\n"
                "- In that case, try a different element ID from the Sentience format list\n"
                "- NEVER use arbitrary index numbers when Sentience format is present\n"
                "- NEVER ignore the ID from the Sentience format - it is the ONLY valid index to use\n"
                "</sentience_format>\n"
            ),
        ).get_system_message()

        return system_prompt

    def _is_done(self, model_output: Any) -> bool:
        """
        Check if task is done based on model output.

        Simplified for Phase 1.

        Args:
            model_output: Model output

        Returns:
            True if done, False otherwise
        """
        # TODO: Parse model output and check for 'done' action
        return False
