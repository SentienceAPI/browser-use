"""
SentienceAgent: Custom agent with full control over prompt construction.

This agent uses Sentience SDK snapshots as the primary, compact prompt format,
with automatic fallback to vision mode when snapshots fail.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import SystemMessage, UserMessage
from browser_use.tokens.service import TokenCost
from browser_use.tokens.views import UsageSummary

if TYPE_CHECKING:
    from browser_use.browser.session import BrowserSession
    from browser_use.tools.service import Tools

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


class SentienceAgent:
    """
    Custom agent with full control over prompt construction.

    Features:
    - Primary: Sentience snapshot as compact prompt (~3K tokens)
    - Fallback: Vision mode when snapshot fails (~40K tokens)
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
        sentience_api_key: str | None = None,
        sentience_use_api: bool | None = None,
        sentience_max_elements: int = 60,
        sentience_show_overlay: bool = False,
        sentience_wait_for_extension_ms: int = 5000,
        sentience_retries: int = 2,
        sentience_retry_delay_s: float = 1.0,
        # Vision fallback configuration
        vision_fallback_enabled: bool = True,
        vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
        vision_include_screenshots: bool = True,
        # Token tracking
        calculate_cost: bool = True,
        # Agent settings
        max_steps: int = 100,
        max_failures: int = 3,
        llm_timeout: int = 60,
        step_timeout: int = 120,
        **kwargs,
    ):
        """
        Initialize SentienceAgent.

        Args:
            task: The task for the agent to complete
            llm: Language model to use
            browser_session: Browser session instance
            tools: Tools registry (optional)
            sentience_api_key: Sentience API key for gateway mode
            sentience_use_api: Force API vs extension mode
            sentience_max_elements: Maximum elements in snapshot
            sentience_show_overlay: Show visual overlay
            sentience_wait_for_extension_ms: Wait time for extension
            sentience_retries: Number of snapshot retries
            sentience_retry_delay_s: Delay between retries
            vision_fallback_enabled: Enable vision fallback
            vision_detail_level: Vision detail level
            vision_include_screenshots: Include screenshots in fallback
            calculate_cost: Track token usage
            max_steps: Maximum steps
            max_failures: Maximum failures
            llm_timeout: LLM timeout
            step_timeout: Step timeout
        """
        self.task = task
        self.llm = llm
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
        sentience_config = SentienceAgentConfig(
            sentience_api_key=sentience_api_key,
            sentience_use_api=sentience_use_api,
            sentience_max_elements=sentience_max_elements,
            sentience_show_overlay=sentience_show_overlay,
            sentience_wait_for_extension_ms=sentience_wait_for_extension_ms,
            sentience_retries=sentience_retries,
            sentience_retry_delay_s=sentience_retry_delay_s,
        )
        vision_fallback = VisionFallbackConfig(
            enabled=vision_fallback_enabled,
            detail_level=vision_detail_level,
            include_screenshots=vision_include_screenshots,
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
        )

        # Initialize SentienceContext (lazy import to avoid hard dependency)
        self._sentience_context: Any | None = None

        # Initialize token cost service
        self.token_cost_service = TokenCost(include_cost=calculate_cost)
        self.token_cost_service.register_llm(llm)

        # Initialize message manager for history tracking
        from browser_use.integrations.sentience.message_manager import CustomMessageManager

        system_message = self._get_system_message()
        self.message_manager = CustomMessageManager(
            task=task,
            system_message=system_message,
            max_history_items=None,  # Keep all history for now
        )

        # Track state
        self._current_step = 0
        self._consecutive_failures = 0
        self._sentience_used_in_last_step = False

        logger.info(
            f"Initialized SentienceAgent: task='{task}', "
            f"sentience_max_elements={sentience_max_elements}, "
            f"vision_fallback={'enabled' if vision_fallback_enabled else 'disabled'}"
        )

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
            # Use Sentience prompt block
            user_message = self._build_sentience_message(sentience_state)
            self._sentience_used_in_last_step = True
            logger.info("✅ Using Sentience snapshot for prompt")
            return user_message, True
        else:
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
            sentience_context = self._get_sentience_context()
            sentience_state = await sentience_context.build(
                self.browser_session,
                goal=self.task,
                wait_for_extension_ms=self.settings.sentience_config.sentience_wait_for_extension_ms,
                retries=self.settings.sentience_config.sentience_retries,
                retry_delay_s=self.settings.sentience_config.sentience_retry_delay_s,
            )
            return sentience_state
        except Exception as e:
            logger.info(f"Sentience snapshot failed: {e}")
            return None

    def _build_sentience_message(self, sentience_state: Any) -> UserMessage:
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

        # Log the Sentience prompt block for debugging
        logger.info(
            f"📋 Sentience prompt block ({len(sentience_state.prompt_block)} chars):\n"
            f"{sentience_state.prompt_block[:500]}..."  # First 500 chars
            if len(sentience_state.prompt_block) > 500
            else sentience_state.prompt_block
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

        # Get AgentOutput type from tools registry
        # Create action model from registered actions
        action_model = self.tools.registry.create_action_model()
        # Create AgentOutput type with custom actions
        from browser_use.agent.views import AgentOutput
        AgentOutputType = AgentOutput.type_with_custom_actions(action_model)

        # Track execution history
        execution_history: list[dict[str, Any]] = []

        # Main agent loop
        for step in range(self.settings.max_steps):
            self._current_step = step
            step_info = AgentStepInfo(step_number=step, max_steps=self.settings.max_steps)
            logger.info(f"📍 Step {step + 1}/{self.settings.max_steps}")

            # Prepare context
            try:
                user_message, sentience_used = await self._prepare_context()
                logger.info(
                    f"Context prepared: sentience_used={sentience_used}, "
                    f"message_length={len(str(user_message.content))}"
                )

                # Get messages from message manager
                messages = self.message_manager.get_messages(user_message=user_message)

                # Call LLM with structured output
                kwargs: dict = {"output_format": AgentOutputType, "session_id": self.browser_session.id}
                response = await asyncio.wait_for(
                    self.llm.ainvoke(messages, **kwargs),
                    timeout=self.settings.llm_timeout,
                )

                # Parse AgentOutput from response
                model_output: AgentOutput = response.completion  # type: ignore[assignment]

                logger.info(
                    f"LLM response received: {len(model_output.action) if model_output.action else 0} actions"
                )

                # Execute actions
                action_results: list[ActionResult] = []
                if model_output.action:
                    action_results = await self._execute_actions(model_output.action)

                # Update history with model output and action results
                self.message_manager.update_history(
                    model_output=model_output,
                    result=action_results,
                    step_info=step_info,
                )

                # Track in execution history
                execution_history.append(
                    {
                        "step": step + 1,
                        "model_output": model_output,
                        "action_results": action_results,
                        "sentience_used": sentience_used,
                    }
                )

                # Check if done
                is_done = any(result.is_done for result in action_results if result.is_done)
                if is_done:
                    logger.info("✅ Task completed")
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

        # Return execution summary (will return AgentHistoryList in future phases)
        return {
            "steps": self._current_step + 1,
            "sentience_used": self._sentience_used_in_last_step,
            "usage": usage_summary,
            "execution_history": execution_history,
        }

    async def _execute_actions(self, actions: list[Any]) -> list[Any]:
        """
        Execute a list of actions.

        Args:
            actions: List of ActionModel instances

        Returns:
            List of ActionResult instances
        """
        from browser_use.agent.views import ActionResult

        results: list[ActionResult] = []
        total_actions = len(actions)

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
                logger.info(f"  ▶️  {action_name}: {action_data.get(action_name, {})}")

                # Execute action
                result = await self.tools.act(
                    action=action,
                    browser_session=self.browser_session,
                    file_system=self.file_system,
                    page_extraction_llm=None,  # TODO: Add page extraction LLM support
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
                "IMPORTANT: When browser_state contains elements in Sentience format (ID|role|text|...), "
                "you MUST use the element ID (first field) as the index parameter for interactions.\n"
                "- The format shows: ID|role|text|imp|is_primary|docYq|ord|DG|href\n"
                "- Use click with index=ID where ID is the first number (e.g., from '65|span|Show HN:...' use click with index: 65)\n"
                "- Use input with index=ID for text inputs (e.g., from '48|textbox|Search...' use input with index: 48)\n"
                "- The ID in the Sentience format IS the index to use - they are the same value\n"
                "- Example: For element '65|span|Show HN: Rocket Launch...', use click with index: 65\n"
                "- DO NOT use arbitrary index numbers when Sentience format is present - always use the ID from the element line\n"
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
