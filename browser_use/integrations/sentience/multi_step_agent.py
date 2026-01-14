"""
Multi-Step SentienceAgent: Uses SentienceAgentAsync from Sentience SDK for multi-step task execution with per-step verification.

This agent provides:
- Multi-step task execution with step-by-step verification
- AgentRuntime integration for declarative assertions
- Tracer support for execution tracking
- Local LLM support (Qwen 2.5 3B via LocalLLMProvider)

Example:
    >>> from browser_use.integrations.sentience import MultiStepSentienceAgent
    >>> from sentience.async_api import AsyncSentienceBrowser
    >>> from sentience.llm_provider import LocalLLMProvider
    >>>
    >>> async with AsyncSentienceBrowser() as browser:
    >>>     llm = LocalLLMProvider(model_name="Qwen/Qwen2.5-3B-Instruct")
    >>>     agent = MultiStepSentienceAgent(
    >>>         browser=browser,
    >>>         llm=llm,
    >>>     )
    >>>
    >>>     task_steps = [
    >>>         {"goal": "Step 1", "task": "Do something"},
    >>>         {"goal": "Step 2", "task": "Do something else"},
    >>>     ]
    >>>
    >>>     results = await agent.run_multi_step(task_steps)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from sentience.agent import SentienceAgentAsync
    from sentience.agent_config import AgentConfig
    from sentience.agent_runtime import AgentRuntime
    from sentience.async_api import AsyncSentienceBrowser
    from sentience.llm_provider import LLMProvider
    from sentience.tracing import Tracer

logger = logging.getLogger(__name__)


class MultiStepSentienceAgent:
    """
    Multi-step agent using SentienceAgentAsync from Sentience SDK.
    
    Features:
    - Multi-step task execution
    - AgentRuntime integration for verification
    - Tracer support for execution tracking
    - Step-by-step assertions using expect() DSL
    - Local LLM support (Qwen 2.5 3B)
    """

    def __init__(
        self,
        browser: AsyncSentienceBrowser,
        llm: LLMProvider,
        runtime: AgentRuntime | None = None,
        tracer: Tracer | None = None,
        trace_dir: str | Path = "traces",
        sentience_api_key: str | None = None,
        agent_config: AgentConfig | None = None,
        default_snapshot_limit: int = 50,
        verbose: bool = True,
        **agent_kwargs: Any,
    ):
        """
        Initialize Multi-Step SentienceAgent.

        Args:
            browser: AsyncSentienceBrowser instance from Sentience SDK
            llm: LLMProvider instance (e.g., LocalLLMProvider for Qwen 2.5 3B)
            runtime: Optional AgentRuntime (will be created if not provided)
            tracer: Optional Tracer (will be created if not provided)
            trace_dir: Directory for trace files
            sentience_api_key: Optional Sentience API key for gateway mode
            agent_config: Optional AgentConfig for SentienceAgentAsync
            default_snapshot_limit: Default snapshot limit for agent
            verbose: Print execution logs
            **agent_kwargs: Additional kwargs passed to SentienceAgentAsync
        """
        self.browser = browser
        self.llm = llm
        self.agent_config = agent_config
        self.default_snapshot_limit = default_snapshot_limit
        self.verbose = verbose
        self.agent_kwargs = agent_kwargs
        self.trace_dir = Path(trace_dir)
        self.sentience_api_key = sentience_api_key or os.getenv("SENTIENCE_API_KEY")
        
        # Runtime and tracer (initialized lazily)
        self._runtime: AgentRuntime | None = runtime
        self._tracer: Tracer | None = tracer
        self._verification_initialized = False

    async def _initialize_verification(self) -> None:
        """Initialize AgentRuntime and Tracer for verification."""
        if self._verification_initialized:
            return

        try:
            from sentience.agent_runtime import AgentRuntime
            from sentience.tracing import JsonlTraceSink, Tracer

            # Create tracer if not provided
            if self._tracer is None:
                self.trace_dir.mkdir(exist_ok=True)
                run_id = f"multi-step-agent-{int(time.time())}"
                sink = JsonlTraceSink(str(self.trace_dir / f"{run_id}.jsonl"))
                self._tracer = Tracer(run_id=run_id, sink=sink)
                logger.info(f"📝 Created tracer: {self.trace_dir / f'{run_id}.jsonl'}")

            # Create AgentRuntime if not provided
            if self._runtime is None:
                # AgentRuntime needs a backend - create PlaywrightBackend directly
                # AsyncSentienceBrowser has a .page property
                page = self.browser.page
                if page is None:
                    logger.warning("⚠️  No page available for AgentRuntime")
                    raise ValueError("AsyncSentienceBrowser must have a page. Call browser.goto() or browser.new_page() first.")
                
                # Create backend directly to avoid legacy path issues
                from sentience.backends.playwright_backend import PlaywrightBackend
                
                backend = PlaywrightBackend(page)
                self._runtime = AgentRuntime(
                    backend=backend,
                    tracer=self._tracer,
                    sentience_api_key=self.sentience_api_key,
                )
                logger.info("✅ Created AgentRuntime for verification")

            self._verification_initialized = True

        except ImportError as e:
            logger.warning(
                f"⚠️  Verification requested but Sentience SDK not fully installed: {e}. "
                "Install with: pip install sentienceapi"
            )
            self._verification_initialized = False
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize verification: {e}")
            import traceback
            logger.debug(f"  📋 Traceback: {traceback.format_exc()}")
            self._verification_initialized = False

    @property
    def runtime(self) -> AgentRuntime | None:
        """Get AgentRuntime instance."""
        return self._runtime

    @property
    def tracer(self) -> Tracer | None:
        """Get Tracer instance."""
        return self._tracer

    async def run_multi_step(
        self,
        task_steps: list[dict[str, str]],
        verification_callbacks: dict[int, Callable[[Any, int, Any], bool]] | None = None,
        max_retries: int = 2,
    ) -> list[Any]:
        """
        Run a multi-step task with step-by-step verification.

        Args:
            task_steps: List of step dictionaries with 'goal' and 'task' keys
            verification_callbacks: Optional dict mapping step_idx to verification function
                                   Each callback receives (runtime, step_idx, snapshot) and returns bool
            max_retries: Maximum retries per step (default: 2)

        Returns:
            List of AgentActionResult objects for each step

        Example:
            >>> task_steps = [
            >>>     {"goal": "Search Google", "task": "Search for 'python'"},
            >>>     {"goal": "Click first result", "task": "Click the first search result"},
            >>> ]
            >>> results = await agent.run_multi_step(task_steps)
        """
        # Initialize verification if needed
        await self._initialize_verification()

        results = []
        verification_callbacks = verification_callbacks or {}

        for step_idx, step_info in enumerate(task_steps, start=1):
            goal = step_info.get("goal", f"Step {step_idx}")
            task = step_info.get("task", goal)
            
            # Record step start time
            step_start_time = time.time()
            step_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            logger.info(f"\n{'=' * 80}")
            logger.info(f"📋 Step {step_idx}: {goal}")
            logger.info(f"⏰ Started at: {step_start_timestamp}")
            logger.info(f"{'=' * 80}")

            # Begin verification step
            if self._runtime:
                self._runtime.begin_step(goal, step_index=step_idx - 1)
                logger.info(f"✅ Began verification step {step_idx}")

            # Determine snapshot limit (higher for last step to capture all posts)
            snapshot_limit = self.default_snapshot_limit
            if step_idx == len(task_steps):
                snapshot_limit = max(self.default_snapshot_limit, 100)  # Increase limit for last step
                logger.info(f"📊 Using increased snapshot limit ({snapshot_limit}) for final step")
            
            # Create SentienceAgentAsync for this step
            from sentience.agent import SentienceAgentAsync
            from sentience.agent_config import AgentConfig
            
            # Merge agent_config with agent_kwargs
            merged_config = self.agent_config
            if merged_config is None:
                merged_config = AgentConfig()
            
            # For last step, use higher snapshot limit in agent config
            if step_idx == len(task_steps):
                merged_config.snapshot_limit = snapshot_limit
            
            # Create agent
            agent = SentienceAgentAsync(
                browser=self.browser,
                llm=self.llm,
                default_snapshot_limit=snapshot_limit,
                verbose=self.verbose,
                tracer=self._tracer,
                config=merged_config,
                **self.agent_kwargs,
            )

            # Take snapshot and log compact prompt before running agent
            logger.info(f"📸 Taking snapshot for step {step_idx}...")
            from sentience.snapshot import snapshot_async
            from sentience.models import SnapshotOptions
            
            # Use the goal from step_info for SnapshotOptions (more descriptive than task)
            step_goal = step_info.get("goal", goal)
            snap_opts = SnapshotOptions(
                limit=snapshot_limit,
                goal=step_goal,  # Use the goal field from step_info
            )
            if self.agent_config:
                if self.agent_config.show_overlay:
                    snap_opts.show_overlay = True
            
            # Take snapshot with error handling for extension injection failures
            try:
                pre_agent_snapshot = await snapshot_async(self.browser, snap_opts)
            except Exception as snapshot_error:
                logger.warning(f"⚠️  Snapshot failed with exception: {snapshot_error}")
                logger.warning(f"   This may be due to extension injection timeout. Continuing without snapshot logging...")
                # Create a failed snapshot object to continue execution
                # Get current URL for the snapshot
                current_url = "unknown"
                try:
                    if self.browser.page:
                        current_url = self.browser.page.url
                except Exception:
                    pass
                
                from sentience.models import Snapshot
                pre_agent_snapshot = Snapshot(
                    status="error",
                    error=str(snapshot_error),
                    elements=[],
                    url=current_url,
                )
            
            if pre_agent_snapshot.status == "success":
                # Log snapshot statistics
                all_element_ids = [el.id for el in pre_agent_snapshot.elements]
                max_element_id = max(all_element_ids) if all_element_ids else 0
                min_element_id = min(all_element_ids) if all_element_ids else 0
                logger.info(f"📊 Snapshot stats: {len(pre_agent_snapshot.elements)} total elements, IDs range: {min_element_id}-{max_element_id}")
                
                # Format snapshot in compact format: ID|role|text|imp|is_primary|docYq|ord|DG|href
                # Use the same logic as SentienceContext._format_snapshot_for_llm
                import re
                
                # Filter to interactive elements only (same as SentienceContext)
                interactive_roles = {
                    "button", "link", "textbox", "searchbox", "combobox", "checkbox",
                    "radio", "slider", "tab", "menuitem", "option", "switch", "cell",
                    "a", "input", "select", "textarea",
                }
                
                interactive_elements = [
                    el for el in pre_agent_snapshot.elements
                    if (el.role or "").lower() in interactive_roles
                ]
                
                # Log interactive elements stats
                interactive_ids = [el.id for el in interactive_elements]
                if interactive_ids:
                    max_interactive_id = max(interactive_ids)
                    min_interactive_id = min(interactive_ids)
                    logger.info(f"📊 Interactive elements: {len(interactive_elements)} elements, IDs range: {min_interactive_id}-{max_interactive_id}")
                else:
                    logger.warning(f"⚠️  No interactive elements found in snapshot!")
                
                # Compute rank_in_group for dominant group elements
                rank_in_group_map: dict[int, int] = {}
                dg_elements_for_rank = [
                    el for el in interactive_elements
                    if el.in_dominant_group is True
                ]
                if not dg_elements_for_rank and pre_agent_snapshot.dominant_group_key:
                    dg_elements_for_rank = [
                        el for el in interactive_elements
                        if el.group_key == pre_agent_snapshot.dominant_group_key
                    ]
                
                # Sort by (doc_y, bbox.y, bbox.x, -importance) for rank
                def rank_sort_key(el):
                    doc_y = el.doc_y if el.doc_y is not None else float("inf")
                    bbox_y = el.bbox.y if el.bbox else float("inf")
                    bbox_x = el.bbox.x if el.bbox else float("inf")
                    neg_importance = -(el.importance or 0)
                    return (doc_y, bbox_y, bbox_x, neg_importance)
                
                dg_elements_for_rank.sort(key=rank_sort_key)
                for rank, el in enumerate(dg_elements_for_rank):
                    rank_in_group_map[el.id] = rank
                
                # Format elements
                compact_lines = []
                # Use the same limit as the snapshot (which may be higher for last step)
                for el in interactive_elements[:snapshot_limit]:
                    # Skip REMOVED elements
                    if hasattr(el, 'diff_status') and el.diff_status == "REMOVED":
                        continue
                    
                    # Get role (override to "link" if element has href)
                    role = el.role or ""
                    if el.href:
                        role = "link"
                    elif not role:
                        role = "element"
                    
                    # Get name/text (truncate aggressively, normalize whitespace)
                    name = el.text or ""
                    name = re.sub(r"\s+", " ", name.strip())
                    if len(name) > 30:
                        name = name[:27] + "..."
                    
                    # Extract fields
                    importance = el.importance or 0
                    doc_y = el.doc_y or 0
                    
                    # is_primary: from visual_cues.is_primary
                    is_primary = False
                    if el.visual_cues:
                        is_primary = el.visual_cues.is_primary or False
                    is_primary_flag = "1" if is_primary else "0"
                    
                    # docYq: bucketed doc_y (round to nearest 200)
                    doc_yq = int(round(doc_y / 200)) if doc_y else 0
                    
                    # Determine if in dominant group
                    in_dg = el.in_dominant_group
                    if in_dg is None and pre_agent_snapshot.dominant_group_key:
                        in_dg = el.group_key == pre_agent_snapshot.dominant_group_key
                    
                    # ord_val: rank_in_group if in dominant group
                    if in_dg and el.id in rank_in_group_map:
                        ord_val = rank_in_group_map[el.id]
                    else:
                        ord_val = "-"
                    
                    # DG: 1 if dominant group, else 0
                    dg_flag = "1" if in_dg else "0"
                    
                    # href: compress (use domain or last path segment)
                    href = el.href or ""
                    if href:
                        # Simple compression: use domain or last path segment
                        if "/" in href:
                            parts = href.split("/")
                            if len(parts) > 1:
                                href = parts[-1] or parts[-2] if len(parts) > 2 else ""
                        if len(href) > 30:
                            href = href[:27] + "..."
                    
                    # Format: ID|role|text|importance|is_primary|docYq|ord|DG|href
                    compact_lines.append(f"{el.id}|{role}|{name}|{importance}|{is_primary_flag}|{doc_yq}|{ord_val}|{dg_flag}|{href}")
                
                compact_prompt = "\n".join(compact_lines)
                
                # Log which element IDs are actually shown to LLM
                shown_ids = [el.id for el in interactive_elements[:self.default_snapshot_limit]]
                if shown_ids:
                    logger.info(f"📋 Showing {len(shown_ids)} elements to LLM, IDs: {min(shown_ids)}-{max(shown_ids)}")
                else:
                    logger.warning(f"⚠️  No elements shown to LLM!")
                
                logger.info(f"\n{'=' * 80}")
                logger.info(f"📋 Compact Snapshot Prompt for Step {step_idx}:")
                logger.info(f"{'=' * 80}")
                logger.info(compact_prompt)
                logger.info(f"{'=' * 80}\n")
            else:
                error_msg = pre_agent_snapshot.error or "Unknown error"
                logger.warning(f"⚠️  Snapshot failed: {error_msg}")
                logger.warning(f"   Continuing without snapshot logging - agent will still run")
                pre_agent_snapshot = None  # Set to None if snapshot failed
            
            # Run agent for this step
            logger.info(f"🤖 Running agent for step {step_idx}...")
            result = await agent.act(task, max_retries=max_retries)
            results.append(result)
            
            if result.success:
                logger.info(f"✅ Agent completed step {step_idx}: {result.action} on element {result.element_id}")
                
                # Special handling for last step: extract element text and validate
                if step_idx == len(task_steps) and result.element_id is not None:
                    # Check if element ID exists in snapshot
                    element_found = False
                    element_text = None
                    if pre_agent_snapshot and pre_agent_snapshot.status == "success":
                        all_ids = [el.id for el in pre_agent_snapshot.elements]
                        if result.element_id in all_ids:
                            element_found = True
                            for el in pre_agent_snapshot.elements:
                                if el.id == result.element_id:
                                    element_text = el.text or ""
                                    logger.info(f"📝 Found element {result.element_id}: role={el.role}, text={element_text[:100] if element_text else 'N/A'}...")
                                    break
                        else:
                            logger.warning(f"⚠️  Element ID {result.element_id} not found in snapshot!")
                            logger.warning(f"   Available element IDs range: {min(all_ids)}-{max(all_ids)}")
                            logger.warning(f"   Total elements in snapshot: {len(pre_agent_snapshot.elements)}")
                    
                    if element_text:
                        if "Show HN" in element_text:
                            logger.info(f"✅ Validation passed: Element text contains 'Show HN'")
                        else:
                            logger.warning(f"⚠️  Validation failed: Element text does not contain 'Show HN'")
                            logger.warning(f"   Element text: {element_text[:200]}")
                    elif not element_found:
                        logger.error(f"❌ Element {result.element_id} does not exist in snapshot - LLM selected invalid element ID!")
            else:
                logger.warning(f"⚠️  Agent step {step_idx} had issues: {result.error or 'Unknown error'}")

            # Take snapshot for verification
            if self._runtime:
                logger.info(f"📸 Taking snapshot for verification...")
                snapshot = None
                try:
                    snapshot = await self._runtime.snapshot()
                    logger.info(f"✅ Snapshot taken: {len(snapshot.elements)} elements found")
                except Exception as e:
                    # Extension might not be loaded or page might have changed
                    # Try to use AsyncSentienceBrowser snapshot as fallback
                    logger.warning(f"⚠️  AgentRuntime.snapshot() failed: {e}")
                    logger.info(f"   Attempting fallback snapshot via AsyncSentienceBrowser...")
                    try:
                        from sentience.snapshot import snapshot_async
                        from sentience.models import SnapshotOptions
                        fallback_snap_opts = SnapshotOptions(limit=50, goal="verification")
                        snapshot = await snapshot_async(self.browser, fallback_snap_opts)
                        if snapshot.status == "success":
                            logger.info(f"✅ Fallback snapshot taken: {len(snapshot.elements)} elements found")
                        else:
                            logger.warning(f"⚠️  Fallback snapshot failed: {snapshot.error}")
                            snapshot = None
                    except Exception as fallback_error:
                        logger.warning(f"⚠️  Fallback snapshot also failed: {fallback_error}")
                        snapshot = None

                # Run verification callback if provided
                if step_idx in verification_callbacks:
                    logger.info(f"🔍 Running custom verification for step {step_idx}...")
                    callback = verification_callbacks[step_idx]
                    if snapshot:
                        passed = callback(self._runtime, step_idx, snapshot)
                        logger.info(f"  {'✅' if passed else '❌'} Custom verification: {'PASSED' if passed else 'FAILED'}")
                    else:
                        logger.warning(f"⚠️  Skipping verification callback - no snapshot available")
                        # Still call callback but with None snapshot
                        try:
                            passed = callback(self._runtime, step_idx, None)
                            logger.info(f"  {'✅' if passed else '❌'} Custom verification: {'PASSED' if passed else 'FAILED'}")
                        except Exception as callback_error:
                            logger.warning(f"⚠️  Verification callback failed: {callback_error}")
            
            # Record step end time and calculate duration
            step_end_time = time.time()
            step_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            step_duration = step_end_time - step_start_time
            
            logger.info(f"{'=' * 80}")
            logger.info(f"⏰ Step {step_idx} completed at: {step_end_timestamp}")
            logger.info(f"⏱️  Step {step_idx} duration: {step_duration:.2f} seconds")
            logger.info(f"{'=' * 80}\n")

        return results

    async def assert_done(
        self,
        predicate: Any,
        label: str = "task_complete",
    ) -> bool:
        """
        Assert that the overall task is complete.

        Args:
            predicate: Predicate from sentience.asserts (e.g., expect(...).to_exist())
            label: Label for the assertion

        Returns:
            True if assertion passed, False otherwise

        Example:
            >>> from sentience.asserts import expect, E, in_dominant_list
            >>> 
            >>> task_complete = await agent.assert_done(
            >>>     expect(in_dominant_list().nth(0)).to_have_text_contains("Show HN"),
            >>>     label="top_post_found",
            >>> )
        """
        if not self._runtime:
            logger.warning("⚠️  AgentRuntime not initialized, cannot assert_done")
            return False

        logger.info("🔍 Verifying task completion...")
        result = self._runtime.assert_done(predicate, label=label)
        
        if result:
            logger.info("✅ Task completion verification passed")
        else:
            logger.info("❌ Task completion verification failed")
        
        return result

    async def get_verification_summary(self) -> dict[str, Any]:
        """
        Get verification summary.

        Returns:
            Dictionary with verification statistics
        """
        if not self._runtime:
            return {
                "runtime_available": False,
                "all_assertions_passed": None,
                "required_assertions_passed": None,
            }

        return {
            "runtime_available": True,
            "all_assertions_passed": self._runtime.all_assertions_passed(),
            "required_assertions_passed": self._runtime.required_assertions_passed(),
            "trace_file": str(self.trace_dir / f"{self._tracer.run_id}.jsonl") if self._tracer else None,
        }
