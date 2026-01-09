"""Inject Sentience semantic geometry into Agent context."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from browser_use.browser.session import BrowserSession
    from sentience.models import Snapshot

logger = logging.getLogger(__name__)


@dataclass
class SentienceState:
    """Sentience state with snapshot and formatted prompt block."""

    url: str
    snapshot: "Snapshot"
    prompt_block: str


def format_snapshot_for_llm(snapshot: "Snapshot", limit: int = 100) -> str:
    """
    Format Sentience snapshot for LLM consumption.

    Creates a compact inventory of interactive elements with IDs, roles, and names.
    Optimized for minimal token usage while maintaining usability.

    Args:
        snapshot: Sentience Snapshot object
        limit: Maximum number of elements to include (default: 100)

    Returns:
        Formatted string for LLM prompt
    """
    # Filter to interactive elements only (buttons, links, inputs, etc.)
    interactive_roles = {
        "button", "link", "textbox", "searchbox", "combobox", "checkbox",
        "radio", "slider", "tab", "menuitem", "option", "switch", "cell"
    }
    
    lines = []
    logger.info(f"received snapshot: {snapshot}")
    for el in snapshot.elements[:limit]:  # Check more, filter down
        # Get role (prefer role, fallback to tag)
        role = getattr(el, "role", None) or getattr(el, "tag", None) or ""
        
        # Skip non-interactive elements to reduce tokens
        if role.lower() not in interactive_roles and role.lower() not in {"a", "button", "input", "select", "textarea"}:
            continue
        
        # Get name/text (truncate aggressively)
        name = (getattr(el, "name", None) or getattr(el, "text", None) or "").strip()
        if len(name) > 40:  # More aggressive truncation
            name = name[:37] + "..."
        
        # Compact format: ID|role|name|importance (no coordinates to save tokens)
        # LLM can use the ID directly, coordinates aren't needed
        # Importance helps LLM prioritize which elements to interact with
        importance = getattr(el, "importance", 0)
        lines.append(f"{el.id}|{role}|{name}|{importance}")
        logger.info(f"Added element: {el.id}|{role}|{name}|{importance}")
        if len(lines) >= limit:
            break

    return "\n".join(lines)


async def build_sentience_state(
    browser_session: "BrowserSession",
) -> Optional[SentienceState]:
    """
    Build Sentience state from browser session.

    Takes a snapshot using the Sentience extension and formats it for LLM consumption.
    If snapshot fails (extension not loaded, timeout, etc.), returns None.

    Args:
        browser_session: Browser-use BrowserSession instance

    Returns:
        SentienceState with snapshot and formatted prompt, or None if snapshot failed
    """
    try:
        # Import here to avoid requiring sentience as a hard dependency
        from sentience.backends import BrowserUseAdapter
        from sentience.backends.snapshot import snapshot
        from sentience.models import SnapshotOptions

        # Create adapter and backend
        adapter = BrowserUseAdapter(browser_session)
        backend = await adapter.create_backend()

        # Give extension a moment to inject (especially after navigation)
        # The snapshot() call has its own timeout, but a small delay helps
        import asyncio
        await asyncio.sleep(0.5)

        # Limit to 50 interactive elements to minimize token usage
        # Only interactive elements are included in the formatted output
        # Note: sentience_api_key is not used here - we're using the backend snapshot
        # which always calls the local extension. API key is only for server-side API calls.
        options = SnapshotOptions(limit=100)  # Get more, filter to ~50 interactive

        # Take snapshot with retry logic (extension may need time to inject after navigation)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                snap = await snapshot(backend, options=options)
                break  # Success
            except Exception as e:
                if attempt < max_retries - 1:
                    # Wait a bit longer before retry
                    logger.debug("Sentience snapshot attempt %d failed, retrying...", attempt + 1)
                    await asyncio.sleep(1.0)
                else:
                    raise  # Re-raise on final attempt

        # Get URL from snapshot or browser state
        url = getattr(snap, "url", "") or ""

        # Format for LLM (limit to 50 interactive elements to minimize tokens)
        formatted = format_snapshot_for_llm(snap, limit=50)

        # Ultra-compact format to minimize tokens
        prompt = (
            "## Elements (ID|role|text|importance)\n"
            "Use click(index=ID) or input_text(index=ID,...). Format: ID|role|text|importance\n"
            "Higher importance = more likely to be relevant for the task.\n"
            f"{formatted}"
        )

        logger.info(f"✅ Sentience snapshot: {len(snap.elements)} elements, URL: {url}")
        return SentienceState(url=url, snapshot=snap, prompt_block=prompt)

    except ImportError:
        logger.warning("⚠️  Sentience SDK not available, skipping snapshot")
        return None
    except Exception as e:
        # Log warning if extension not loaded or snapshot fails
        logger.warning(f"⚠️  Sentience snapshot skipped: {e}")
        return None
