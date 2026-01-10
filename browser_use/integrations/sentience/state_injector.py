"""Inject Sentience semantic geometry into Agent context."""

from __future__ import annotations

import logging
import os
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


def format_snapshot_for_llm(snapshot: "Snapshot", top_by_importance: int = 40, top_from_dominant_group: int = 15, top_by_position: int = 10) -> str:
    """
    Format Sentience snapshot for LLM consumption.

    Creates an ultra-compact inventory of interactive elements optimized for minimal token usage.
    Selects top elements by importance + top elements from dominant group for ordinal tasks.

    Args:
        snapshot: Sentience Snapshot object
        top_by_importance: Number of top elements by importance to include (default: 20)
        top_from_dominant_group: Number of top elements from dominant group to include (default: 15)
        top_by_position: Number of top elements by position (lowest doc_y) to include (default: 10)

    Returns:
        Formatted string for LLM prompt with format: ID|role|text|imp|docYq|ord|DG|href
    """
    # Filter to interactive elements only (buttons, links, inputs, etc.)
    interactive_roles = {
        "button", "link", "textbox", "searchbox", "combobox", "checkbox",
        "radio", "slider", "tab", "menuitem", "option", "switch", "cell"
    }
    
    dominant_group_key = snapshot.dominant_group_key or ""
    
    # Extract and filter interactive elements
    interactive_elements = []
    for el in snapshot.elements:
        # Get role (prefer role, fallback to tag)
        role = getattr(el, "role", None) or getattr(el, "tag", None) or ""
        
        # Skip non-interactive elements
        if role.lower() not in interactive_roles and role.lower() not in {"a", "button", "input", "select", "textarea"}:
            continue
        
        interactive_elements.append(el)
    
    # Sort by importance (descending) for importance-based selection
    interactive_elements.sort(key=lambda el: getattr(el, "importance", 0), reverse=True)
    
    # Get top N by importance (track by ID for deduplication)
    top_by_imp_ids = set()
    top_by_imp = []
    for el in interactive_elements[:top_by_importance]:
        el_id = getattr(el, "id", None)
        if el_id and el_id not in top_by_imp_ids:
            top_by_imp_ids.add(el_id)
            top_by_imp.append(el)
    
    # Get top elements from dominant group (sorted by group_index for ordinal tasks)
    dominant_group_elements = [
        el for el in interactive_elements
        if getattr(el, "group_key", "") == dominant_group_key
    ]
    dominant_group_elements.sort(key=lambda el: getattr(el, "group_index", 999))
    
    # Get top N by position (lowest doc_y = top of page) - critical for ordinal tasks
    # Sort by doc_y ascending (smaller = higher on page)
    elements_by_position = sorted(
        interactive_elements,
        key=lambda el: (getattr(el, "doc_y", 0) or 0, getattr(el, "importance", 0))
    )
    
    # Combine all selections (deduplicate by element ID)
    selected_elements = top_by_imp.copy()
    selected_ids = top_by_imp_ids.copy()
    
    # Add dominant group elements
    for el in dominant_group_elements[:top_from_dominant_group]:
        el_id = getattr(el, "id", None)
        if el_id and el_id not in selected_ids:
            selected_ids.add(el_id)
            selected_elements.append(el)
    
    # Add top elements by position (ensures we capture items at top of page)
    for el in elements_by_position[:top_by_position]:
        el_id = getattr(el, "id", None)
        if el_id and el_id not in selected_ids:
            selected_ids.add(el_id)
            selected_elements.append(el)
    
    # Format lines with pre-encoded compact fields
    lines = []
    for el in selected_elements:
        # Get role (prefer role, fallback to tag)
        role = getattr(el, "role", None) or getattr(el, "tag", None) or ""
        
        # Get name/text (truncate aggressively)
        name = (getattr(el, "name", None) or getattr(el, "text", None) or "").strip()
        if len(name) > 30:  # Aggressive truncation
            name = name[:27] + "..."
        
        # Extract fields
        importance = int(getattr(el, "importance", 0))
        doc_y = getattr(el, "doc_y", 0) or 0
        group_key = getattr(el, "group_key", "") or ""
        group_index = getattr(el, "group_index", 0) or 0
        
        # Pre-encode fields for compactness
        # docYq: bucketed doc_y (round to nearest 200 for smaller numbers)
        doc_yq = int(round(doc_y / 200)) if doc_y else 0

        # Phase 3.2: Use pre-computed in_dominant_group field (uses fuzzy matching)
        # This is computed by the gateway so we don't need to implement fuzzy matching here
        in_dg = getattr(el, "in_dominant_group", None)
        if in_dg is None:
            # Fallback for older gateway versions: use exact string match
            in_dg = group_key == dominant_group_key if dominant_group_key else False

        # ord: group_index if in dominant group, else "-"
        ord_val = group_index if in_dg else "-"

        # DG: 1 if dominant group, else 0
        dg_flag = "1" if in_dg else "0"
        
        # href: short token (domain or last path segment, or blank)
        href = ""
        el_href = getattr(el, "href", None)
        if el_href:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(el_href)
                if parsed.netloc:
                    href = parsed.netloc.split(".")[-2] if "." in parsed.netloc else parsed.netloc[:10]
                elif parsed.path:
                    href = parsed.path.split("/")[-1][:10] or "item"
            except Exception:
                href = "item"
        
        # Ultra-compact format: ID|role|text|imp|docYq|ord|DG|href
        cur_line = f"{el.id}|{role}|{name}|{importance}|{doc_yq}|{ord_val}|{dg_flag}|{href}"
        lines.append(cur_line)
    
    logger.debug(
        "Formatted %d elements (top %d by importance + top %d from dominant group + top %d by position)",
        len(lines),
        top_by_importance,
        top_from_dominant_group,
        top_by_position,
    )
    
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

        # Get API key from environment if available
        api_key = os.getenv("SENTIENCE_API_KEY")
        # Limit to 50 interactive elements to minimize token usage
        # Only interactive elements are included in the formatted output
        if api_key:
            options = SnapshotOptions(use_api=True, sentience_api_key=api_key, limit=50, show_overlay=True, goal="Click the first ShowHN link")  # Get more, filter to ~50 interactive
        else:
            options = SnapshotOptions(limit=50, show_overlay=True, goal="Click the first ShowHN link")  # Get more, filter to ~50 interactive

        # Take snapshot with retry logic (extension may need time to inject after navigation)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                snap = await snapshot(backend, options=options)
                break  # Success
            except Exception:
                if attempt < max_retries - 1:
                    # Wait a bit longer before retry
                    logger.debug("Sentience snapshot attempt %d failed, retrying...", attempt + 1)
                    await asyncio.sleep(1.0)
                else:
                    raise  # Re-raise on final attempt

        # Get URL from snapshot or browser state
        url = getattr(snap, "url", "") or ""

        # Format for LLM (top 20 by importance + top 15 from dominant group + top 10 by position)
        formatted = format_snapshot_for_llm(snap, top_by_importance=40, top_from_dominant_group=15, top_by_position=10)
        print(f"formatted: {formatted}")

        # Ultra-compact per-step prompt (minimal token usage)
        # Format: ID|role|text|imp|docYq|ord|DG|href
        # Rules: ordinal→DG=1 then ord asc; otherwise imp desc. Use click(ID)/input_text(ID,...).
        prompt = (
            "Elements: ID|role|text|imp|docYq|ord|DG|href\n"
            "Rules: ordinal→DG=1 then ord asc; otherwise imp desc. Use click(ID)/input_text(ID,...).\n"
            f"{formatted}"
        )


        logger.info("✅ Sentience snapshot: %d elements, URL: %s", len(snap.elements), url)
        return SentienceState(url=url, snapshot=snap, prompt_block=prompt)

    except ImportError:
        logger.warning("⚠️  Sentience SDK not available, skipping snapshot")
        return None
    except Exception as e:
        # Log warning if extension not loaded or snapshot fails
        logger.warning("⚠️  Sentience snapshot skipped: %s", e)
        return None
