"""Inject Sentience semantic geometry into Agent context."""

from __future__ import annotations

import logging
import os
import re
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
        Formatted string for LLM prompt with format: ID|role|text|imp|is_primary|docYq|ord|DG|href
    """
    # Filter to interactive elements only (buttons, links, inputs, etc.)
    # NOTE: Be more permissive - if element has an ID and importance > 0, it's likely interactive
    interactive_roles = {
        "button", "link", "textbox", "searchbox", "combobox", "checkbox",
        "radio", "slider", "tab", "menuitem", "option", "switch", "cell"
    }
    
    dominant_group_key = snapshot.dominant_group_key or ""
    
    # Extract and filter interactive elements
    interactive_elements = []
    role_counts = {}  # Track what roles we're seeing
    logger.info("Total elements in snapshot: %d", len(snapshot.elements))
    for el in snapshot.elements:
        # Get role (prefer role, fallback to tag)
        role = getattr(el, "role", None) or getattr(el, "tag", None) or ""
        role_lower = role.lower() if role else ""
        
        # Track role distribution
        role_counts[role_lower] = role_counts.get(role_lower, 0) + 1
        
        # More permissive filter: include if:
        # 1. Has known interactive role
        # 2. Has an ID (elements with IDs are usually interactive)
        # 3. Has importance > 0 (interactive elements usually have importance)
        # 4. Has href (links are interactive)
        el_id = getattr(el, "id", None)
        importance = getattr(el, "importance", 0) or 0
        has_href = bool(getattr(el, "href", None))
        
        # is_interactive = (
        #     role_lower in interactive_roles or
        #     role_lower in {"a", "button", "input", "select", "textarea"} or
        #     (el_id and importance > 0) or  # Has ID and importance = likely interactive
        #     has_href  # Has href = it's a link
        # )
        
        # if not is_interactive:
        #     continue
        
        interactive_elements.append(el)
    
    # Log role distribution for debugging
    logger.info("Role distribution (top 10): %s", dict(sorted(role_counts.items(), key=lambda x: x[1], reverse=True)[:10]))
    logger.info("Interactive elements found: %d", len(interactive_elements))
    
    # If we found very few interactive elements, log sample of filtered-out roles
    if len(interactive_elements) < 10:
        filtered_roles = {r: c for r, c in role_counts.items() 
                         if r not in interactive_roles and r not in {"a", "button", "input", "select", "textarea"}}
        logger.info("Filtered out roles (top 10): %s", dict(sorted(filtered_roles.items(), key=lambda x: x[1], reverse=True)[:10]))
        # Also log sample of elements that were filtered out
        sample_filtered = []
        for el in snapshot.elements[:10]:
            role = getattr(el, "role", None) or getattr(el, "tag", None) or ""
            el_id = getattr(el, "id", None)
            importance = getattr(el, "importance", 0) or 0
            sample_filtered.append(f"role={role}, id={el_id}, importance={importance}")
        logger.info("Sample filtered elements: %s", sample_filtered)
    
    # Sort by importance (descending) for importance-based selection
    interactive_elements.sort(key=lambda el: getattr(el, "importance", 0), reverse=True)
    
    # Log top elements by importance for debugging
    logger.info("Top 10 elements by importance:")
    for i, el in enumerate(interactive_elements[:10]):
        el_id = getattr(el, "id", "no-id")
        el_role = getattr(el, "role", None) or getattr(el, "tag", None) or ""
        el_name = (getattr(el, "name", None) or getattr(el, "text", None) or "")[:50]
        el_importance = getattr(el, "importance", 0)
        el_href = getattr(el, "href", None)
        logger.info("  [%d] ID=%s, role=%s, imp=%s, href=%s, name='%s'", 
                    i+1, el_id, el_role, el_importance, bool(el_href), el_name)
    
    # Get top N by importance (track by ID for deduplication)
    # Note: If elements don't have IDs, we'll include them all (no deduplication possible)
    top_by_imp_ids = set()
    top_by_imp = []
    elements_without_id = 0
    for el in interactive_elements[:top_by_importance]:
        el_id = getattr(el, "id", None)
        if el_id:
            if el_id not in top_by_imp_ids:
                top_by_imp_ids.add(el_id)
                top_by_imp.append(el)
        else:
            # Element without ID - include it (can't deduplicate)
            top_by_imp.append(el)
            elements_without_id += 1
    
    logger.info("Top by importance: %d elements (%d without IDs)", len(top_by_imp), elements_without_id)
    
    # Get top elements from dominant group (sorted by group_index for ordinal tasks)
    # Use in_dominant_group flag (computed by gateway with fuzzy matching) instead of exact group_key match
    # Gateway uses fuzzy matching (x-bucket ±1) to compute in_dominant_group, so we must use that flag
    dominant_group_elements = []
    for el in interactive_elements:
        # Phase 3.2: Use pre-computed in_dominant_group field (uses fuzzy matching from gateway)
        in_dg = getattr(el, "in_dominant_group", None)
        if in_dg is None:
            # Fallback for older gateway versions: use exact string match
            in_dg = getattr(el, "group_key", "") == dominant_group_key if dominant_group_key else False
        if in_dg:
            dominant_group_elements.append(el)
    
    dominant_group_elements.sort(key=lambda el: getattr(el, "group_index", 999))
    logger.info("Dominant group elements: %d (using in_dominant_group flag, dominant_group_key: %s)", 
                 len(dominant_group_elements), dominant_group_key)
    
    # Log top dominant group elements for debugging
    logger.info("Top 10 dominant group elements (by group_index):")
    for i, el in enumerate(dominant_group_elements[:10]):
        el_id = getattr(el, "id", "no-id")
        el_role = getattr(el, "role", None) or getattr(el, "tag", None) or ""
        el_name = (getattr(el, "name", None) or getattr(el, "text", None) or "")[:50]
        el_importance = getattr(el, "importance", 0)
        el_group_index = getattr(el, "group_index", 0)
        el_href = getattr(el, "href", None)
        logger.info("  [%d] ID=%s, role=%s, imp=%s, ord=%s, href=%s, name='%s'", 
                    i+1, el_id, el_role, el_importance, el_group_index, bool(el_href), el_name)
    
    # Get top N by position (lowest doc_y = top of page) - critical for ordinal tasks
    # Sort by doc_y ascending (smaller = higher on page)
    elements_by_position = sorted(
        interactive_elements,
        key=lambda el: (getattr(el, "doc_y", 0) or 0, getattr(el, "importance", 0))
    )
    
    # Log top elements by position for debugging
    logger.info("Top 10 elements by position (lowest doc_y):")
    for i, el in enumerate(elements_by_position[:10]):
        el_id = getattr(el, "id", "no-id")
        el_role = getattr(el, "role", None) or getattr(el, "tag", None) or ""
        el_name = (getattr(el, "name", None) or getattr(el, "text", None) or "")[:50]
        el_doc_y = getattr(el, "doc_y", 0)
        el_importance = getattr(el, "importance", 0)
        el_href = getattr(el, "href", None)
        logger.info("  [%d] ID=%s, role=%s, doc_y=%s, imp=%s, href=%s, name='%s'", 
                    i+1, el_id, el_role, el_doc_y, el_importance, bool(el_href), el_name)
    
    # Combine all selections (deduplicate by element ID)
    selected_elements = top_by_imp.copy()
    selected_ids = top_by_imp_ids.copy()
    
    # Add dominant group elements
    added_from_dg = 0
    for el in dominant_group_elements[:top_from_dominant_group]:
        el_id = getattr(el, "id", None)
        if el_id:
            if el_id not in selected_ids:
                selected_ids.add(el_id)
                selected_elements.append(el)
                added_from_dg += 1
        else:
            # Element without ID - include it if we haven't seen it before (by object reference)
            # This is a fallback for elements without IDs
            if el not in selected_elements:
                selected_elements.append(el)
                added_from_dg += 1
    
    # Add top elements by position (ensures we capture items at top of page)
    added_from_position = 0
    for el in elements_by_position[:top_by_position]:
        el_id = getattr(el, "id", None)
        if el_id:
            if el_id not in selected_ids:
                selected_ids.add(el_id)
                selected_elements.append(el)
                added_from_position += 1
        else:
            # Element without ID - include it if we haven't seen it before (by object reference)
            if el not in selected_elements:
                selected_elements.append(el)
                added_from_position += 1
    
    logger.info("Selected elements: %d total (%d from importance, +%d from dominant group, +%d from position)", 
                 len(selected_elements), len(top_by_imp), added_from_dg, added_from_position)
    
    # Check if specific element (ID 49) is selected
    element_49 = next((el for el in selected_elements if getattr(el, "id", None) == 49), None)
    if element_49:
        el_in_dg = getattr(element_49, "in_dominant_group", None)
        el_group_index = getattr(element_49, "group_index", 0)
        el_importance = getattr(element_49, "importance", 0)
        el_doc_y = getattr(element_49, "doc_y", 0)
        logger.info("✅ Element ID 49 (Librario) is selected: imp=%s, in_dg=%s, ord=%s, doc_y=%s", 
                   el_importance, el_in_dg, el_group_index, el_doc_y)
    else:
        logger.warning("❌ Element ID 49 (Librario) is NOT in selected_elements!")
    
    # Sort selected elements: prioritize dominant group elements (by ord), then by importance
    if dominant_group_key:
        def sort_key(el):
            in_dg = getattr(el, "in_dominant_group", None)
            if in_dg is None:
                in_dg = getattr(el, "group_key", "") == dominant_group_key if dominant_group_key else False
            if in_dg:
                # Dominant group elements: sort by group_index (ord) first, then importance
                return (0, getattr(el, "group_index", 999), -getattr(el, "importance", 0))
            else:
                # Non-dominant group: sort by importance (descending)
                return (1, -getattr(el, "importance", 0))
        selected_elements.sort(key=sort_key)
        logger.info("Sorted selected elements: dominant group first (by ord), then by importance")
    
    # Log breakdown of selected elements by type
    selected_with_href = sum(1 for el in selected_elements if getattr(el, "href", None))
    selected_in_dg = sum(1 for el in selected_elements if getattr(el, "in_dominant_group", None))
    selected_buttons = sum(1 for el in selected_elements if (getattr(el, "role", None) or getattr(el, "tag", None) or "").lower() == "button")
    selected_links = sum(1 for el in selected_elements if (getattr(el, "role", None) or getattr(el, "tag", None) or "").lower() in ("link", "a"))
    selected_spans = sum(1 for el in selected_elements if (getattr(el, "role", None) or getattr(el, "tag", None) or "").lower() == "span")
    logger.info("Selected breakdown: %d with href, %d in dominant group, %d buttons, %d links, %d spans", 
                 selected_with_href, selected_in_dg, selected_buttons, selected_links, selected_spans)
    
    # Log first 20 selected elements to see what's being prioritized
    logger.info("First 20 selected elements (after sorting):")
    for i, el in enumerate(selected_elements[:20]):
        el_id = getattr(el, "id", "no-id")
        el_role = getattr(el, "role", None) or getattr(el, "tag", None) or ""
        el_name = (getattr(el, "name", None) or getattr(el, "text", None) or "")[:40]
        el_importance = getattr(el, "importance", 0)
        el_href = getattr(el, "href", None)
        el_in_dg = getattr(el, "in_dominant_group", None)
        el_group_index = getattr(el, "group_index", 0) if el_in_dg else None
        logger.info("  [%d] ID=%s, role=%s, imp=%s, href=%s, in_dg=%s, ord=%s, name='%s'", 
                    i+1, el_id, el_role, el_importance, bool(el_href), el_in_dg, el_group_index, el_name)
    
    # Format lines with pre-encoded compact fields
    lines = []
    for el in selected_elements:
        # Get href directly from element
        href_value = getattr(el, "href", None)
        
        # Log element details for first 10 elements to debug missing post titles
        if len(lines) < 10:
            el_id = getattr(el, "id", "no-id")
            el_role = getattr(el, "role", None)
            el_tag = getattr(el, "tag", None)
            el_name = getattr(el, "name", None) or getattr(el, "text", None) or ""
            el_importance = getattr(el, "importance", 0)
            el_in_dg = getattr(el, "in_dominant_group", None)
            logger.info("Element %s: role=%s, tag=%s, href=%s, importance=%s, in_dg=%s, name='%s'", 
                       el_id, el_role, el_tag, href_value, el_importance, el_in_dg, el_name[:50])
        
        # Get role (prefer role, fallback to tag)
        # Since Sentience extension now filters out structural tags, we can trust the elements it sends
        role = getattr(el, "role", None) or getattr(el, "tag", None) or ""
        
        # Override: if element has href, it's always a link
        if href_value:
            role = "link"
        elif not role:
            # If still no role and element is in our selection, it's likely interactive but missing role/tag
            # Use a generic fallback for elements that passed our interactive filter
            role = "element"  # Generic fallback for interactive elements without explicit role/tag
        
        # Get name/text (truncate aggressively, remove newlines and normalize whitespace)
        name = (getattr(el, "name", None) or getattr(el, "text", None) or "")
        # Remove newlines and normalize whitespace (replace all whitespace with single space)
        name = re.sub(r'\s+', ' ', name.strip())
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
        
        # is_primary: from visual_cues.is_primary (boolean)
        visual_cues = getattr(el, "visual_cues", None)
        logger.info("Visual cues: %s", visual_cues)
        is_primary = False
        if visual_cues:
            is_primary = getattr(visual_cues, "is_primary", False)
            logger.info("Is primary: %s", is_primary)
        is_primary_flag = "1" if is_primary else "0"
        
        # href: short token (domain or last path segment, or blank)
        href = ""
        if href_value:
            try:
                from urllib.parse import urlparse
                href_str = str(href_value).strip()
                parsed = urlparse(href_str)
                if parsed.netloc:
                    # Extract domain (second-to-last part of domain, e.g., "ycombinator" from "news.ycombinator.com")
                    parts = parsed.netloc.split(".")
                    if len(parts) >= 2:
                        href = parts[-2][:10]  # e.g., "ycombinator" from "news.ycombinator.com"
                    else:
                        href = parsed.netloc[:10]
                elif parsed.path:
                    # Relative URL - use last path segment
                    path_parts = [p for p in parsed.path.split("/") if p]
                    if path_parts:
                        href = path_parts[-1][:10] or "item"
                    else:
                        href = "item"
                else:
                    # Just a fragment or query - use "item" as fallback
                    href = "item"
                
                # Log parsed href for first few elements
                if len(lines) < 5:
                    logger.info("Element %s: parsed href='%s' -> token='%s'", getattr(el, "id", "no-id"), href_str, href)
            except Exception as e:
                logger.warning("Failed to parse href '%s' for element %s: %s", href_value, getattr(el, "id", "no-id"), e)
                href = "item"
        
        # Ultra-compact format: ID|role|text|imp|is_primary|docYq|ord|DG|href
        cur_line = f"{el.id}|{role}|{name}|{importance}|{is_primary_flag}|{doc_yq}|{ord_val}|{dg_flag}|{href}"
        lines.append(cur_line)
    
    logger.info(
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

        # Check current page URL first - skip extension check on about:blank or non-HTTP pages
        import asyncio
        try:
            current_url = await browser_session.get_current_page_url()
            logger.debug("Current page URL: %s", current_url)
            
            # Skip networkidle wait and extension check for non-HTTP pages
            is_meaningful_page = current_url.lower().split(':', 1)[0] in ('http', 'https')
            if not is_meaningful_page:
                logger.debug("Skipping Sentience snapshot - page is not HTTP/HTTPS (url: %s)", current_url)
                return None
        except Exception as e:
            logger.debug("Could not get current page URL: %s, proceeding anyway...", e)
            current_url = None
            is_meaningful_page = True  # Assume meaningful page if we can't check

        # Wait for page to be fully loaded (networkidle) before checking extension
        # Similar to sdk-python: page.goto() + wait_for_load_state("networkidle")
        if is_meaningful_page:
            logger.info("Waiting for page to reach networkidle state...")
            networkidle_timeout = 30.0  # 30 seconds timeout (same as sdk-python)
            poll_interval = 0.1  # Check every 100ms
            
            # Get current page target_id and CDP session
            target_id = browser_session.agent_focus_target_id
            if not target_id:
                # Fallback: get first page target
                pages = await browser_session._cdp_get_all_pages(include_http=True, include_pages=True)
                if pages:
                    target_id = pages[0]['targetId']
            
            if target_id:
                try:
                    cdp_session = await browser_session.get_or_create_cdp_session(target_id, focus=False)
                    if hasattr(cdp_session, '_lifecycle_events'):
                        start_time = asyncio.get_event_loop().time()
                        networkidle_reached = False
                        
                        while (asyncio.get_event_loop().time() - start_time) < networkidle_timeout:
                            # Check for networkIdle event in stored lifecycle events
                            for event_data in list(cdp_session._lifecycle_events):
                                event_name = event_data.get('name')
                                if event_name == 'networkIdle':
                                    duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                                    logger.info("✅ Page reached networkidle state (%.0fms)", duration_ms)
                                    networkidle_reached = True
                                    break
                            
                            if networkidle_reached:
                                break
                            
                            await asyncio.sleep(poll_interval)
                        
                        if not networkidle_reached:
                            logger.warning("⚠️  Page did not reach networkidle state within %.0fs, proceeding anyway...", networkidle_timeout)
                    else:
                        logger.debug("Lifecycle events not available, skipping networkidle wait")
                except Exception as e:
                    logger.debug("Error waiting for networkidle: %s, proceeding anyway...", e)
            else:
                logger.debug("No target_id available, skipping networkidle wait")

            # Give extension a moment to inject after networkidle (similar to sdk-python's wait_for_load_state)
            # Some extensions need a small delay after page load to inject content scripts
            await asyncio.sleep(0.5)
            logger.debug("Waited 500ms after networkidle for extension injection")

        # Check if extension is loaded before attempting snapshot
        max_wait_time = 10.0  # Maximum time to wait for extension (seconds)
        check_interval = 0.2  # Check every 200ms
        extension_loaded = False
        
        logger.info("Checking if Sentience extension is loaded...")
        for check_num in range(int(max_wait_time / check_interval)):
            try:
                # Check if extension is injected by evaluating JavaScript
                # Also check for extension runtime to diagnose if extension is loaded but not injecting
                diag = await backend.eval("""
                    (() => {
                        const hasSentience = typeof window.sentience !== 'undefined';
                        const hasSnapshot = hasSentience && typeof window.sentience.snapshot === 'function';
                        const extId = document.documentElement.dataset.sentienceExtensionId || null;
                        
                        // Check if chrome.runtime is available (indicates extension context)
                        let chromeRuntimeAvailable = false;
                        try {
                            chromeRuntimeAvailable = typeof chrome !== 'undefined' && typeof chrome.runtime !== 'undefined';
                        } catch (e) {}
                        
                        return {
                            window_sentience: hasSentience,
                            window_sentience_snapshot: hasSnapshot,
                            extension_id_attr: extId,
                            ready_state: document.readyState,
                            chrome_runtime_available: chromeRuntimeAvailable,
                            url: window.location.href
                        };
                    })()
                """)
                
                current_diag_url = diag.get("url", "")
                # Skip extension check if we're on about:blank or non-HTTP pages
                if current_diag_url and current_diag_url.lower().split(':', 1)[0] not in ('http', 'https'):
                    logger.debug("Page is on %s, skipping extension check", current_diag_url)
                    return None
                
                if diag.get("window_sentience") and diag.get("window_sentience_snapshot"):
                    extension_loaded = True
                    logger.info("✅ Sentience extension is loaded (ready_state: %s)", diag.get("ready_state"))
                    break
                
                # Log diagnostic info every 2 seconds (every 10th check)
                if check_num % 10 == 0 and check_num > 0:
                    logger.info("Extension check #%d: window.sentience=%s, chrome.runtime=%s, ready_state=%s, url=%s", 
                               check_num, diag.get("window_sentience"), diag.get("chrome_runtime_available"),
                               diag.get("ready_state"), diag.get("url")[:50])
            except Exception as e:
                logger.debug("Error checking extension status: %s", e)
            
            await asyncio.sleep(check_interval)
        
        if not extension_loaded:
            logger.warning("⚠️  Sentience extension not detected after %.1fs, attempting snapshot anyway...", max_wait_time)

        # Get API key from environment if available
        api_key = os.getenv("SENTIENCE_API_KEY")
        # Limit to 50 interactive elements to minimize token usage
        # Only interactive elements are included in the formatted output
        if api_key:
            options = SnapshotOptions(use_api=True, sentience_api_key=api_key, limit=50, show_overlay=True, goal="Click the first ShowHN link")  # Get more, filter to ~50 interactive
        else:
            options = SnapshotOptions(limit=50, show_overlay=True, goal="Click the first ShowHN link")  # Get more, filter to ~50 interactive

        # Take snapshot with retry logic (extension may need time to inject after navigation)
        max_retries = 3  # Increased retries
        for attempt in range(max_retries):
            try:
                snap = await snapshot(backend, options=options)
                if attempt > 0:
                    logger.info("✅ Sentience snapshot succeeded on attempt %d", attempt + 1)
                break  # Success
            except Exception as e:
                if attempt < max_retries - 1:
                    # Wait progressively longer: 1s, 2s, 3s
                    wait_time = (attempt + 1) * 1.0
                    logger.info("Sentience snapshot attempt %d failed: %s, retrying in %.1fs...", 
                               attempt + 1, str(e)[:100], wait_time)
                    await asyncio.sleep(wait_time)
                else:
                    raise  # Re-raise on final attempt

        # Get URL from snapshot or browser state
        url = getattr(snap, "url", "") or ""

        # Format for LLM (top 20 by importance + top 15 from dominant group + top 10 by position)
        formatted = format_snapshot_for_llm(snap, top_by_importance=60, top_from_dominant_group=15, top_by_position=10)
        logger.info("Formatted output: %d lines, %d chars", len(formatted.split('\n')), len(formatted))
        logger.info("Formatted content:\n%s", formatted)

        # Ultra-compact per-step prompt (minimal token usage)
        # Format: ID|role|text|imp|is_primary|docYq|ord|DG|href
        # Rules: ordinal→DG=1 then ord asc; otherwise imp desc. Use click(ID)/input_text(ID,...).
        prompt = (
            "Elements: ID|role|text|imp|is_primary|docYq|ord|DG|href\n"
            "Rules: ordinal→DG=1 then ord asc; otherwise imp desc. Use click(ID)/input_text(ID,...).\n"
            f"{formatted}"
        )


        logger.info("✅ Sentience snapshot: %d elements, URL: %s", len(snap.elements), url)
        return SentienceState(url=url, snapshot=snap, prompt_block=prompt)

    except ImportError:
        logger.info("⚠️  Sentience SDK not available, skipping snapshot")
        return None
    except Exception as e:
        # Log warning if extension not loaded or snapshot fails
        logger.info("⚠️  Sentience snapshot skipped: %s", e)
        return None
