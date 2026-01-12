"""
Example usage of SentienceAgent with Verification.

This example demonstrates how to use SentienceAgent with:
- Sentience snapshot as primary prompt (compact, token-efficient)
- Vision fallback when snapshot fails
- Token usage tracking
- **NEW: Machine-verifiable assertions via Sentience SDK AgentRuntime**
- **NEW: Declarative task completion verification**

The verification feature showcases the full power of the Sentience SDK:
- Per-step assertions (url_contains, exists, not_exists, etc.)
- Predicate combinators (all_of, any_of)
- Machine-verifiable task completion (assert_done)
- Trace output for observability (Studio timeline)
"""

import asyncio
import os
from pathlib import Path

import glob
from dotenv import load_dotenv

from browser_use import BrowserProfile, BrowserSession, ChatBrowserUse
from browser_use.integrations.sentience import SentienceAgent, SentienceAgentConfig
from sentience import get_extension_dir

# Import Sentience SDK verification helpers
from sentience.verification import (
    url_contains,
    exists,
    not_exists,
    all_of,
    any_of,
)
# Import the assertion DSL for expressive queries
from sentience.asserts import E, expect, in_dominant_list

# Note: This example requires:
# 1. Sentience SDK installed: pip install sentienceapi
# 2. Sentience extension loaded in browser
# 3. Optional: SENTIENCE_API_KEY in .env for gateway mode

load_dotenv()


def log(msg: str) -> None:
    """Print with flush for immediate output."""
    print(msg, flush=True)


async def main():
    """Example: Use SentienceAgent to find the top Show HN post."""
    try:
        # Get path to Sentience extension
        extension_path = get_extension_dir()
        log(f"Loading Sentience extension from: {extension_path}")

        # Verify extension exists
        if not os.path.exists(extension_path):
            raise FileNotFoundError(f"Sentience extension not found at: {extension_path}")
        if not os.path.exists(os.path.join(extension_path, "manifest.json")):
            raise FileNotFoundError(
                f"Sentience extension manifest not found at: {extension_path}/manifest.json"
            )
        log(f"✅ Sentience extension verified at: {extension_path}")

        # Find browser executable (optional - browser-use will find one if not specified)
        # This example looks for Playwright-installed browsers (Chromium-based, work with CDP)
        playwright_cache = Path.home() / "Library/Caches/ms-playwright"
        browser_patterns = [
            playwright_cache
            / "chromium-*/chrome-mac*/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
            playwright_cache / "chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium",
        ]

        browser_executable = None
        for pattern in browser_patterns:
            matches = glob.glob(str(pattern))
            if matches:
                matches.sort()
                browser_executable = matches[-1]  # Use latest version
                if Path(browser_executable).exists():
                    log(f"✅ Found browser: {browser_executable}")
                    break

        if not browser_executable:
            log("⚠️  Browser not found, browser-use will try to install it")

        # Get default extension paths and combine with Sentience extension
        # Chrome only uses the LAST --load-extension arg, so we must combine all extensions
        log("Collecting all extension paths...")
        extension_paths = [extension_path]

        # Create a temporary profile to ensure default extensions are downloaded
        # This ensures extensions exist before we try to load them
        temp_profile = BrowserProfile(enable_default_extensions=True)
        default_extensions = temp_profile._ensure_default_extensions_downloaded()

        if default_extensions:
            extension_paths.extend(default_extensions)
            log(f"  ✅ Found {len(default_extensions)} default extensions")
        else:
            log("  ⚠️  No default extensions found (this is OK, Sentience will still work)")

        log(f"Total extensions to load: {len(extension_paths)} (including Sentience)")

        # Combine all extensions into a single --load-extension arg
        combined_extensions = ",".join(extension_paths)
        log(f"Combined extension paths (first 100 chars): {combined_extensions[:100]}...")

        # Create browser profile with ALL extensions combined
        # Strategy: Disable default extensions, manually load all together
        browser_profile = BrowserProfile(
            headless=False,  # Run with visible browser for demo
            executable_path=browser_executable,  # Use found browser if available
            enable_default_extensions=False,  # Disable auto-loading, we'll load manually
            ignore_default_args=[
                "--enable-automation",
                "--disable-extensions",  # Important: don't disable extensions
                "--hide-scrollbars",
                # Don't disable component extensions - we need background pages for Sentience
            ],
            args=[
                "--enable-extensions",
                "--disable-extensions-file-access-check",  # Allow extension file access
                "--disable-extensions-http-throttling",  # Don't throttle extension HTTP
                "--extensions-on-chrome-urls",  # Allow extensions on chrome:// URLs
                f"--load-extension={combined_extensions}",  # Load ALL extensions together
            ],
        )

        log("Browser profile configured with Sentience extension")

        # Start browser session
        log("Creating BrowserSession...")
        browser_session = BrowserSession(browser_profile=browser_profile)
        await browser_session.start()
        log("✅ Browser session started")

        # Initialize SentienceAgent
        llm = ChatBrowserUse()
        task = """Go to HackerNews Show at https://news.ycombinator.com/show and find the top 1 Show HN post.

IMPORTANT: Do NOT click the post. Instead:
1. Identify the top post from the Sentience snapshot (it will be the first post in the list)
2. Note its element ID (index number) and title from the snapshot
3. Call the done action with the element ID and title in this format: "Top post: element ID [index], title: [title]"
"""

        log(f"\n🚀 Starting SentienceAgent with Verification: {task}\n")

        # Define verification assertions
        # These will be checked after each step to provide machine-verifiable observability
        step_assertions = [
            # Verify we're on Hacker News
            {
                "predicate": url_contains("news.ycombinator.com"),
                "label": "on_hackernews",
                "required": True,  # Required: agent fails if this doesn't pass
            },
            # Verify Show HN posts are visible
            {
                "predicate": exists("role=link text~'Show HN'"),
                "label": "show_hn_posts_visible",
            },
            # Verify no error messages
            {
                "predicate": not_exists("text~'Error'"),
                "label": "no_error_message",
            },
        ]

        # Define task completion assertion
        # This is machine-verifiable: if this passes, the task is done!
        done_assertion = all_of(
            url_contains("news.ycombinator.com/show"),
            exists("role=link text~'Show HN'"),
        )

        log("📋 Verification assertions configured:")
        log("  - on_hackernews (required): URL contains 'news.ycombinator.com'")
        log("  - show_hn_posts_visible: Show HN links are visible")
        log("  - no_error_message: No error text on page")
        log("  - done_assertion: URL is /show AND Show HN links visible\n")

        # Create Sentience configuration
        sentience_config = SentienceAgentConfig(
            sentience_api_key=os.getenv("SENTIENCE_API_KEY"),
            sentience_use_api=True,  # Use gateway/API mode
            sentience_max_elements=40,
            sentience_show_overlay=True,
        )

        agent = SentienceAgent(
            task=task,
            llm=llm,
            browser_session=browser_session,
            tools=None,  # Will use default tools
            sentience_config=sentience_config,
            # Vision fallback configuration
            vision_fallback_enabled=True,
            vision_detail_level="auto",
            vision_include_screenshots=True,
            # Token tracking
            calculate_cost=True,
            # Agent settings
            max_steps=10,  # Limit steps for example
            max_failures=3,
            # ✨ NEW: Verification configuration (Sentience SDK AgentRuntime)
            enable_verification=True,
            step_assertions=step_assertions,
            done_assertion=done_assertion,
            trace_dir="traces",  # Trace output for Studio timeline
        )

        # Run agent
        result = await agent.run()

        # Get token usage
        usage_summary = await agent.token_cost_service.get_usage_summary()
        log(f"\n📊 Token Usage Summary:")
        log(f"  Total tokens: {usage_summary.total_tokens}")
        log(f"  Total cost: ${usage_summary.total_cost:.6f}")
        log(f"  Steps: {result.get('steps', 'unknown')}")

        # Show detailed Sentience usage stats
        sentience_stats = result.get("sentience_usage_stats", {})
        if sentience_stats:
            steps_using = sentience_stats.get("steps_using_sentience", 0)
            total_steps = sentience_stats.get("total_steps", 0)
            percentage = sentience_stats.get("sentience_percentage", 0)
            log(f"  Sentience used: {result.get('sentience_used', False)}")
            log(f"  Sentience usage: {steps_using}/{total_steps} steps ({percentage:.1f}%)")
        else:
            log(f"  Sentience used: {result.get('sentience_used', 'unknown')}")

        # ✨ NEW: Show verification results
        verification = result.get("verification")
        if verification:
            log(f"\n🔍 Verification Summary:")
            log(f"  All assertions passed: {verification.get('all_assertions_passed', 'N/A')}")
            log(f"  Required assertions passed: {verification.get('required_assertions_passed', 'N/A')}")
            log(f"  Task verified complete: {verification.get('task_verified_complete', False)}")

            # Show individual assertions
            assertions = verification.get("assertions", [])
            if assertions:
                log(f"\n  Assertion Details ({len(assertions)} total):")
                for assertion in assertions:
                    status = "✅" if assertion.get("passed") else "❌"
                    label = assertion.get("label", "unnamed")
                    required = " (required)" if assertion.get("required") else ""
                    log(f"    {status} {label}{required}")
        else:
            log(f"\n🔍 Verification: disabled")

    except ImportError as e:
        log(f"❌ Import error: {e}")
        log("Make sure Sentience SDK is installed: pip install sentienceapi")
    except Exception as e:
        log(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
