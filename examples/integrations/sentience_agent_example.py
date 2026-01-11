"""
Example usage of SentienceAgent.

This example demonstrates how to use SentienceAgent with:
- Sentience snapshot as primary prompt (compact, token-efficient)
- Vision fallback when snapshot fails
- Token usage tracking
"""

import asyncio
import os

from dotenv import load_dotenv

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
        from browser_use import BrowserProfile, ChatBrowserUse, BrowserSession
        from browser_use.integrations.sentience import SentienceAgent
        from sentience import get_extension_dir
        from pathlib import Path
        import glob

        # Get path to Sentience extension
        sentience_ext_path = get_extension_dir()
        log(f"Loading Sentience extension from: {sentience_ext_path}")

        # Verify extension exists
        if not os.path.exists(sentience_ext_path):
            raise FileNotFoundError(f"Sentience extension not found at: {sentience_ext_path}")
        if not os.path.exists(os.path.join(sentience_ext_path, "manifest.json")):
            raise FileNotFoundError(
                f"Sentience extension manifest not found at: {sentience_ext_path}/manifest.json"
            )
        log(f"✅ Sentience extension verified at: {sentience_ext_path}")

        # Find browser executable (optional - browser-use will find one if not specified)
        # This example looks for Playwright-installed browsers (Chromium-based, work with CDP)
        playwright_path = Path.home() / "Library/Caches/ms-playwright"
        chromium_patterns = [
            playwright_path
            / "chromium-*/chrome-mac*/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
            playwright_path / "chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium",
        ]

        executable_path = None
        for pattern in chromium_patterns:
            matches = glob.glob(str(pattern))
            if matches:
                matches.sort()
                executable_path = matches[-1]  # Use latest version
                if Path(executable_path).exists():
                    log(f"✅ Found browser: {executable_path}")
                    break

        if not executable_path:
            log("⚠️  Browser not found, browser-use will try to install it")

        # Get default extension paths and combine with Sentience extension
        # Chrome only uses the LAST --load-extension arg, so we must combine all extensions
        log("Collecting all extension paths...")
        all_extension_paths = [sentience_ext_path]

        # Create a temporary profile to ensure default extensions are downloaded
        # This ensures extensions exist before we try to load them
        temp_profile = BrowserProfile(enable_default_extensions=True)
        default_ext_paths = temp_profile._ensure_default_extensions_downloaded()

        if default_ext_paths:
            all_extension_paths.extend(default_ext_paths)
            log(f"  ✅ Found {len(default_ext_paths)} default extensions")
        else:
            log("  ⚠️  No default extensions found (this is OK, Sentience will still work)")

        log(f"Total extensions to load: {len(all_extension_paths)} (including Sentience)")

        # Combine all extensions into a single --load-extension arg
        combined_extensions = ",".join(all_extension_paths)
        log(f"Combined extension paths (first 100 chars): {combined_extensions[:100]}...")

        # Create browser profile with ALL extensions combined
        # Strategy: Disable default extensions, manually load all together
        browser_profile = BrowserProfile(
            headless=False,  # Run with visible browser for demo
            executable_path=executable_path,  # Use found browser if available
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
        task = "Find the top 1 post on Show HN"

        log(f"\n🚀 Starting SentienceAgent: {task}\n")

        agent = SentienceAgent(
            task=task,
            llm=llm,
            browser_session=browser_session,
            tools=None,  # Will use default tools in later phases
            # Sentience configuration
            sentience_api_key=os.getenv("SENTIENCE_API_KEY"),
            sentience_use_api=True,  # use gateway/API mode
            sentience_max_elements=40,
            sentience_show_overlay=True,
            # Vision fallback configuration
            vision_fallback_enabled=True,
            vision_detail_level="auto",
            vision_include_screenshots=True,
            # Token tracking
            calculate_cost=True,
            # Agent settings
            max_steps=10,  # Limit steps for example
            max_failures=3,
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
        sentience_stats = result.get('sentience_usage_stats', {})
        if sentience_stats:
            steps_using = sentience_stats.get('steps_using_sentience', 0)
            total_steps = sentience_stats.get('total_steps', 0)
            percentage = sentience_stats.get('sentience_percentage', 0)
            log(f"  Sentience used: {result.get('sentience_used', False)}")
            log(f"  Sentience usage: {steps_using}/{total_steps} steps ({percentage:.1f}%)")
        else:
            log(f"  Sentience used: {result.get('sentience_used', 'unknown')}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure Sentience SDK is installed: pip install sentienceapi")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
