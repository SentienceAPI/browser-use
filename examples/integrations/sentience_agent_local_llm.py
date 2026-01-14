"""
Example: SentienceAgent with multi-step verification using AgentRuntime.

This example demonstrates how to use SentienceAgent with:
- Primary: Local LLM (Qwen 2.5 3B) for Sentience snapshots (fast, free)
- Fallback: Cloud vision model (GPT-4o) for vision mode when Sentience fails
- **NEW: Multi-step task with step-by-step verification via AgentRuntime**
- **NEW: Declarative task completion verification using expect() DSL**

Requirements:
1. Install transformers: pip install transformers torch accelerate
2. Optional: pip install bitsandbytes (for 4-bit/8-bit quantization)
3. Sentience SDK installed: pip install sentienceapi
4. Sentience extension loaded in browser
5. OPENAI_API_KEY in .env for GPT-4o vision fallback

Note: Local models will be downloaded from Hugging Face on first use.
Note: `accelerate` is required when using `device_map="auto"`.
"""

import asyncio
import logging
import os
import traceback
from pathlib import Path

import glob
from dotenv import load_dotenv

from browser_use import BrowserProfile, BrowserSession
from browser_use.integrations.sentience import SentienceAgent, SentienceAgentConfig
from browser_use.llm import ChatHuggingFace, ChatOpenAI
from browser_use.llm.messages import SystemMessage, UserMessage
from sentience import get_extension_dir

# Import Sentience SDK AgentRuntime and verification helpers
from sentience.backends import BrowserUseAdapter
from sentience.agent_runtime import AgentRuntime
from sentience.tracing import Tracer, JsonlTraceSink
from sentience.verification import url_contains
from sentience.asserts import E, expect, in_dominant_list

load_dotenv()

# Enable debug logging to see detailed Sentience extension errors
# Uncomment the next line to see more diagnostic information
logging.getLogger("browser_use.integrations.sentience").setLevel(logging.DEBUG)


def log(msg: str) -> None:
    """Print with flush for immediate output."""
    print(msg, flush=True)


async def main():
    """Example: Multi-step task with step-by-step verification."""
    browser_session = None
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
            # Increase wait times to reduce stale element issues
            minimum_wait_page_load_time=0.5,  # Wait longer before capturing page state
            wait_for_network_idle_page_load_time=1.0,  # Wait longer for network to be idle
            wait_between_actions=0.3,  # Wait longer between actions to let page stabilize
        )

        log("Browser profile configured with Sentience extension")

        # Start browser session
        log("Creating BrowserSession...")
        browser_session = BrowserSession(browser_profile=browser_profile)
        await browser_session.start()
        log("✅ Browser session started")

        # Initialize local LLM via Hugging Face transformers
        log("\n" + "=" * 80)
        log("🤖 Initializing Local LLM (Hugging Face transformers)")
        log("=" * 80)

        # Option 1: Qwen 2.5 3B (recommended for small models)
        log("📦 Creating ChatHuggingFace instance...")
        log("   Model: Qwen/Qwen2.5-3B-Instruct")
        log("   ⚠️  IMPORTANT: Model download happens on FIRST LLM call")
        log("   This means it will download when agent makes first decision")
        llm = ChatHuggingFace(
            model="Qwen/Qwen2.5-3B-Instruct",
            device_map="auto",  # Automatically use GPU if available
            torch_dtype="float16",  # Use float16 for faster inference
            max_new_tokens=2048,  # Increased for complete JSON responses
            temperature=0.1,  # Very low temperature for deterministic structured output
        )
        log("✅ ChatHuggingFace instance created (model not loaded yet)")

        # OPTIONAL: Pre-load the model now (before agent starts)
        # This will download the model immediately so you can see progress
        log("\n🔄 Pre-loading model (this will download if not cached)...")
        log("   ⚠️  This is where the download happens - watch for progress!")
        log("   You can skip this by commenting out the next block")
        try:
            # Trigger model loading by calling ainvoke with a simple message
            # This will download/load the model now
            test_messages = [
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="Say 'ready'"),
            ]
            log("   📞 Calling model to trigger download/loading...")
            log("   ⏳ This may take 5-15 minutes on first run (~6GB download)")
            log("   💡 Watch for 'Loading Hugging Face model' messages above")
            response = await llm.ainvoke(test_messages)
            log(f"   ✅ Model loaded successfully! Response: {response.completion[:50]}...")
        except Exception as e:
            log(f"   ❌ Model loading failed: {e}")
            log("   Continuing anyway - model will load on first agent call")
            traceback.print_exc()

        log(f"✅ Using local LLM: {llm.model}")
        log(f"   Device: {llm.device_map}")
        log("\n⏳ Note: Model will be downloaded from Hugging Face on first use (~6GB)")
        log("   This may take 5-15 minutes depending on your internet speed...")
        log("   Model will be cached locally for future runs.\n")

        # Initialize vision LLM for fallback (cloud vision model)
        log("\n" + "=" * 80)
        log("👁️ Initializing Vision LLM (Cloud model for vision fallback)")
        log("=" * 80)
        log("📦 Creating ChatOpenAI instance for vision fallback...")
        log("   Model: gpt-4o (vision-capable)")
        log("   ⚠️  This will only be used when Sentience snapshot fails")
        vision_llm = ChatOpenAI(model="gpt-4o")
        log("✅ Vision LLM configured (will be used only for vision fallback)")

        # ========================================================================
        # SETUP AGENTRUNTIME FOR VERIFICATION
        # ========================================================================
        log("\n" + "=" * 80)
        log("🔍 Setting up AgentRuntime for Multi-Step Verification")
        log("=" * 80)

        # Create BrowserBackend using BrowserUseAdapter
        adapter = BrowserUseAdapter(browser_session)
        backend = await adapter.create_backend()
        log("✅ Created BrowserBackend from browser-use session")

        # Create tracer for verification events
        trace_dir = Path("traces")
        trace_dir.mkdir(exist_ok=True)
        sink = JsonlTraceSink(str(trace_dir / "verification_trace.jsonl"))
        tracer = Tracer(run_id="multi-step-task", sink=sink)
        log("✅ Created Tracer for verification events")

        # Create AgentRuntime with backend
        runtime = AgentRuntime(
            backend=backend,
            tracer=tracer,
            sentience_api_key=os.getenv("SENTIENCE_API_KEY"),
        )
        log("✅ Created AgentRuntime for step-by-step verification")

        # ========================================================================
        # MULTI-STEP TASK WITH VERIFICATION
        # ========================================================================
        log("\n" + "=" * 80)
        log("🚀 Starting Multi-Step Task with Verification")
        log("=" * 80)

        # Define the multi-step task
        task_steps = [
            {
                "goal": "Go to Google and search for 'HackerNews Show'",
                "task": """Go to google.com using the navigate action. 
                After the page loads, you MUST complete these TWO ACTIONS IN ORDER:

                ACTION 1 - Type the search query into the search input box:
                - Find the search input box on the page (it's usually the main text input field)
                - Use the input_text action to type "HackerNews Show" directly into the search box
                - The text to type is exactly: HackerNews Show
                - DO NOT click the input box first - just use input_text action directly
                - The input_text action will automatically focus and type into the search box

                ACTION 2 - Click the Search button:
                - After ACTION 1 completes (after typing), find the Search button on the page
                - The Search button is usually located near the search input box
                - Look for a button with text like "Google Search", "Search", or a search icon
                - Use the click action to click the Search button
                - This will submit the search query

                IMPORTANT:
                - The search query text is: "HackerNews Show" (only these words, nothing else)
                - Do NOT click the search input box before typing - use input_text action directly
                - After typing, you must click the Search button to submit the search
                - Do NOT press Enter key - find and click the Search button instead
                - Action sequence: 1) input_text, 2) click Search button (only 2 actions total)""",
            },
            {
                "goal": "Click the Show HN link in search results",
                "task": "In the search results, find and click the link to 'Show | Hacker News'",
            },
            {
                "goal": "Find the top 1 Show HN post",
                "task": "On the Show HN page, identify the top 1 Show HN post (first post in the list). Do NOT click it. Just identify it.",
            },
        ]

        # Create Sentience configuration
        sentience_config = SentienceAgentConfig(
            sentience_api_key=os.getenv("SENTIENCE_API_KEY"),
            sentience_use_api=True,  # Use gateway/API mode
            sentience_max_elements=40,
            sentience_show_overlay=True,
        )

        # Run each step with verification
        for step_idx, step_info in enumerate(task_steps, start=1):
            log(f"\n{'=' * 80}")
            log(f"📋 Step {step_idx}: {step_info['goal']}")
            log(f"{'=' * 80}")

            # Begin verification step
            runtime.begin_step(step_info["goal"], step_index=step_idx - 1)
            log(f"✅ Began verification step {step_idx}")

            # Create agent for this step
            agent = SentienceAgent(
                task=step_info["task"],
                llm=llm,  # Primary LLM: Qwen 3B for Sentience snapshots
                vision_llm=vision_llm,  # Fallback LLM: GPT-4o for vision mode
                browser_session=browser_session,
                tools=None,  # Will use default tools
                sentience_config=sentience_config,
                # Vision fallback configuration
                vision_fallback_enabled=True,
                vision_detail_level="auto",
                vision_include_screenshots=True,
                # Token tracking
                calculate_cost=True,
                # Agent settings - increased to handle stale element retries
                max_steps=10,  # Increased to allow more retries with fresh snapshots
                max_failures=5,  # Increased to handle stale element indices (page changes between snapshot and action)
                # Local LLM specific settings
                max_history_items=5,
                llm_timeout=300,
                step_timeout=360,
                # Disable built-in verification (we're using AgentRuntime)
                enable_verification=False,
            )

            # Run agent for this step
            log(f"🤖 Running agent for step {step_idx}...")
            result = await agent.run()
            log(f"✅ Agent completed step {step_idx}")

            # Take snapshot for verification
            log(f"📸 Taking snapshot for verification...")
            snapshot = await runtime.snapshot()
            log(f"✅ Snapshot taken: {len(snapshot.elements)} elements found")

            # Step-specific verification
            log(f"🔍 Verifying step {step_idx}...")
            all_passed = True

            if step_idx == 1:
                # Step 1: Verify we're on Google
                log("  Verifying: URL contains google.com")
                passed = runtime.assert_(
                    url_contains("google.com"),
                    label="on_google",
                    required=True,
                )
                all_passed = all_passed and passed
                log(f"  {'✅' if passed else '❌'} URL contains google.com: {passed}")

                # Verify search results contain "Show | Hacker News"
                log("  Verifying: Search results contain 'Show | Hacker News'")
                passed = runtime.assert_(
                    expect(E(text_contains="Show")).to_exist(),
                    label="search_results_contain_show",
                )
                all_passed = all_passed and passed
                log(f"  {'✅' if passed else '❌'} Search results contain 'Show': {passed}")

                # Also check for "Hacker News" text
                passed = runtime.assert_(
                    expect.text_present("Hacker News"),
                    label="hacker_news_text_present",
                )
                all_passed = all_passed and passed
                log(f"  {'✅' if passed else '❌'} 'Hacker News' text present: {passed}")

            elif step_idx == 2:
                # Step 2: Verify we're on Show HN page
                log("  Verifying: URL contains news.ycombinator.com/show")
                passed = runtime.assert_(
                    url_contains("news.ycombinator.com/show"),
                    label="on_show_hn_page",
                    required=True,
                )
                all_passed = all_passed and passed
                log(f"  {'✅' if passed else '❌'} URL contains news.ycombinator.com/show: {passed}")

                # Verify Show HN posts are visible
                log("  Verifying: Show HN posts are visible")
                passed = runtime.assert_(
                    expect(E(text_contains="Show HN")).to_exist(),
                    label="show_hn_posts_visible",
                )
                all_passed = all_passed and passed
                log(f"  {'✅' if passed else '❌'} Show HN posts visible: {passed}")

            elif step_idx == 3:
                # Step 3: Verify we found the top post
                log("  Verifying: Top 1 Show HN post contains 'Show HN' in title")
                # Check if the first item in dominant list contains "Show HN"
                passed = runtime.assert_(
                    expect(in_dominant_list().nth(0)).to_have_text_contains("Show HN"),
                    label="top_post_contains_show_hn",
                    required=True,
                )
                all_passed = all_passed and passed
                log(f"  {'✅' if passed else '❌'} Top post contains 'Show HN': {passed}")

                # Verify we're still on Show HN page
                passed = runtime.assert_(
                    url_contains("news.ycombinator.com/show"),
                    label="still_on_show_hn_page",
                )
                all_passed = all_passed and passed
                log(f"  {'✅' if passed else '❌'} Still on Show HN page: {passed}")

            log(f"\n{'✅' if all_passed else '❌'} Step {step_idx} verification: {'PASSED' if all_passed else 'FAILED'}")

        # ========================================================================
        # FINAL TASK COMPLETION VERIFICATION
        # ========================================================================
        log(f"\n{'=' * 80}")
        log("🎯 Final Task Completion Verification")
        log(f"{'=' * 80}")

        # Take final snapshot
        final_snapshot = await runtime.snapshot()
        log(f"📸 Final snapshot: {len(final_snapshot.elements)} elements")

        # Verify task completion
        log("🔍 Verifying task completion...")
        task_complete = runtime.assert_done(
            expect(in_dominant_list().nth(0)).to_have_text_contains("Show HN"),
            label="task_complete_top_post_found",
        )

        if task_complete:
            log("✅ Task completed successfully!")
            log(f"   Top post title contains 'Show HN'")
        else:
            log("❌ Task completion verification failed")
            log("   Top post may not contain 'Show HN' in title")

        # ========================================================================
        # SUMMARY
        # ========================================================================
        log(f"\n{'=' * 80}")
        log("📊 Summary")
        log(f"{'=' * 80}")

        # Get token usage from last agent
        usage_summary = await agent.token_cost_service.get_usage_summary()
        log(f"Token Usage:")
        log(f"  Total tokens: {usage_summary.total_tokens}")
        log(f"  Total cost: ${usage_summary.total_cost:.6f}")

        # Show verification summary
        log(f"\nVerification Summary:")
        log(f"  Task completed: {task_complete}")
        log(f"  All assertions passed: {runtime.all_assertions_passed()}")
        log(f"  Required assertions passed: {runtime.required_assertions_passed()}")

        # Show trace file location
        log(f"\nTrace file: {trace_dir / 'verification_trace.jsonl'}")
        log("  You can view this in Sentience Studio for detailed verification timeline")

    except ImportError as e:
        log(f"❌ Import error: {e}")
        log("\nPlease install required packages:")
        log("  pip install transformers torch accelerate sentienceapi")
    except Exception as e:
        log(f"❌ Error: {e}")
        traceback.print_exc()
    finally:
        if browser_session is not None:
            try:
                await browser_session.stop()  # Gracefully stop the browser session
            except Exception as e:
                log(f"⚠️  Error stopping browser session: {e}")


if __name__ == "__main__":
    asyncio.run(main())
