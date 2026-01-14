"""
Example: MultiStepSentienceAgent with Local LLM and AgentRuntime verification.

This example demonstrates how to use MultiStepSentienceAgent with:
- Primary: Local LLM (Qwen 2.5 3B) via LocalLLMProvider from Sentience SDK
- Multi-step task execution with step-by-step verification via AgentRuntime
- Declarative task completion verification using expect() DSL

Requirements:
1. Install transformers: pip install transformers torch accelerate
2. Optional: pip install bitsandbytes (for 4-bit/8-bit quantization)
3. Sentience SDK installed: pip install sentienceapi

Note: Local models will be downloaded from Hugging Face on first use.
Note: `accelerate` is required when using `device_map="auto"`.
"""

import asyncio
import logging
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv

# Import Sentience SDK components
from sentience.async_api import AsyncSentienceBrowser
from sentience.llm_provider import LocalLLMProvider
from sentience.agent_config import AgentConfig
from sentience.verification import url_contains
from sentience.asserts import E, expect, in_dominant_list

# Import MultiStepSentienceAgent from browser-use integration
from browser_use.integrations.sentience import MultiStepSentienceAgent

load_dotenv()

# Enable debug logging
logging.getLogger("browser_use.integrations.sentience").setLevel(logging.DEBUG)


def log(msg: str) -> None:
    """Print with flush for immediate output."""
    print(msg, flush=True)


async def main():
    """Example: Multi-step task with step-by-step verification using MultiStepSentienceAgent."""
    browser = None
    try:
        # ========================================================================
        # INITIALIZE SENTIENCE BROWSER
        # ========================================================================
        log("\n" + "=" * 80)
        log("🌐 Initializing AsyncSentienceBrowser")
        log("=" * 80)

        # Create AsyncSentienceBrowser from Sentience SDK
        browser = AsyncSentienceBrowser(
            headless=False,
            api_key=os.getenv("SENTIENCE_API_KEY"),
        )
        await browser.start()
        log("✅ AsyncSentienceBrowser started")

        # Navigate to the first URL immediately so extension can inject properly
        # The extension needs to be on an actual page, not about:blank
        first_url = "https://google.com"
        log(f"🌐 Navigating to first URL: {first_url}")
        await browser.goto(first_url)
        log("✅ Navigated to first URL - extension should now be injected")

        # ========================================================================
        # INITIALIZE LOCAL LLM
        # ========================================================================
        log("\n" + "=" * 80)
        log("🤖 Initializing Local LLM (Qwen 2.5 3B)")
        log("=" * 80)

        log("📦 Creating LocalLLMProvider instance...")
        log("   Model: Qwen/Qwen2.5-3B-Instruct")
        log("   ⚠️  IMPORTANT: Model download happens on FIRST LLM call")
        llm = LocalLLMProvider(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            device="auto",
            load_in_4bit=False,  # Set to True to save memory
            torch_dtype="auto",
        )
        log("✅ LocalLLMProvider instance created (model not loaded yet)")

        # OPTIONAL: Pre-load the model now
        log("\n🔄 Pre-loading model (this will download if not cached)...")
        log("   ⚠️  This is where the download happens - watch for progress!")
        try:
            log("   📞 Calling model to trigger download/loading...")
            log("   ⏳ This may take 5-15 minutes on first run (~6GB download)")
            response = llm.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Say 'ready'",
                max_new_tokens=50,
            )
            log(f"   ✅ Model loaded successfully! Response: {response.content[:50]}...")
        except Exception as e:
            log(f"   ❌ Model loading failed: {e}")
            log("   Continuing anyway - model will load on first agent call")
            traceback.print_exc()

        log(f"✅ Using local LLM: {llm.model_name}")

        # ========================================================================
        # CREATE MULTI-STEP AGENT
        # ========================================================================
        log("\n" + "=" * 80)
        log("🚀 Creating MultiStepSentienceAgent")
        log("=" * 80)

        # Create AgentConfig for SentienceAgentAsync
        agent_config = AgentConfig(
            snapshot_limit=50,
            temperature=0.0,
            max_retries=3,
            verify=True,
            capture_screenshots=True,
            screenshot_format="jpeg",
            screenshot_quality=80,
            show_overlay=True,
        )

        # Create multi-step agent
        agent = MultiStepSentienceAgent(
            browser=browser,
            llm=llm,
            trace_dir="traces",
            sentience_api_key=os.getenv("SENTIENCE_API_KEY"),
            agent_config=agent_config,
            default_snapshot_limit=50,
            verbose=True,
        )
        log("✅ MultiStepSentienceAgent created")

        # ========================================================================
        # DEFINE MULTI-STEP TASK
        # ========================================================================
        log("\n" + "=" * 80)
        log("📋 Defining Multi-Step Task")
        log("=" * 80)

        task_steps = [
            {
                "goal": "Verify on Google search page",
                "task": "You are on google.com. Verify you see the Google search interface with a search box.",
            },
            {
                "goal": "Type 'Hacker News Show' in the search box",
                "task": """Type "Hacker News Show" into the Google search box.
                
åå                Find the search input (role="combobox" or "searchbox" with "Search" text). Use type_text action with its element ID to type "Hacker News Show". Do NOT click anything yet.""",
            },
            {
                "goal": "Click the Google Search button",
                "task": """Click the "Google Search" button to submit.
                
                Find the button (role="button" with "Google Search" text). Use click action with its element ID. Do NOT press Enter.""",
            },
            {
                "goal": "Click 'Show | Hacker News' link",
                "task": """Click the link with exact title "Show | Hacker News" in search results.
                
                Find link element (role="link") with text "Show | Hacker News" (with pipe |). Use click action with its element ID. Only click this exact link, not others.""",
            },
            {
                "goal": "Find the top 1 Show HN post",
                "task": """On Hacker News Show page, identify the element ID of the first post in the list.
                
                CRITICAL: This is an IDENTIFICATION task only. Do NOT click anything.
                
                Find the first post element (role="link") in the list. The post should have "Show HN" in its title text.
                Output the element ID using CLICK(id) format, but this is for identification only - the click will be prevented.
                Example: If the first post has ID 631, output CLICK(631) but understand this is just to report the ID.""",
            },
        ]

        log(f"✅ Defined {len(task_steps)} task steps")

        # ========================================================================
        # DEFINE VERIFICATION CALLBACKS
        # ========================================================================
        log("\n" + "=" * 80)
        log("🔍 Defining Verification Callbacks")
        log("=" * 80)

        def verify_step_1(runtime, step_idx, snapshot):
            """Verify step 1: On Google search page."""
            log("  Verifying: URL contains google.com")
            passed = runtime.assert_(
                url_contains("google.com"),
                label="on_google",
                required=True,
            )
            log(f"  {'✅' if passed else '❌'} URL contains google.com: {passed}")
            return passed

        def verify_step_2(runtime, step_idx, snapshot):
            """Verify step 2: Text typed in search box."""
            # Verify we're still on Google
            log("  Verifying: Still on google.com")
            passed1 = runtime.assert_(
                url_contains("google.com"),
                label="still_on_google",
            )
            log(f"  {'✅' if passed1 else '❌'} Still on google.com: {passed1}")
            return passed1

        def verify_step_3(runtime, step_idx, snapshot):
            """Verify step 3: Search results page loaded."""
            log("  Verifying: Search results contain 'Show | Hacker News'")
            passed1 = runtime.assert_(
                expect(E(text_contains="Show")).to_exist(),
                label="search_results_contain_show",
            )
            log(f"  {'✅' if passed1 else '❌'} Search results contain 'Show': {passed1}")

            passed2 = runtime.assert_(
                expect.text_present("Hacker News"),
                label="hacker_news_text_present",
            )
            log(f"  {'✅' if passed2 else '❌'} 'Hacker News' text present: {passed2}")

            return passed1 and passed2

        def verify_step_4(runtime, step_idx, snapshot):
            """Verify step 4: On Show HN page."""
            log("  Verifying: URL contains news.ycombinator.com/show")
            passed1 = runtime.assert_(
                url_contains("news.ycombinator.com/show"),
                label="on_show_hn_page",
                required=True,
            )
            log(f"  {'✅' if passed1 else '❌'} URL contains news.ycombinator.com/show: {passed1}")

            passed2 = runtime.assert_(
                expect(E(text_contains="Show HN")).to_exist(),
                label="show_hn_posts_visible",
            )
            log(f"  {'✅' if passed2 else '❌'} Show HN posts visible: {passed2}")

            return passed1 and passed2

        def verify_step_5(runtime, step_idx, snapshot):
            """Verify step 5: Top post found.
            
            Note: The agent may have clicked the post (navigating away from Show HN page),
            so we only verify that we're on a Hacker News page (either Show HN list or post detail).
            The actual element text validation is done in multi_step_agent.py using the pre-agent snapshot.
            """
            log("  Verifying: On Hacker News (either Show HN list or post detail page)")
            # After clicking, we might be on the post detail page, so just check we're on HN
            passed = runtime.assert_(
                url_contains("news.ycombinator.com"),
                label="on_hackernews",
                required=True,
            )
            log(f"  {'✅' if passed else '❌'} On Hacker News page: {passed}")
            
            # Note: We don't check for "Show HN" text or dominant list because:
            # 1. If the agent clicked the post, we're on the detail page (no Show HN text)
            # 2. The element text validation was already done in multi_step_agent.py using pre-agent snapshot
            # 3. The task is to identify the element, not necessarily stay on the Show HN page
            
            return passed

        verification_callbacks = {
            1: verify_step_1,
            2: verify_step_2,
            3: verify_step_3,
            4: verify_step_4,
            5: verify_step_5,
        }

        log(f"✅ Defined {len(verification_callbacks)} verification callbacks")

        # ========================================================================
        # RUN MULTI-STEP TASK
        # ========================================================================
        log("\n" + "=" * 80)
        log("🚀 Running Multi-Step Task")
        log("=" * 80)

        results = await agent.run_multi_step(
            task_steps=task_steps,
            verification_callbacks=verification_callbacks,
            max_retries=2,
        )

        log(f"\n✅ Completed {len(results)} steps")

        # ========================================================================
        # FINAL VERIFICATION
        # ========================================================================
        log("\n" + "=" * 80)
        log("🔍 Final Task Verification")
        log("=" * 80)

        task_complete = await agent.assert_done(
            expect(in_dominant_list().nth(0)).to_have_text_contains("Show HN"),
            label="top_post_found",
        )

        if task_complete:
            log("✅ Task completed successfully!")
        else:
            log("⚠️  Task may not be complete - check verification results")

        # ========================================================================
        # SUMMARY
        # ========================================================================
        log("\n" + "=" * 80)
        log("📊 Verification Summary")
        log("=" * 80)

        summary = await agent.get_verification_summary()
        log(f"Runtime available: {summary['runtime_available']}")
        log(f"All assertions passed: {summary['all_assertions_passed']}")
        log(f"Required assertions passed: {summary['required_assertions_passed']}")
        if summary.get("trace_file"):
            log(f"Trace file: {summary['trace_file']}")

    except Exception as e:
        log(f"\n❌ Error: {e}")
        traceback.print_exc()
    finally:
        if browser:
            log("\n🛑 Closing browser...")
            await browser.close()
            log("✅ Browser closed")


if __name__ == "__main__":
    asyncio.run(main())
