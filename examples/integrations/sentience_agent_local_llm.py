"""
Example: SentienceAgent with dual-model setup (local LLM + cloud vision model).

This example demonstrates how to use SentienceAgent with:
- Primary: Local LLM (Qwen 2.5 3B) for Sentience snapshots (fast, free)
- Fallback: Cloud vision model (GPT-4o) for vision mode when Sentience fails

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
from browser_use.integrations.sentience import SentienceAgent
from browser_use.llm import ChatHuggingFace, ChatOpenAI
from browser_use.llm.messages import SystemMessage, UserMessage
from sentience import get_extension_dir

load_dotenv()

# Enable debug logging to see detailed Sentience extension errors
# Uncomment the next line to see more diagnostic information
logging.getLogger("browser_use.integrations.sentience").setLevel(logging.DEBUG)


def log(msg: str) -> None:
    """Print with flush for immediate output."""
    print(msg, flush=True)


async def main():
    """Example: Use SentienceAgent with local LLM (Qwen 2.5 3B or BitNet)."""
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

        # Option 2: BitNet B1.58 2B 4T (if available on Hugging Face)
        # llm = ChatHuggingFace(
        #     model="microsoft/bitnet-b1.58-2B",  # Check actual model name on HF
        #     device_map="auto",
        #     torch_dtype="float16",
        # )

        # Option 3: Other small models
        # llm = ChatHuggingFace(
        #     model="meta-llama/Llama-3.2-3B-Instruct",
        #     device_map="auto",
        #     torch_dtype="float16",
        # )

        # Option 4: Use 4-bit quantization to save memory (requires bitsandbytes)
        # llm = ChatHuggingFace(
        #     model="Qwen/Qwen2.5-3B-Instruct",
        #     device_map="auto",
        #     load_in_4bit=True,  # Reduces memory usage significantly
        #     max_new_tokens=2048,
        # )

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

        # Initialize SentienceAgent
        task = """Go to HackerNews Show at https://news.ycombinator.com/show and find the top 1 Show HN post.

IMPORTANT: Do NOT click the post. Instead:
1. Identify the top post from the Sentience snapshot (it will be the first post in the list)
2. Note its element ID (index number) and title from the snapshot
3. Call the done action with the element ID and title in this format: "Top post: element ID [index], title: [title]"
"""

        log(f"\n🚀 Starting SentienceAgent: {task}\n")

        agent = SentienceAgent(
            task=task,
            llm=llm,  # Primary LLM: Qwen 3B for Sentience snapshots
            vision_llm=vision_llm,  # Fallback LLM: GPT-4o for vision mode
            browser_session=browser_session,
            tools=None,  # Will use default tools
            # Sentience configuration
            sentience_api_key=os.getenv("SENTIENCE_API_KEY"),
            sentience_use_api=True,  # Use gateway/API mode
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
            # Local LLM specific settings (keep these for local model compatibility)
            max_history_items=5,  # Keep minimal history for small models
            llm_timeout=300,  # Increased timeout for local LLMs (5 minutes)
            step_timeout=360,  # Increased step timeout (6 minutes)
        )

        # Run agent
        result = await agent.run()

        # Get token usage
        usage_summary = await agent.token_cost_service.get_usage_summary()
        log("\n📊 Token Usage Summary:")
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
