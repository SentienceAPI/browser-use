# Custom Agent Design: Full Control Over LLM Prompts

## Executive Summary

This document outlines the design for implementing a custom browser automation agent with full control over prompt construction, enabling:
1. **Primary**: Sentience SDK snapshot elements as compact, token-efficient prompts
2. **Fallback**: Vision-based prompts when Sentience snapshots fail
3. **Token tracking**: Integration with browser-use's built-in token usage utilities
4. **SDK integration**: Leveraging `SentienceContext` and other SDK backend modules

## Current Architecture Analysis

### Existing Agent Flow

The current `browser_use.Agent` class follows this flow:

```
Agent.run()
  └─> _prepare_context()
      ├─> build_sentience_state() [optional, if Sentience SDK available]
      │   └─> Injects Sentience prompt block via _add_context_message()
      └─> _message_manager.create_state_messages()
          └─> AgentMessagePrompt.get_user_message()
              ├─> Builds browser state (DOM tree, screenshots)
              ├─> Combines agent history, state, browser state
              └─> Returns UserMessage with text + optional images
  └─> _get_next_action()
      └─> LLM.ainvoke(messages)
          └─> TokenCostService tracks usage automatically
```

### Key Components

1. **Agent** (`browser_use/agent/service.py`):
   - Orchestrates the agent loop
   - Manages browser session, tools, and state
   - Calls `_prepare_context()` before each LLM call
   - Handles action execution and retries

2. **MessageManager** (`browser_use/agent/message_manager/service.py`):
   - Manages conversation history
   - Creates state messages via `create_state_messages()`
   - Detects Sentience injection and reduces DOM size accordingly
   - Handles vision mode (screenshots vs. text-only)

3. **AgentMessagePrompt** (`browser_use/agent/prompts.py`):
   - Builds the complete user message
   - Combines: agent history, agent state, browser state, read state
   - Handles vision mode (text + images vs. text-only)
   - Formats DOM tree and screenshots

4. **TokenCostService** (`browser_use/tokens/service.py`):
   - Automatically tracks token usage when LLMs are registered
   - Calculates costs based on model pricing
   - Provides usage summaries and statistics

5. **SentienceContext** (`sentience/backends/sentience_context.py`):
   - Provides `build()` method that returns `SentienceContextState`
   - `SentienceContextState` contains: `url`, `snapshot`, `prompt_block`
   - Handles extension waiting, snapshot retries, and formatting

## Design Goals

### 1. Primary: Sentience Snapshot as Preferred Prompt

**Requirement**: Use Sentience SDK snapshot elements as the primary, compact prompt format.

**Implementation Strategy**:
- Use `SentienceContext.build()` to get snapshot and formatted prompt
- Inject the `prompt_block` as the primary browser state representation
- Skip or minimize DOM tree extraction when Sentience is available
- Format: `ID|role|text|imp|is_primary|docYq|ord|DG|href`

**Benefits**:
- **Token efficiency**: ~60 elements × ~50 chars = ~3K tokens vs. ~40K tokens for full DOM
- **Semantic accuracy**: Importance scores and dominant group detection
- **Ordinal support**: Built-in support for "first", "third", etc. via `ord` and `DG` fields

### 2. Fallback: Vision Mode When Snapshot Fails

**Requirement**: Automatically fall back to vision-based prompts if Sentience snapshot fails.

**Failure Scenarios**:
- Extension not loaded
- Snapshot timeout
- Network errors
- Invalid snapshot response

**Implementation Strategy**:
- Try `SentienceContext.build()` first
- If `None` returned, fall back to vision mode:
  - Enable screenshots (`use_vision=True`)
  - Use full DOM tree (truncated to ~40K chars)
  - Include browser state summary

**Decision Logic**:
```python
sentience_state = await sentience_context.build(browser_session, goal=task)
if sentience_state:
    # Use Sentience prompt block
    prompt = sentience_state.prompt_block
    use_vision = False
else:
    # Fall back to vision
    prompt = build_dom_state(browser_session)
    use_vision = True
```

### 3. Token Usage Tracking

**Requirement**: Use browser-use's built-in token usage utilities.

**Implementation Strategy**:
- Initialize `TokenCost` service with `calculate_cost=True`
- Register LLM instance: `token_cost_service.register_llm(llm)`
- Token tracking happens automatically via wrapped `ainvoke()` method
- Access usage via:
  - `agent.token_cost_service.get_usage_summary()`
  - `history.usage` (from `agent.run()`)

**Token Tracking Flow**:
```
LLM.ainvoke(messages)
  └─> [wrapped by TokenCostService]
      ├─> original_ainvoke(messages)
      │   └─> Returns result with result.usage
      └─> token_cost_service.add_usage(model, usage)
          └─> Tracks in usage_history
```

### 4. SDK Integration

**Requirement**: Use `SentienceContext` and other SDK backend modules.

**Available SDK Components**:
- `SentienceContext` (`sentience/backends/sentience_context.py`):
  - `build(browser_session, goal=...)` → `SentienceContextState | None`
  - Handles extension waiting, snapshot, formatting
- `BrowserUseAdapter` (`sentience/backends/browser_use_adapter.py`):
  - Adapts browser-use `BrowserSession` to Sentience backend interface
- `snapshot()` (`sentience/backends/snapshot.py`):
  - Low-level snapshot function (used by `SentienceContext`)

**Integration Points**:
- Use `SentienceContext` as the primary interface (recommended)
- Or use `BrowserUseAdapter` + `snapshot()` directly for more control

## Proposed Architecture

### Custom Agent Class Structure

```python
class CustomSentienceAgent:
    """
    Custom agent with full control over prompt construction.
    
    Features:
    - Primary: Sentience snapshot as compact prompt
    - Fallback: Vision mode when snapshot fails
    - Token usage tracking
    - Full control over message construction
    """
    
    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        browser_session: BrowserSession,
        tools: Tools,
        # Sentience configuration
        sentience_api_key: str | None = None,
        sentience_use_api: bool | None = None,
        sentience_max_elements: int = 60,
        sentience_show_overlay: bool = False,
        # Vision fallback configuration
        vision_fallback_enabled: bool = True,
        vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
        # Token tracking
        calculate_cost: bool = True,
        # Other agent settings
        max_steps: int = 100,
        use_vision: bool = False,  # Default: prefer Sentience over vision
        ...
    ):
        self.task = task
        self.llm = llm
        self.browser_session = browser_session
        self.tools = tools
        
        # Initialize SentienceContext
        self.sentience_context = SentienceContext(
            sentience_api_key=sentience_api_key,
            use_api=sentience_use_api,
            max_elements=sentience_max_elements,
            show_overlay=sentience_show_overlay,
        )
        
        # Initialize token cost service
        self.token_cost_service = TokenCost(include_cost=calculate_cost)
        self.token_cost_service.register_llm(llm)
        
        # Vision fallback settings
        self.vision_fallback_enabled = vision_fallback_enabled
        self.vision_detail_level = vision_detail_level
        self.use_vision = use_vision  # Can be overridden by fallback logic
        
        # Message manager for conversation history
        self.message_manager = CustomMessageManager(...)
        
    async def run(self) -> AgentHistoryList:
        """Main agent loop with custom prompt construction."""
        # Similar to Agent.run() but with custom _prepare_context()
        ...
    
    async def _prepare_context(self) -> tuple[UserMessage, bool]:
        """
        Prepare context with Sentience-first, vision-fallback strategy.
        
        Returns:
            (user_message, sentience_used): Tuple of message and whether Sentience was used
        """
        # Try Sentience first
        sentience_state = await self.sentience_context.build(
            self.browser_session,
            goal=self.task,
        )
        
        if sentience_state:
            # Use Sentience prompt block
            user_message = self._build_sentience_message(sentience_state)
            return user_message, True
        else:
            # Fall back to vision
            if self.vision_fallback_enabled:
                user_message = await self._build_vision_message()
                return user_message, False
            else:
                # No fallback: return minimal message
                user_message = self._build_minimal_message()
                return user_message, False
    
    def _build_sentience_message(self, sentience_state: SentienceContextState) -> UserMessage:
        """Build user message using Sentience prompt block."""
        # Combine agent history + Sentience prompt block
        content = (
            f"<agent_history>\n{self.message_manager.get_history_description()}\n</agent_history>\n\n"
            f"<browser_state>\n{sentience_state.prompt_block}\n</browser_state>\n"
        )
        return UserMessage(content=content, cache=True)
    
    async def _build_vision_message(self) -> UserMessage:
        """Build user message using vision (screenshots + DOM)."""
        # Get browser state summary with screenshots
        browser_state = await self.browser_session.get_browser_state_summary(
            include_screenshot=True
        )
        
        # Build DOM state description
        dom_state = self._build_dom_state(browser_state)
        
        # Combine with screenshots
        content_parts = [
            ContentPartTextParam(text=dom_state),
            # Add screenshots...
        ]
        
        return UserMessage(content=content_parts, cache=True)
```

### Message Construction Flow

```
_prepare_context()
  ├─> Try: sentience_context.build(browser_session, goal=task)
  │   ├─> Success: _build_sentience_message()
  │   │   └─> Returns: UserMessage with Sentience prompt block
  │   └─> Failure: None returned
  │
  └─> Fallback (if sentience_state is None):
      ├─> vision_fallback_enabled?
      │   ├─> Yes: _build_vision_message()
      │   │   └─> Returns: UserMessage with screenshots + DOM
      │   └─> No: _build_minimal_message()
      │       └─> Returns: UserMessage with minimal state
```

### Integration with Existing Components

#### 1. Browser Session
- **Reuse**: `BrowserSession` from browser-use
- **No changes needed**: Works with existing browser session

#### 2. Tools
- **Reuse**: `Tools` registry from browser-use
- **No changes needed**: Same tool interface

#### 3. Token Cost Service
- **Reuse**: `TokenCost` from browser-use
- **Integration**: Register LLM and access usage summaries

#### 4. Message Manager
- **Custom**: Create `CustomMessageManager` that:
  - Manages conversation history (similar to existing `MessageManager`)
  - Does NOT automatically inject Sentience (we handle it explicitly)
  - Provides history description for prompt construction

## Implementation Plan

### Phase 1: Core Custom Agent (Week 1)

**Tasks**:
1. Create `CustomSentienceAgent` class skeleton
2. Implement `_prepare_context()` with Sentience-first logic
3. Implement `_build_sentience_message()` using `SentienceContext`
4. Implement basic agent loop (`run()` method)
5. Integrate token cost service

**Deliverables**:
- `custom_sentience_agent.py` with basic functionality
- Unit tests for prompt construction logic

### Phase 2: Vision Fallback (Week 1-2)

**Tasks**:
1. Implement `_build_vision_message()` with screenshots
2. Implement `_build_dom_state()` for DOM tree extraction
3. Add fallback decision logic
4. Test fallback scenarios (extension not loaded, timeout, etc.)

**Deliverables**:
- Complete fallback implementation
- Integration tests for fallback scenarios

### Phase 3: Message Manager Integration (Week 2)

**Tasks**:
1. Create `CustomMessageManager` for history management
2. Integrate with agent loop
3. Handle system messages and tool definitions
4. Test conversation history tracking

**Deliverables**:
- `custom_message_manager.py`
- History tracking tests

### Phase 4: Advanced Features (Week 2-3)

**Tasks**:
1. Add configuration options (max_elements, show_overlay, etc.)
2. Add logging and observability
3. Add error handling and retries
4. Performance optimization

**Deliverables**:
- Production-ready custom agent
- Documentation and examples

## Code Structure

```
browser_use/
  integrations/
    sentience/
      custom_agent.py          # CustomSentienceAgent class
      custom_message_manager.py # CustomMessageManager class
      prompt_builder.py         # Prompt construction utilities
      examples/
        custom_agent_example.py # Example usage
```

## Example Usage

```python
from browser_use import BrowserSession, Tools, ChatBrowserUse
from browser_use.integrations.sentience.custom_agent import CustomSentienceAgent
from sentience import get_extension_dir
from browser_use import BrowserProfile

async def main():
    # Setup browser with Sentience extension
    sentience_ext_path = get_extension_dir()
    browser_profile = BrowserProfile(
        args=[f"--load-extension={sentience_ext_path}"]
    )
    browser_session = BrowserSession(browser_profile=browser_profile)
    await browser_session.start()
    
    # Initialize custom agent
    llm = ChatBrowserUse()
    tools = Tools()  # Use default tools
    
    agent = CustomSentienceAgent(
        task="Find the number 1 post on Show HN",
        llm=llm,
        browser_session=browser_session,
        tools=tools,
        # Sentience configuration
        sentience_api_key=os.getenv("SENTIENCE_API_KEY"),
        sentience_max_elements=60,
        sentience_show_overlay=True,
        # Vision fallback
        vision_fallback_enabled=True,
        vision_detail_level='auto',
        # Token tracking
        calculate_cost=True,
        # Agent settings
        max_steps=100,
        use_vision=False,  # Prefer Sentience over vision
    )
    
    # Run agent
    history = await agent.run()
    
    # Get token usage
    usage_summary = await agent.token_cost_service.get_usage_summary()
    print(f"Token usage: {usage_summary}")
    
    # Check if Sentience was used
    sentience_used = history.metadata.get('sentience_used', False)
    print(f"Sentience used: {sentience_used}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Benefits of This Design

### 1. Token Efficiency
- **Sentience mode**: ~3K tokens per step (60 elements × ~50 chars)
- **Vision mode**: ~40K tokens per step (full DOM + screenshots)
- **Savings**: ~92% token reduction when Sentience is available

### 2. Reliability
- **Automatic fallback**: No manual intervention needed
- **Graceful degradation**: Works even if extension fails
- **Error handling**: Robust retry logic for snapshots

### 3. Flexibility
- **Full control**: Customize prompt construction
- **Configurable**: Adjust Sentience and vision settings
- **Extensible**: Easy to add new prompt strategies

### 4. Integration
- **Reuses existing components**: Browser session, tools, token tracking
- **SDK compatibility**: Uses official Sentience SDK interfaces
- **Backward compatible**: Can coexist with existing Agent class

## Challenges and Mitigations

### Challenge 1: Extension Loading Timing
**Issue**: Extension may not be ready when agent starts.

**Mitigation**:
- `SentienceContext.build()` already handles extension waiting
- Can increase `wait_for_extension_ms` parameter
- Fallback to vision if extension never loads

### Challenge 2: Snapshot Failures
**Issue**: Snapshot may fail due to network, timeout, or extension issues.

**Mitigation**:
- Automatic fallback to vision mode
- Retry logic in `SentienceContext.build()`
- Configurable retry count and delays

### Challenge 3: Token Tracking Accuracy
**Issue**: Need to track tokens for both Sentience and vision modes.

**Mitigation**:
- `TokenCostService` automatically tracks all LLM calls
- No manual token counting needed
- Usage summaries include both modes

### Challenge 4: Message Format Consistency
**Issue**: Sentience and vision messages have different formats.

**Mitigation**:
- Use consistent message structure (agent_history + browser_state)
- LLM adapts to different browser_state formats
- Can add format indicators if needed

## Testing Strategy

### Unit Tests
- Prompt construction logic
- Fallback decision logic
- Message formatting

### Integration Tests
- Full agent loop with Sentience
- Full agent loop with vision fallback
- Token usage tracking
- Extension loading scenarios

### Performance Tests
- Token usage comparison (Sentience vs. vision)
- Latency comparison
- Memory usage

## Future Enhancements

1. **Hybrid Mode**: Use both Sentience and vision (Sentience for structure, vision for visual confirmation)
2. **Adaptive Selection**: Automatically choose best mode based on page type
3. **Caching**: Cache Sentience snapshots to reduce API calls
4. **Streaming**: Stream snapshot results as they become available
5. **Multi-page**: Handle multiple pages/tabs with different strategies

## Conclusion

This design provides a clean, flexible architecture for implementing a custom agent with full control over prompt construction. The Sentience-first, vision-fallback strategy maximizes token efficiency while maintaining reliability. Integration with existing browser-use components minimizes code duplication and leverages proven functionality.

The implementation can be done incrementally, starting with core functionality and adding advanced features over time. The modular design allows for easy testing and maintenance.
