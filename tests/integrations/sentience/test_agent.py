"""Unit tests for SentienceAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from browser_use.integrations.sentience.agent import (
    SentienceAgent,
    SentienceAgentConfig,
    SentienceAgentSettings,
    VisionFallbackConfig,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="test response"))
    llm.model = "test-model"
    llm.provider = "test-provider"
    return llm


@pytest.fixture
def mock_browser_session():
    """Create a mock browser session."""
    session = MagicMock()
    session.is_connected.return_value = True
    session.get_browser_state_summary = AsyncMock(
        return_value=MagicMock(
            url="https://example.com",
            screenshot=None,
            page_info=MagicMock(title="Test Page"),
            dom_state=MagicMock(selector_map={}),
        )
    )
    session.get_current_page_url = AsyncMock(return_value="https://example.com")
    return session


@pytest.fixture
def mock_tools():
    """Create a mock tools registry."""
    return MagicMock()


class TestSentienceAgentConfig:
    """Test SentienceAgentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SentienceAgentConfig()
        assert config.sentience_api_key is None
        assert config.sentience_use_api is None
        assert config.sentience_max_elements == 60
        assert config.sentience_show_overlay is False
        assert config.sentience_wait_for_extension_ms == 5000
        assert config.sentience_retries == 2
        assert config.sentience_retry_delay_s == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SentienceAgentConfig(
            sentience_api_key="test-key",
            sentience_max_elements=100,
            sentience_show_overlay=True,
        )
        assert config.sentience_api_key == "test-key"
        assert config.sentience_max_elements == 100
        assert config.sentience_show_overlay is True


class TestVisionFallbackConfig:
    """Test VisionFallbackConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VisionFallbackConfig()
        assert config.enabled is True
        assert config.detail_level == "auto"
        assert config.include_screenshots is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = VisionFallbackConfig(
            enabled=False,
            detail_level="high",
            include_screenshots=False,
        )
        assert config.enabled is False
        assert config.detail_level == "high"
        assert config.include_screenshots is False


class TestSentienceAgentSettings:
    """Test SentienceAgentSettings Pydantic model."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = SentienceAgentSettings(task="test task")
        assert settings.task == "test task"
        assert settings.max_steps == 100
        assert settings.max_failures == 3
        assert settings.calculate_cost is True
        assert isinstance(settings.sentience_config, SentienceAgentConfig)
        assert isinstance(settings.vision_fallback, VisionFallbackConfig)

    def test_custom_settings(self):
        """Test custom settings values."""
        settings = SentienceAgentSettings(
            task="custom task",
            max_steps=50,
            max_failures=5,
            calculate_cost=False,
        )
        assert settings.task == "custom task"
        assert settings.max_steps == 50
        assert settings.max_failures == 5
        assert settings.calculate_cost is False


class TestSentienceAgent:
    """Test SentienceAgent class."""

    def test_init(self, mock_llm, mock_browser_session, mock_tools):
        """Test agent initialization."""
        agent = SentienceAgent(
            task="test task",
            llm=mock_llm,
            browser_session=mock_browser_session,
            tools=mock_tools,
        )
        assert agent.task == "test task"
        assert agent.llm == mock_llm
        assert agent.browser_session == mock_browser_session
        assert agent.tools == mock_tools
        assert agent._current_step == 0
        assert agent._consecutive_failures == 0

    def test_init_with_custom_config(self, mock_llm, mock_browser_session):
        """Test agent initialization with custom configuration."""
        agent = SentienceAgent(
            task="test task",
            llm=mock_llm,
            browser_session=mock_browser_session,
            sentience_max_elements=100,
            vision_fallback_enabled=False,
        )
        assert agent.settings.sentience_config.sentience_max_elements == 100
        assert agent.settings.vision_fallback.enabled is False

    def test_get_sentience_context_success(self, mock_llm, mock_browser_session):
        """Test getting SentienceContext when SDK is available."""
        with patch("browser_use.integrations.sentience.agent.SentienceContext") as mock_context:
            agent = SentienceAgent(
                task="test task",
                llm=mock_llm,
                browser_session=mock_browser_session,
            )
            context = agent._get_sentience_context()
            assert context is not None
            mock_context.assert_called_once()

    def test_get_sentience_context_import_error(self, mock_llm, mock_browser_session):
        """Test getting SentienceContext when SDK is not available."""
        with patch(
            "browser_use.integrations.sentience.agent.SentienceContext",
            side_effect=ImportError("No module named 'sentience'"),
        ):
            agent = SentienceAgent(
                task="test task",
                llm=mock_llm,
                browser_session=mock_browser_session,
            )
            with pytest.raises(ImportError, match="Sentience SDK is required"):
                agent._get_sentience_context()

    @pytest.mark.asyncio
    async def test_try_sentience_snapshot_success(
        self, mock_llm, mock_browser_session
    ):
        """Test successful Sentience snapshot."""
        mock_state = MagicMock()
        mock_state.prompt_block = "test prompt block"

        with patch.object(
            SentienceAgent, "_get_sentience_context", return_value=MagicMock()
        ) as mock_get_context:
            mock_context = MagicMock()
            mock_context.build = AsyncMock(return_value=mock_state)
            mock_get_context.return_value = mock_context

            agent = SentienceAgent(
                task="test task",
                llm=mock_llm,
                browser_session=mock_browser_session,
            )
            result = await agent._try_sentience_snapshot()

            assert result == mock_state
            mock_context.build.assert_called_once()

    @pytest.mark.asyncio
    async def test_try_sentience_snapshot_failure(
        self, mock_llm, mock_browser_session
    ):
        """Test failed Sentience snapshot."""
        with patch.object(
            SentienceAgent, "_get_sentience_context", return_value=MagicMock()
        ) as mock_get_context:
            mock_context = MagicMock()
            mock_context.build = AsyncMock(return_value=None)
            mock_get_context.return_value = mock_context

            agent = SentienceAgent(
                task="test task",
                llm=mock_llm,
                browser_session=mock_browser_session,
            )
            result = await agent._try_sentience_snapshot()

            assert result is None

    @pytest.mark.asyncio
    async def test_build_sentience_message(self, mock_llm, mock_browser_session):
        """Test building message with Sentience prompt block."""
        mock_state = MagicMock()
        mock_state.prompt_block = "Elements: ID|role|text|...\n1|button|Click|...\n"

        agent = SentienceAgent(
            task="test task",
            llm=mock_llm,
            browser_session=mock_browser_session,
        )
        message = agent._build_sentience_message(mock_state)

        assert isinstance(message.content, str)
        assert "agent_history" in message.content
        assert "browser_state" in message.content
        assert mock_state.prompt_block in message.content

    @pytest.mark.asyncio
    async def test_build_vision_message_without_screenshot(
        self, mock_llm, mock_browser_session
    ):
        """Test building vision message without screenshot."""
        agent = SentienceAgent(
            task="test task",
            llm=mock_llm,
            browser_session=mock_browser_session,
            vision_include_screenshots=False,
        )
        message = await agent._build_vision_message()

        assert isinstance(message.content, str)
        assert "agent_history" in message.content
        assert "browser_state" in message.content

    @pytest.mark.asyncio
    async def test_build_vision_message_with_screenshot(
        self, mock_llm, mock_browser_session
    ):
        """Test building vision message with screenshot."""
        mock_browser_session.get_browser_state_summary.return_value = MagicMock(
            url="https://example.com",
            screenshot="base64_screenshot_data",
            page_info=MagicMock(title="Test Page"),
            dom_state=MagicMock(selector_map={}),
        )

        agent = SentienceAgent(
            task="test task",
            llm=mock_llm,
            browser_session=mock_browser_session,
            vision_include_screenshots=True,
        )
        message = await agent._build_vision_message()

        # Should be a list of content parts when screenshot is included
        assert isinstance(message.content, list)
        assert len(message.content) == 3  # text, label, image

    @pytest.mark.asyncio
    async def test_prepare_context_sentience_success(
        self, mock_llm, mock_browser_session
    ):
        """Test context preparation with successful Sentience snapshot."""
        mock_state = MagicMock()
        mock_state.prompt_block = "test prompt block"

        with patch.object(
            SentienceAgent, "_try_sentience_snapshot", return_value=mock_state
        ):
            agent = SentienceAgent(
                task="test task",
                llm=mock_llm,
                browser_session=mock_browser_session,
            )
            message, sentience_used = await agent._prepare_context()

            assert sentience_used is True
            assert isinstance(message.content, str)
            assert agent._sentience_used_in_last_step is True

    @pytest.mark.asyncio
    async def test_prepare_context_vision_fallback(
        self, mock_llm, mock_browser_session
    ):
        """Test context preparation with vision fallback."""
        with patch.object(
            SentienceAgent, "_try_sentience_snapshot", return_value=None
        ):
            agent = SentienceAgent(
                task="test task",
                llm=mock_llm,
                browser_session=mock_browser_session,
                vision_fallback_enabled=True,
            )
            message, sentience_used = await agent._prepare_context()

            assert sentience_used is False
            assert agent._sentience_used_in_last_step is False

    @pytest.mark.asyncio
    async def test_prepare_context_no_fallback(
        self, mock_llm, mock_browser_session
    ):
        """Test context preparation without fallback."""
        with patch.object(
            SentienceAgent, "_try_sentience_snapshot", return_value=None
        ):
            agent = SentienceAgent(
                task="test task",
                llm=mock_llm,
                browser_session=mock_browser_session,
                vision_fallback_enabled=False,
            )
            message, sentience_used = await agent._prepare_context()

            assert sentience_used is False
            assert isinstance(message.content, str)
            assert "agent_history" in message.content

    def test_get_agent_history_description(self, mock_llm, mock_browser_session):
        """Test agent history description generation."""
        agent = SentienceAgent(
            task="test task",
            llm=mock_llm,
            browser_session=mock_browser_session,
        )
        agent._current_step = 0
        history = agent._get_agent_history_description()
        assert "test task" in history
        assert "Step: 1" in history

    def test_build_dom_state(self, mock_llm, mock_browser_session):
        """Test DOM state building."""
        mock_browser_state = MagicMock()
        mock_browser_state.url = "https://example.com"
        mock_browser_state.page_info = MagicMock(title="Test Page")
        mock_browser_state.dom_state = MagicMock(selector_map={"1": "button"})

        agent = SentienceAgent(
            task="test task",
            llm=mock_llm,
            browser_session=mock_browser_session,
        )
        dom_state = agent._build_dom_state(mock_browser_state)

        assert "https://example.com" in dom_state
        assert "Test Page" in dom_state
        assert "Interactive elements: 1" in dom_state
