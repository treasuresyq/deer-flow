"""Unit tests for the Unbrowse community web fetch tool."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def reset_api_key_warned():
    """Reset the module-level warning flag before each test."""
    import deerflow.community.unbrowse.tools as unbrowse_mod

    unbrowse_mod._api_key_warned = set()
    yield
    unbrowse_mod._api_key_warned = set()


@pytest.fixture
def mock_config_with_key():
    with patch("deerflow.community.unbrowse.tools.get_app_config") as mock:
        tool_config = MagicMock()
        tool_config.model_extra = {"api_key": "test-unbrowse-key"}
        mock.return_value.get_tool_config.return_value = tool_config
        yield mock


@pytest.fixture
def mock_config_no_key():
    with patch("deerflow.community.unbrowse.tools.get_app_config") as mock:
        tool_config = MagicMock()
        tool_config.model_extra = {}
        mock.return_value.get_tool_config.return_value = tool_config
        yield mock


def _make_response(payload: object) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _make_scrape_response(scrape: dict, *, structured: bool = False) -> MagicMock:
    """Build a JSON-RPC tools/call response the way Unbrowse's MCP endpoint returns it."""
    result: dict = {"content": [{"type": "text", "text": json.dumps(scrape)}], "isError": False}
    if structured:
        result["structuredContent"] = scrape
    return _make_response({"jsonrpc": "2.0", "id": 1, "result": result})


def _scrape(markdown: object = "# Example Domain\n\nBody", title: object = "Example Domain") -> dict:
    return {
        "url": "https://example.com",
        "finalUrl": "https://example.com/",
        "via": "http",
        "metadata": {"title": title, "sourceURL": "https://example.com/", "statusCode": 200},
        "markdown": markdown,
    }


def _patch_client(mock_resp: MagicMock):
    """Patch httpx.AsyncClient so the context-managed ``post`` returns ``mock_resp``."""
    patcher = patch("deerflow.community.unbrowse.tools.httpx.AsyncClient")
    mock_client_cls = patcher.start()
    mock_client_cls.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)
    return patcher, mock_client_cls


def _post_call(mock_client_cls):
    return mock_client_cls.return_value.__aenter__.return_value.post.call_args


async def _fetch(url: str = "https://example.com") -> str:
    from deerflow.community.unbrowse.tools import web_fetch_tool

    return await web_fetch_tool.ainvoke({"url": url})


class TestGetApiKey:
    def test_returns_config_key_when_present(self):
        with patch("deerflow.community.unbrowse.tools.get_app_config") as mock:
            tool_config = MagicMock()
            tool_config.model_extra = {"api_key": "from-config"}
            mock.return_value.get_tool_config.return_value = tool_config

            from deerflow.community.unbrowse.tools import _get_api_key

            assert _get_api_key("web_fetch") == "from-config"

    def test_falls_back_to_env_when_config_key_whitespace(self):
        with patch("deerflow.community.unbrowse.tools.get_app_config") as mock:
            tool_config = MagicMock()
            tool_config.model_extra = {"api_key": "   "}
            mock.return_value.get_tool_config.return_value = tool_config
            with patch.dict("os.environ", {"UNBROWSE_API_KEY": "env-key"}):
                from deerflow.community.unbrowse.tools import _get_api_key

                assert _get_api_key("web_fetch") == "env-key"

    def test_uses_env_when_tool_is_not_configured(self):
        with patch("deerflow.community.unbrowse.tools.get_app_config") as mock:
            mock.return_value.get_tool_config.return_value = None
            with patch.dict("os.environ", {"UNBROWSE_API_KEY": "env-only"}):
                from deerflow.community.unbrowse.tools import _get_api_key

                assert _get_api_key("web_fetch") == "env-only"

    def test_returns_none_when_no_key_anywhere(self):
        with patch("deerflow.community.unbrowse.tools.get_app_config") as mock:
            mock.return_value.get_tool_config.return_value = None
            with patch.dict("os.environ", {}, clear=True):
                from deerflow.community.unbrowse.tools import _get_api_key

                assert _get_api_key("web_fetch") is None

    def test_returns_none_when_env_key_whitespace(self):
        with patch("deerflow.community.unbrowse.tools.get_app_config") as mock:
            mock.return_value.get_tool_config.return_value = None
            with patch.dict("os.environ", {"UNBROWSE_API_KEY": "  "}, clear=True):
                from deerflow.community.unbrowse.tools import _get_api_key

                assert _get_api_key("web_fetch") is None


class TestResolveRender:
    def test_defaults_to_auto(self):
        from deerflow.community.unbrowse.tools import _resolve_render

        assert _resolve_render(None) == "auto"

    def test_normalizes_supported_modes(self):
        from deerflow.community.unbrowse.tools import _resolve_render

        assert _resolve_render(" Never ") == "never"
        assert _resolve_render("always") == "always"

    def test_unsupported_mode_falls_back_with_warning(self, caplog):
        from deerflow.community.unbrowse.tools import _resolve_render

        with caplog.at_level(logging.WARNING, logger="deerflow.community.unbrowse.tools"):
            assert _resolve_render("headless") == "auto"

        assert "headless" in caplog.text


class TestMissingKeyMessage:
    def test_warns_once_per_tool_name(self, caplog):
        from deerflow.community.unbrowse.tools import _missing_key_message

        with caplog.at_level(logging.WARNING, logger="deerflow.community.unbrowse.tools"):
            assert _missing_key_message("web_fetch") == "UNBROWSE_API_KEY is not configured"
            assert _missing_key_message("web_fetch") == "UNBROWSE_API_KEY is not configured"

        assert sum("Unbrowse API key is not set" in r.message for r in caplog.records) == 1


class TestWebFetchTool:
    async def test_returns_title_and_markdown(self, mock_config_with_key):
        patcher, _ = _patch_client(_make_scrape_response(_scrape()))
        try:
            result = await _fetch()
        finally:
            patcher.stop()

        assert result == "# Example Domain\n\n# Example Domain\n\nBody"

    async def test_sends_one_json_rpc_tools_call(self, mock_config_with_key):
        patcher, mock_client_cls = _patch_client(_make_scrape_response(_scrape()))
        try:
            await _fetch()
            call = _post_call(mock_client_cls)
        finally:
            patcher.stop()

        assert call.args[0] == "https://unbrowse.ai/api/mcp"
        assert call.kwargs["headers"]["Authorization"] == "Bearer test-unbrowse-key"
        assert call.kwargs["json"] == {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "unbrowse.scrape",
                "arguments": {"url": "https://example.com", "formats": ["markdown"], "render": "auto"},
            },
        }

    async def test_render_can_be_set_from_config(self, mock_config_with_key):
        mock_config_with_key.return_value.get_tool_config.return_value.model_extra = {"api_key": "k", "render": "never"}

        patcher, mock_client_cls = _patch_client(_make_scrape_response(_scrape()))
        try:
            await _fetch()
            call = _post_call(mock_client_cls)
        finally:
            patcher.stop()

        assert call.kwargs["json"]["params"]["arguments"]["render"] == "never"

    async def test_prefers_structured_content(self, mock_config_with_key):
        response = _make_scrape_response(_scrape(markdown="from text"), structured=True)
        response.json.return_value["result"]["structuredContent"] = _scrape(markdown="from structured")

        patcher, _ = _patch_client(response)
        try:
            result = await _fetch()
        finally:
            patcher.stop()

        assert result == "# Example Domain\n\nfrom structured"

    async def test_truncates_long_content(self, mock_config_with_key):
        patcher, _ = _patch_client(_make_scrape_response(_scrape(markdown="x" * 9000, title="Long")))
        try:
            result = await _fetch()
        finally:
            patcher.stop()

        assert len(result) == len("# Long\n\n") + 4096

    async def test_non_string_markdown_does_not_raise(self, mock_config_with_key):
        patcher, _ = _patch_client(_make_scrape_response(_scrape(markdown=12345, title="Numeric")))
        try:
            result = await _fetch()
        finally:
            patcher.stop()

        assert result == "# Numeric\n\n12345"

    async def test_falls_back_to_untitled(self, mock_config_with_key):
        scrape = _scrape(markdown="Body")
        scrape["metadata"] = None

        patcher, _ = _patch_client(_make_scrape_response(scrape))
        try:
            result = await _fetch()
        finally:
            patcher.stop()

        assert result == "# Untitled\n\nBody"

    @pytest.mark.parametrize("markdown", ["", None])
    async def test_empty_markdown_returns_error(self, mock_config_with_key, markdown):
        patcher, _ = _patch_client(_make_scrape_response(_scrape(markdown=markdown)))
        try:
            result = await _fetch()
        finally:
            patcher.stop()

        assert result == "Error: No content found"

    async def test_json_rpc_error_returns_its_message(self, mock_config_with_key):
        payload = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "page.goto: net::ERR_NAME_NOT_RESOLVED", "data": None}}

        patcher, _ = _patch_client(_make_response(payload))
        try:
            result = await _fetch("https://nope.invalid")
        finally:
            patcher.stop()

        assert result == "Error: Unbrowse error: page.goto: net::ERR_NAME_NOT_RESOLVED"

    async def test_tool_error_result_returns_its_text(self, mock_config_with_key):
        payload = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "render failed"}], "isError": True}}

        patcher, _ = _patch_client(_make_response(payload))
        try:
            result = await _fetch()
        finally:
            patcher.stop()

        assert result == "Error: Unbrowse error: render failed"

    @pytest.mark.parametrize(
        "payload",
        [
            ["not", "a", "dict"],
            {"jsonrpc": "2.0", "id": 1, "result": "oops"},
            {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "not json"}]}},
            {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "[1, 2]"}]}},
        ],
        ids=["non-dict-body", "non-dict-result", "non-json-text", "non-object-json"],
    )
    async def test_malformed_response_returns_error(self, mock_config_with_key, payload):
        patcher, _ = _patch_client(_make_response(payload))
        try:
            result = await _fetch()
        finally:
            patcher.stop()

        assert result == "Error: Unbrowse returned an unexpected response format"

    async def test_http_error_returns_error_string(self, mock_config_with_key):
        request = httpx.Request("POST", "https://unbrowse.ai/api/mcp")
        response = httpx.Response(401, text='{"error":"invalid_token"}', request=request)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("error", request=request, response=response)

        patcher, _ = _patch_client(mock_resp)
        try:
            result = await _fetch()
        finally:
            patcher.stop()

        assert result == "Error: Unbrowse API error: HTTP 401"

    async def test_network_error_returns_error_string(self, mock_config_with_key):
        patcher, mock_client_cls = _patch_client(MagicMock())
        mock_client_cls.return_value.__aenter__.return_value.post.side_effect = httpx.ConnectError("boom")
        try:
            result = await _fetch()
        finally:
            patcher.stop()

        assert result == "Error: boom"

    async def test_missing_key_returns_error_string(self, mock_config_no_key):
        with patch.dict("os.environ", {}, clear=True):
            patcher = patch("deerflow.community.unbrowse.tools.httpx.AsyncClient")
            mock_client_cls = patcher.start()
            try:
                result = await _fetch()
            finally:
                patcher.stop()

        assert result == "Error: UNBROWSE_API_KEY is not configured"
        mock_client_cls.assert_not_called()
