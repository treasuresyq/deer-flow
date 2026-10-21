"""Unit tests for the Serply community web search tool."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def reset_api_key_warned():
    """Reset the module-level warning flag before each test."""
    import deerflow.community.serply.tools as serply_mod

    serply_mod._api_key_warned = set()
    yield
    serply_mod._api_key_warned = set()


def _patch_config(extra: dict | None):
    """Patch get_app_config so web_search resolves to a tool config with ``extra``."""
    patcher = patch("deerflow.community.serply.tools.get_app_config")
    mock = patcher.start()
    if extra is None:
        mock.return_value.get_tool_config.return_value = None
    else:
        tool_config = MagicMock()
        tool_config.model_extra = extra
        mock.return_value.get_tool_config.return_value = tool_config
    return patcher, mock


@pytest.fixture
def mock_config_with_key():
    patcher, mock = _patch_config({"api_key": "test-serply-key", "max_results": 5})
    yield mock
    patcher.stop()


@pytest.fixture
def mock_config_no_key():
    patcher, mock = _patch_config({})
    yield mock
    patcher.stop()


def _make_response(payload: object) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _search_rows(n: int) -> list[dict]:
    return [{"title": f"Result {i}", "link": f"https://example.com/{i}", "description": f"Snippet {i}", "position": i} for i in range(1, n + 1)]


def _patch_client(mock_resp: MagicMock):
    """Patch httpx.AsyncClient so the context-managed ``get`` returns ``mock_resp``."""
    patcher = patch("deerflow.community.serply.tools.httpx.AsyncClient")
    mock_client_cls = patcher.start()
    mock_client_cls.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_resp)
    return patcher, mock_client_cls


def _get_call(mock_client_cls):
    return mock_client_cls.return_value.__aenter__.return_value.get.call_args


async def _run(query: str = "test query", max_results: int = 5) -> dict:
    from deerflow.community.serply.tools import web_search_tool

    return json.loads(await web_search_tool.ainvoke({"query": query, "max_results": max_results}))


class TestResolveToolSettings:
    def test_returns_config_key_when_present(self):
        patcher, _ = _patch_config({"api_key": "from-config"})
        try:
            from deerflow.community.serply.tools import _resolve_tool_settings

            assert _resolve_tool_settings("web_search")[0] == "from-config"
        finally:
            patcher.stop()

    def test_falls_back_to_env_when_config_key_blank(self):
        patcher, _ = _patch_config({"api_key": "   "})
        try:
            with patch.dict("os.environ", {"SERPLY_API_KEY": "env-key"}):
                from deerflow.community.serply.tools import _resolve_tool_settings

                assert _resolve_tool_settings("web_search")[0] == "env-key"
        finally:
            patcher.stop()

    def test_falls_back_to_env_when_no_config(self):
        patcher, _ = _patch_config(None)
        try:
            with patch.dict("os.environ", {"SERPLY_API_KEY": "env-only"}):
                from deerflow.community.serply.tools import _resolve_tool_settings

                assert _resolve_tool_settings("web_search")[0] == "env-only"
        finally:
            patcher.stop()

    def test_returns_none_when_no_key_anywhere(self):
        patcher, _ = _patch_config(None)
        try:
            with patch.dict("os.environ", {}, clear=True):
                from deerflow.community.serply.tools import _resolve_tool_settings

                assert _resolve_tool_settings("web_search")[0] is None
        finally:
            patcher.stop()


class TestCoerceMaxResults:
    def test_returns_value_when_valid(self):
        from deerflow.community.serply.tools import _coerce_max_results

        assert _coerce_max_results(3) == 3
        assert _coerce_max_results("7") == 7

    def test_caps_at_serply_maximum(self):
        from deerflow.community.serply.tools import _coerce_max_results

        assert _coerce_max_results(999) == 100

    def test_invalid_values_fall_back_to_default(self):
        from deerflow.community.serply.tools import _coerce_max_results

        assert _coerce_max_results("oops") == 5
        assert _coerce_max_results(None) == 5
        assert _coerce_max_results(0) == 5
        assert _coerce_max_results(-3) == 5


class TestCoerceVertical:
    def test_accepts_known_verticals(self):
        from deerflow.community.serply.tools import _coerce_vertical

        assert _coerce_vertical(None) == "search"
        assert _coerce_vertical("news") == "news"
        assert _coerce_vertical(" Scholar ") == "scholar"

    def test_unknown_vertical_falls_back_to_search(self, caplog):
        from deerflow.community.serply.tools import _coerce_vertical

        with caplog.at_level(logging.WARNING):
            assert _coerce_vertical("images") == "search"
        assert "Invalid Serply vertical" in caplog.text


class TestWebSearchTool:
    async def test_basic_search_returns_normalized_results(self, mock_config_with_key):
        patcher, _ = _patch_client(_make_response({"results": _search_rows(2)}))
        try:
            result = await _run("test query")
        finally:
            patcher.stop()

        assert result["query"] == "test query"
        assert result["total_results"] == 2
        assert result["results"][0] == {"title": "Result 1", "url": "https://example.com/1", "content": "Snippet 1"}

    async def test_sends_correct_headers_and_params(self, mock_config_with_key):
        patcher, mock_client_cls = _patch_client(_make_response({"results": _search_rows(1)}))
        try:
            await _run("  padded query  ")
            call = _get_call(mock_client_cls)
        finally:
            patcher.stop()

        assert call.args[0] == "https://api.serply.io/v1/search/"
        assert call.kwargs["headers"]["X-Api-Key"] == "test-serply-key"
        assert call.kwargs["headers"]["Accept"] == "application/json"
        assert call.kwargs["params"] == {"q": "padded query", "num": 5}

    async def test_config_max_results_overrides_parameter(self):
        patcher, _ = _patch_config({"api_key": "k", "max_results": 2})
        try:
            client_patcher, mock_client_cls = _patch_client(_make_response({"results": _search_rows(5)}))
            try:
                result = await _run("q", max_results=5)
            finally:
                client_patcher.stop()
        finally:
            patcher.stop()

        assert _get_call(mock_client_cls).kwargs["params"]["num"] == 2
        assert result["total_results"] == 2

    async def test_passes_through_locale_params_from_config(self):
        patcher, _ = _patch_config({"api_key": "k", "gl": "fr", "hl": "fr", "ignored": "x"})
        try:
            client_patcher, mock_client_cls = _patch_client(_make_response({"results": _search_rows(1)}))
            try:
                await _run("q")
            finally:
                client_patcher.stop()
        finally:
            patcher.stop()

        params = _get_call(mock_client_cls).kwargs["params"]
        assert params["gl"] == "fr"
        assert params["hl"] == "fr"
        assert "ignored" not in params

    async def test_news_vertical_uses_news_endpoint_and_trims_client_side(self):
        entries = [
            {
                "title": f"Story {i}",
                "link": f"https://news.example.com/{i}",
                "summary": "<a href='x'>Genuine&nbsp;attention</a> &amp; chatbots",
                "published": "Mon, 01 Jun 2026 08:00:00 GMT",
                "source": {"title": "Example Times"},
            }
            for i in range(10)
        ]
        patcher, _ = _patch_config({"api_key": "k", "max_results": 3, "vertical": "news"})
        try:
            client_patcher, mock_client_cls = _patch_client(_make_response({"entries": entries}))
            try:
                result = await _run("q")
            finally:
                client_patcher.stop()
        finally:
            patcher.stop()

        assert _get_call(mock_client_cls).args[0] == "https://api.serply.io/v1/news/"
        assert result["total_results"] == 3
        first = result["results"][0]
        assert first["content"] == "Genuine\xa0attention & chatbots"
        assert first["published"] == "Mon, 01 Jun 2026 08:00:00 GMT"
        assert first["source"] == "Example Times"

    async def test_scholar_vertical_maps_authors_and_citations(self):
        articles = [
            {
                "title": "Attention Is All You Need",
                "link": "https://arxiv.org/abs/1706.03762",
                "description": "The dominant sequence transduction models...",
                "author": {
                    "names": "A Vaswani, N Shazeer - NeurIPS, 2017",
                    "authors": [{"name": "A Vaswani", "link": "https://openalex.org/A1"}, {"name": "N Shazeer", "link": "https://openalex.org/A2"}],
                },
                "extras": {"citations": {"count": 120000}},
                "doc": {"link": "https://arxiv.org/pdf/1706.03762", "type": "PDF"},
            },
            {"title": "No metadata", "link": "https://example.org/paper"},
        ]
        patcher, _ = _patch_config({"api_key": "k", "vertical": "scholar"})
        try:
            client_patcher, mock_client_cls = _patch_client(_make_response({"articles": articles}))
            try:
                result = await _run("transformers")
            finally:
                client_patcher.stop()
        finally:
            patcher.stop()

        assert _get_call(mock_client_cls).args[0] == "https://api.serply.io/v1/scholar/"
        assert result["results"][0]["authors"] == ["A Vaswani", "N Shazeer"]
        assert result["results"][0]["cited_by"] == 120000
        assert result["results"][0]["pdf_url"] == "https://arxiv.org/pdf/1706.03762"
        assert result["results"][1] == {"title": "No metadata", "url": "https://example.org/paper", "content": "", "authors": [], "cited_by": 0, "pdf_url": ""}

    async def test_empty_results_returns_error_json(self, mock_config_with_key):
        patcher, _ = _patch_client(_make_response({"results": []}))
        try:
            assert await _run("nothing") == {"error": "No results found", "query": "nothing"}
        finally:
            patcher.stop()

    async def test_missing_results_key_is_treated_as_no_results(self, mock_config_with_key):
        patcher, _ = _patch_client(_make_response({"results": None}))
        try:
            assert (await _run("nothing"))["error"] == "No results found"
        finally:
            patcher.stop()

    async def test_unexpected_payload_shape_returns_error_json(self, mock_config_with_key):
        patcher, _ = _patch_client(_make_response({"results": "not-a-list"}))
        try:
            assert "unexpected response format" in (await _run("q"))["error"]
        finally:
            patcher.stop()

        patcher, _ = _patch_client(_make_response(["not", "a", "dict"]))
        try:
            assert "unexpected response format" in (await _run("q"))["error"]
        finally:
            patcher.stop()

    @pytest.mark.parametrize("rows", [["junk", "more junk"], [None, None], [1, 2], [[]], [False]])
    async def test_rows_without_any_objects_return_format_error(self, mock_config_with_key, rows, caplog):
        """A non-empty rows container holding no objects is malformed, not an empty search.

        Reporting "No results found" for it would tell the agent the query
        legitimately matched nothing, hiding a broken upstream payload.
        """
        patcher, _ = _patch_client(_make_response({"results": rows}))
        try:
            with caplog.at_level(logging.ERROR):
                result = await _run("q")
        finally:
            patcher.stop()

        assert "unexpected response format" in result["error"]
        assert any("no usable result objects" in r.getMessage() for r in caplog.records if r.levelno == logging.ERROR)

    async def test_mixed_rows_keep_the_object_entries(self, mock_config_with_key):
        rows = ["junk", {"title": "Kept", "link": "https://example.com/kept", "description": "Body"}, 7]

        patcher, _ = _patch_client(_make_response({"results": rows}))
        try:
            result = await _run("q")
        finally:
            patcher.stop()

        assert result["total_results"] == 1
        assert result["results"][0] == {"title": "Kept", "url": "https://example.com/kept", "content": "Body"}

    @pytest.mark.parametrize(
        ("vertical", "rows_key"),
        [("news", "entries"), ("scholar", "articles")],
    )
    async def test_vertical_rows_without_any_objects_return_format_error(self, vertical, rows_key):
        """The news and scholar verticals share the helper, so they contain malformed rows too."""
        patcher, _ = _patch_config({"api_key": "test-serply-key", "max_results": 5, "vertical": vertical})
        try:
            client_patcher, _ = _patch_client(_make_response({rows_key: ["junk", 3]}))
            try:
                result = await _run("q")
            finally:
                client_patcher.stop()
        finally:
            patcher.stop()

        assert "unexpected response format" in result["error"]

    @pytest.mark.parametrize(
        ("vertical", "rows_key"),
        [("news", "entries"), ("scholar", "articles")],
    )
    async def test_vertical_missing_and_empty_rows_keep_no_results(self, vertical, rows_key):
        """A missing or empty vertical container stays a legitimate empty search."""
        patcher, _ = _patch_config({"api_key": "test-serply-key", "max_results": 5, "vertical": vertical})
        try:
            client_patcher, _ = _patch_client(_make_response({rows_key: []}))
            try:
                assert (await _run("nothing"))["error"] == "No results found"
            finally:
                client_patcher.stop()

            client_patcher, _ = _patch_client(_make_response({}))
            try:
                assert (await _run("nothing"))["error"] == "No results found"
            finally:
                client_patcher.stop()
        finally:
            patcher.stop()

    async def test_missing_api_key_returns_error_json_and_warns_once(self, mock_config_no_key, caplog):
        with patch.dict("os.environ", {}, clear=True), caplog.at_level(logging.WARNING):
            first = await _run("q")
            second = await _run("q")

        assert first == {"error": "SERPLY_API_KEY is not configured", "query": "q"}
        assert second == first
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "SERPLY_API_KEY" in warnings[0].getMessage()

    async def test_http_error_returns_structured_error(self, mock_config_with_key):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("403", request=MagicMock(), response=mock_resp)
        patcher, _ = _patch_client(mock_resp)
        try:
            result = await _run("q")
        finally:
            patcher.stop()

        assert result == {"error": "Serply API error: HTTP 403", "query": "q"}

    async def test_network_exception_returns_error_json(self, mock_config_with_key):
        patcher, mock_client_cls = _patch_client(MagicMock())
        mock_client_cls.return_value.__aenter__.return_value.get.side_effect = httpx.ConnectError("boom")
        try:
            result = await _run("q")
        finally:
            patcher.stop()

        assert result == {"error": "boom", "query": "q"}

    async def test_long_query_is_truncated(self, mock_config_with_key):
        patcher, mock_client_cls = _patch_client(_make_response({"results": _search_rows(1)}))
        try:
            await _run("x" * 600)
        finally:
            patcher.stop()

        assert len(_get_call(mock_client_cls).kwargs["params"]["q"]) == 500

    async def test_uses_env_key_when_config_absent(self):
        patcher, _ = _patch_config(None)
        try:
            with patch.dict("os.environ", {"SERPLY_API_KEY": "env-key"}):
                client_patcher, mock_client_cls = _patch_client(_make_response({"results": _search_rows(1)}))
                try:
                    result = await _run("q")
                finally:
                    client_patcher.stop()
        finally:
            patcher.stop()

        assert result["total_results"] == 1
        assert _get_call(mock_client_cls).kwargs["headers"]["X-Api-Key"] == "env-key"


def test_coerce_max_results_inf_falls_back_to_default():
    """A YAML `.inf` max_results must fall back to the default, not crash."""
    import deerflow.community.serply.tools as serply_mod

    assert serply_mod._coerce_max_results(float("inf")) == serply_mod._DEFAULT_MAX_RESULTS
