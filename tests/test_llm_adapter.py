"""Azure OpenAI adapter: message/tool translation and proxy wiring (no network calls)."""
import json

from src.llm.azure_openai_adapter import proxied_http_client, to_openai_messages, to_openai_tools
from src.llm.gateway import ToolSpec
from src.utils.config import get_settings


def test_message_translation():
    out = to_openai_messages("SYS", [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "name": "get_my_reservation", "arguments": {"confirmation_code": "DLRAV4"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "get_my_reservation", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Done."},
    ])
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[2]["tool_calls"][0]["function"] == {
        "name": "get_my_reservation", "arguments": json.dumps({"confirmation_code": "DLRAV4"})}
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'}
    assert out[4] == {"role": "assistant", "content": "Done."}


def test_tool_translation():
    (tool,) = to_openai_tools([ToolSpec(name="t", description="d", input_schema={"type": "object"})])
    assert tool == {"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object"}}}


def test_proxy_only_when_configured():
    settings = get_settings()
    assert proxied_http_client(settings.model_copy(update={"http_proxy": "", "https_proxy": ""})) is None
    client = proxied_http_client(settings.model_copy(update={"https_proxy": "http://proxy.example.com:912"}))
    assert client is not None
    # NO_PROXY from the environment may add host-specific mounts; ours are the scheme-wide ones.
    assert {"http://", "https://"} <= {p.pattern for p in client._mounts}
