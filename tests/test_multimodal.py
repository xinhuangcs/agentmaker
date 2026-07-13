"""Multimodal content parts: neutral helpers, per-protocol adapter translation, capability
guards, session-store round-trip, and token accounting. Hermetic except the final real-API
vision smoke (skipped without OPENAI_API_KEY)."""

import asyncio
import base64
import json
import os
import struct
import zlib

import pytest

from agentmaker import (LLMClient, LLMConfigError, LLMError, Message, Scope, SqliteSessionStore,
                          content_text, image_part_from_bytes, image_part_from_file,
                          image_part_from_url, messages_have_images, text_part)
from agentmaker.core.adapters.anthropic import AnthropicAdapter
from agentmaker.core.adapters.openai_compat import _to_openai_messages
from agentmaker.core.multimodal import IMAGE_TOKEN_ESTIMATE, content_tokens

RED_PIXEL = base64.b64encode(b"fake-png-bytes").decode("ascii")

PARTS = [text_part("what color is this?"),
         {"type": "image", "media_type": "image/png", "data": RED_PIXEL}]


def _solid_png(width: int, height: int, rgb: tuple) -> bytes:
    """Build a minimal solid-color PNG (pure stdlib; keeps the real-API smoke deterministic)."""
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)   # 8-bit RGB
    row = b"\x00" + bytes(rgb) * width                               # filter 0 + pixels
    body = zlib.compress(row * height)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", body)
            + chunk(b"IEND", b""))


def test_part_builders_and_text_flattening(tmp_path):
    """Builders validate media types; content_text flattens parts with image placeholders."""
    png = tmp_path / "shot.png"
    png.write_bytes(b"png-bytes")
    part = image_part_from_file(png)
    assert part["media_type"] == "image/png"
    assert base64.b64decode(part["data"]) == b"png-bytes"
    with pytest.raises(ValueError):
        image_part_from_bytes(b"x", "image/bmp")
    with pytest.raises(ValueError):
        image_part_from_file(tmp_path / "noext")

    flattened = content_text(PARTS)
    assert "what color is this?" in flattened and "[image: image/png]" in flattened
    assert content_text("plain") == "plain"
    assert messages_have_images([{"role": "user", "content": PARTS}])
    assert not messages_have_images([{"role": "user", "content": "hi"}])


def test_content_tokens_budgets_images_flat():
    """Token accounting: text via the counter, plus a flat estimate per image part."""
    text_only = content_tokens("hello world")
    with_image = content_tokens(PARTS)
    assert with_image >= IMAGE_TOKEN_ESTIMATE
    assert content_tokens(None) == 0
    assert text_only > 0


def test_openai_translation_builds_data_url():
    """OpenAI protocol: neutral parts become text / image_url parts (base64 -> data URL); str passes through."""
    out = _to_openai_messages([{"role": "system", "content": "sys"},
                               {"role": "user", "content": PARTS},
                               {"role": "user", "content": [image_part_from_url("https://x/i.png")]}])
    assert out[0]["content"] == "sys"
    translated = out[1]["content"]
    assert translated[0] == {"type": "text", "text": "what color is this?"}
    assert translated[1]["type"] == "image_url"
    assert translated[1]["image_url"]["url"] == f"data:image/png;base64,{RED_PIXEL}"
    assert out[2]["content"][0]["image_url"]["url"] == "https://x/i.png"


def test_anthropic_translation_builds_source_blocks():
    """Anthropic protocol: parts become content blocks (base64 / url sources); system flattens to text."""
    system, convo = AnthropicAdapter._to_anthropic([
        {"role": "system", "content": [text_part("be brief")]},
        {"role": "user", "content": PARTS},
        {"role": "user", "content": [image_part_from_url("https://x/i.png")]},
    ])
    assert system == "be brief"
    blocks = convo[0]["content"]
    assert blocks[0] == {"type": "text", "text": "what color is this?"}
    assert blocks[1]["source"] == {"type": "base64", "media_type": "image/png", "data": RED_PIXEL}
    assert convo[1]["content"][0]["source"] == {"type": "url", "url": "https://x/i.png"}


def test_gemini_translation_inline_only():
    """Gemini protocol: base64 parts become inline Parts; URL parts fail loud (API takes no remote URLs)."""
    genai_types = pytest.importorskip("google.genai").types
    from agentmaker.core.adapters.gemini import _to_gemini_parts

    parts = _to_gemini_parts(PARTS, genai_types)
    assert parts[0].text == "what color is this?"
    assert parts[1].inline_data.mime_type == "image/png"
    with pytest.raises(LLMConfigError):
        _to_gemini_parts([image_part_from_url("https://x/i.png")], genai_types)


def test_llmclient_vision_guard_fails_loud_before_network():
    """supports_vision=False (deepseek profile) raises a clear config error before any request."""
    client = LLMClient("deepseek", api_key="k")
    assert client.supports_vision is False
    with pytest.raises(LLMConfigError, match="image input"):
        asyncio.run(client.chat([{"role": "user", "content": PARTS}]))
    # Explicit override wins over the profile; plain text is never blocked.
    assert LLMClient("deepseek", api_key="k", supports_vision=True).supports_vision is True
    assert LLMClient("openai", api_key="k").supports_vision is True


def test_emulate_tools_rejects_images():
    """Text emulation flattens everything to a prompt; image parts must fail loud, not vanish."""
    client = LLMClient("deepseek", api_key="k", supports_vision=True, emulate_tools=True)
    with pytest.raises(LLMError, match="text-only"):
        asyncio.run(client.chat([{"role": "user", "content": PARTS}],
                                tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}]))


def test_session_store_round_trips_parts():
    """SqliteSessionStore persists part lists as JSON and restores them faithfully (metadata untouched)."""
    store = SqliteSessionStore(":memory:")
    scope = Scope(user="u", session="s")
    store.append(Message(content=PARTS, role="user", metadata={"k": "v"}), scope=scope)
    store.append(Message(content="plain reply", role="assistant"), scope=scope)
    loaded = store.load(scope=scope)
    assert loaded[0].content == PARTS
    assert loaded[0].metadata == {"k": "v"}   # internal format flag stripped on load
    assert loaded[1].content == "plain reply"


# ── Real-API vision smoke (the actual "does a picture reach the model" verification) ──────

@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY for the real vision smoke")
def test_real_openai_vision_smoke():
    """Send a solid red PNG through the full client -> adapter path and expect the model to name the color."""
    client = LLMClient("openai", model="gpt-4o-mini")
    content = [image_part_from_bytes(_solid_png(64, 64, (255, 0, 0)), "image/png"),
               text_part("What is the dominant color of this image? Answer with the color name only.")]
    reply = asyncio.run(client.chat([{"role": "user", "content": content}], max_tokens=1000))
    assert "red" in reply.content.lower() or "红" in reply.content


def test_solid_png_is_valid():
    """The test PNG builder emits a decodable PNG signature and chunks (guards the smoke's input)."""
    data = _solid_png(4, 4, (255, 0, 0))
    assert data.startswith(b"\x89PNG\r\n\x1a\n") and b"IHDR" in data and b"IEND" in data
    assert json.loads(json.dumps(PARTS)) == PARTS   # parts stay JSON-serializable (session store contract)