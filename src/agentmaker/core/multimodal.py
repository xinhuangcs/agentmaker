"""agentmaker.core.multimodal: provider-neutral multimodal content parts and helpers.

A message's "content" is either a plain str (text-only, the common case) or a list of
content parts. Parts use a small provider-neutral shape; each protocol adapter translates
them into its wire format (shapes verified against the official docs of all three
protocols, 2026-07):

    {"type": "text", "text": "..."}
    {"type": "image", "media_type": "image/png", "data": "<base64>"}   # inline image
    {"type": "image", "url": "https://..."}                            # remote image

Anything that consumes message content as text (token estimation, history summaries,
keyword indexing, logging) must go through content_text() / content_tokens() instead of
assuming str.
"""

import base64
from pathlib import Path
from typing import Any, Callable, List, Union

from .text import count_tokens

# Image formats accepted by all three protocols (the strictest common set; Anthropic's
# documented list). Reject others at part-construction time instead of failing server-side.
IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})

_SUFFIX_TO_MEDIA_TYPE = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                         ".gif": "image/gif", ".webp": "image/webp"}

# Flat per-image token estimate for context accounting. Providers bill by resolution
# (patch/tile based); an exact figure is impossible without decoding the image, so budget
# a mid-size-image order of magnitude to keep window math conservative rather than blind.
IMAGE_TOKEN_ESTIMATE = 800

MessageContent = Union[str, List[dict]]


def text_part(text: str) -> dict:
    """Build a text content part."""
    return {"type": "text", "text": text}


def image_part_from_bytes(data: bytes, media_type: str) -> dict:
    """Build an inline image part from raw bytes (base64-encoded for transport).

    Args:
        data: Raw image bytes.
        media_type: One of IMAGE_MEDIA_TYPES, e.g. "image/png".

    Raises:
        ValueError: Unsupported media type (fail at construction, not server-side).
    """
    if media_type not in IMAGE_MEDIA_TYPES:
        raise ValueError(f"Unsupported image media type: {media_type} (supported: {sorted(IMAGE_MEDIA_TYPES)})")
    return {"type": "image", "media_type": media_type,
            "data": base64.b64encode(data).decode("ascii")}


def image_part_from_file(path: Union[str, Path], media_type: str = None) -> dict:
    """Build an inline image part from a local file (media type inferred from the suffix when omitted).

    Raises:
        ValueError: Suffix not recognized and media_type not given, or unsupported media type.
    """
    path = Path(path)
    if media_type is None:
        media_type = _SUFFIX_TO_MEDIA_TYPE.get(path.suffix.lower())
        if media_type is None:
            raise ValueError(f"Cannot infer image media type from suffix {path.suffix!r}; pass media_type explicitly")
    return image_part_from_bytes(path.read_bytes(), media_type)


def image_part_from_url(url: str) -> dict:
    """Build a remote-image part (the provider fetches the URL; Gemini's adapter does not support this, see its docs)."""
    return {"type": "image", "url": url}


def is_image_part(part: Any) -> bool:
    """True when the value is an image content part."""
    return isinstance(part, dict) and part.get("type") == "image"


def content_has_image(content: Any) -> bool:
    """True when the content (str or part list) contains at least one image part."""
    return isinstance(content, list) and any(is_image_part(p) for p in content)


def messages_have_images(messages) -> bool:
    """True when any message in the list carries image parts (accepts dicts or Message-likes)."""
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if content_has_image(content):
            return True
    return False


def content_text(content: MessageContent) -> str:
    """Flatten content into plain text: str passes through; part lists join text parts with
    newlines and render each image as an "[image: ...]" placeholder (so summaries / keyword
    indexes / logs stay meaningful instead of showing a Python repr)."""
    if not isinstance(content, list):
        return content or ""
    pieces = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            pieces.append(part.get("text", ""))
        elif part.get("type") == "image":
            pieces.append(f"[image: {part.get('media_type') or part.get('url', 'inline')}]")
    return "\n".join(pieces)


def content_tokens(content: MessageContent, counter: Callable[[str], int] = count_tokens) -> int:
    """Estimate the token count of content: text via the pluggable counter, plus a flat
    IMAGE_TOKEN_ESTIMATE per image part (see the constant's comment). None counts as 0."""
    if content is None:
        return 0
    if not isinstance(content, list):
        return counter(content)
    images = sum(1 for p in content if is_image_part(p))
    return counter(content_text(content)) + images * IMAGE_TOKEN_ESTIMATE
