from ..common import Msg

import base64
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

Role = Literal["user", "assistant"]


class Part:
    @dataclass(frozen=True)
    class Text:
        text: str

    @dataclass(frozen=True)
    class UrlImage:
        url: str

    @dataclass(frozen=True)
    class DataImage:
        media_type: str
        data_base64: str
        data_bytes: bytes


type PartUnion = Part.Text | Part.UrlImage | Part.DataImage


@dataclass(frozen=True)
class NormalizedMessage:
    role: Role
    parts: tuple[PartUnion, ...]


def _as_mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _extract_image_url(part: Mapping[str, Any]) -> str | None:
    image_url = part.get("image_url")
    if isinstance(image_url, str):
        return image_url

    image_url_mapping = _as_mapping(image_url)
    if image_url_mapping is not None:
        nested_url = image_url_mapping.get("url")
        if isinstance(nested_url, str):
            return nested_url

    return None


def _parse_data_url(url: str) -> Part.DataImage | None:
    if not url.startswith("data:"):
        return None

    header, _, payload = url.partition(",")
    if not payload or ";base64" not in header:
        return None

    media_type = header[5:].split(";", 1)[0] or "image/png"
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception:
        return None

    return Part.DataImage(media_type=media_type, data_base64=payload, data_bytes=raw)


def normalize_parts(content: object) -> tuple[PartUnion, ...]:
    if isinstance(content, str):
        return (Part.Text(text=content),) if content else ()

    if not isinstance(content, list):
        return ()

    parts: list[PartUnion] = []
    for raw_part in content:
        part = _as_mapping(raw_part)
        if part is None:
            continue

        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if isinstance(text, str) and text:
                parts.append(Part.Text(text=text))
            continue

        if part_type in ("image_url", "input_image"):
            url = _extract_image_url(part)
            if not url:
                continue

            data_image = _parse_data_url(url)
            if data_image is not None:
                parts.append(data_image)
            else:
                parts.append(Part.UrlImage(url=url))

    return tuple(parts)


def extract_text(content: object) -> str:
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    texts: list[str] = []
    for raw_part in content:
        part = _as_mapping(raw_part)
        if part is None:
            continue
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str):
            texts.append(text)

    return "".join(texts)


def normalize_messages(messages: Sequence[Msg]) -> tuple[str, list[NormalizedMessage]]:
    system_chunks: list[str] = []
    normalized_messages: list[NormalizedMessage] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role in ("system", "developer"):
            system_text = extract_text(content)
            if system_text:
                system_chunks.append(system_text)
            continue

        parts = normalize_parts(content)
        if not parts:
            continue

        normalized_role: Role = "assistant" if role == "assistant" else "user"
        normalized_messages.append(NormalizedMessage(role=normalized_role, parts=parts))

    return "\n\n".join(system_chunks), normalized_messages
