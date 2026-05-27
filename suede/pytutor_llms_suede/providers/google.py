from ..common import Msg
from .base import GetPydanticStream, GetTextStream, Provider
from .msg_content import Part, normalize_messages

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
import instructor
from pydantic import BaseModel

from dataclasses import dataclass
import mimetypes
import os
from typing import Literal, Sequence, cast

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
instructor_client = instructor.from_genai(client, use_async=True)


type ThinkingLevelValue = Literal[
    "low",
    "medium",
    "high",
    "minimal",
    "unspecified",
]


def _to_thinking_level_enum(
    thinking_level: ThinkingLevelValue | None,
) -> genai_types.ThinkingLevel | None:
    if thinking_level is None or thinking_level == "":
        return None

    level_map: dict[ThinkingLevelValue, genai_types.ThinkingLevel] = {
        "low": genai_types.ThinkingLevel.LOW,
        "medium": genai_types.ThinkingLevel.MEDIUM,
        "high": genai_types.ThinkingLevel.HIGH,
        "minimal": genai_types.ThinkingLevel.MINIMAL,
        "unspecified": genai_types.ThinkingLevel.THINKING_LEVEL_UNSPECIFIED,
    }

    return level_map[thinking_level]


def _config_from_metadata(
    model_metadata: "GoogleProvider.ModelMetadata | None",
) -> genai_types.GenerateContentConfigDict:
    config: genai_types.GenerateContentConfigDict = {}
    if model_metadata is None:
        return config

    thinking_config: genai_types.ThinkingConfigDict = {}

    enum_thinking_level = _to_thinking_level_enum(model_metadata.thinking_level)
    if enum_thinking_level is not None:
        thinking_config["thinking_level"] = enum_thinking_level

    if thinking_config:
        config["thinking_config"] = thinking_config

    return config


def _to_messages(
    request: "Provider.TextStream.Request | Provider.PydanticStream.Request",
) -> tuple[Sequence[genai_types.Content], genai_types.GenerateContentConfigDict | None]:
    system_prompt, normalized = normalize_messages(request.messages)
    contents: Sequence[genai_types.Content] = []

    for message in normalized:
        parts: list[genai_types.Part] = []
        for part in message.parts:
            if isinstance(part, Part.Text):
                parts.append(genai_types.Part.from_text(text=part.text))
                continue

            if isinstance(part, Part.DataImage):
                parts.append(
                    genai_types.Part.from_bytes(
                        data=part.data_bytes,
                        mime_type=part.media_type,
                    )
                )
                continue

            if isinstance(part, Part.UrlImage):
                guessed_mime, _ = mimetypes.guess_type(part.url)
                parts.append(
                    genai_types.Part.from_uri(
                        file_uri=part.url,
                        mime_type=guessed_mime,
                    )
                )

        if not parts:
            continue

        google_role = "model" if message.role == "assistant" else "user"
        contents.append(genai_types.Content(role=google_role, parts=parts))

    if not contents:
        contents.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text="")],
            )
        )

    config = _config_from_metadata(
        Provider.model_metadata(request, GoogleProvider.ModelMetadata)
    )
    if system_prompt:
        config["system_instruction"] = system_prompt

    final_config: genai_types.GenerateContentConfigDict | None = config or None

    return contents, final_config


def delta_content_from_chunk(chunk: genai_types.GenerateContentResponse) -> str | None:
    text = getattr(chunk, "text", None)
    if text:
        return text

    candidates = getattr(chunk, "candidates", None) or []
    if not candidates:
        return None

    candidate = candidates[0]
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    texts: list[str] = []
    for part in parts:
        part_text = getattr(part, "text", None)
        if part_text:
            texts.append(part_text)

    if not texts:
        return None

    return "".join(texts)


async def produce_raw_chunks(request: "Provider.TextStream.Request"):
    contents, config = _to_messages(request)
    return await client.aio.models.generate_content_stream(
        model=request.model,
        contents=contents,
        config=config,
    )


async def produce_pydantic_models[ModelT: BaseModel](
    request: "Provider.PydanticStream.Request[ModelT]",
):
    contents, config = _to_messages(request)
    return instructor_client.create_partial(
        config=config,
        response_model=request.type,
        model=request.model,
        messages=cast(list[Msg], contents),
    )


class GoogleProvider(Provider):
    @dataclass(frozen=True, kw_only=True)
    class ModelMetadata:
        thinking_level: ThinkingLevelValue | None = None

    async def try_prepare_text_stream(self, request) -> GetTextStream | None:
        if not request.model.startswith(("gemini", "learnlm")):
            return None

        async def stream():
            return Provider.TextStream.FromChunks(
                request,
                raw_chunk_producer=produce_raw_chunks,
                delta_content_from_chunk=delta_content_from_chunk,
            )

        return stream

    async def try_prepare_pydantic_stream[ModelT: BaseModel](
        self,
        request: "Provider.PydanticStream.Request[ModelT]",
    ) -> GetPydanticStream[ModelT] | None:
        if not request.model.startswith(("gemini", "learnlm")):
            return None

        async def stream():
            return Provider.PydanticStream.FromModels(
                request,
                model_producer=produce_pydantic_models,
            )

        return stream
