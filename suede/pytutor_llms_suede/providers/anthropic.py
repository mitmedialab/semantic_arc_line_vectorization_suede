from .base import GetPydanticStream, GetTextStream, Provider
from .msg_content import Part, normalize_messages

from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    RawContentBlockDeltaEvent,
    RawMessageStreamEvent,
    TextDelta,
)
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel

from dataclasses import dataclass
import os
from typing import Any, cast

load_dotenv()
api_key = os.getenv("CLAUDE_API_KEY")

client = AsyncAnthropic(api_key=api_key)
instructor_client = instructor.from_anthropic(client)


SUPPORTED_IMAGE_MEDIA_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}


def _to_messages(request: "Provider.TextStream.Request"):
    system_prompt, normalized = normalize_messages(request.messages)
    messages: list[MessageParam] = []

    for message in normalized:
        blocks: list[Any] = []
        for part in message.parts:
            if isinstance(part, Part.Text):
                blocks.append({"type": "text", "text": part.text})
                continue

            if isinstance(part, Part.UrlImage):
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": part.url,
                        },
                    }
                )
                continue

            if isinstance(part, Part.DataImage):
                if part.media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
                    continue
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type,
                            "data": part.data_base64,
                        },
                    }
                )

        if blocks:
            messages.append(
                cast(
                    MessageParam,
                    {
                        "role": message.role,
                        "content": blocks,
                    },
                )
            )

    return system_prompt, messages


def delta_content_from_chunk(chunk: RawMessageStreamEvent) -> str | None:
    if not isinstance(chunk, RawContentBlockDeltaEvent):
        return None

    if not isinstance(chunk.delta, TextDelta):
        return None

    if not chunk.delta.text:
        return None

    return chunk.delta.text


async def produce_raw_chunks(request: "Provider.TextStream.Request"):
    system_prompt, messages = _to_messages(request)
    anthropic_metadata = Provider.model_metadata(
        request, AnthropicProvider.ModelMetadata
    )
    max_tokens = (
        anthropic_metadata.max_tokens
        if anthropic_metadata and anthropic_metadata.max_tokens is not None
        else 4096
    )

    create_args = {
        "model": request.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if system_prompt:
        create_args["system"] = system_prompt

    return await client.messages.create(**create_args)


async def produce_pydantic_models[ModelT: BaseModel](
    request: "Provider.PydanticStream.Request[ModelT]",
):
    anthropic_metadata = Provider.model_metadata(
        request, AnthropicProvider.ModelMetadata
    )
    max_tokens = (
        anthropic_metadata.max_tokens
        if anthropic_metadata and anthropic_metadata.max_tokens is not None
        else 4096
    )

    return instructor_client.create_partial(
        response_model=request.type,
        model=request.model,
        messages=request.messages,
        max_tokens=max_tokens,
    )


class AnthropicProvider(Provider):
    @dataclass(frozen=True, kw_only=True)
    class ModelMetadata:
        max_tokens: int | None = None

    async def try_prepare_text_stream(self, request) -> GetTextStream | None:
        if not request.model.startswith("claude"):
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
        if not request.model.startswith("claude"):
            return None

        async def stream():
            return Provider.PydanticStream.FromModels(
                request,
                model_producer=produce_pydantic_models,
            )

        return stream
