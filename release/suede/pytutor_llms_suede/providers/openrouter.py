from .base import GetPydanticStream, GetTextStream, Provider
from .openai import delta_content_from_chunk

from dotenv import load_dotenv
import instructor
from openai import AsyncOpenAI, api_key

from dataclasses import dataclass
import os
from pydantic import BaseModel
from typing import Any, Literal

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
instructor_client = instructor.from_openai(client)


def _openrouter_reasoning_from_metadata(
    metadata: "OpenRouterProvider.ModelMetadata | None",
) -> dict[str, Any] | None:
    if metadata is None:
        return None

    if metadata.reasoning_enabled is False:
        return {"enabled": False}

    reasoning: dict[str, Any] = {}
    if metadata.reasoning_effort is not None:
        reasoning["effort"] = metadata.reasoning_effort
    if metadata.reasoning_max_tokens is not None:
        reasoning["max_tokens"] = metadata.reasoning_max_tokens

    if not reasoning:
        return None

    return reasoning


async def produce_raw_chunks(request: "Provider.TextStream.Request"):
    openrouter_metadata = Provider.model_metadata(
        request, OpenRouterProvider.ModelMetadata
    )

    create_args: dict[str, Any] = {
        "model": request.model,
        "messages": request.messages,
        "stream": True,
    }

    reasoning = _openrouter_reasoning_from_metadata(openrouter_metadata)
    if reasoning is not None:
        create_args["extra_body"] = {"reasoning": reasoning}

    return await client.chat.completions.create(**create_args)


async def produce_pydantic_models[ModelT: BaseModel](
    request: "Provider.PydanticStream.Request[ModelT]",
):
    metadata = Provider.model_metadata(request, OpenRouterProvider.ModelMetadata)

    create_args: dict[str, Any] = {
        "response_model": request.type,
        "model": request.model,
        "messages": request.messages,
    }

    reasoning = _openrouter_reasoning_from_metadata(metadata)
    if reasoning is not None:
        create_args["extra_body"] = {"reasoning": reasoning}

    return instructor_client.create_partial(**create_args)


class OpenRouterProvider(Provider):
    @dataclass(frozen=True, kw_only=True)
    class ModelMetadata:
        reasoning_enabled: bool | None = None
        reasoning_effort: (
            Literal["xhigh", "high", "medium", "low", "minimal", "none"] | None
        ) = None
        reasoning_max_tokens: int | None = None

    async def try_prepare_text_stream(self, request) -> GetTextStream | None:
        if not request.model.startswith(("qwen", "moonshot", "openrouter/")):
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
        if not request.model.startswith(("qwen", "moonshot", "openrouter/")):
            return None

        async def stream():
            return Provider.PydanticStream.FromModels(
                request,
                model_producer=produce_pydantic_models,
            )

        return stream
