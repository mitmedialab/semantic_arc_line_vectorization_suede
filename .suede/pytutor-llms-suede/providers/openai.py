from .base import Provider, GetPydanticStream

from dotenv import load_dotenv
import instructor
from openai import AsyncOpenAI, api_key
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from pydantic import BaseModel

from dataclasses import dataclass
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(api_key=api_key)
instructor_client = instructor.from_openai(client)


def delta_content_from_chunk(chunk: ChatCompletionChunk) -> str | None:
    if not getattr(chunk, "choices", None):
        return None

    if len(chunk.choices) == 0:
        return None

    content = chunk.choices[0].delta.content

    if content is None:
        return None

    return content


async def produce_raw_chunks(request: "Provider.TextStream.Request"):
    return await client.chat.completions.create(
        model=request.model,
        messages=request.messages,
        stream=True,
    )


async def produce_pydantic_models[ModelT: BaseModel](
    request: "Provider.PydanticStream.Request[ModelT]",
):
    return instructor_client.create_partial(
        response_model=request.type,
        model=request.model,
        messages=request.messages,
    )


class OpenAIProvider(Provider):
    @dataclass(frozen=True, kw_only=True)
    class ModelMetadata:
        pass

    async def try_prepare_text_stream(self, request):
        if not request.model.startswith("gpt"):
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
        if not request.model.startswith("gpt"):
            return None

        async def stream():
            return Provider.PydanticStream.FromModels(
                request,
                model_producer=produce_pydantic_models,
            )

        return stream
