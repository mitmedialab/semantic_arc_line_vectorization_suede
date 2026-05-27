from ..common import Msg, Chunk
from ..utils import build_from_json_schema

from dataclasses import dataclass, field, replace
from pydantic import BaseModel

from abc import ABC, abstractmethod
from typing import (
    Awaitable,
    Callable,
    AsyncIterator,
    Literal,
    List,
    NamedTuple,
    Optional,
    Sequence,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .anthropic import AnthropicProvider
    from .google import GoogleProvider
    from .openai import OpenAIProvider
    from .openrouter import OpenRouterProvider

    type ModelMetadata = (
        OpenAIProvider.ModelMetadata
        | AnthropicProvider.ModelMetadata
        | GoogleProvider.ModelMetadata
        | OpenRouterProvider.ModelMetadata
    )
else:
    type ModelMetadata = object


class Event:
    class Text(NamedTuple):
        payload: Chunk.Text
        type: Literal["chunk"]

    class Pydantic[ModelT: BaseModel](NamedTuple):
        payload: Chunk.Pydantic[ModelT]
        type: Literal["chunk"]

    class Error(NamedTuple):
        payload: Exception
        type: Literal["error"]


type TextStream = AsyncIterator[Event.Text | Event.Error]
type GetTextStream = Callable[[], Awaitable[TextStream]]

type PydanticStream[ModelT: BaseModel] = AsyncIterator[
    Event.Pydantic[ModelT] | Event.Error
]
type GetPydanticStream[ModelT: BaseModel] = Callable[
    [], Awaitable[PydanticStream[ModelT]]
]
type SchemaStream = PydanticStream[BaseModel]

type Interrupt = Callable[[], bool]

type TextProducer[T] = Callable[
    ["Provider.TextStream.Request"], Awaitable[AsyncIterator[T]]
]
type ModelProducer[ModelT: BaseModel] = Callable[
    ["Provider.PydanticStream.Request[ModelT]"], Awaitable[AsyncIterator[ModelT]]
]
type DeltaContentFromChunk[T] = Callable[[T], str | None]


class Base:
    @dataclass(frozen=True, kw_only=True)
    class Request:
        messages: List[Msg]
        model: str
        model_metadata: List[ModelMetadata] = field(default_factory=list)
        interrupt: Optional[Interrupt] = None
        fallback_model: Optional[str] = None


class Provider(ABC):
    @staticmethod
    async def _Collect[ItemT](stream: AsyncIterator[ItemT]) -> list[ItemT]:
        items: list[ItemT] = []
        async for item in stream:
            items.append(item)
        return items

    @staticmethod
    async def _Select[RequestT: Base.Request, StreamT](
        request: RequestT,
        providers: Sequence["Provider"],
        try_prepare: Callable[
            ["Provider", RequestT],
            Awaitable[Callable[[], Awaitable[StreamT]] | None],
        ],
    ) -> StreamT:
        for provider in providers:
            starter = await try_prepare(provider, request)
            if starter is not None:
                return await starter()  # only one stream actually starts

        if request.fallback_model is not None:
            fallback_request = replace(
                request,
                model=request.fallback_model,
                fallback_model=None,
            )
            return await Provider._Select(
                fallback_request,
                providers,
                try_prepare,
            )

        raise ValueError(f"No provider accepted model={request.model}")

    @classmethod
    def model_metadata[MetadataT](
        cls,
        request: Base.Request,
        metadata_type: type[MetadataT],
    ) -> MetadataT | None:
        return next(
            (
                metadata
                for metadata in request.model_metadata
                if isinstance(metadata, metadata_type)
            ),
            None,
        )

    class TextStream:
        @dataclass(frozen=True, kw_only=True)
        class Request(Base.Request):
            pass

        @classmethod
        async def FromChunks[ChunkT](
            cls,
            request: Request,
            *,
            raw_chunk_producer: TextProducer[ChunkT],
            delta_content_from_chunk: DeltaContentFromChunk[ChunkT],
        ) -> TextStream:
            i = 0
            accumulated = ""

            try:
                async for chunk in await raw_chunk_producer(request):
                    if request.interrupt and request.interrupt():
                        break

                    content = delta_content_from_chunk(chunk)
                    if content is None:
                        continue

                    accumulated += content
                    yield Event.Text(
                        type="chunk",
                        payload=Chunk.Text(delta=content, accumulated=accumulated, i=i),
                    )
                    i += 1
            except Exception as e:
                yield Event.Error(type="error", payload=e)

        @classmethod
        async def Select(
            cls,
            request: "Provider.TextStream.Request",
            *providers: "Provider",
        ) -> TextStream:
            return await Provider._Select(
                request,
                providers,
                lambda provider, req: provider.try_prepare_text_stream(req),
            )

        @classmethod
        async def Collect(
            cls,
            stream: TextStream,
        ) -> list[Event.Text | Event.Error]:
            return await Provider._Collect(stream)

    class PydanticStream:
        @dataclass(frozen=True, kw_only=True)
        class Request[ModelT: BaseModel](Base.Request):
            type: type[ModelT]

        @classmethod
        async def FromModels[ModelT: BaseModel](
            cls,
            request: "Provider.PydanticStream.Request[ModelT]",
            *,
            model_producer: ModelProducer[ModelT],
        ) -> PydanticStream[ModelT]:
            i = 0
            previous_model: ModelT | None = None
            try:
                async for model in await model_producer(request):
                    if request.interrupt and request.interrupt():
                        break

                    yield Event.Pydantic[ModelT](
                        type="chunk",
                        payload=Chunk.Pydantic[ModelT](
                            current=model, previous=previous_model, i=i
                        ),
                    )
                    previous_model = model
                    i += 1
            except Exception as e:
                yield Event.Error(type="error", payload=e)

        @classmethod
        async def Select[ModelT: BaseModel](
            cls,
            request: "Provider.PydanticStream.Request[ModelT]",
            *providers: "Provider",
        ) -> PydanticStream[ModelT]:
            return await Provider._Select(
                request,
                providers,
                lambda provider, req: provider.try_prepare_pydantic_stream(req),
            )

        @classmethod
        async def Collect[ModelT: BaseModel](
            cls,
            stream: PydanticStream[ModelT],
        ) -> list[Event.Pydantic[ModelT] | Event.Error]:
            return await Provider._Collect(stream)

    class SchemaStream:
        @dataclass(frozen=True, kw_only=True)
        class Request(Base.Request):
            schema: str

        @classmethod
        async def Select(
            cls,
            request: "Provider.SchemaStream.Request",
            *providers: "Provider",
        ) -> SchemaStream:
            model_type = build_from_json_schema(request.schema)
            pydantic_request = Provider.PydanticStream.Request[BaseModel](
                messages=request.messages,
                model=request.model,
                model_metadata=request.model_metadata,
                interrupt=request.interrupt,
                fallback_model=request.fallback_model,
                type=model_type,
            )
            return await Provider.PydanticStream.Select(pydantic_request, *providers)

        @classmethod
        async def Collect(
            cls,
            stream: SchemaStream,
        ) -> list[Event.Pydantic[BaseModel] | Event.Error]:
            return await Provider._Collect(stream)

    @abstractmethod
    async def try_prepare_text_stream(
        self,
        request: "Provider.TextStream.Request",
    ) -> Optional[GetTextStream]:
        raise NotImplementedError

    @abstractmethod
    async def try_prepare_pydantic_stream[ModelT: BaseModel](
        self,
        request: "Provider.PydanticStream.Request[ModelT]",
    ) -> Optional[GetPydanticStream[ModelT]]:
        raise NotImplementedError
