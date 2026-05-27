from .utils import compute_delta

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel

from typing import NamedTuple

type Msg = ChatCompletionMessageParam


class Chunk:
    class Text(NamedTuple):
        """Represents one text-stream update.

        Attributes:
            delta: New text emitted for this chunk.
            accumulated: Full text accumulated up to and including this chunk.
            i: Zero-based chunk index in the stream.
        """

        delta: str
        accumulated: str
        i: int

    class Pydantic[ModelT: BaseModel](NamedTuple):
        """Represents one structured-model stream update.

        Attributes:
            current: Current model snapshot emitted by the stream.
            previous: Previous snapshot (None for the first emitted chunk).
            i: Zero-based chunk index in the stream.

        Methods:
            compute_delta: Returns changed fields from previous -> current as a
                nested dictionary. For the first chunk (previous is None), returns
                the full current model dump.
        """

        current: ModelT
        previous: ModelT | None
        i: int

        def compute_delta(self) -> dict[str, object]:
            if self.previous is None:
                return self.current.model_dump(mode="python")

            current_dump = self.current.model_dump(mode="python")
            previous_dump = self.previous.model_dump(mode="python")
            delta_obj = compute_delta(current_dump, previous_dump)
            return delta_obj if isinstance(delta_obj, dict) else {}
