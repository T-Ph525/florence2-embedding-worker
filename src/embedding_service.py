from config import EmbeddingServiceConfig
from infinity_emb.engine import AsyncEngineArray, EngineArgs
from utils import (
    OpenAIModelInfo,
    ModelInfo,
    list_embeddings_to_response,
    to_rerank_response,
    to_caption_response,
    to_image_embedding_response,
)

import asyncio
import base64
from typing import Any

class EmbeddingService:
    def __init__(self):
        self.config = EmbeddingServiceConfig()
        engine_args = []
        for model_name, batch_size, dtype in zip(
            self.config.model_names, self.config.batch_sizes, self.config.dtypes
        ):
            engine_args.append(
                EngineArgs(
                    model_name_or_path=model_name,
                    batch_size=batch_size,
                    engine=self.config.backend,
                    dtype=dtype,
                    model_warmup=False,
                    lengths_via_tokenize=True,
                )
            )

        self.engine_array = AsyncEngineArray.from_args(engine_args)
        self.is_running = False
        self.semaphore = asyncio.Semaphore(1)

    async def start(self):
        """Starts the engine background loop."""
        async with self.semaphore:
            if not self.is_running:
                await self.engine_array.astart()
                self.is_running = True

    async def stop(self):
        """Stops the engine background loop."""
        async with self.semaphore:
            if self.is_running:
                await self.engine_array.astop()
                self.is_running = False

    async def route_openai_models(self) -> OpenAIModelInfo:
        return OpenAIModelInfo(
            data=[ModelInfo(id=model_id, stats={}) for model_id in self.list_models()]
        ).model_dump()

    def list_models(self) -> list[str]:
        return list(self.engine_array.engines_dict.keys())

    async def route_openai_get_embeddings(
        self,
        embedding_input: str | list[str],
        model_name: str,
        return_as_list: bool = False,
    ):
        """Returns embeddings for the input text."""
        if not self.is_running:
            await self.start()
        if not isinstance(embedding_input, list):
            embedding_input = [embedding_input]

        embeddings, usage = await self.engine_array[model_name].embed(embedding_input)
        if return_as_list:
            return [
                list_embeddings_to_response(embeddings, model=model_name, usage=usage)
            ]
        else:
            return list_embeddings_to_response(
                embeddings, model=model_name, usage=usage
            )

    async def route_florence2_get_embeddings(
        self,
        embedding_input: str | list[str],
        model_name: str,
        return_as_list: bool = False,
    ):
        """Returns embeddings for the input image using Florence-2."""
        if not self.is_running:
            await self.start()
        if not isinstance(embedding_input, list):
            embedding_input = [embedding_input]

        # Assuming embedding_input is a base64 encoded image
        embeddings, usage = await self.engine_array[model_name].embed(embedding_input)
        if return_as_list:
            return [
                to_image_embedding_response(embeddings, model=model_name, usage=usage)
            ]
        else:
            return to_image_embedding_response(
                embeddings, model=model_name, usage=usage
            )

    async def route_florence2_caption(
        self,
        image: str,
        prompt: str = "",
    ):
        """Returns a caption for the input image using Florence-2."""
        if not self.is_running:
            await self.start()

        # Assuming image is a base64 encoded image
        caption, usage = await self.engine_array["florence-2"].caption(image, prompt=prompt)
        return to_caption_response(
            caption=caption, model="florence-2", usage=usage
        )

    async def infinity_rerank(
        self, query: str, docs: list[str], return_docs: bool, model_name: str
    ):
        """Rerank the documents based on the query."""
        if not self.is_running:
            await self.start()
        scores, usage = await self.engine_array[model_name].rerank(
            query=query, docs=docs, raw_scores=False
        )
        if not return_docs:
            docs = None
        return to_rerank_response(
            scores=scores, documents=docs, model=model_name, usage=usage
        )
