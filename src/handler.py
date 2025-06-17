import runpod
from utils import create_error_response
from typing import Any
from embedding_service import EmbeddingService

# Gracefully catch configuration errors (e.g., missing env vars) so the user sees
# a clean message instead of a full Python traceback when the container starts.
try:
    embedding_service = EmbeddingService()
except Exception as e:  # noqa: BLE001 (intercept everything on startup)
    import sys

    sys.stderr.write(f"\nstartup failed: {e}\n")
    sys.exit(1)

async def async_generator_handler(job: dict[str, Any]):
    """Handle the requests and process them asynchronously using Florence-2."""
    job_input = job["input"]

    if job_input.get("openai_route"):
        openai_route, openai_input = job_input.get("openai_route"), job_input.get("openai_input")

        if openai_route and openai_route == "/v1/models":
            call_fn, kwargs = embedding_service.route_openai_models, {}
        elif openai_route and openai_route == "/v1/embeddings":
            model_name = openai_input.get("model")
            if not openai_input:
                return create_error_response("Missing input").model_dump()
            if not model_name:
                return create_error_response("Did not specify model in openai_input").model_dump()

            # Handle image embedding with Florence-2
            call_fn, kwargs = embedding_service.route_florence2_get_embeddings, {
                "embedding_input": openai_input.get("input"),
                "model_name": model_name,
                "return_as_list": True,
            }
        else:
            return create_error_response(f"Invalid OpenAI Route: {openai_route}").model_dump()
    else:
        # Handle image captioning or other Florence-2 specific tasks
        if job_input.get("image"):
            call_fn, kwargs = embedding_service.route_florence2_caption, {
                "image": job_input.get("image"),
                "prompt": job_input.get("prompt", ""),
            }
        elif job_input.get("input"):
            call_fn, kwargs = embedding_service.route_florence2_get_embeddings, {
                "embedding_input": job_input.get("input"),
                "model_name": job_input.get("model"),
            }
        else:
            return create_error_response(f"Invalid input: {job}").model_dump()

    try:
        out = await call_fn(**kwargs)
        return out
    except Exception as e:
        return create_error_response(str(e)).model_dump()

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": async_generator_handler,
        "concurrency_modifier": lambda x: embedding_service.config.runpod_max_concurrency,
    })
