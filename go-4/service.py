"""go-4 embedding service.

A custom FastAPI service for nomic-embed-vision-v1.5 + nomic-embed-text-v1.5.
Both models share the same latent space, so this service exposes a single
/v1/embeddings endpoint that accepts either text or image inputs and returns
embeddings in either modality.

ik_llama.cpp can't do this standalone — its vision support is for multimodal
CHAT (vision feeds into an LLM decoder), not standalone image-embedding
servers. So we wrap the HuggingFace transformers model directly.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

import numpy as np
import requests
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("go-4")

TEXT_MODEL_PATH = os.environ.get("TEXT_MODEL_PATH", "/models/nomic-embed-text-v1.5")
VISION_MODEL_PATH = os.environ.get("VISION_MODEL_PATH", "/models/nomic-embed-vision-v1.5")
EMBED_DIM = 768  # nomic-embed-text-v1.5 and the aligned vision tower both produce 768-d

app = FastAPI(title="go-4", description="Nomic multimodal embedding service")


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def load_models() -> tuple[Any, Any, Any, Any]:
    log.info("Loading text model from %s", TEXT_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
    text_model = AutoModel.from_pretrained(TEXT_MODEL_PATH)
    text_model.eval()

    log.info("Loading vision model from %s", VISION_MODEL_PATH)
    image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_PATH)
    vision_model = AutoModel.from_pretrained(VISION_MODEL_PATH, trust_remote_code=True)
    vision_model.eval()

    log.info("Models loaded; embed dim = %d", EMBED_DIM)
    return tokenizer, text_model, image_processor, vision_model


tokenizer, text_model, image_processor, vision_model = load_models()


@torch.no_grad()
def embed_text(texts: list[str]) -> np.ndarray:
    # Nomic requires the search_query: / search_document: / clustering: prefix
    # for asymmetric retrieval. We don't enforce a prefix here — clients
    # supply it in their text when needed.
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    output = text_model(**encoded)
    embeddings = mean_pooling(output, encoded["attention_mask"])
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_images(images: list[Image.Image]) -> np.ndarray:
    inputs = image_processor(images=images, return_tensors="pt")
    output = vision_model(**inputs)
    embeddings = output.last_hidden_state[:, 0]  # CLS token
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy().astype(np.float32)


def decode_image(item: dict[str, Any]) -> Image.Image:
    url = item.get("image_url", {}).get("url", "")
    if not url:
        raise HTTPException(status_code=400, detail="image_url missing url")
    if url.startswith("data:image"):
        try:
            header, b64 = url.split(",", 1)
            raw = base64.b64decode(b64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"bad data URL: {exc}") from exc
        return Image.open(io.BytesIO(raw)).convert("RGB")
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"image fetch failed: {exc}") from exc
    return Image.open(resp.raw).convert("RGB")


class EmbeddingRequest(BaseModel):
    input: str | list[str | dict[str, Any]]
    model: str = "nomic-embed-vision-v1.5"
    encoding_format: str = Field(default="float")


class EmbeddingData(BaseModel):
    embedding: list[float]
    index: int
    object: str = "embedding"


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingData]
    model: str
    object: str = "list"
    usage: dict[str, int]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": "nomic-embed-vision-v1.5"}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    inputs = req.input if isinstance(req.input, list) else [req.input]
    if not inputs:
        raise HTTPException(status_code=400, detail="input is empty")

    text_items: list[int] = []  # indexes into inputs
    image_items: list[tuple[int, Image.Image]] = []
    for idx, item in enumerate(inputs):
        if isinstance(item, str):
            text_items.append(idx)
        elif isinstance(item, dict):
            image_items.append((idx, decode_image(item)))
        else:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported input item at index {idx}: must be str or dict",
            )

    embeddings: list[np.ndarray | None] = [None] * len(inputs)

    if text_items:
        text_vecs = embed_text([inputs[i] for i in text_items])
        for slot, vec in zip(text_items, text_vecs):
            embeddings[slot] = vec

    if image_items:
        img_vecs = embed_images([img for _, img in image_items])
        for (slot, _), vec in zip(image_items, img_vecs):
            embeddings[slot] = vec

    if any(e is None for e in embeddings):
        raise HTTPException(status_code=500, detail="internal: missing embedding")

    return EmbeddingResponse(
        data=[
            EmbeddingData(
                embedding=vec.tolist(),  # type: ignore[union-attr]
                index=i,
            )
            for i, vec in enumerate(embeddings)
        ],
        model=req.model,
        usage={"prompt_tokens": 0, "total_tokens": 0},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
