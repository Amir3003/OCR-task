"""JUZ40 OCR Engine: prototype backend.

Serves the single-page demo UI at ``/`` and exposes ``POST /v1/recognize``.
The recognition itself is a simulation: the endpoint waits 3-4 seconds (to mimic
GPU inference) and returns a fixed, schema-conformant payload chosen by subject.
The uploaded image is validated for presence only and is not analysed.
"""

import asyncio
import base64
import random
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
MODEL_VERSION = "1.0.0"

Subject = Literal["Math", "Physics", "Chemistry", "Geometry"]


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class RecognizeRequest(BaseModel):
    subject: Subject = Field(description="Subject of the handwritten document.")
    image_data: str = Field(
        min_length=1,
        description="Document image: a base64 string or a `data:image/...;base64,` URL.",
    )

    @field_validator("subject", mode="before")
    @classmethod
    def _normalize_subject(cls, value: object) -> object:
        return value.strip().capitalize() if isinstance(value, str) else value


class Quality(BaseModel):
    usable: bool


class Label(BaseModel):
    text: str
    xy: tuple[float, float]


class Block(BaseModel):
    # Field order mirrors the order of keys in the tender's reference payloads.
    id: str
    order: int
    type: Literal["text", "formula", "chem_structure", "drawing", "answer"]
    drawingType: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    content: str | None = None
    latex: str | None = None
    smiles: str | None = None
    image: str | None = None
    labels: list[Label] | None = None
    confidence: float


class Page(BaseModel):
    index: int
    quality: Quality
    blocks: list[Block]


class RecognizeResponse(BaseModel):
    jobId: str
    modelVersion: str
    pages: list[Page]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
CHEMISTRY_RESULT = {
    "jobId": "j-000451",
    "modelVersion": MODEL_VERSION,
    "pages": [
        {
            "index": 1,
            "quality": {"usable": True},
            "blocks": [
                {
                    "id": "b1",
                    "order": 1,
                    "type": "text",
                    "bbox": [0.08, 0.06, 0.62, 0.04],
                    "content": "Уравнение реакции этерификации:",
                    "confidence": 0.98,
                },
                {
                    "id": "b2",
                    "order": 2,
                    "type": "formula",
                    "bbox": [0.08, 0.12, 0.70, 0.09],
                    "latex": r"\ce{CH3COOH + C2H5OH <=>[H2SO4] CH3COOC2H5 + H2O}",
                    "confidence": 0.92,
                },
                {
                    "id": "b3",
                    "order": 3,
                    "type": "chem_structure",
                    "bbox": [0.10, 0.30, 0.35, 0.18],
                    "smiles": "CC(=O)OCC",
                    "image": "s3://ocr/j-000451/b3.png",
                    "confidence": 0.89,
                },
            ],
        }
    ],
}

PHYSICS_MATH_RESULT = {
    "jobId": "j-000452",
    "modelVersion": MODEL_VERSION,
    "pages": [
        {
            "index": 1,
            "quality": {"usable": True},
            "blocks": [
                {
                    "id": "b1",
                    "order": 1,
                    "type": "text",
                    "bbox": [0.08, 0.06, 0.62, 0.04],
                    "content": "Дано: m = 2 кг, v_0 = 10 м/с",
                    "confidence": 0.94,
                },
                {
                    "id": "b2",
                    "order": 2,
                    "type": "formula",
                    "bbox": [0.08, 0.12, 0.70, 0.09],
                    "latex": r"E_k = \frac{mv_0^2}{2} = \frac{2 \cdot 10^2}{2} = 100",
                    "confidence": 0.96,
                },
                {
                    "id": "b3",
                    "order": 3,
                    "type": "drawing",
                    "drawingType": "schema_forces",
                    "bbox": [0.55, 0.30, 0.38, 0.30],
                    "image": "s3://ocr/j-000452/b3.png",
                    "labels": [
                        {"text": "mg", "xy": [0.12, 0.91]},
                        {"text": "v_0", "xy": [0.20, 0.80]},
                    ],
                    "confidence": 0.90,
                },
                {
                    "id": "b4",
                    "order": 4,
                    "type": "answer",
                    "latex": r"100\,\text{Дж}",
                    "confidence": 0.99,
                },
            ],
        }
    ],
}

# The tender's reference payloads cover Chemistry and Physics/Math only; Geometry
# gets an equivalent fixture in the same schema so the demo stays coherent.
GEOMETRY_RESULT = {
    "jobId": "j-000453",
    "modelVersion": MODEL_VERSION,
    "pages": [
        {
            "index": 1,
            "quality": {"usable": True},
            "blocks": [
                {
                    "id": "b1",
                    "order": 1,
                    "type": "text",
                    "bbox": [0.08, 0.06, 0.66, 0.04],
                    "content": "В прямоугольном треугольнике ABC: AB = 6 см, BC = 8 см, ∠B = 90°",
                    "confidence": 0.95,
                },
                {
                    "id": "b2",
                    "order": 2,
                    "type": "formula",
                    "bbox": [0.08, 0.13, 0.72, 0.09],
                    "latex": r"AC = \sqrt{AB^2 + BC^2} = \sqrt{6^2 + 8^2} = \sqrt{100} = 10",
                    "confidence": 0.95,
                },
                {
                    "id": "b3",
                    "order": 3,
                    "type": "drawing",
                    "drawingType": "geometry_figure",
                    "bbox": [0.50, 0.28, 0.40, 0.32],
                    "image": "s3://ocr/j-000453/b3.png",
                    "labels": [
                        {"text": "A", "xy": [0.10, 0.12]},
                        {"text": "B", "xy": [0.10, 0.88]},
                        {"text": "C", "xy": [0.90, 0.88]},
                    ],
                    "confidence": 0.91,
                },
                {
                    "id": "b4",
                    "order": 4,
                    "type": "answer",
                    "latex": r"AC = 10\,\text{см}",
                    "confidence": 0.98,
                },
            ],
        }
    ],
}

# Validated once at import time so a malformed fixture fails fast.
RESULTS: dict[str, RecognizeResponse] = {
    "Chemistry": RecognizeResponse.model_validate(CHEMISTRY_RESULT),
    "Physics": RecognizeResponse.model_validate(PHYSICS_MATH_RESULT),
    "Math": RecognizeResponse.model_validate(PHYSICS_MATH_RESULT),
    "Geometry": RecognizeResponse.model_validate(GEOMETRY_RESULT),
}


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="JUZ40 OCR Engine API",
    version="1.0.0",
    description="Handwritten educational OCR: Math, Physics, Chemistry, Geometry.",
)

REQUEST_BODY_DOC = {
    "required": True,
    "content": {
        "application/json": {"schema": RecognizeRequest.model_json_schema()},
        "multipart/form-data": {"schema": RecognizeRequest.model_json_schema()},
        "application/x-www-form-urlencoded": {"schema": RecognizeRequest.model_json_schema()},
    },
}


async def parse_payload(request: Request) -> RecognizeRequest:
    """Accept the request as JSON or as form data and validate it."""
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    raw: dict[str, str] = {}

    if content_type == "application/json":
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(status_code=400, detail="Malformed JSON body.") from None
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="JSON body must be an object.")
        raw = body
    elif content_type in ("multipart/form-data", "application/x-www-form-urlencoded"):
        form = await request.form()
        for key, value in form.multi_items():
            if isinstance(value, str):
                raw[key] = value
            else:  # uploaded file part: forward it as base64 image data
                content = await value.read()
                raw[key] = base64.b64encode(content).decode("ascii") if content else ""
        for alias in ("file", "image"):
            if "image_data" not in raw and alias in raw:
                raw["image_data"] = raw[alias]
    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported Content-Type. Use application/json or form data.",
        )

    try:
        return RecognizeRequest.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False, include_context=False, include_input=False),
        ) from None


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse(
        INDEX_FILE.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/health", tags=["System"], summary="Liveness probe")
async def health() -> dict[str, str]:
    return {"status": "ok", "modelVersion": MODEL_VERSION}


@app.post(
    "/v1/recognize",
    response_model=RecognizeResponse,
    response_model_exclude_none=True,
    tags=["OCR"],
    summary="Recognize a handwritten document",
    description=(
        "Accepts a document image and a subject, and returns structured blocks "
        "(text, formulas, chemical structures, drawings, answers). "
        "Processing takes 3-4 seconds."
    ),
    openapi_extra={"requestBody": REQUEST_BODY_DOC},
)
async def recognize(request: Request, response: Response) -> RecognizeResponse:
    started = time.perf_counter()
    payload = await parse_payload(request)

    await asyncio.sleep(random.uniform(3.0, 4.0))  # simulated GPU inference

    response.headers["X-Process-Time"] = f"{(time.perf_counter() - started) * 1000:.0f}"
    response.headers["X-Model-Version"] = MODEL_VERSION
    return RESULTS[payload.subject]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
