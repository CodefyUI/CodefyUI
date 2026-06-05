"""System / environment introspection endpoints (compute devices, etc.)."""
from fastapi import APIRouter

from ..core.device_utils import describe_accelerator

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/devices")
async def list_devices() -> dict:
    """Compute devices available for graph execution.

    Returns ``describe_accelerator()`` — the best-available ``default`` plus a
    ``devices`` list with human-friendly labels that distinguish NVIDIA CUDA
    from AMD ROCm and label Apple MPS. The frontend's global device selector
    renders this.
    """
    return describe_accelerator()
