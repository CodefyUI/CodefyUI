import logging
from typing import TYPE_CHECKING, Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

if TYPE_CHECKING:
    # `pathlib` is imported lazily inside the methods below (startup cost);
    # this makes the "Path" annotations resolvable to a type checker without
    # paying that cost at import time.
    from pathlib import Path

logger = logging.getLogger(__name__)


class ImageBatchReaderNode(BaseNode):
    NODE_NAME = "ImageBatchReader"
    CATEGORY = "IO"
    DESCRIPTION = "Read all images from a directory and stack them into a batch tensor (N, C, H, W)"

    # #144: cacheable again -- cache_fingerprint() below folds an aggregate
    # fingerprint (count, total size, latest mtime) of exactly the files
    # `directory`+`pattern`(+`max_images`) select into the cache key, so
    # images being added, removed, or edited busts the cache instead of the
    # key surviving unchanged because only the directory/pattern strings
    # are hashed.
    cacheable = True

    @staticmethod
    def _resolve_directory(directory: str) -> "Path":
        from pathlib import Path

        from ...config import settings

        d = Path(directory)
        if not d.is_absolute():
            d = settings.MODELS_DIR.parent / d
        return d.resolve()

    @classmethod
    def _selected_paths(cls, params: dict[str, Any]) -> list["Path"] | None:
        """The exact file list execute() would glob and read, or None if
        params do not name a readable directory."""
        directory = str(params.get("directory", "") or "")
        if not directory:
            return None
        try:
            d = cls._resolve_directory(directory)
            pattern = str(params.get("pattern", "*.png") or "*.png")
            paths = sorted(d.glob(pattern))
        except Exception:
            return None
        max_images = params.get("max_images", 0) or 0
        try:
            max_images = int(max_images)
        except (TypeError, ValueError):
            max_images = 0
        if max_images > 0:
            paths = paths[:max_images]
        return paths

    @classmethod
    def cache_fingerprint(cls, params: dict[str, Any]) -> Any:
        from ...core.cache_fingerprint import paths_fingerprint

        paths = cls._selected_paths(params)
        if paths is None:
            return None
        return paths_fingerprint(paths)

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="images", data_type=DataType.TENSOR, description="Batch of images (N, C, H, W)"),
            PortDefinition(name="count", data_type=DataType.SCALAR, description="Number of images loaded"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="directory",
                param_type=ParamType.STRING,
                default="",
                description="Directory containing image files",
            ),
            ParamDefinition(
                name="pattern",
                param_type=ParamType.STRING,
                default="*.png",
                description="Glob pattern to match files (e.g. *.png, *.jpg, *.{png,jpg})",
            ),
            ParamDefinition(
                name="resize",
                param_type=ParamType.INT,
                default=224,
                description="Resize all images to this square size (required for batching)",
                min_value=1,
            ),
            ParamDefinition(
                name="max_images",
                param_type=ParamType.INT,
                default=0,
                description="Maximum number of images to load (0 = all)",
                min_value=0,
            ),
            ParamDefinition(
                name="mode",
                param_type=ParamType.SELECT,
                default="RGB",
                description="Color mode",
                options=["RGB", "L", "RGBA"],
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch
        from PIL import Image
        from torchvision import transforms

        directory = params.get("directory", "")
        pattern = params.get("pattern", "*.png")
        resize = params.get("resize", 224)
        max_images = params.get("max_images", 0)
        mode = params.get("mode", "RGB")

        from ...config import settings

        if not directory:
            raise ValueError("Directory path is required")

        d = self._resolve_directory(directory)

        # Restrict reads to project data directories
        allowed_roots = [
            settings.MODELS_DIR.parent.resolve(),
            settings.EXAMPLES_DIR.resolve(),
        ]
        if not any(d.is_relative_to(root) for root in allowed_roots):
            raise ValueError("Directory must be within the project data or examples directories")

        if not d.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Collect image paths
        paths = sorted(d.glob(pattern))
        if max_images > 0:
            paths = paths[:max_images]

        if not paths:
            raise ValueError(f"No images found matching '{pattern}' in {directory}")

        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])

        tensors = []
        for p in paths:
            try:
                img = Image.open(p).convert(mode)
                tensors.append(transform(img))
            except Exception as e:
                logger.warning("Skipping %s: %s", p.name, e)

        if not tensors:
            raise ValueError("No images could be loaded")

        batch = torch.stack(tensors)  # (N, C, H, W)
        logger.info("Loaded %d images, batch shape: %s", len(tensors), batch.shape)

        return {"images": batch, "count": len(tensors)}
