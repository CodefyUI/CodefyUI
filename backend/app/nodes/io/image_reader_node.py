from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class ImageReaderNode(BaseNode):
    NODE_NAME = "ImageReader"
    CATEGORY = "IO"
    DESCRIPTION = "Read an image file from disk and output as a tensor (C, H, W) with values in [0, 1]"

    # #144: cacheable again -- cache_fingerprint() below folds the resolved
    # file's (size, mtime, and for small files a content hash) into the
    # cache key, so a replaced image busts the cache instead of the key
    # surviving unchanged because only the `path` string is hashed.
    cacheable = True

    @staticmethod
    def _resolve_path(path: str) -> "Path":
        from pathlib import Path

        from ...config import settings

        p = Path(path)
        if not p.is_absolute():
            # Relative paths resolve against IMAGES_DIR so filenames picked
            # from the uploaded-files dropdown work without the user typing
            # a full path.
            p = settings.IMAGES_DIR / p
        return p

    @classmethod
    def cache_fingerprint(cls, params: dict[str, Any]) -> Any:
        from ...core.cache_fingerprint import path_fingerprint

        path = str(params.get("path", "") or "")
        if not path:
            return None
        try:
            resolved = cls._resolve_path(path)
        except Exception:
            return None
        return path_fingerprint(resolved)

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="image", data_type=DataType.IMAGE, description="Image as tensor (C, H, W)"),
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Same image as generic tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.IMAGE_FILE,
                default="",
                description="Image file — select an uploaded file or upload a new one",
            ),
            ParamDefinition(
                name="mode",
                param_type=ParamType.SELECT,
                default="RGB",
                description="Color mode for the loaded image",
                options=["RGB", "L", "RGBA"],
            ),
            ParamDefinition(
                name="resize",
                param_type=ParamType.INT,
                default=0,
                description="Resize image to a (resize, resize) square (0 = no resize)",
                min_value=0,
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image, ImageFile
        from torchvision import transforms

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        path = params.get("path", "")
        mode = params.get("mode", "RGB")
        resize = params.get("resize", 0)

        if not path:
            raise ValueError("Image path is required")

        p = self._resolve_path(path)

        if not p.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Some PNGs contain non-standard trailing chunks (e.g. 'ghtt') that
        # cause Pillow's load_end() to fail with "Truncated File Read".
        # The pixel data is already decoded at that point, so we catch the
        # error and proceed.
        img = Image.open(p)
        try:
            img.load()
        except OSError:
            pass
        img = img.convert(mode)

        transform_list = []
        if resize > 0:
            # Resize to (resize, resize) square — required by downstream
            # convolutional models that expect a fixed input shape.
            transform_list.append(transforms.Resize((resize, resize)))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)

        tensor = transform(img)  # (C, H, W), values in [0, 1]

        return {"image": tensor, "tensor": tensor}
