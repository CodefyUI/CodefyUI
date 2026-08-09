import logging
from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)


class ImageWriterNode(BaseNode):
    NODE_NAME = "ImageWriter"
    CATEGORY = "IO"
    DESCRIPTION = "Save a tensor as an image file (PNG, JPEG, etc.)"

    # The write to disk IS this node's output. A cache hit returns the
    # recorded {"path": ...} without calling execute() again, which is
    # correct for a pure node and wrong here: if the user deleted the file
    # (or it never ran with Rec off), a cache hit would hand back a path
    # that no longer points at anything (#143).
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="image", data_type=DataType.IMAGE, description="Image tensor (C, H, W) or (H, W)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="path", data_type=DataType.STRING, description="Path to the saved image"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="path",
                param_type=ParamType.STRING,
                default="output.png",
                description="Output file path",
            ),
            ParamDefinition(
                name="format",
                param_type=ParamType.SELECT,
                default="PNG",
                description="Image format",
                options=["PNG", "JPEG", "BMP", "TIFF"],
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        from torchvision.utils import save_image

        from ...config import settings
        from ...core.data_paths import resolve_data_path

        image = inputs["image"]
        path = params.get("path", "output.png")
        fmt = params.get("format", "PNG")

        # Inside the data directory, and not over CodefyUI's own storage --
        # ``core.data_paths`` owns both halves of that rule and is shared
        # with ModelSaver and the checkpoint writers (#224). A relative path
        # lands under <data>/output rather than MODELS_DIR, which is why the
        # base is passed rather than assumed.
        #
        # This node was never able to reach the database (the forced
        # extension below rewrote the name first); what it could overwrite
        # was any file under the data root ending in an image extension.
        p = resolve_data_path(path, base=settings.MODELS_DIR.parent / "output")

        # Ensure correct extension, then re-validate: the path written must
        # be the path checked. Not theoretical -- DB_PATH is env-overridable,
        # so a database named `store.png` is reachable by asking for
        # `store.jpg` with format=PNG and letting the rewrite do the rest.
        ext_map = {"PNG": ".png", "JPEG": ".jpg", "BMP": ".bmp", "TIFF": ".tiff"}
        expected_ext = ext_map.get(fmt, ".png")
        if p.suffix.lower() != expected_ext:
            p = resolve_data_path(p.with_suffix(expected_ext),
                                  base=settings.MODELS_DIR.parent / "output")

        p.parent.mkdir(parents=True, exist_ok=True)

        # Handle batched tensors: save first image
        if image.dim() == 4:
            image = image[0]

        save_image(image, str(p))
        logger.info("Saved image to %s", p)

        return {"path": str(p)}
