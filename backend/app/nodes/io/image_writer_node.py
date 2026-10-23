import logging
from typing import Any

from ...core.advisories import emit_advisory
from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

logger = logging.getLogger(__name__)

#: Prefix on the canvas-log line this node's advisory carries.
_LOG_PREFIX = "[WARN] ImageWriter: "


class ImageWriterNode(BaseNode):
    NODE_NAME = "ImageWriter"
    CATEGORY = "IO"
    DESCRIPTION = "Save an image tensor to a PNG, JPEG, BMP or TIFF file"
    DETAILS = (
        "The extension follows the chosen format (.jpeg and .tif are kept as "
        "written). A relative path lands under the data directory's output "
        "folder. A batch of N > 1 images, (N, C, H, W), writes one file per "
        "image with a zero-padded index after the name (shape_0.png, "
        "shape_1.png, ...); files from an earlier, larger batch are left in "
        "place. Float values are read as [0, 1] and clipped to it, with a "
        "run-log warning when any are negative; a uint8 image is read as "
        "0-255. Wire `directory` into ImageBatchReader's `directory` to read "
        "the files back after they are written."
    )

    # The write to disk IS this node's output. A cache hit returns the
    # recorded {"path": ...} without calling execute() again, which is
    # correct for a pure node and wrong here: if the user deleted the file
    # (or it never ran with Rec off), a cache hit would hand back a path
    # that no longer points at anything (#143).
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            # TENSOR, which an IMAGE source also connects to: a synthetic or
            # computed batch (TensorCreate, DatasetBatch) is a TENSOR, and the
            # type system does not convert TENSOR to IMAGE.
            PortDefinition(
                name="image",
                data_type=DataType.TENSOR,
                description="Image (C, H, W) or (H, W), or a batch (N, C, H, W)",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="path", data_type=DataType.STRING,
                           description="Path to the saved image (the first one, for a batch)"),
            PortDefinition(name="paths", data_type=DataType.LIST,
                           description="Every file written, in index order"),
            PortDefinition(name="directory", data_type=DataType.STRING,
                           description="Folder the files were written to"),
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

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        import torch
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
        # A spelling of the format's own extension (photo.jpeg, scan.tif) is
        # kept as written; only a different extension is replaced.
        ext_map = {"PNG": (".png",), "JPEG": (".jpg", ".jpeg"), "BMP": (".bmp",),
                   "TIFF": (".tiff", ".tif")}
        accepted_ext = ext_map.get(fmt, (".png",))
        if p.suffix.lower() not in accepted_ext:
            p = resolve_data_path(p.with_suffix(accepted_ext[0]),
                                  base=settings.MODELS_DIR.parent / "output")

        p.parent.mkdir(parents=True, exist_ok=True)

        # One file per image of a batch. A batch of one keeps the plain name,
        # which is what a single image unsqueezed to (1, C, H, W) used to get.
        if image.dim() == 4 and image.shape[0] > 1:
            width = len(str(image.shape[0] - 1))
            targets = [
                resolve_data_path(p.with_name(f"{p.stem}_{i:0{width}d}{p.suffix}"),
                                  base=settings.MODELS_DIR.parent / "output")
                for i in range(image.shape[0])
            ]
            images = list(image)
        else:
            targets = [p]
            images = [image[0] if image.dim() == 4 else image]

        result: dict[str, Any] = {}
        if image.numel() and float(image.min()) < 0:
            lo, hi = float(image.min()), float(image.max())
            result["__log__"] = emit_advisory(
                f"the image's values span [{lo:.3g}, {hi:.3g}], and everything "
                f"below 0 is saved as black (the file stores [0, 1]). If it is "
                f"normalized to [-1, 1], rescale it first with (x + 1) / 2; "
                f"if it was standardized with a mean and std, undo that "
                f"before this node.",
                kind="image_writer_negative_values",
                prefix=_LOG_PREFIX, context=context, logger=logger)

        for target, single in zip(targets, images):
            # save_image expects floats in [0, 1]. A uint8 image (0..255, as
            # a PIL or NumPy conversion produces) crashed inside it with
            # "result type Float can't be cast to ... Byte"; dividing by 255
            # makes the round trip exact.
            if single.dtype == torch.uint8:
                single = single.float() / 255.0
            elif not single.is_floating_point():
                single = single.float()
            save_image(single, str(target))
        logger.info("Saved %d image(s) to %s", len(targets), p.parent)

        result["path"] = str(targets[0])
        result["paths"] = [str(t) for t in targets]
        result["directory"] = str(p.parent)
        return result
