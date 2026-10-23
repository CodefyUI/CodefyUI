import logging
from typing import TYPE_CHECKING, Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from .image_reader_node import TRANSFORM_INPUT_DESC, apply_wired_transform

if TYPE_CHECKING:
    # `pathlib` is imported lazily inside the methods below (startup cost);
    # this makes the "Path" annotations resolvable to a type checker without
    # paying that cost at import time.
    from pathlib import Path

logger = logging.getLogger(__name__)


def expand_braces(pattern: str) -> list[str]:
    """Expand shell-style ``{a,b}`` alternatives into separate glob patterns.

    ``pathlib`` globbing has no brace expansion, so the documented
    ``*.{png,jpg}`` matched nothing at all. Nested and repeated groups expand
    left to right; a pattern without a complete ``{...,...}`` group is
    returned unchanged.
    """
    start = pattern.find("{")
    if start < 0:
        return [pattern]
    depth = 0
    for end in range(start, len(pattern)):
        if pattern[end] == "{":
            depth += 1
        elif pattern[end] == "}":
            depth -= 1
            if depth == 0:
                break
    else:
        return [pattern]
    body = pattern[start + 1:end]
    # Split the group on top-level commas only.
    options, depth, last = [], 0, 0
    for i, ch in enumerate(body):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            options.append(body[last:i])
            last = i + 1
    options.append(body[last:])
    if len(options) < 2:
        head = pattern[:end + 1]
        return [head + rest for rest in expand_braces(pattern[end + 1:])]
    prefix, suffix = pattern[:start], pattern[end + 1:]
    expanded: list[str] = []
    for option in options:
        for candidate in expand_braces(prefix + option + suffix):
            if candidate not in expanded:
                expanded.append(candidate)
    return expanded


def glob_sorted(directory: "Path", pattern: str) -> list["Path"]:
    """Every file under *directory* matching *pattern*, braces expanded,
    without duplicates, in sorted order."""
    found: set["Path"] = set()
    for single in expand_braces(pattern):
        found.update(directory.glob(single))
    return sorted(found)


class ImageBatchReaderNode(BaseNode):
    NODE_NAME = "ImageBatchReader"
    CATEGORY = "IO"
    DESCRIPTION = "Stack every image in a directory into (N, C, H, W)"
    DETAILS = (
        "Files matching the glob pattern are read in sorted order and each is "
        "resized to the same square size so they can be stacked; an image that "
        "fails to open is skipped. max_images caps how many are read. A chain "
        "wired into `transform` (the training dataset's eval_transform) replaces "
        "resize and the [0, 1] conversion, so the batch is preprocessed exactly "
        "as the training data was; every image must then come out the same size. "
        "A wired `directory` (ImageWriter's `directory` output) replaces the "
        "directory param and runs this node after the writer, under the same "
        "rule: inside the data or examples directories."
    )

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
            paths = glob_sorted(d, pattern)
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
        return [
            PortDefinition(
                name="transform",
                data_type=DataType.TRANSFORM,
                description=TRANSFORM_INPUT_DESC,
                optional=True,
            ),
            PortDefinition(
                name="directory",
                data_type=DataType.STRING,
                description=(
                    "Directory to read (overrides the directory param when "
                    "wired). Wire ImageWriter.directory here to read its "
                    "files after it writes them."
                ),
                optional=True,
            ),
        ]

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
                description="Resize all images to this square size (required for batching). Ignored when a transform is wired.",
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

        # The wired directory wins, as VideoLoad's path does. The edge is
        # also what orders this read after an upstream ImageWriter: a
        # trigger edge does not order execution.
        directory = str(inputs.get("directory") or params.get("directory", "") or "")
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
        paths = glob_sorted(d, pattern)
        if max_images > 0:
            paths = paths[:max_images]

        if not paths:
            raise ValueError(f"No images found matching '{pattern}' in {directory}")

        wired = inputs.get("transform")
        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])

        tensors = []
        names = []
        for p in paths:
            try:
                img = Image.open(p).convert(mode)
            except Exception as e:
                logger.warning("Skipping %s: %s", p.name, e)
                continue
            if wired is not None:
                # Outside the skip above: a chain that is wrong for these
                # images is wrong for all of them, so it fails the node.
                tensors.append(apply_wired_transform(wired, img, "ImageBatchReader", p.name))
            else:
                tensors.append(transform(img))
            names.append(p.name)

        if not tensors:
            raise ValueError("No images could be loaded")

        first = tuple(tensors[0].shape)
        for name, t in zip(names, tensors):
            if tuple(t.shape) != first:
                raise ValueError(
                    f"ImageBatchReader: the images do not share one size after "
                    f"preprocessing ({names[0]} is {first}, {name} is "
                    f"{tuple(t.shape)}), so they cannot be stacked. Add "
                    f"ResizeTransform to the wired chain."
                )

        batch = torch.stack(tensors)  # (N, C, H, W)
        logger.info("Loaded %d images, batch shape: %s", len(tensors), batch.shape)

        return {"images": batch, "count": len(tensors)}
