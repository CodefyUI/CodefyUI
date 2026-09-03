"""One re-discovery call, for every place that has to trigger one.

``rediscover_all`` takes seven arguments and every caller passes the same
seven: the process-wide node and preset registries, the three directories
``settings`` names, and the two plugin roots. That full spelling was written
out four times -- in ``POST /api/nodes/reload``, ``POST /api/plugins/reload``,
the plugin enable/disable handler and the custom-node upload/delete handler --
which made "reload everything" a nine-line quotation rather than a call, and
made a fifth caller a copy-paste job. Worse, the copies were free to drift:
a route that forgot ``presets_dir`` would clear the presets and never put
them back, and nothing but a reader comparing all four would notice.

So the arguments are fixed here, once, and the routes ask for the ANSWER.
The plugin install and uninstall routes of the Plugin Center are that fifth
and sixth caller; they get the same five counts without repeating a line.

The settings and the plugin roots are read at CALL time, not at import time:
a test redirects ``settings.CUSTOM_NODES_DIR`` or the user-data root for the
duration of one test, and a re-discovery run inside it has to see the
redirect rather than whatever was true when this module was first imported.
"""

from __future__ import annotations

from app.config import settings
from app.core import plugin_loader
from app.core.node_registry import registry
from app.core.preset_registry import preset_registry


def rediscover_now() -> dict[str, int]:
    """Clear and re-discover every node and preset source.

    Returns ``{"builtin", "custom", "plugins", "presets", "total"}`` -- the
    counts ``POST /api/nodes/reload`` and ``POST /api/plugins/reload`` return
    verbatim -- and bumps the reload generation the editor polls.
    """
    return plugin_loader.rediscover_all(
        registry,
        preset_registry,
        nodes_dir=settings.NODES_DIR,
        custom_nodes_dir=settings.CUSTOM_NODES_DIR,
        presets_dir=settings.PRESETS_DIR,
        builtin_root=plugin_loader.plugins_builtin_root(),
        user_root=plugin_loader.plugins_user_root(),
    )
