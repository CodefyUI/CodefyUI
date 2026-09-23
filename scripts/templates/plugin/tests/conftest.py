"""Shared pytest setup for this plugin's tests.

The tests import the nodes straight from the plugin root
(``from nodes.example_node import ExampleNode``), which ``pytest.ini`` makes
possible with ``pythonpath = .``, so nothing here has to touch ``sys``.

Keep this file and every test free of ``sys``, ``os``, ``pathlib`` and the
other modules CodefyUI's install-time security scan refuses: the scan reads
every ``.py`` file in the plugin, tests included, and one refused import stops
the whole plugin from installing. Read files with plain ``open()``, which
needs no grant.
"""
