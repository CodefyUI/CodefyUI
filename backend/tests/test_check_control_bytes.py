"""Unit tests for scripts/check_control_bytes.py (#209 review follow-up).

The guard itself has no other tests exercising it -- CI just runs it as a
script -- so a change that silently narrowed what it catches (the UTF-8
skip broadening, the vendored-dump exemption growing) would stay green
forever. These pin ``_scan`` and the vendored-dump predicate directly, the
same way ``backend/tests/test_dev_cli.py`` pins ``scripts/dev.py``'s pure
helpers.
"""

from __future__ import annotations

import check_control_bytes as ccb  # scripts/check_control_bytes.py -- conftest puts scripts/ on sys.path


def test_scan_finds_a_raw_nul_with_its_line_number():
    assert ccb._scan(b"a\x00b") == [(1, 0x00)]


def test_scan_clean_text_reports_nothing():
    assert ccb._scan(b"ok\n") == []


def test_scan_allows_tab_lf_cr():
    # Tab, LF and CR are C0 too, but legitimate in text source (CR shows up
    # in CRLF-checked-out files) -- must never be flagged.
    assert ccb._scan(b"a\tb\nc\rd\n") == []


def test_scan_flags_the_whole_c0_range_not_just_nul():
    # The same hazard construct has also produced 0x01 and 0x02 in this
    # repo's history (issue #209's own comment thread) -- a NUL-only scan
    # would have missed them.
    assert ccb._scan(b"a\x01b\x02c") == [(1, 0x01), (1, 0x02)]


def test_scan_reports_the_correct_line_across_multiple_lines():
    data = b"line one\nline two\x00\nline three\n"
    assert ccb._scan(data) == [(2, 0x00)]


def test_scan_skips_binary_data_entirely():
    # Not valid UTF-8 (0x89 is an invalid UTF-8 leading byte on its own --
    # the PNG signature byte) -- must be skipped rather than scanned, even
    # though it contains plenty of C0-range bytes.
    png_like = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    assert ccb._scan(png_like) == []


def test_is_vendored_binary_dump_matches_only_the_mnist_ubyte_files():
    assert ccb._is_vendored_binary_dump(
        "backend/data/MNIST/raw/train-labels-idx1-ubyte")
    assert ccb._is_vendored_binary_dump(
        "backend/data/MNIST/raw/t10k-images-idx3-ubyte")
    # The suffix match is deliberately narrow: a DIFFERENT file added later
    # under the same directory (a README, say) must not be silently
    # exempted along with the actual binary dumps.
    assert not ccb._is_vendored_binary_dump(
        "backend/data/MNIST/raw/README.md")
    assert not ccb._is_vendored_binary_dump(
        "backend/data/MNIST/raw/train-labels-idx1-ubyte.gz")
    # Same suffix, wrong directory -- also not exempt.
    assert not ccb._is_vendored_binary_dump("some/other/dir/train-ubyte")
