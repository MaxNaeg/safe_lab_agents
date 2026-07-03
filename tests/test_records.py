"""Tests for the format-neutral record-serialization helpers
(:mod:`safe_lab_agents.mcp.predefined.records`)."""

from __future__ import annotations

import h5py
import numpy as np

from safe_lab_agents.mcp.predefined.records import (
    extract_arrays,
    is_quantity,
    json_safe,
    quantity,
    split_quantity,
)


class TestJsonSafe:
    def test_passthrough_scalars(self):
        assert json_safe(1) == 1
        assert json_safe("x") == "x"
        assert json_safe(True) is True
        assert json_safe(None) is None

    def test_nested(self):
        assert json_safe({"a": [1, (2, 3)]}) == {"a": [1, [2, 3]]}

    def test_unknown_stringified(self):
        assert json_safe(object()).startswith("<object")


class TestQuantity:
    def test_quantity_builds_dict(self):
        assert quantity(2.5, "W") == {"value": 2.5, "unit": "W"}

    def test_quantity_with_term(self):
        q = quantity(2.5, "W", term="http://qudt.org/vocab/unit/W")
        assert q["term"] == "http://qudt.org/vocab/unit/W"

    def test_is_quantity(self):
        assert is_quantity(quantity(1, "V"))
        assert not is_quantity({"value": 1})
        assert not is_quantity({"a": 1})
        # An ndarray reference is not a quantity even with a unit key.
        assert not is_quantity({"_type": "ndarray", "value": 1, "unit": "V"})

    def test_split_quantity(self):
        value, unit, term = split_quantity(quantity(2.5, "W", term="iri"))
        assert (value, unit, term) == (2.5, "W", "iri")


class TestExtractArrays:
    def test_bare_array(self, tmp_path):
        h5 = tmp_path / "x.h5"
        ref = extract_arrays(np.arange(3), h5, "")
        assert ref["_type"] == "ndarray"
        assert ref["shape"] == [3]
        assert "unit" not in ref

    def test_dict_arrays(self, tmp_path):
        h5 = tmp_path / "x.h5"
        out = extract_arrays({"a": np.arange(4), "b": 2}, h5, "/grp")
        assert out["a"]["_type"] == "ndarray"
        assert out["a"]["dataset"] == "/grp/a"
        assert out["b"] == 2

    def test_array_quantity_carries_unit_and_hdf5_attr(self, tmp_path):
        h5 = tmp_path / "x.h5"
        out = extract_arrays({"trace": quantity(np.arange(5), "V")}, h5, "")
        ref = out["trace"]
        assert ref["_type"] == "ndarray"
        assert ref["unit"] == "V"
        with h5py.File(str(h5), "r") as f:
            assert f[ref["dataset"].lstrip("/")].attrs["units"] == "V"

    def test_scalar_quantity_untouched(self, tmp_path):
        h5 = tmp_path / "x.h5"
        out = extract_arrays({"power": quantity(2.5, "W")}, h5, "")
        assert out["power"] == {"value": 2.5, "unit": "W"}
        assert not h5.exists()
