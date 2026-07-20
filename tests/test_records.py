"""Tests for the format-neutral record-serialization helpers
(:mod:`safe_lab_agents.mcp.predefined.records`)."""

from __future__ import annotations

import h5py
import numpy as np

from safe_lab_agents.mcp.predefined.records import (
    extract_arrays,
    flatten_record,
    is_quantity,
    json_safe,
    quantity,
    split_quantity,
)


class TestFlattenRecord:
    def test_flat_mapping_unchanged(self):
        assert flatten_record({"a": 1, "b": "x"}) == {"a": 1, "b": "x"}

    def test_nested_dict_dotted_keys(self):
        assert flatten_record({"scan": {"x": 1, "y": 2}}) == {"scan.x": 1, "scan.y": 2}

    def test_nested_list_indexed_keys(self):
        assert flatten_record({"pts": [10, 20]}) == {"pts.0": 10, "pts.1": 20}

    def test_quantity_kept_whole(self):
        q = quantity(2.5, "W")
        assert flatten_record({"power": q}) == {"power": q}

    def test_ndarray_ref_kept_whole(self):
        ref = {"_type": "ndarray", "dataset": "/g/x", "shape": [3]}
        assert flatten_record({"scan": {"x": ref}}) == {"scan.x": ref}

    def test_deeply_nested(self):
        assert flatten_record({"a": {"b": [{"c": 1}]}}) == {"a.b.0.c": 1}

    def test_none_mapping(self):
        assert flatten_record(None) == {}


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

    def test_numpy_scalars_kept_numeric(self):
        """numpy scalars must survive as native numbers, not be stringified."""
        out = json_safe(np.int64(5))
        assert out == 5 and isinstance(out, int) and not isinstance(out, bool)

        out = json_safe(np.float32(1.5))
        assert out == 1.5 and isinstance(out, float)

        out = json_safe(np.bool_(True))
        assert out is True and isinstance(out, bool)

        # np.float64 (a float subclass) also stays numeric.
        assert isinstance(json_safe(np.float64(2.0)), float)

    def test_numpy_scalar_nested_in_dict(self):
        assert json_safe({"count": np.int64(3)}) == {"count": 3}
        assert isinstance(json_safe({"count": np.int64(3)})["count"], int)


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

    def test_array_in_nested_list_is_extracted(self, tmp_path):
        """Arrays inside a list value (previously stringified) are now written."""
        h5 = tmp_path / "x.h5"
        out = extract_arrays({"traces": [np.arange(3), np.arange(3, 6)]}, h5, "/g")
        refs = out["traces"]
        assert [r["_type"] for r in refs] == ["ndarray", "ndarray"]
        with h5py.File(str(h5), "r") as f:
            np.testing.assert_array_equal(f[refs[0]["dataset"].lstrip("/")][()], np.arange(3))
            np.testing.assert_array_equal(f[refs[1]["dataset"].lstrip("/")][()], np.arange(3, 6))
        assert refs[0]["dataset"] == "/g/traces/0"
        assert refs[1]["dataset"] == "/g/traces/1"

    def test_array_in_nested_dict_is_extracted(self, tmp_path):
        """Arrays inside a nested dict value (previously stringified) are written."""
        h5 = tmp_path / "x.h5"
        out = extract_arrays({"scan": {"x": np.arange(4), "n": 2}}, h5, "")
        assert out["scan"]["x"]["_type"] == "ndarray"
        assert out["scan"]["n"] == 2  # non-array sibling preserved
        with h5py.File(str(h5), "r") as f:
            np.testing.assert_array_equal(
                f[out["scan"]["x"]["dataset"].lstrip("/")][()], np.arange(4)
            )

    def test_nested_array_quantity_keeps_unit(self, tmp_path):
        h5 = tmp_path / "x.h5"
        out = extract_arrays({"scan": {"trace": quantity(np.arange(5), "V")}}, h5, "")
        ref = out["scan"]["trace"]
        assert ref["unit"] == "V"
        with h5py.File(str(h5), "r") as f:
            assert f[ref["dataset"].lstrip("/")].attrs["units"] == "V"

    def test_no_array_anywhere_writes_no_file(self, tmp_path):
        h5 = tmp_path / "x.h5"
        out = extract_arrays({"scan": {"x": [1, 2, 3], "label": "a"}}, h5, "")
        assert out == {"scan": {"x": [1, 2, 3], "label": "a"}}
        assert not h5.exists()

    def test_has_arrays_is_recursive(self):
        from safe_lab_agents.mcp.predefined.records import has_arrays

        assert has_arrays({"scan": {"x": np.arange(3)}})
        assert has_arrays({"traces": [1, np.arange(3)]})
        assert not has_arrays({"scan": {"x": [1, 2], "n": 3}})

    def test_concurrent_appends_to_one_file_do_not_corrupt(self, tmp_path):
        """Many threads appending distinct groups to the SAME .h5 (the batch
        scenario) must not corrupt libhdf5; ``_h5_lock`` serializes the writes."""
        import concurrent.futures

        h5 = tmp_path / "batch.h5"
        n = 64
        arrays = {f"g{i}": np.arange(i, i + 5) for i in range(n)}

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(lambda g: extract_arrays(arrays[g], h5, f"/{g}"), arrays))

        # Every dataset must be present and readable with the value it was
        # written with — no lost or truncated writes.
        with h5py.File(str(h5), "r") as f:
            assert sorted(f.keys()) == sorted(arrays)
            for g, arr in arrays.items():
                assert np.array_equal(f[f"{g}/data"][()], arr)
