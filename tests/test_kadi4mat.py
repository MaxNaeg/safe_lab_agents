"""Tests for Kadi4Mat ELN integration.

All tests run without a real Kadi4Mat instance.
"""

from __future__ import annotations

import re
import time

import numpy as np

from safe_lab_agents import quantity
from safe_lab_agents.mcp.predefined.kadi4mat_utils import (
    make_collection_identifier,
    make_record_identifier,
    make_user_slug,
    slugify,
)


# ======================================================================
# slugify
# ======================================================================


class TestSlugify:
    def test_basic(self):
        assert slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        assert slugify("Transmission @ 2.5V!") == "transmission-2-5v"

    def test_max_length(self):
        result = slugify("a very long title that should be truncated", max_length=10)
        assert len(result) <= 10
        assert not result.endswith("-")

    def test_consecutive_hyphens(self):
        assert slugify("a---b___c") == "a-b-c"

    def test_empty(self):
        assert slugify("") == ""

    def test_only_special_chars(self):
        assert slugify("@#$%") == ""


# ======================================================================
# make_user_slug
# ======================================================================


class TestMakeUserSlug:
    def test_email_takes_local_part(self):
        assert make_user_slug("mpt240@uni-erlangen.de") == "mpt240"

    def test_simple_username(self):
        assert make_user_slug("admin") == "admin"

    def test_dots_in_local_part(self):
        assert make_user_slug("john.doe@kit.edu") == "john-doe"

    def test_truncates_to_max_length(self):
        result = make_user_slug("verylongusername@example.com")
        assert len(result) <= 8

    def test_custom_max_length(self):
        result = make_user_slug("mpt240@uni-erlangen.de", max_length=4)
        assert result == "mpt2"


# ======================================================================
# make_collection_identifier
# ======================================================================


class TestMakeCollectionIdentifier:
    def test_basic(self):
        result = make_collection_identifier("mpt240", "optical-resonances")
        assert result == "mpt240-optical-resonances"

    def test_slugifies_project(self):
        result = make_collection_identifier("user", "My Cool Project!")
        assert result == "user-my-cool-project"

    def test_respects_50_char_limit(self):
        result = make_collection_identifier(
            "mpt240", "a-very-long-project-name-that-might-exceed-limits"
        )
        assert len(result) <= 50


# ======================================================================
# make_record_identifier
# ======================================================================


class TestMakeRecordIdentifier:
    def test_format(self):
        result = make_record_identifier("mpt240", "optical", "Scan at 2V")
        # Should match: user_slug-project_slug-YYYYMMDD-HHMMSS-microseconds
        pattern = r"^mpt240-optical-\d{8}-\d{6}-\d{6}$"
        assert re.match(
            pattern, result
        ), f"Identifier '{result}' does not match expected pattern"

    def test_within_50_chars(self):
        result = make_record_identifier("mpt240", "my-project", "some title")
        assert len(result) <= 50

    def test_uniqueness_across_calls(self):
        # Two calls produce different identifiers (different microsecond timestamps)
        import time

        id1 = make_record_identifier("user", "proj", "title")
        time.sleep(0.001)  # ensure different microsecond
        id2 = make_record_identifier("user", "proj", "title")
        assert id1 != id2

    def test_title_not_in_identifier(self):
        # Title is stored in the record's title field, not the identifier
        result = make_record_identifier("user", "proj", "My Long Title")
        assert "my-long-title" not in result


# ======================================================================
# auto-log → Kadi4Mat push integration
# ======================================================================


class TestAutoLogKadiIntegration:
    """Tests for the auto-log → Kadi4Mat push integration.

    Uses a mock KadiClient injected directly into an AutoLogger instance.
    No real Kadi4Mat instance required.
    """

    def _make_mock_client(self):
        from unittest.mock import MagicMock

        client = MagicMock()
        client.create_record.return_value = "test-record-id"
        return client

    def _setup(self, tmp_path, mock_client=None):
        from safe_lab_agents.mcp.predefined.autolog import AutoLogger

        auto_logger = AutoLogger(output_dir=tmp_path, kadi_client=mock_client)
        return auto_logger.wrapper, auto_logger

    def _teardown(self, auto_logger):
        # Nothing to clean up: all state lives on the instance, which is dropped
        # when the test returns.  Kept for symmetry with the try/finally callers.
        pass

    def test_individual_record_calls_create_record(self, tmp_path):
        mock_client = self._make_mock_client()
        wrapper, auto_logger = self._setup(tmp_path, mock_client)
        try:

            def measure(voltage: float) -> dict:
                """Measure something."""
                return {"voltage": voltage, "reading": 0.5}

            wrapper(measure)(voltage=3.0)

            mock_client.create_record.assert_called_once()
            kwargs = mock_client.create_record.call_args[1]
            assert kwargs["title"] == "measure"
            assert kwargs["call_args"] == {"voltage": 3.0}
            assert kwargs["result"]["voltage"] == 3.0
            assert kwargs["result"]["reading"] == 0.5
        finally:
            self._teardown(auto_logger)

    def test_nested_array_flattened_into_dotted_extra(self, tmp_path):
        """An array nested in a dict result reaches Kadi as its own dotted extra
        (scan.x) with an ndarray summary, not a stringified reference dict."""
        import numpy as np

        mock_client = self._make_mock_client()
        wrapper, auto_logger = self._setup(tmp_path, mock_client)
        try:

            def measure() -> dict:
                """Measure a nested trace."""
                return {"scan": {"x": np.arange(3), "n": 5}}

            wrapper(measure)()

            result = mock_client.create_record.call_args[1]["result"]
            assert result["scan.x"].startswith("ndarray[3]")
            assert result["scan.n"] == 5
            # No value is a raw reference dict / nested container.
            assert not any(isinstance(v, (dict, list)) for v in result.values())
        finally:
            self._teardown(auto_logger)

    def test_kadi_failure_does_not_break_tool(self, tmp_path):
        mock_client = self._make_mock_client()
        mock_client.create_record.side_effect = RuntimeError("Kadi is down")
        wrapper, auto_logger = self._setup(tmp_path, mock_client)
        try:

            def measure(voltage: float) -> float:
                """Read voltage."""
                return voltage * 2

            result = wrapper(measure)(voltage=1.5)
            assert result == 3.0
        finally:
            self._teardown(auto_logger)

    def test_no_kadi_client_skips_push(self, tmp_path):
        wrapper, auto_logger = self._setup(tmp_path, mock_client=None)
        try:
            call_count = {"n": 0}

            def measure(v: float) -> dict:
                return {"v": v}

            wrapped = wrapper(measure)
            wrapped(v=1.0)
            # No exception means push was silently skipped
            assert call_count["n"] == 0
        finally:
            self._teardown(auto_logger)

    def test_hdf5_file_passed_to_kadi_when_arrays_present(self, tmp_path):
        mock_client = self._make_mock_client()
        wrapper, auto_logger = self._setup(tmp_path, mock_client)
        try:

            def scan() -> dict:
                return {"data": np.array([1.0, 2.0, 3.0]), "label": "test"}

            wrapper(scan)()

            mock_client.create_record.assert_called_once()
            kwargs = mock_client.create_record.call_args[1]
            # ndarray represented as shape/dtype summary string
            assert kwargs["result"]["data"] == "ndarray[3] float64"
            assert kwargs["result"]["label"] == "test"
            # JSON + HDF5 both attached
            suffixes = {f.suffix for f in kwargs["files"]}
            assert ".h5" in suffixes
            assert ".json" in suffixes
        finally:
            self._teardown(auto_logger)

    def test_no_autolog_decorator_skips_kadi(self, tmp_path):
        from safe_lab_agents.mcp.predefined.autolog import no_autolog

        mock_client = self._make_mock_client()
        wrapper, auto_logger = self._setup(tmp_path, mock_client)
        try:

            @no_autolog
            def helper(x: int) -> int:
                return x * 2

            result = wrapper(helper)(3)
            assert result == 6
            mock_client.create_record.assert_not_called()
        finally:
            self._teardown(auto_logger)

    def test_batch_pushes_to_kadi_on_stop(self, tmp_path):
        mock_client = self._make_mock_client()
        wrapper, auto_logger = self._setup(tmp_path, mock_client)
        try:
            auto_logger.start_batch("Voltage sweep")

            def measure(v: float) -> dict:
                return {"v": v}

            wrapper(measure)(v=1.0)
            wrapper(measure)(v=2.0)
            # No kadi push during batch
            mock_client.create_record.assert_not_called()

            auto_logger.stop_batch()
            # One kadi push for the whole batch
            mock_client.create_record.assert_called_once()
            kwargs = mock_client.create_record.call_args[1]
            assert kwargs["title"] == "Voltage sweep"
        finally:
            self._teardown(auto_logger)

    def test_log_analysis_pushes_to_kadi(self, tmp_path):
        mock_client = self._make_mock_client()
        wrapper, auto_logger = self._setup(tmp_path, mock_client)
        try:
            result = auto_logger.log_analysis(
                title="Linear fit",
                text="Power scales linearly.",
                data={"slope": 0.023, "residuals": np.array([0.1, -0.1, 0.05])},
                references=["exp_20260522_111149_616781"],
            )
            assert "Linear fit" in result

            mock_client.create_record.assert_called_once()
            kwargs = mock_client.create_record.call_args[1]
            assert kwargs["title"] == "Linear fit"
            # Scalar data value passed as kadi metadata
            assert kwargs["result"].get("slope") == 0.023
            # ndarray represented as shape/dtype string, JSON + HDF5 both attached
            assert kwargs["result"]["residuals"] == "ndarray[3] float64"
            suffixes = {f.suffix for f in kwargs["files"]}
            assert ".h5" in suffixes
            assert ".json" in suffixes
            # text forwarded to kadi
            assert kwargs["result"].get("text") == "Power scales linearly."
        finally:
            self._teardown(auto_logger)


# ======================================================================
# KadiClient rate limiting
# ======================================================================


class TestKadiClientRateLimiting:
    def test_session_limit_blocks_next_call(self):
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        client = KadiClient(project="test", max_per_session=3, max_per_minute=100)
        # Simulate that 3 records were already created (limit reached).
        client._session_count = 3

        # The NEXT call should be blocked entirely.
        msg = client._check_rate_limit_before()
        assert msg is not None
        assert "NOT saved" in msg

    def test_session_limit_warns_after_reaching(self):
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        client = KadiClient(project="test", max_per_session=3, max_per_minute=100)
        # Simulate that this record just made it to the limit.
        client._session_count = 3

        msg = client._check_rate_limit_after()
        assert msg is not None
        assert "NEXT tool call will NOT be recorded" in msg

    def test_per_minute_warns_after_reaching(self):
        import time as time_mod
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        client = KadiClient(project="test", max_per_session=1000, max_per_minute=2)
        # Simulate 2 recent records — limit just reached.
        now = time_mod.monotonic()
        client._recent_timestamps = [now - 10, now - 5]

        msg = client._check_rate_limit_after()
        assert msg is not None
        assert "NEXT tool call will NOT be recorded" in msg
        assert "seconds" in msg

    def test_per_minute_blocks_next_call(self):
        import time as time_mod
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        client = KadiClient(project="test", max_per_session=1000, max_per_minute=2)
        # Simulate 2 recent records within the window — limit reached.
        now = time_mod.monotonic()
        client._recent_timestamps = [now - 10, now - 5]

        # The NEXT call must be blocked entirely, not merely warned about.
        msg = client._check_rate_limit_before()
        assert msg is not None
        assert "NOT saved" in msg
        assert "per minute" in msg

    def test_per_minute_block_clears_when_window_ages_out(self):
        import time as time_mod
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        client = KadiClient(project="test", max_per_session=1000, max_per_minute=2)
        # Both timestamps are older than 60s — the window should be empty.
        now = time_mod.monotonic()
        client._recent_timestamps = [now - 120, now - 90]

        assert client._check_rate_limit_before() is None
        assert len(client._recent_timestamps) == 0

    def test_within_limits_no_warning(self):
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        client = KadiClient(project="test", max_per_session=100, max_per_minute=10)
        client._session_count = 5

        assert client._check_rate_limit_before() is None
        assert client._check_rate_limit_after() is None

    def test_old_timestamps_expire(self):
        import time as time_mod
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        client = KadiClient(project="test", max_per_session=1000, max_per_minute=2)
        # Timestamps older than 60 seconds should be pruned.
        now = time_mod.monotonic()
        client._recent_timestamps = [now - 120, now - 90]

        msg = client._check_rate_limit_after()
        assert msg is None
        assert len(client._recent_timestamps) == 0


# ======================================================================
# _add_metadatum — units & terms
# ======================================================================


class TestAddMetadatumUnits:
    def _record(self):
        from unittest.mock import MagicMock

        return MagicMock()

    def test_quantity_sets_unit_on_numeric(self):
        from safe_lab_agents import quantity
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        record = self._record()
        KadiClient._add_metadatum(record, "result_power", quantity(2.5, "W"))
        meta = record.add_metadatum.call_args[0][0]
        assert meta == {
            "key": "result_power",
            "value": 2.5,
            "type": "float",
            "unit": "W",
        }

    def test_quantity_term_propagated(self):
        from safe_lab_agents import quantity
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        record = self._record()
        KadiClient._add_metadatum(
            record,
            "result_power",
            quantity(2.5, "W", term="http://qudt.org/vocab/unit/W"),
        )
        meta = record.add_metadatum.call_args[0][0]
        assert meta["term"] == "http://qudt.org/vocab/unit/W"

    def test_plain_value_has_no_unit(self):
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        record = self._record()
        KadiClient._add_metadatum(record, "result_status", "ok")
        meta = record.add_metadatum.call_args[0][0]
        assert meta == {"key": "result_status", "value": "ok", "type": "str"}
        assert "unit" not in meta

    def test_numpy_scalars_classified_numerically(self):
        """np.int64/np.float32/np.bool_ must not be stringified into str extras."""
        import numpy as np
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        cases = [
            (np.int64(5), "int", 5),
            (np.float32(1.5), "float", 1.5),
            (np.bool_(True), "bool", True),
        ]
        for value, expected_type, expected_value in cases:
            record = self._record()
            KadiClient._add_metadatum(record, "result_x", value)
            meta = record.add_metadatum.call_args[0][0]
            assert meta["type"] == expected_type
            assert meta["value"] == expected_value

    def test_numpy_valued_quantity_keeps_unit(self):
        """A quantity whose value is a numpy scalar must still carry its unit
        (a str-classified value would drop it)."""
        import numpy as np
        from safe_lab_agents import quantity
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        record = self._record()
        KadiClient._add_metadatum(record, "result_power", quantity(np.float32(2.5), "W"))
        meta = record.add_metadatum.call_args[0][0]
        assert meta["type"] == "float"
        assert meta["value"] == 2.5
        assert meta["unit"] == "W"


# ======================================================================
# create_record — real serialization contract (client NOT mocked)
# ======================================================================


class _FakeRecord:
    """Records what ``create_record`` builds so the real KadiClient
    serialization path (metadata typing, unit retention, file upload)
    can be asserted without a live Kadi instance."""

    def __init__(self):
        self.metadata: list[dict] = []
        self.collection_links: list[int] = []
        self.uploaded: list[str] = []

    def add_collection_link(self, collection_id):
        self.collection_links.append(collection_id)

    def add_metadatum(self, metadatum):
        self.metadata.append(metadatum)

    def upload_file(self, filepath):
        self.uploaded.append(filepath)


class _FakeManager:
    def __init__(self, record):
        self._record = record
        self.record_calls: list[dict] = []

    def record(self, **kwargs):
        self.record_calls.append(kwargs)
        return self._record


class TestCreateRecordSerialization:
    """Drive the real ``create_record`` (only the network manager is faked) so
    the numpy/quantity serialization contract and rate-limit enforcement are
    exercised end-to-end, not through a fully-mocked client."""

    def _connected_client(self, record):
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        client = KadiClient(project="proj", max_per_session=1000, max_per_minute=1000)
        # Bypass the real network connect: pretend we are already connected.
        client._manager = _FakeManager(record)
        client._user_slug = "user"
        client._collection_id = 42
        return client

    def test_real_serialization_of_numpy_and_quantities(self):
        record = _FakeRecord()
        client = self._connected_client(record)

        rid = client.create_record(
            title="measure",
            call_args={"channel": np.int64(3)},
            result={
                "power": quantity(np.float32(2.5), "W"),
                "count": np.int64(5),
                "ok": np.bool_(True),
                "note": "done",
            },
            files=[],
        )

        # Returned id is exactly the one the manager was asked to create
        # (the identifier embeds a timestamp, so it can't be recomputed).
        assert rid == client._manager.record_calls[0]["identifier"]
        assert rid.startswith("user-proj-")
        by_key = {m["key"]: m for m in record.metadata}

        # numpy int arg stays numeric (not stringified into a str extra).
        assert by_key["param_channel"]["type"] == "int"
        assert by_key["param_channel"]["value"] == 3

        # numpy-valued quantity keeps its unit and numeric type.
        assert by_key["result_power"]["type"] == "float"
        assert by_key["result_power"]["value"] == 2.5
        assert by_key["result_power"]["unit"] == "W"

        assert by_key["result_count"]["type"] == "int"
        assert by_key["result_count"]["value"] == 5
        assert by_key["result_ok"]["type"] == "bool"
        assert by_key["result_ok"]["value"] is True

        # A plain string carries no unit field.
        assert by_key["result_note"]["type"] == "str"
        assert "unit" not in by_key["result_note"]

        # Session bookkeeping advanced through the real code path.
        assert client._session_count == 1
        assert len(client._recent_timestamps) == 1

    def test_files_uploaded_through_real_path(self, tmp_path):
        record = _FakeRecord()
        client = self._connected_client(record)
        f = tmp_path / "trace.h5"
        f.write_text("x", encoding="utf-8")

        client.create_record(title="m", call_args={}, result={"x": 1}, files=[f])

        assert record.uploaded == [str(f)]

    def test_create_record_blocked_by_per_minute_limit_never_connects(self):
        """A per-minute-limited call must return the block string from
        ``create_record`` itself and never attempt to connect."""
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        client = KadiClient(project="proj", max_per_session=1000, max_per_minute=2)
        now = time.monotonic()
        client._recent_timestamps = [now - 10, now - 5]  # window full

        msg = client.create_record(
            title="m", call_args={}, result={"x": 1}, files=[]
        )

        assert msg is not None and "NOT saved" in msg and "per minute" in msg
        # The block short-circuits before _ensure_connected: no manager, no count.
        assert client._manager is None
        assert client._session_count == 0

    def test_create_record_blocked_by_session_limit_never_connects(self):
        from safe_lab_agents.mcp.predefined.kadi4mat import KadiClient

        client = KadiClient(project="proj", max_per_session=3, max_per_minute=1000)
        client._session_count = 3  # limit already reached

        msg = client.create_record(
            title="m", call_args={}, result={"x": 1}, files=[]
        )

        assert msg is not None and "NOT saved" in msg
        assert client._manager is None
        assert client._session_count == 3  # unchanged — nothing recorded
