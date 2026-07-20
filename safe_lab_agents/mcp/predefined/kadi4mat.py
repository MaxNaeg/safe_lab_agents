"""Kadi4Mat ELN integration — KadiClient and helpers.

``KadiClient`` is used by the auto-log module to push each ELN record to a
Kadi4Mat instance after it has been written locally.  Enable Kadi4Mat by
passing ``--kadi4mat-project <name>`` when starting a session; auto-log is
enabled automatically.

**Requirements:** ``pip install safe-lab-agents[kadi4mat]`` (installs
``kadi-apy``).  The user must have configured ``kadi-apy`` with a host
and personal access token (see ``kadi-apy config create``).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from safe_lab_agents.mcp.predefined.kadi4mat_utils import (
    make_collection_identifier,
    make_record_identifier,
    make_user_slug,
)
from safe_lab_agents.mcp.predefined.records import is_quantity, split_quantity

logger = logging.getLogger(__name__)

# ---- Guard the kadi-apy import ----------------------------------------

try:
    from kadi_apy import KadiManager  # type: ignore[import-untyped]

    KADI_AVAILABLE = True
except ImportError:
    KADI_AVAILABLE = False


# ======================================================================
# KadiClient – thin wrapper around KadiManager
# ======================================================================


# Rate limits to prevent overloading the Kadi4Mat server.
_DEFAULT_MAX_RECORDS_PER_MINUTE = 10
_DEFAULT_MAX_RECORDS_PER_SESSION = 500


class KadiClient:
    """Manages the connection to a Kadi4Mat instance.

    The connection is established lazily on the first call to
    :meth:`create_record`.  If the connection fails, the client disables
    itself and logs a warning — tool execution is never affected.

    Rate limiting prevents overloading the server: a per-minute limit
    catches runaway loops, and a per-session limit catches slow-burn
    issues.  When a limit is hit, ``create_record`` returns an error
    message instead of ``None`` so the agent can see what happened.
    """

    def __init__(
        self,
        project: str,
        max_per_minute: int = _DEFAULT_MAX_RECORDS_PER_MINUTE,
        max_per_session: int | float = _DEFAULT_MAX_RECORDS_PER_SESSION,
    ) -> None:
        self._project = project
        self._manager: Any = None
        self._user_slug: str | None = None
        self._collection_id: int | None = None
        self._disabled = False

        # Rate limiting state.  ``create_record`` can run on parallel threads
        # (each auto-log push happens on the calling tool's thread), so the
        # counters below are guarded by ``_lock``.  The lock is held only around
        # the counter reads/writes, never across the Kadi network calls.
        self._max_per_minute = max_per_minute
        self._max_per_session = max_per_session
        self._session_count = 0
        self._recent_timestamps: list[float] = []
        self._lock = threading.Lock()

    # ---- Lazy init ---------------------------------------------------

    def _ensure_connected(self) -> bool:
        """Connect to Kadi4Mat and resolve/create the project collection.

        Returns ``True`` if the client is ready, ``False`` if it has been
        disabled due to an error.
        """
        if self._disabled:
            return False
        if self._manager is not None:
            return True

        if not KADI_AVAILABLE:
            logger.warning(
                "kadi-apy is not installed. Install with: "
                "pip install safe-lab-agents[kadi4mat]"
            )
            self._disabled = True
            return False

        try:
            self._manager = KadiManager()
            raw_username = self._manager.pat_user.meta["identity"]["username"]
            self._user_slug = make_user_slug(raw_username)

            # Resolve or create the project collection.
            coll_id = make_collection_identifier(self._user_slug, self._project)
            collection = self._manager.collection(
                identifier=coll_id, title=self._project, create=True
            )
            self._collection_id = collection.meta["id"]
            logger.info(
                "Kadi4Mat connected: user=%s, collection=%s (id=%s)",
                self._user_slug,
                coll_id,
                self._collection_id,
            )
        except Exception:
            logger.warning("Failed to connect to Kadi4Mat", exc_info=True)
            self._disabled = True
            return False

        return True

    # ---- Record creation ---------------------------------------------

    def _check_rate_limit_before(self) -> str | None:
        """Check whether this record must be blocked by a rate limit.

        The session limit is a hard stop — once reached, no more records
        are created.  The per-minute limit is a soft stop that catches
        runaway loops: once the sliding window is full the record is
        rejected until enough old entries age out.  Returns an error
        message if blocked, or ``None`` if the record may be created.
        """
        with self._lock:
            if self._session_count >= self._max_per_session:
                return (
                    f"Kadi4Mat: session limit of {self._max_per_session} "
                    f"records was already reached.  This record was NOT saved. "
                    f" No more records will be created in this session."
                )

            # Per-minute sliding window: hard-block while the window is full so
            # a runaway loop cannot keep creating records at full speed.
            now = time.monotonic()
            self._recent_timestamps = [
                t for t in self._recent_timestamps if now - t < 60.0
            ]
            if len(self._recent_timestamps) >= self._max_per_minute:
                oldest = min(self._recent_timestamps)
                wait_seconds = int(60.0 - (now - oldest)) + 1
                return (
                    f"Kadi4Mat: rate limit of {self._max_per_minute} records "
                    f"per minute reached.  This record was NOT saved.  Wait at "
                    f"least {wait_seconds} seconds before the next tool call.  "
                    f"This is a safety measure — please check whether your "
                    f"experiment is stuck in a loop."
                )
        return None

    def _check_rate_limit_after(self) -> str | None:
        """Check rate limits *after* a record was successfully created.

        Returns a warning message about the *next* call if a limit is
        about to be reached, or ``None`` if everything is fine.
        """
        with self._lock:
            # Session limit warning.
            if self._session_count >= self._max_per_session:
                return (
                    f"WARNING: Kadi4Mat session limit of "
                    f"{self._max_per_session} records has been reached.  This "
                    f"record was saved, but the NEXT tool call will NOT be "
                    f"recorded.  No more records will be created in this "
                    f"session.  This is a safety measure to avoid overloading "
                    f"the server — please check whether your experiment is "
                    f"running as expected."
                )

            # Per-minute sliding window warning.
            now = time.monotonic()
            self._recent_timestamps = [
                t for t in self._recent_timestamps if now - t < 60.0
            ]
            if len(self._recent_timestamps) >= self._max_per_minute:
                oldest = min(self._recent_timestamps)
                wait_seconds = int(60.0 - (now - oldest)) + 1
                return (
                    f"WARNING: Kadi4Mat rate limit of {self._max_per_minute} "
                    f"records per minute reached.  This record was saved, but "
                    f"the NEXT tool call will NOT be recorded unless you wait "
                    f"at least {wait_seconds} seconds.  This is a safety "
                    f"measure — please check whether your experiment is stuck "
                    f"in a loop."
                )

        return None

    def create_record(
        self,
        title: str,
        call_args: dict[str, Any],
        result: Any,
        files: list,
    ) -> str | None:
        """Create a Kadi4Mat record documenting a single tool call.

        Returns the record identifier on success, or ``None`` on failure.
        If a rate limit is about to be hit, the record is still created
        but the return value includes a warning about the next call.
        Failures are logged but never raised.
        """
        # Hard stop: session limit was already reached on a previous call.
        block_msg = self._check_rate_limit_before()
        if block_msg is not None:
            logger.warning(block_msg)
            return block_msg

        if not self._ensure_connected():
            return None

        assert self._user_slug is not None  # guaranteed by _ensure_connected
        identifier = make_record_identifier(self._user_slug, self._project, title)

        try:
            record = self._manager.record(
                identifier=identifier, title=title, create=True
            )

            # Link to project collection.
            if self._collection_id is not None:
                record.add_collection_link(collection_id=self._collection_id)

            # Add call arguments as metadata (param_*).
            for key, value in call_args.items():
                self._add_metadatum(record, f"param_{key}", value)

            # Add return values as metadata (result_*).
            if isinstance(result, dict):
                for key, value in result.items():
                    self._add_metadatum(record, f"result_{key}", value)
            else:
                self._add_metadatum(record, "result", result)

            # Upload extracted files.
            for filepath in files:
                record.upload_file(str(filepath))

            with self._lock:
                self._session_count += 1
                self._recent_timestamps.append(time.monotonic())
            logger.info(
                "Kadi4Mat record created: %s (%d/%d this session)",
                identifier,
                self._session_count,
                self._max_per_session,
            )

            # Check if we are about to hit a limit — warn about the NEXT call.
            warning = self._check_rate_limit_after()
            if warning is not None:
                logger.warning(warning)
                return f"{identifier}\n\n{warning}"

            return identifier

        except Exception:
            logger.warning(
                "Failed to create Kadi4Mat record '%s'", identifier, exc_info=True
            )
            return None

    # ---- Helpers -----------------------------------------------------

    @staticmethod
    def _add_metadatum(record: Any, key: str, value: Any) -> None:
        """Add a single metadata entry, inferring the Kadi4Mat type.

        A quantity dict (``{"value": …, "unit": …}``) is unwrapped so the unit
        lands in Kadi's native ``unit`` field (allowed only on int/float
        extras) and an optional ontology IRI in the ``term`` field.
        """
        unit: str | None = None
        term: str | None = None
        if is_quantity(value):
            value, unit, term = split_quantity(value)

        if isinstance(value, bool):
            kadi_type = "bool"
        elif isinstance(value, int):
            kadi_type = "int"
        elif isinstance(value, float):
            kadi_type = "float"
        else:
            kadi_type = "str"
            value = str(value)

        metadatum: dict[str, Any] = {"key": key, "value": value, "type": kadi_type}
        # Kadi4Mat permits a unit only on numeric extras.
        if unit and kadi_type in ("int", "float"):
            metadatum["unit"] = unit
        if term:
            metadatum["term"] = term

        record.add_metadatum(metadatum)
