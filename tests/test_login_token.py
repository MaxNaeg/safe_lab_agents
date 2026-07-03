"""Tests for OAuth-token extraction from a `claude setup-token` recording."""

from safe_lab_agents.cli import _extract_oauth_token


# A trimmed snippet of a real `script`-captured `claude setup-token` session,
# with control codes around the token (token value is fabricated).
_RECORDING = (
    "^[[1C^[[1B^[[93m"
    "sk-ant-oat01-AbC0_dEf-GhIjKlMnOpQrStUvWxYz0123456789-_AbCdEfGhIjKlMnOpQrStUvWxYzABCD"
    "\r^[[1C^[[2B^[[37mStore this token securely. You won't be able to see it again."
)


def test_extracts_token_from_recording() -> None:
    assert (
        _extract_oauth_token(_RECORDING)
        == "sk-ant-oat01-AbC0_dEf-GhIjKlMnOpQrStUvWxYz0123456789-_AbCdEfGhIjKlMnOpQrStUvWxYzABCD"
    )


def test_returns_none_when_no_token() -> None:
    assert _extract_oauth_token("Browser didn't open? Use the url below to sign in") is None


def test_token_stops_at_carriage_return() -> None:
    # The token must not absorb trailing terminal output after the CR.
    token = _extract_oauth_token("prefix ^[[93msk-ant-oat42-XYZ_abc-123\rmore output here")
    assert token == "sk-ant-oat42-XYZ_abc-123"
