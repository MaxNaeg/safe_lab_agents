"""End-to-end pipeline tests.

These tests boot **real** containers and launch **real** agents across the
``{docker, podman} × {claude-code, openclaw} × {autonomous, interactive}`` matrix
(each cell also exercises ``resume``). They cost tokens and take minutes, so they
are gated behind the ``SLA_E2E=1`` environment variable and the ``e2e`` pytest
marker — a plain ``pytest`` run never collects them. See ``docs/E2E_TESTING.md``.
"""
