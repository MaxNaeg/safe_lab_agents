"""Cross-platform pseudo-console for driving interactive CLI sessions.

The e2e driver types into, and reads the screen of, a real interactive terminal
program (the agent TUIs won't work over a plain pipe). That needs a pseudo-
terminal, which is platform-specific:

* **POSIX** (Linux, macOS): the stdlib ``pty`` — ``pty.fork`` + a master fd read
  with ``select``, ``SIGINT`` for graceful interrupt, ``waitpid`` for exit.
* **Windows**: ConPTY via the ``pywinpty`` package (``pip install pywinpty``).

Both are exposed behind one :class:`Console` interface so :mod:`_driver` stays
platform-agnostic. ``import pty`` is POSIX-only, so it lives in the POSIX backend
and is imported lazily — this module (and the whole e2e package) imports cleanly
on Windows; ``pywinpty`` is likewise imported only when a Windows console spawns.
"""

from __future__ import annotations

import os
import time
from typing import Optional

IS_WINDOWS = os.name == "nt"


class Console:
    """A child process running under a pseudo-terminal.

    ``read`` returns whatever output is available within a timeout (``""`` if
    none); ``write`` sends keystrokes; ``interrupt`` delivers Ctrl-C so the
    child's SIGINT handler runs its graceful teardown; ``poll`` returns the exit
    code once the child has exited; ``kill`` force-terminates and frees resources.
    """

    def read(self, timeout: float) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def write(self, data: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def interrupt(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def poll(self) -> Optional[int]:  # pragma: no cover - interface
        raise NotImplementedError

    def kill(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


def spawn(argv: list[str], env: Optional[dict] = None) -> Console:
    """Start ``argv`` under a pseudo-terminal and return a :class:`Console`."""
    if IS_WINDOWS:
        return _WindowsConsole(argv, env)
    return _PosixConsole(argv, env)


# ----------------------------------------------------------------------
# POSIX backend (stdlib pty) — behaviour-identical to the original driver.
# ----------------------------------------------------------------------
class _PosixConsole(Console):
    def __init__(self, argv: list[str], env: Optional[dict]) -> None:
        import pty
        import select
        import signal

        self._select = select
        self._signal = signal

        pid, master_fd = pty.fork()
        if pid == 0:  # child
            os.environ.setdefault("TERM", "xterm-256color")
            if env:
                os.environ.update(env)
            os.execvp(argv[0], argv)
            os._exit(127)  # unreachable on a successful exec
        self.pid = pid
        self.master_fd = master_fd

    def read(self, timeout: float) -> str:
        r, _, _ = self._select.select([self.master_fd], [], [], timeout)
        if self.master_fd not in r:
            return ""
        try:
            chunk = os.read(self.master_fd, 65536)
        except OSError:
            return ""  # slave closed (child exited)
        return chunk.decode("utf-8", "replace")

    def write(self, data: str) -> None:
        os.write(self.master_fd, data.encode("utf-8"))

    def interrupt(self) -> None:
        # SIGINT straight to the child (the CLI): its handler unwinds to a
        # graceful commit regardless of what the TUI is showing. This is
        # deliberately NOT a Ctrl-C injected into the TUI, which a modal dialog
        # can swallow.
        try:
            os.kill(self.pid, self._signal.SIGINT)
        except ProcessLookupError:
            pass

    def poll(self) -> Optional[int]:
        try:
            done, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            return 0
        if done == 0:
            return None
        return os.waitstatus_to_exitcode(status)

    def kill(self) -> None:
        try:
            os.kill(self.pid, self._signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            os.waitpid(self.pid, 0)
        except (ChildProcessError, OSError):
            pass
        try:
            os.close(self.master_fd)
        except OSError:
            pass


# ----------------------------------------------------------------------
# Windows backend (ConPTY via pywinpty).
# ----------------------------------------------------------------------
class _WindowsConsole(Console):
    """ConPTY-backed console.

    ``pywinpty``'s ``read`` blocks, so a daemon reader thread pumps output into a
    queue and :meth:`read` pops it with a timeout — giving the same non-blocking
    semantics the POSIX ``select`` path has.
    """

    def __init__(self, argv: list[str], env: Optional[dict]) -> None:
        import queue
        import threading

        try:
            from winpty import PtyProcess  # type: ignore
        except ImportError as exc:  # pragma: no cover - Windows-only path
            raise RuntimeError(
                "Native Windows e2e needs the 'pywinpty' package "
                "(pip install pywinpty). See docs/E2E_TESTING.md."
            ) from exc

        environ = os.environ.copy()
        environ.setdefault("TERM", "xterm-256color")
        if env:
            environ.update(env)

        self._proc = PtyProcess.spawn(argv, env=environ, dimensions=(50, 200))
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:  # pragma: no cover - Windows-only path
        while True:
            try:
                data = self._proc.read(65536)
            except EOFError:
                break
            except Exception:
                break
            if data:
                self._queue.put(data)
            else:
                time.sleep(0.02)  # avoid a busy spin when idle

    def read(self, timeout: float) -> str:  # pragma: no cover - Windows-only path
        import queue

        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return ""

    def write(self, data: str) -> None:  # pragma: no cover - Windows-only path
        self._proc.write(data)

    def interrupt(self) -> None:  # pragma: no cover - Windows-only path
        # ConPTY has no direct SIGINT-to-child; send Ctrl-C into the console,
        # which the terminal converts to a CTRL_C_EVENT for the foreground
        # process (the CLI), triggering its graceful-teardown handler.
        try:
            self._proc.sendintr()
        except Exception:
            try:
                self._proc.write("\x03")
            except Exception:
                pass

    def poll(self) -> Optional[int]:  # pragma: no cover - Windows-only path
        if self._proc.isalive():
            return None
        status = self._proc.exitstatus
        return status if status is not None else 0

    def kill(self) -> None:  # pragma: no cover - Windows-only path
        try:
            self._proc.terminate(force=True)
        except Exception:
            pass
