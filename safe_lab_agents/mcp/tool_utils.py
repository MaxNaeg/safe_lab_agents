import datetime
from pathlib import Path
import pickle
import ast
import numpy as np
import inspect
import functools
import textwrap
SHARED_DATA_DIR: str | None = None # will be overwritten at runtime


class _LazyExperiment:
    """Lazily construct a stateful experiment object and proxy access to it.

    The wrapped object is built on the first *use* (in the MCP server subprocess,
    where tools actually run) and cached.  Construction is deferred because the
    tools file is imported several times and in two processes (each MCP subprocess
    plus the parent process for client generation); constructing eagerly would open
    the hardware on every one of those imports.

    Accessing a method that is defined on the class factory returns a deferred
    callable that carries the method's name/docstring/signature (with ``self``
    stripped) — so it can be registered directly in ``MCP_TOOLS``/``PYTHON_TOOLS``
    without constructing the object:

        exp = experiment(Setup)
        PYTHON_TOOLS = [exp.get_power]

    Accessing a property or data attribute (or any attribute when a non-class
    factory such as a lambda is used) constructs the object and returns the real
    value.

    Teardown is intentionally **not** handled here.  Define a module-level
    ``GRACEFUL_EXPERIMENT_SHUTDOWN`` to release the experiment on exit:

        def GRACEFUL_EXPERIMENT_SHUTDOWN():
            exp.close()
    """

    def __init__(self, factory, args, kwargs):
        # Use object.__setattr__ so these don't route through __getattr__.
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_kwargs", kwargs)
        object.__setattr__(self, "_instance", None)

    def _ensure(self):
        """Construct the wrapped object on first use and cache it."""
        if self._instance is None:
            object.__setattr__(self, "_instance", self._factory(*self._args, **self._kwargs))
        return self._instance

    def __getattr__(self, name):
        cls = self._factory if isinstance(self._factory, type) else None
        raw = inspect.getattr_static(cls, name, None) if cls is not None else None
        # Method on the class -> deferred callable, no construction yet.
        if isinstance(raw, (staticmethod, classmethod)) or inspect.isfunction(raw):
            return self._deferred(name, raw)
        # Property / data attribute / non-class factory -> build now, return value.
        return getattr(self._ensure(), name)

    def _deferred(self, name, raw):
        """Wrap a class method so it constructs the object lazily when called.

        The returned callable advertises the method's metadata (name, docstring,
        type hints) with ``self``/``cls`` stripped from the signature, so tool
        registration and schema generation work without an instance.
        """
        func = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw

        @functools.wraps(func)
        def call(*args, **kwargs):
            # getattr on the instance binds self/cls correctly.
            return getattr(self._ensure(), name)(*args, **kwargs)

        sig = inspect.signature(func)
        drop = () if isinstance(raw, staticmethod) else tuple(sig.parameters)[:1]
        call.__signature__ = sig.replace(
            parameters=[p for p in sig.parameters.values() if p.name not in drop]
        )
        # functools.wraps set __wrapped__ = func; remove it so inspect.signature
        # uses our stripped __signature__ instead of following back to func.
        if hasattr(call, "__wrapped__"):
            del call.__wrapped__
        return call


def experiment(factory, *args, **kwargs):
    """Lazily construct a stateful experiment, forwarding ``*args``/``**kwargs``.

    The experiment is built on first use, in the process that actually runs the
    tools, and cached.  Pass the class (with any constructor arguments) so its
    methods can be registered directly as tools:

        exp = experiment(Setup)
        exp = experiment(Setup, "/dev/ttyUSB0", baud=9600)

        def get_position(component: str) -> float:
            \"\"\"...\"\"\"
            return float(exp.get_position(loc=component))

        # or register a method directly (requires a class factory):
        PYTHON_TOOLS = [exp.get_position, get_position]
        MCP_TOOLS    = [exp.move_to]

    For factories that are not classes (e.g. a lambda or classmethod), the object
    is built on first attribute access, so direct method registration is not
    available — use a wrapper function for those tools:

        exp = experiment(lambda: Setup.from_config("cfg.yaml"))

    Release the experiment on exit via the single shutdown hook:

        def GRACEFUL_EXPERIMENT_SHUTDOWN():
            exp.close()

    Args:
        factory: A class (recommended) or any callable returning the experiment.
        *args, **kwargs: Passed to ``factory`` when the experiment is constructed.

    Returns:
        A lazy proxy that forwards attribute access to the experiment.
    """
    return _LazyExperiment(factory, args, kwargs)

def _iter_own_returns(node):
    """Yield value-returning ``ast.Return`` nodes belonging directly to *node*.

    Descends through control-flow (if/for/with/try) in source order but does NOT
    enter nested functions/lambdas, so a helper's ``return`` is never mistaken
    for the decorated function's own.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(child, ast.Return) and child.value is not None:
            yield child
        yield from _iter_own_returns(child)


def _element_name(elt, index: int) -> str:
    """Name one returned element: a plain variable/attribute keeps its name;
    any other expression gets a positional ``result{index}`` fallback so the
    saved filename stays valid and aligned with its value."""
    if isinstance(elt, ast.Name):
        return elt.id
    if isinstance(elt, ast.Attribute):
        return elt.attr
    return f"result{index}"


def get_return_names(func) -> list[str]:
    """Return the names *func* returns, recovered structurally from its AST.

    Used by :func:`results_to_shared`.  Parses the decorated function's first
    own ``return``: a tuple return yields one name per element (via the AST
    ``Tuple`` elements, so commas inside calls like ``return combine(a, b), c``
    don't over-split), and a single-value return yields one name.  Raises
    ``ValueError`` with a clear message at decoration time when the source
    cannot be read or there is no ``return <value>`` statement.
    """
    try:
        # dedent so methods / nested functions (indented source) still parse.
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"results_to_shared cannot read the source of "
            f"{getattr(func, '__name__', func)!r}; it must be a named function "
            "defined in a file (not a lambda, built-in, or interactively-typed "
            "function)."
        ) from exc

    tree = ast.parse(source)
    func_def = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == getattr(func, "__name__", None)
        ),
        None,
    )
    ret = next(_iter_own_returns(func_def), None) if func_def is not None else None
    if ret is None:
        raise ValueError(
            f"results_to_shared requires {getattr(func, '__name__', func)!r} to "
            "have a `return <names>` statement, e.g. `return x, y, z`."
        )

    elts = ret.value.elts if isinstance(ret.value, ast.Tuple) else [ret.value]
    return [_element_name(elt, i) for i, elt in enumerate(elts)]


def results_to_shared(results_to_save: list[bool] | None = None):
    '''Decorator to save function return values to the shared data directory.
    The decorated function must have a return statement with a comma-separated list of variable names, e.g.: return x, y, z
    The decorator will save the corresponding return values to {SHARED_DATA_DIR}/{variable_name}/ with a timestamped filename.
    Args:
        results_to_save: A list of booleans indicating which return values to save (True) and pass directly (False). If None, all results will be saved.
    Returns:
        The decorated function.
    '''
    def decorator(func):
        return_names = get_return_names(func)
        save_mask = results_to_save if results_to_save is not None else [True] * len(return_names)
        assert len(save_mask) == len(return_names), f"Length of results_to_save ({len(save_mask)}) must match number of return values ({len(return_names)})."

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if SHARED_DATA_DIR is None:
                raise RuntimeError(
                    "results_to_shared requires a shared directory. "
                    "Pass --shared <dir> to agent start."
                )
            to_return = []
            result = func(*args, **kwargs)
            if len(return_names) != 1:
                assert isinstance(result, tuple) and len(result) == len(return_names), \
                    f"Expected {len(return_names)} return values (one for each name in {return_names}), got {len(result)}"
            if len(return_names) == 1:
                result = (result,)  # make it a tuple for consistent processing
            for name, value, save_res in zip(return_names, result, save_mask):
                if save_res:
                    save_type = 'pkl'
                    if isinstance(value, np.ndarray):
                        save_type = 'npy'
                    basepath = Path(f"{SHARED_DATA_DIR}/{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.{save_type}") 
                    # add dateime and underscore if already exists:
                    while basepath.exists():
                        basepath = basepath.with_name(basepath.stem + "_").with_suffix(f".{save_type}")
                    if save_type == 'npy':
                        np.save(basepath, value)
                    else:
                        with open(basepath, "wb") as f:
                            pickle.dump(value, f)
                    to_return.append(f"Saved result {name} to shared data directory with name {basepath.name}")
                else:
                    to_return.append(value)
            if len(return_names) == 1:
                return to_return[0]
            return tuple(to_return)
        return wrapper
    return decorator