"""Process-local coordination shared by managers that own the same resource."""

from contextlib import ExitStack, contextmanager
import threading
from typing import Callable, TypeVar, cast
import weakref

T = TypeVar("T")


class StripedCoordinator:
    """Serialize overlapping keys and allow whole-resource maintenance locks."""

    def __init__(self, stripes: int = 64):
        self._locks = tuple(threading.RLock() for _ in range(stripes))

    @contextmanager
    def hold(self, keys):
        """Hold the stripes covering keys in a stable order."""
        indexes = sorted({hash(key) % len(self._locks) for key in keys})
        with ExitStack() as stack:
            for index in indexes:
                stack.enter_context(self._locks[index])
            yield

    def hold_all(self):
        """Hold every stripe for a whole-resource maintenance operation."""
        return self.hold(range(len(self._locks)))


_registry_lock = threading.RLock()
_registry: dict[int, tuple[weakref.ReferenceType, StripedCoordinator]] = {}
_values: dict[tuple[int, str], tuple[weakref.ReferenceType, object]] = {}
_named: weakref.WeakValueDictionary[str, StripedCoordinator] = weakref.WeakValueDictionary()


def shared_coordinator(owner) -> StripedCoordinator:
    """Return the process-local coordinator associated with an object identity."""
    named_key = getattr(owner, "_coordination_key", None)
    if named_key is not None:
        with _registry_lock:
            coordinator = _named.get(named_key)
            if coordinator is None:
                coordinator = StripedCoordinator()
                _named[named_key] = coordinator
            return coordinator
    key = id(owner)
    with _registry_lock:
        current = _registry.get(key)
        if current is not None and current[0]() is owner:
            return current[1]
        coordinator = StripedCoordinator()
        try:
            owner_ref = weakref.ref(owner, lambda ref, k=key: _discard(k, ref))
        except TypeError:
            return coordinator
        _registry[key] = (owner_ref, coordinator)
        return coordinator


def _discard(key: int, owner_ref: weakref.ReferenceType) -> None:
    """Remove a coordinator after its owner is collected."""
    with _registry_lock:
        current = _registry.get(key)
        if current is not None and current[0] is owner_ref:
            _registry.pop(key, None)
        for value_key, current_value in list(_values.items()):
            if value_key[0] == key and current_value[0] is owner_ref:
                _values.pop(value_key, None)


def shared_value(owner, name: str, factory: Callable[[], T]) -> T:
    """Return a lazily created process-local value associated with an object."""
    key = (id(owner), name)
    with _registry_lock:
        current = _values.get(key)
        if current is not None and current[0]() is owner:
            return cast(T, current[1])
        value = factory()
        try:
            owner_ref = weakref.ref(owner, lambda ref, k=id(owner): _discard(k, ref))
        except TypeError:
            return value
        _values[key] = (owner_ref, value)
        return value
