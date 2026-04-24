from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from nr_phy_simu.common.types import PlotArtifact


@dataclass
class SimulationRuntimeContext:
    """Shared per-run scratch space for intermediate variables and plot artifacts."""

    namespaces: dict[str, dict[str, Any]] = field(default_factory=dict)
    plot_artifacts: list[PlotArtifact] = field(default_factory=list)

    def clear(self) -> None:
        """Remove all runtime variables and plot artifacts from this context."""
        self.namespaces.clear()
        self.plot_artifacts.clear()

    def set(self, namespace: str, key: str, value: Any) -> None:
        """Store one runtime value under a namespace/key pair.

        Args:
            namespace: Logical owner, for example ``"channel_estimation"``.
            key: Variable name inside the namespace.
            value: Runtime value to store.
        """
        self.namespaces.setdefault(namespace, {})[key] = value

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """Read one runtime value.

        Args:
            namespace: Logical owner used when writing the value.
            key: Variable name inside the namespace.
            default: Value returned when the namespace/key is not present.

        Returns:
            Stored value or ``default``.
        """
        return self.namespaces.get(namespace, {}).get(key, default)

    def namespace(self, namespace: str) -> dict[str, Any]:
        """Return a mutable namespace dictionary, creating it if needed."""
        return self.namespaces.setdefault(namespace, {})

    def add_plot_artifact(self, artifact: PlotArtifact) -> None:
        """Register an intermediate variable for plotting."""
        self.plot_artifacts.append(artifact)


_DEFAULT_RUNTIME_CONTEXT = SimulationRuntimeContext()
_CURRENT_RUNTIME_CONTEXT: ContextVar[SimulationRuntimeContext] = ContextVar(
    "nr_phy_simu_runtime_context",
    default=_DEFAULT_RUNTIME_CONTEXT,
)


def get_runtime_context() -> SimulationRuntimeContext:
    """Return the active simulation runtime context."""
    return _CURRENT_RUNTIME_CONTEXT.get()


def set_runtime_context(context: SimulationRuntimeContext) -> Token[SimulationRuntimeContext]:
    """Activate a runtime context for subsequent module calls."""
    return _CURRENT_RUNTIME_CONTEXT.set(context)


def reset_runtime_context(token: Token[SimulationRuntimeContext]) -> None:
    """Restore the context that was active before ``set_runtime_context``."""
    _CURRENT_RUNTIME_CONTEXT.reset(token)


def clear_runtime_context() -> None:
    """Clear the active runtime context in place."""
    get_runtime_context().clear()
