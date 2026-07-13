"""Config aggregation plus serialization primitives.

AgentmakerConfig: the "set defaults in one place / tune in one file" entry point that bundles the
per-subsystem frozen sub-configs together.

It is a pure holder: frozen, with no module-level instance, and components do not depend on it (each
component depends on its own narrow sub-config, which the from_config classmethods
(Memory/RagRetriever/IngestionPipeline) slice out and hand down). to_dict/from_dict use a
hand-written recursion (not dataclasses.asdict, whose internal deepcopy raises TypeError on
MappingProxyType).

Where config lives (a two-layer rule):
    - Subsystem knobs (top_k / chunk / weights / halflife / keep_recent / rrf_k / mq /
      summary_top_k ...) go into the sub-configs here.
    - Single-instance settings (CLITool(timeout=) / OpenAIEmbedder(dimensions=) /
      Tracer(max_value_len=) ...) go into that component's own constructor argument, not here.
"""

from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from typing import Any, Mapping

from .context.types import CompactionConfig, ContextConfig, ReducerConfig
from .context.window_budget import WindowBudgetConfig
from .memory.types import MemoryConfig
from .rag.types import ChunkingConfig, RagConfig
from .retrieval.types import RetrievalConfig
from .tools.tool_retriever import ToolRetrievalConfig


def _to_plain(v: Any) -> Any:
    """Recursively reduce to a JSON-friendly structure: dataclass -> dict, Mapping (including MappingProxyType) -> dict, tuple -> list.

    Hand-written (not dataclasses.asdict: asdict deepcopies internally and raises TypeError on
    MappingProxyType).
    """
    if is_dataclass(v) and not isinstance(v, type):
        return {f.name: _to_plain(getattr(v, f.name)) for f in fields(v)}
    if isinstance(v, Mapping):
        return {k: _to_plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_plain(x) for x in v]
    return v


def _sub_config_classes(cls: type) -> dict[str, type]:
    """Introspect the "field name -> sub-config class" mapping from AgentmakerConfig's dataclass fields, as the single source of truth.

    Convention: every sub-config field is declared with field(default_factory=SubClass), whose
    default_factory is exactly that class. If any field lacks a default_factory (misusing default=
    or a bare annotation), fail loud on the spot, avoiding the silent failure of "added a sub-config
    but forgot to register / validate it".
    """
    out: dict[str, type] = {}
    for f in fields(cls):
        factory = f.default_factory
        if factory is MISSING or not isinstance(factory, type):
            raise TypeError(f"AgentmakerConfig.{f.name} must be declared with field(default_factory=SubConfigClass)")
        out[f.name] = factory
    return out


def _build(cfg_cls: type, d: Mapping) -> Any:
    """Construct a sub-config class: fail loud on unknown keys, fall back to defaults on missing keys, minimal str->int/float coercion.

    Coercion is judged by the type of the field's default value, not get_type_hints: the latter
    raises when evaluating things like `int | None` and requires every annotation to be resolvable,
    which is too fragile.
    """
    if not isinstance(d, Mapping):
        raise ValueError(f"{cfg_cls.__name__} config must be an object, got {type(d).__name__}")
    fmap = {f.name: f for f in fields(cfg_cls)}
    unknown = set(d) - set(fmap)
    if unknown:
        raise ValueError(f"{cfg_cls.__name__} unknown config key(s) {sorted(unknown)}; allowed {sorted(fmap)}")
    kw = {}
    for k, v in d.items():
        default = fmap[k].default
        if isinstance(default, bool):              # bool is a subclass of int, so check it first or the int branch below swallows it
            if isinstance(v, bool):
                kw[k] = v
            elif isinstance(v, str) and v.strip().lower() in ("true", "false", "1", "0", "yes", "no"):
                kw[k] = v.strip().lower() in ("true", "1", "yes")
            else:
                raise ValueError(f"{cfg_cls.__name__}.{k} needs bool, got {v!r}")
        elif isinstance(default, int) and isinstance(v, str):
            kw[k] = int(v)
        elif isinstance(default, float) and isinstance(v, (str, int)):
            kw[k] = float(v)
        else:
            kw[k] = v                               # Optional (e.g. max_tokens=None) / containers are not coerced; passed as-is for __post_init__/constructor to handle
    return cfg_cls(**kw)


@dataclass(frozen=True)
class AgentmakerConfig:
    """The developer's "set defaults in one place / tune in one file" aggregation entry: a pure holder of the narrow sub-configs.

    No logic, no module-level instance, and components do not depend on it (each component depends on
    its own narrow sub-config, sliced out and handed down by that class's from_config). Usage: an
    app/script instantiates one at the assembly root and passes it explicitly to each component (or wires it
    in one line via each class's from_config); to persist to a file, use to_dict / from_dict.
    """
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)   # shared default; memory/rag can each pass their own override if they need heterogeneous settings
    rag: RagConfig = field(default_factory=RagConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    reducer: ReducerConfig = field(default_factory=ReducerConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    window_budget: WindowBudgetConfig = field(default_factory=WindowBudgetConfig)  # window-budget allocation knobs (output reserve + retrieval-chunk share)
    tool_retrieval: ToolRetrievalConfig = field(default_factory=ToolRetrievalConfig)  # Tool-RAG knobs (top_k / always-on list / zero-hit fallback)

    def to_dict(self) -> dict:
        """Export to a plain dict (json.dumps-able / persistable to a file)."""
        return _to_plain(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "AgentmakerConfig":
        """Restore from a dict (read from JSON / YAML / env-assembled): fail loud on unknown top-level keys, fall back to defaults on missing keys."""
        if not isinstance(d, Mapping):
            raise ValueError(f"AgentmakerConfig config must be an object (dict), got {type(d).__name__}")
        sub = _sub_config_classes(cls)                      # single source: the list is the dataclass fields, no more hand-copying
        unknown = set(d) - sub.keys()
        if unknown:
            raise ValueError(f"Unknown config key(s) {sorted(unknown)}; allowed {sorted(sub)}")
        return cls(**{k: _build(sub[k], d[k]) for k in d})

    def for_window(self, context_window, *, use_ratio: float = 0.5, fallback_window=None) -> "AgentmakerConfig":
        """Derive a new instance with context.max_tokens set from the model's window (solving the AgentmakerConfig() out-of-the-box problem).

        Keeps the other context fields (mmr_lambda / source_ratios ...) and only sets max_tokens.
        """
        window = context_window or fallback_window
        if not window:
            raise ValueError("both context_window and fallback_window are empty, cannot derive max_tokens")
        return replace(self, context=replace(self.context, max_tokens=int(window * use_ratio)))

    def validate(self) -> None:
        """Independent range validation of each sub-config (pure read-only assertions, no business logic: this is the god-object boundary).

        Note: cross-subsystem checks that need _PREFIX_TOKENS (such as "per-source context quota vs
        chunk-render accounting") stay in ContextBuilder (only there is the render accounting
        available); AgentmakerConfig does not duplicate them, to avoid accounting drift.
        """
        for f in fields(self):                              # single source: iterate the dataclass fields, so a missing .validate() raises AttributeError on the spot
            sub = getattr(self, f.name)
            if f.name == "context":
                # context is special-cased: the aggregate config uses the "wired to an Agent" accounting, where the
                # retrieval-chunk budget is supplied by window_budget, so max_tokens may be omitted (calling the
                # builder standalone does require max_tokens, and that path has its own _max_tokens_or_raise fallback).
                sub.validate(require_max_tokens=False)
            else:
                sub.validate()
