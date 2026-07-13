"""agentmaker.core.exceptions: the framework's unified exceptions.

All agentmaker exceptions inherit from AgentmakerError, which itself inherits RuntimeError. To catch any
framework exception at a higher layer, use `except AgentmakerError`; it is more precise than
`except RuntimeError`, which would also swallow third-party failures. The hierarchy includes LLMError
(with LLMConfigError / LLMRequestError / LLMResponseError / ContextWindowExceeded), RetrievalError,
SessionError, ToolError, GuardrailTripwireError, RunLimitExceeded, and RunCancelled.
"""


class AgentmakerError(RuntimeError):
    """Common framework exception base and RuntimeError subclass."""


class LLMError(AgentmakerError):
    """Unified exception for LLM configuration or invocation. Both network failures and "malformed response structure" are normalized here for uniform catching by higher layers."""


class ContextWindowExceeded(LLMError):
    """Raised when, after loss-aware reduction of the trajectory/history, the mandatory-to-keep portion (protected head + the most recent few entries) still exceeds the model's context window budget.

    This is an actionable error meaning "this task really is too large for this model" (consider splitting the task / switching to a model with a larger window), and it never silently truncates away signal.
    Inherits LLMError, so `except LLMError` also catches it.
    """


class LLMConfigError(LLMError):
    """LLM configuration / dependency error: unknown provider / missing key / missing model / missing base_url at construction time, or the SDK not installed (ImportError).
    These are "developer configuration problems" where retrying is pointless: they should alert / fix the config directly, as distinct from the runtime LLMRequestError."""


class LLMRequestError(LLMError):
    """LLM runtime call failure (network / rate limit / auth / timeout). Carries structured attributes for precise higher-layer decisions:

    provider / model: which call failed; status_code: the HTTP status code (extracted from the underlying SDK exception by duck typing, None if not obtainable);
    retryable: whether it is worth retrying (True for 408 / 429 / 5xx / timeout). Third parties use this for exponential backoff without parsing message text.
    """

    def __init__(self, message: str, *, provider=None, model=None, status_code=None, retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retryable = retryable


class LLMResponseError(LLMError):
    """LLM response structure / parsing failure (empty choices / no candidates / request assembly failure / structured-output validation retries exhausted).
    Usually the fix is to change the prompt or retry the structured output rather than back off as for a network error, so it is kept separate from LLMRequestError."""


class RetrievalError(AgentmakerError):
    """Unified exception for the retrieval subsystem (embedding / vector store / retrieval). Configuration, missing dependencies, and invocation and storage failures are all normalized here."""


class SessionError(AgentmakerError):
    """Unified exception for session persistence (SessionStore). Failures opening / reading / writing the session store are all normalized here."""


class GuardrailTripwireError(AgentmakerError):
    """Raised when a guardrail trips (input / output violation); the exception's str is the readable block explanation shown to the user."""


class RunLimitExceeded(AgentmakerError):
    """Raised when a single run exceeds a RunPolicy limit (wall-time / number of LLM calls / number of tool calls / token cap), aborting this round."""


class RunCancelled(AgentmakerError):
    """Raised when a single run is aborted by RunPolicy's cooperative cancellation (the cancel callback returns True)."""


class ToolError(AgentmakerError):
    """Unified exception for the tool subsystem (tool integration / registration / execution): missing dependencies (e.g. mcp not installed) and registration failures are all normalized here
    (the same convention as RetrievalError, where "a domain exception covers that domain's missing dependencies", making `except ToolError` precise for tool-domain problems)."""


class ToolRegistrationError(ToolError, ValueError):
    """Tool registration failure and ValueError subclass."""
