"""agentmaker.tools.integrations.cli: wrap local commands as a Tool (with safety gates).

Lets an agent run locally installed commands (git / ls / grep, etc.), but the CLI is high-risk, so
safety is the core concern. The safety gates:
    1. Allowlist (deny by default): only programs the app explicitly permits are allowed; everything else is refused.
    2. Injection defense: no shell=True. shlex (with punctuation_chars) tokenizes the command, then subprocess
       runs it directly. Unquoted shell operators (pipe / redirect / multiple commands / subshell) are tokenized
       into standalone tokens and refused; metacharacters inside quotes stay literal and are allowed.
    3. Dangerous-argument gate: the allowlist only checks the program name, so allowing git / find / interpreters
       is effectively opening RCE. A default denylist adds a second layer against high-risk flags of the
       "execute a code string / exfiltrate / reverse shell" kind (overridable/disable via arg_policy).
    4. Environment isolation: the subprocess gets only a minimal env (PATH / HOME / LANG), not the whole
       os.environ; otherwise API keys in .env would flow back to the model via command output.
    5. Human confirmation: requires_confirmation=True, reusing the confirm callback already provided by ToolRegistry.
Additionally, a timeout prevents hangs (on timeout it kills the whole process group, cleaning up grandchild
processes the command spawned) and output truncation guards against blowing up the context. OS sandboxes /
containers (Seatbelt / Bubblewrap / Docker, as in Claude Code / OpenHands) offer stronger isolation but are
platform-specific and heavy; left for later / the app layer.

A synchronous run plus a native async arun (asyncio.create_subprocess_exec) share the same validation and
output-assembly logic so they cannot drift apart.
"""

import asyncio
import os
import shlex
import signal
import subprocess
from typing import Callable, List, Optional, Tuple

from ...prompts import DEFAULT_PROMPTS
from ..base import Tool, ToolParameter
from ..response import ToolResponse

# Dangerous arguments denied by default (a defense-in-depth layer, not absolute safety); the app can override via arg_policy.
_INTERPRETERS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish",
                           "python", "python2", "python3", "perl", "ruby",
                           "node", "nodejs", "php", "lua"})       # interpreters with an "execute a code string" switch
_CODE_FLAGS = frozenset({"-c", "-e", "-E", "--eval", "--exec"})  # the "run inline code" switch of interpreters
_DANGEROUS_ARGS = frozenset({"-exec", "-execdir", "-delete", "-fdelete",  # find: run a command / delete files
                             "--upload-file", "-K", "--config"})          # curl: exfiltrate / read a config script

# Environment variable names the subprocess inherits by default (minimal set): only what the command needs, never leaking secrets from .env.
_DEFAULT_ENV_KEYS = ("PATH", "HOME", "LANG")


class CLITool(Tool):
    """Wrap "run one allowlisted local command" as a Tool (high-risk, requires confirmation)."""

    requires_confirmation = True  # High-risk: must pass confirm before execution, enforced by ToolRegistry.

    # Shell operator characters: when unquoted, these (pipe / redirect / multiple commands / subshell) denote
    # shell control flow, which this tool does not support and always refuses; the same characters inside quotes
    # stay literal arguments after shlex and are allowed (and under shell=False they are harmless anyway).
    _PUNCT = frozenset("();<>|&")

    def __init__(self, allowed_commands: List[str], *, timeout: float = 10.0,
                 max_output_chars: int = 4000,
                 env: Optional[dict] = None,
                 arg_policy: Optional[Callable[[List[str]], Optional[str]]] = None,
                 prompts=None):
        """Build a CLITool restricted to an allowlist of programs.

        Args:
            allowed_commands: Allowlist of program names permitted to run (e.g. ["git", "ls", "grep"]); empty = deny all.
            timeout: Per-command timeout in seconds; on timeout the command is terminated (killing spawned grandchildren too).
            max_output_chars: Maximum characters retained for each of stdout / stderr; the excess is truncated.
            env: Environment variable mapping for the subprocess. By default only a minimal set is passed
                 (PATH / HOME / LANG, taken from the current process), never leaking the whole os.environ
                 (which includes .env secrets); if passed explicitly, exactly what you pass is used (app's own responsibility).
            arg_policy: Dangerous-argument gate hook `(tokens) -> error text | None` (None = allow). By default a
                 batch of high-risk flags is refused; pass a custom function to tighten/loosen, or pass
                 `lambda tokens: None` to disable this gate.
        """
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if max_output_chars <= 0:
            raise ValueError(f"max_output_chars must be a positive integer, got {max_output_chars}")
        self._allowed = set(allowed_commands or [])
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        if env is not None:
            self._env = dict(env)
        else:
            self._env = {k: os.environ[k] for k in _DEFAULT_ENV_KEYS if k in os.environ}
        self._arg_policy = arg_policy if arg_policy is not None else self._default_arg_policy
        self.prompts = prompts or DEFAULT_PROMPTS
        allowed_str = ", ".join(sorted(self._allowed)) or self.prompts.text("tool.none")
        super().__init__(name="shell", description=self.prompts.render("tool.desc.shell", allowed=allowed_str))

    def get_parameters(self) -> List[ToolParameter]:
        """Declare parameters: a single command string."""
        return [ToolParameter("command", "string", self.prompts.text("tool.param.shell.command"))]

    def run(self, parameters: dict) -> ToolResponse:
        """Run synchronously: after validation, run the command via subprocess.Popen (shell=False, own process group), returning the exit code plus output.

        Using Popen (not subprocess.run) with start_new_session makes the command a process-group leader, so on
        timeout the whole process group is killed, including grandchildren it spawned (subprocess.run's built-in
        timeout only kills the direct child and would miss grandchildren).
        """
        tokens, err = self._validate(parameters.get("command", ""))
        if err is not None:
            return ToolResponse.error(err)
        try:
            proc = subprocess.Popen(
                tokens, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, errors="replace", env=self._env, start_new_session=True)
        except FileNotFoundError:
            return ToolResponse.error(self._msg_not_found(tokens[0]))
        except OSError as e:
            return ToolResponse.error(self._msg_unrunnable(tokens[0], e))
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._kill_group(proc.pid)
            proc.communicate()          # Drain and close the pipes, reap the zombie.
            return ToolResponse.error(self._msg_timeout())
        return self._format(proc.returncode, stdout, stderr)

    async def arun(self, parameters: dict) -> ToolResponse:
        """Run natively async: create_subprocess_exec (shell=False, own process group) with a wait_for timeout; validation/assembly shared with run."""
        tokens, err = self._validate(parameters.get("command", ""))
        if err is not None:
            return ToolResponse.error(err)
        try:
            proc = await asyncio.create_subprocess_exec(
                *tokens, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=self._env, start_new_session=True)
        except FileNotFoundError:
            return ToolResponse.error(self._msg_not_found(tokens[0]))
        except OSError as e:
            return ToolResponse.error(self._msg_unrunnable(tokens[0], e))
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            self._kill_group(proc.pid)
            await proc.wait()
            return ToolResponse.error(self._msg_timeout())
        return self._format(proc.returncode,
                            stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace"))

    def _validate(self, command: str) -> Tuple[Optional[List[str]], Optional[str]]:
        """Validate a command: allowlist plus injection defense. Returns (tokens, None) if runnable, or (None, error text) if refused.

        Tokenize with shlex (punctuation_chars): unquoted shell operators are split into standalone tokens, and
        any token made entirely of operator characters is refused (pipe / redirect / multiple commands / subshell);
        metacharacters inside quotes stay literal arguments and are allowed.
        """
        command = (command or "").strip()
        if not command:
            return None, self.prompts.text("tool.msg.shell.empty_cmd")
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError as e:
            return None, self.prompts.render("tool.msg.shell.parse_failed", err=e)
        if not tokens:
            return None, self.prompts.text("tool.msg.shell.empty_cmd")
        bad = [t for t in tokens if t and all(ch in self._PUNCT for ch in t)]
        if bad:
            return None, self.prompts.render("tool.msg.shell.operator", bad=bad)
        if tokens[0] not in self._allowed:
            allowed = ", ".join(sorted(self._allowed)) or self.prompts.text("tool.none")
            return None, self.prompts.render("tool.msg.shell.not_allowed", program=tokens[0], allowed=allowed)
        policy_error = self._arg_policy(tokens)
        if policy_error is not None:
            return None, policy_error
        return tokens, None

    def _default_arg_policy(self, tokens: List[str]) -> Optional[str]:
        """Default dangerous-argument gate: return error text to refuse, or None to allow.

        Three high-risk classes are refused: (1) an interpreter plus its execute-a-code-string switch
        (sh -c / python -c, etc.); (2) find's -exec / -delete etc. and curl's --upload-file / --config;
        (3) ssh's -o ProxyCommand / LocalCommand (which run an argument as a command). This is defense in
        depth, not absolute safety; the app can override or disable it via the arg_policy constructor argument.
        """
        program = os.path.basename(tokens[0])
        args = tokens[1:]
        if program in _INTERPRETERS:
            hit = [a for a in args if a in _CODE_FLAGS]
            if hit:
                return self.prompts.render("tool.msg.shell.dangerous_arg", bad=hit)
        hit = [a for a in args if a in _DANGEROUS_ARGS]
        if hit:
            return self.prompts.render("tool.msg.shell.dangerous_arg", bad=hit)
        for i, arg in enumerate(args):
            low = arg.lower()
            following = args[i + 1].lower() if i + 1 < len(args) else ""
            if (arg == "-o" and ("proxycommand" in following or "localcommand" in following)) or \
               (low.startswith("-o") and ("proxycommand" in low or "localcommand" in low)):
                return self.prompts.render("tool.msg.shell.dangerous_arg", bad=[arg])
        return None

    @staticmethod
    def _kill_group(pid: int) -> None:
        """SIGKILL the entire process group led by pid (including grandchildren the command spawned); ignore if the process has already exited."""
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def _format(self, returncode: int, stdout: str, stderr: str) -> ToolResponse:
        """Assemble the result: exit code plus truncated stdout / stderr; mark partial if anything was truncated, otherwise success (data carries the exit code)."""
        out, out_trunc = self._truncate((stdout or "").strip())
        err, err_trunc = self._truncate((stderr or "").strip())
        parts = [self.prompts.render("tool.msg.shell.exit_code", code=returncode)]
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        if not out and not err:
            parts.append(self.prompts.text("tool.msg.shell.no_output"))
        text = "\n".join(parts)
        if out_trunc or err_trunc:
            return ToolResponse.partial(text, data={"returncode": returncode, "truncated": True})
        return ToolResponse.ok(text, data={"returncode": returncode})

    def _truncate(self, text: str) -> Tuple[str, bool]:
        """Truncate and annotate if longer than max_output_chars; returns (text, whether truncated)."""
        if len(text) > self.max_output_chars:
            return text[:self.max_output_chars] + self.prompts.render("tool.msg.shell.truncated", max=self.max_output_chars), True
        return text, False

    def _msg_timeout(self) -> str:
        """Timeout error message (shared by run / arun to avoid drift)."""
        return self.prompts.render("tool.msg.shell.timeout", timeout=self.timeout)

    def _msg_not_found(self, program: str) -> str:
        """Command-not-found error message."""
        return self.prompts.render("tool.msg.shell.cmd_not_found", program=program)

    def _msg_unrunnable(self, program: str, e: OSError) -> str:
        """Error message for a command that cannot be executed (permission denied and other OSError)."""
        return self.prompts.render("tool.msg.shell.unrunnable", program=program, err=e)


