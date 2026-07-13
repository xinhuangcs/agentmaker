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
application deployment controls outside this tool.

A synchronous run plus a native async arun (asyncio.create_subprocess_exec) share the same validation and
output-assembly logic so they cannot drift apart.
"""

import asyncio
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import time
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
                             "--upload-file", "-T", "-K", "--config"})    # curl: exfiltrate / read a config script
_GIT_EXEC_ARGS = frozenset({"-c", "--config-env", "--exec-path", "--extcmd",
                            "--upload-pack", "--receive-pack", "--exec"})

# Environment variable names the subprocess inherits by default (minimal set): only what the command needs, never leaking secrets from .env.
_DEFAULT_ENV_KEYS = ("PATH", "HOME", "LANG")


class CLITool(Tool):
    """Wrap "run one allowlisted local command" as a Tool (high-risk, requires confirmation)."""

    requires_confirmation = True  # High-risk: must pass confirm before execution, enforced by ToolRegistry.
    external_content = True

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
        if os.name != "posix":
            raise OSError("CLITool requires POSIX process-group isolation")
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
        path = self._env.get("PATH") or os.defpath
        self._executables = {
            command: os.path.realpath(resolved)
            for command in self._allowed
            if (resolved := shutil.which(command, path=path)) is not None
        }
        self._max_output_bytes = max_output_chars * 4
        self._arg_policy = arg_policy if arg_policy is not None else self._default_arg_policy
        self.prompts = prompts or DEFAULT_PROMPTS
        allowed_str = ", ".join(sorted(self._allowed)) or self.prompts.text("tool.none")
        super().__init__(name="shell", description=self.prompts.render("tool.desc.shell", allowed=allowed_str))

    def get_parameters(self) -> List[ToolParameter]:
        """Declare parameters: a single command string."""
        return [ToolParameter("command", "string", self.prompts.text("tool.param.shell.command"))]

    def run(self, parameters: dict) -> ToolResponse:
        """Run synchronously: after validation, run the command via subprocess.Popen (shell=False, own process group), returning the exit code plus output.

        The two pipes are drained concurrently with a hard byte cap. Timeout or excess output kills and reaps the
        process group, so neither an unbounded ``communicate()`` buffer nor orphaned grandchildren remain.
        """
        tokens, err = self._validate(parameters.get("command", ""))
        if err is not None:
            return ToolResponse.error(err)
        assert tokens is not None
        try:
            proc = subprocess.Popen(
                tokens, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=self._env, start_new_session=True)
        except FileNotFoundError:
            return ToolResponse.error(self._msg_not_found(tokens[0]))
        except OSError as e:
            return ToolResponse.error(self._msg_unrunnable(tokens[0], e))
        stdout, stderr, out_limited, err_limited, timed_out = self._collect_sync(proc, proc.pid)
        if timed_out:
            return ToolResponse.error(self._msg_timeout())
        returncode = proc.returncode
        assert returncode is not None
        return self._format(returncode, stdout, stderr,
                            stdout_truncated=out_limited, stderr_truncated=err_limited)

    async def arun(self, parameters: dict) -> ToolResponse:
        """Run natively async with bounded pipe readers; validation and result assembly are shared with run."""
        tokens, err = self._validate(parameters.get("command", ""))
        if err is not None:
            return ToolResponse.error(err)
        assert tokens is not None
        try:
            proc = await asyncio.create_subprocess_exec(
                *tokens, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=self._env, start_new_session=True)
        except FileNotFoundError:
            return ToolResponse.error(self._msg_not_found(tokens[0]))
        except OSError as e:
            return ToolResponse.error(self._msg_unrunnable(tokens[0], e))
        stdout, stderr, out_limited, err_limited, timed_out = await self._collect_async(proc, proc.pid)
        if timed_out:
            return ToolResponse.error(self._msg_timeout())
        returncode = proc.returncode
        assert returncode is not None
        return self._format(returncode, stdout, stderr,
                            stdout_truncated=out_limited, stderr_truncated=err_limited)

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
        executable = self._executables.get(tokens[0])
        if executable is None:
            return None, self._msg_not_found(tokens[0])
        tokens[0] = executable
        return tokens, None

    def _default_arg_policy(self, tokens: List[str]) -> Optional[str]:
        """Default dangerous-argument gate: return error text to refuse, or None to allow.

        Three high-risk classes are refused: (1) an interpreter plus its execute-a-code-string switch
        (sh -c / python -c, etc.); (2) find's -exec / -delete etc. and curl's --upload-file / --config;
        (3) ssh's -o ProxyCommand / LocalCommand (which run an argument as a command). This is defense in
        depth, not absolute safety; the app can override or disable it via the arg_policy constructor argument.
        """
        programs = {
            os.path.basename(tokens[0]),
            os.path.basename(self._executables.get(tokens[0], tokens[0])),
        }
        args = tokens[1:]
        interpreter = any(
            program in _INTERPRETERS
            or re.fullmatch(r"(?:python|perl|ruby|node|nodejs|php|lua)\d+(?:\.\d+)*", program)
            for program in programs
        )
        if interpreter:
            hit = [a for a in args if a in _CODE_FLAGS
                   or any(a.startswith(flag) and len(a) > len(flag) for flag in ("-c", "-e", "-E"))
                   or any(a.startswith(f"{flag}=") for flag in ("--eval", "--exec"))]
            if hit:
                return self.prompts.render("tool.msg.shell.dangerous_arg", bad=hit)
        hit = [a for a in args if a in _DANGEROUS_ARGS
               or any(a.startswith(flag) and len(a) > len(flag) for flag in ("-T", "-K"))
               or any(a.startswith(f"{flag}=") for flag in ("--upload-file", "--config"))]
        if hit:
            return self.prompts.render("tool.msg.shell.dangerous_arg", bad=hit)
        if "git" in programs:
            hit = [a for a in args
                   if a in _GIT_EXEC_ARGS
                   or a.startswith("-c=")
                   or (a.startswith("-c") and len(a) > 2)
                   or any(a.startswith(f"{flag}=") for flag in _GIT_EXEC_ARGS if flag != "-c")]
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
    def _kill_group(pgid: int) -> None:
        """Kill a captured process group, including descendants."""
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    @classmethod
    def _terminate(cls, proc, pgid: int) -> None:
        """Kill the process group and fall back to the direct child when needed."""
        cls._kill_group(pgid)
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def _collect_sync(self, proc, pgid: int) -> tuple[str, str, bool, bool, bool]:
        """Collect bounded output, terminating on timeout or overflow."""
        buffers = {proc.stdout: bytearray(), proc.stderr: bytearray()}
        limited = {proc.stdout: False, proc.stderr: False}
        selector = selectors.DefaultSelector()
        for pipe in buffers:
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, pipe)
        deadline = time.monotonic() + self.timeout
        timed_out = False
        try:
            while proc.poll() is None or selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    self._terminate(proc, pgid)
                    break
                events = selector.select(min(0.02, remaining)) if selector.get_map() else ()
                for key, _ in events:
                    pipe = key.data
                    target = buffers[pipe]
                    room = self._max_output_bytes - len(target)
                    try:
                        chunk = os.read(pipe.fileno(), min(65536, room + 1))
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(pipe)
                        continue
                    target.extend(chunk[:room])
                    if len(chunk) > room:
                        limited[pipe] = True
                        self._terminate(proc, pgid)
                        break
                if any(limited.values()):
                    break
                if not events and not selector.get_map() and proc.poll() is None:
                    time.sleep(min(0.02, remaining))
            proc.wait()
        except BaseException:
            self._terminate(proc, pgid)
            proc.wait()
            raise
        finally:
            selector.close()
            proc.stdout.close()
            proc.stderr.close()
        return (bytes(buffers[proc.stdout]).decode("utf-8", "replace"),
                bytes(buffers[proc.stderr]).decode("utf-8", "replace"),
                limited[proc.stdout], limited[proc.stderr], timed_out)

    async def _read_stream_limited(self, stream, exceeded: asyncio.Event) -> tuple[bytes, bool]:
        """Drain one asyncio stream up to the hard byte limit."""
        data = bytearray()
        truncated = False
        while True:
            remaining = self._max_output_bytes - len(data)
            chunk = await stream.read(65536 if truncated else min(65536, remaining + 1))
            if not chunk:
                return bytes(data), truncated
            if truncated:
                continue
            if len(chunk) > remaining:
                data.extend(chunk[:remaining])
                exceeded.set()
                truncated = True
            else:
                data.extend(chunk)

    @staticmethod
    def _close_async_pipes(proc) -> None:
        """Close local pipe transports so escaped descendants cannot hold the call open."""
        for stream in (proc.stdout, proc.stderr):
            transport = getattr(stream, "_transport", None)
            if transport is not None:
                transport.close()

    async def _collect_async(self, proc, pgid: int) -> tuple[str, str, bool, bool, bool]:
        """Collect bounded async output and always reap the child on cancellation."""
        exceeded = asyncio.Event()
        out_task = asyncio.create_task(self._read_stream_limited(proc.stdout, exceeded))
        err_task = asyncio.create_task(self._read_stream_limited(proc.stderr, exceeded))
        wait_task = asyncio.create_task(proc.wait())
        exceeded_task = asyncio.create_task(exceeded.wait())
        complete_task = asyncio.gather(wait_task, out_task, err_task)
        timed_out = False
        try:
            done, _ = await asyncio.wait(
                {complete_task, exceeded_task}, timeout=self.timeout,
                return_when=asyncio.FIRST_COMPLETED)
            if complete_task in done:
                _, (stdout, out_limited), (stderr, err_limited) = await complete_task
            else:
                timed_out = not done
                self._terminate(proc, pgid)
                self._close_async_pipes(proc)
                _, (stdout, out_limited), (stderr, err_limited) = await complete_task
        except asyncio.CancelledError:
            self._terminate(proc, pgid)
            self._close_async_pipes(proc)
            await asyncio.shield(asyncio.gather(complete_task, return_exceptions=True))
            raise
        except BaseException:
            self._terminate(proc, pgid)
            self._close_async_pipes(proc)
            await asyncio.shield(asyncio.gather(complete_task, return_exceptions=True))
            raise
        finally:
            if not exceeded_task.done():
                exceeded_task.cancel()
            await asyncio.gather(exceeded_task, return_exceptions=True)
        return (stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace"),
                out_limited, err_limited, timed_out)

    def _format(self, returncode: int, stdout: str, stderr: str, *,
                stdout_truncated: bool = False, stderr_truncated: bool = False) -> ToolResponse:
        """Assemble the result: exit code plus truncated stdout / stderr; mark partial if anything was truncated, otherwise success (data carries the exit code)."""
        out, out_trunc = self._truncate((stdout or "").strip(), forced=stdout_truncated)
        err, err_trunc = self._truncate((stderr or "").strip(), forced=stderr_truncated)
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

    def _truncate(self, text: str, *, forced: bool = False) -> Tuple[str, bool]:
        """Truncate and annotate if longer than max_output_chars; returns (text, whether truncated)."""
        if len(text) > self.max_output_chars:
            return text[:self.max_output_chars] + self.prompts.render("tool.msg.shell.truncated", max=self.max_output_chars), True
        if forced:
            return text + self.prompts.render("tool.msg.shell.truncated", max=self.max_output_chars), True
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
