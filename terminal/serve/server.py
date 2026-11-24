"""serve/server.py - plan v0.3 P8: OpenAI and Anthropic endpoints over any engine that maps token ids to tokens.

    python -m serve.server --engine mock --port 8095            (a scripted engine, for clients and tests)
    python -m serve.server --engine strata --config strata.json --port 8080   (the real engine, resident)

Endpoints: POST /v1/chat/completions (OpenAI, stream and non-stream), POST /v1/messages (Anthropic, stream and
non-stream), GET /v1/models, GET /models, GET /props, GET /slots, GET /health, GET /mcp. One sequence at a time behind a FIFO (plan: one resident sequence).
Image generation (optional, when the config has an "images" section): POST /v1/images/generations (OpenAI's shape,
plus seed and steps) runs stable-diffusion.cpp's sd-cli for Qwen-Image-2.1-Turbo, one job at a time on the GPU the
text engine uses (serve/images.py).
Tools from MCP servers (serve/mcp.py, `"mcp_servers"` in the config or --mcp-config) are offered only to requests that
ask for them with `"strata_mcp": true` - the web app does; other clients see exactly the API they always saw.
Images (optional, when the config has a "vision" entry): OpenAI image_url parts and Anthropic image blocks (base64
data, http(s) URLs or local file paths) go through `strata-vision` (the model's mmproj file) and reach the engine as
embeddings (`GENI`).  JPEG/PNG/BMP/GIF go straight in; WebP, TIFF, AVIF, ... (agents like omp send WebP) are
converted to PNG first with Pillow.
A request whose prompt alone does not fit the engine's context is REJECTED with 400, never truncated.  When the prompt
fits but prompt plus max tokens does not (Claude Code asks for 32000, the Harness for 16384), max tokens is shortened to
the room left (less CTX_SLACK) and the thinking budget follows it; `--no-fit-max-tokens` / `"fit_max_tokens": false`
brings back the strict 400 of llama.cpp.
An unset (or 0, or -1) max tokens means "unlimited": whatever the prompt leaves of the context.

The engine boundary is `Engine.generate(prompt_ids, max_new, sampling, cancel) -> iterator of token ids`.
`StrataEngine` keeps one `strata --serve` process resident (weights, expert arena and VRAM tier load once) and
talks to it over stdin/stdout; `MockEngine` is a scripted stand-in that makes every API path testable without a GPU.
"""
from __future__ import annotations
import sys
if sys.platform == "win32":
    # CPython 3.12's WMI helper (_wmi, behind platform.system() / uname()) takes an access violation in a worker thread when WMI refuses the
    # account (0x80041003) and the caller has already gone (DIAG-WINHOST-0216). Without _wmi, platform uses the registry and the environment.
    sys.modules["_wmi"] = None

import argparse
import contextlib
import collections
import base64
import hashlib
import hmac
import codecs
import json
import math
import os
import platform
import queue
import re
import select
import signal
import socket
import socketserver
import struct
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.request
import uuid
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Protocol
from urllib.parse import parse_qs, urlsplit, unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))   # run as a script (run-<model>.bat) as well as a module
from serve.server_parts.anthropic_api import (  # noqa: F401  (moved unchanged, file split)
    anthropic_collect,
)
from serve.server_parts.base import (  # noqa: F401  (moved unchanged, file split)
    ANSWER_RESERVE_MIN,
    INKLING_ROOM_MIN_CAP,
    NO_ANSWER_LENGTH,
    NO_ANSWER_STOP,
    INKLING_BUDGET_BY_EFFORT,
    ANSWER_ROOM_MIN_CAP,
    COALESCE_BURST_MS,
    COALESCE_MIN_REQS,
    COALESCE_PREFILL,
    COALESCE_TOKENS,
    COALESCE_WAIT_MS,
    CTX_SLACK,
    ContextOverflow,
    EFFORT_TEXT,
    EFFORT_TURN,
    ENGINE_SILENCE_S,
    FIT_MIN_ROOM,
    HIGH_EFFORT,
    HOLD_KEY,
    IMAGE_PAD,
    IM_END,
    KEEPALIVE_S,
    KEEP_CKPT_KEY,
    LEFTOVER_SETTLE_S,
    LOOPBACK_NAMES,
    LOOP_CHECK_EVERY,
    LOOP_COVERAGE,
    LOOP_HISTORY_WORDS,
    LOOP_TAIL_CHARS,
    LOOP_TAIL_PERIOD,
    LOW_EFFORT,
    PAR_COUNTERS,
    PICTURE_FORMATS,
    PP_CHUNK_MAX,
    PP_FLOOR_TOK_S,
    PP_SLACK,
    RATE_MIN_SPAN_S,
    RATE_WINDOW_S,
    REASONING_CLOSE,
    REASONING_WRAP_UP,
    REPEAT_STOP_TOKENS,
    SESSION_WAIT_MAX_S,
    STALL_CPU_EPS_S,
    STALL_IO_EPS_B,
    V1_V1_ALIASES,
    VISION_START,
    _timeout_env,
    engine_frozen,
    fit_floor,
    focused_recovery_prompt,
    listen_problem,
    reasoning_repeat_coverage,
    rttime_limit_note,
    write_temporary,
)
from serve.server_parts.core import (  # noqa: F401  (moved unchanged, file split)
    Detokenizer,
    NO_LINE,
    OPENAI_MAX_STOP,
    SPLIT_REFUSED_FEATURES,
    StopMatcher,
    TOOL_RETRY_MIN_ROOM,
    ToolMarkupHold,
    _MARKUP_OPENER,
    _WIN_DEVICES,
    _line_start,
    add_carry,
    allowed_ids,
    answer_speed,
    batch_slots_on,
    carry_leg,
    closest_declared,
    dead_markup_tail,
    markup_names,
    prefill_leg_stats,
    prompt_tokens_seen,
    request_stats,
    request_timings,
    slot_filename_problem,
    slot_save_dir,
    stop_strings,
    token_limit,
    tool_retry_note,
    top_logprobs,
)
from serve.server_parts.end_tokens import (  # noqa: F401  (moved unchanged, file split)
    end_token_ids,
)
from serve.server_parts.engines import (  # noqa: F401  (moved unchanged, file split)
    ByteTokenizer,
    ClientGone,
    DRAFT_COSTS_DEFAULT_PLATFORMS,
    DRAFT_HEAD_FAIL,
    DRAFT_HEAD_HINT,
    ENGINE_REQUEST,
    Engine,
    EngineClosing,
    EngineDied,
    EngineNonFinite,
    EngineSilent,
    EngineStarting,
    EngineStuck,
    FATAL_PREFIXES,
    GpuBusy,
    HIP_IGPU_ARCHS,
    HOOK_REASON_MAX,
    ModelBusy,
    PAGE_CSP,
    PARALLEL_MAX,
    QueueCancelled,
    QueueWaitExpired,
    SPLIT_MAX_LANES,
    ServerStopping,
    SessionRefused,
    TRANSIENT_START_MARKS,
    VisionNoRoom,
    _BTRACE,
    _POOL_CORES_OK,
    _RequestOwned,
    _VISION_ARCH_SIGNS,
    _WIN_CRASH,
    _dir_writable,
    _echoing,
    _engine_texts,
    _gpu_numbers,
    btrace,
    card_note,
    card_roles_lines,
    conversation_cache_view,
    conversation_slots,
    cpu_isa_words,
    died_note,
    draft_costs_default,
    draft_costs_dir,
    engine_arch_note,
    engine_binary,
    engine_crashed,
    engine_exit_words,
    engine_has_text,
    engine_silence_s,
    experts_loading_words,
    gpu_all_off,
    gpu_list,
    gpu_roles_active,
    gpu_speed_scores,
    hip_no_card_message,
    hip_speed_scores,
    hook_echo,
    hook_reason,
    hook_text,
    igpu_cards,
    layer_split_of,
    layer_split_value,
    learned_profile_args,
    monitored_gpu,
    network_path,
    own_path_context_warning,
    parallel_args,
    parallel_flag_conflicts,
    parallel_slots,
    parse_batch_window_sample,
    parse_engine_stage_line,
    parse_hip_devices,
    parse_stage_in_flight_sample,
    pick_hip_devices,
    pool_cores_args,
    profile_shape,
    read_engine_stage_log,
    sh_stream_warning,
    start_exit_note,
    start_failure_hint,
    start_log_marks,
    start_log_tail,
    start_log_text,
    unload_wait,
    helper_gpus,
    visible_device_order,
    vision_log_size,
    vision_log_tail,
    vision_start_error,
)
from serve.server_parts.http_api import (  # noqa: F401  (moved unchanged, file split)
    BadBody,
    CARD_PENALTIES,
    DESKTOP_FREE_MIB,
    DESKTOP_RESERVE_MIB,
    FAMILY_MODEL_NAMES,
    FAULTHANDLER_OFF,
    MODEL_CARD_SAMPLING,
    READER_NEEDS_MIB,
    SAMPLING_ALIASES,
    SHARED_KEYS,
    SPLIT_KEYS,
    Server,
    VISION_ON_DEMAND_BY_DEFAULT,
    VISION_RESERVE_MIB,
    _ON_DEMAND_NOT_WITH,
    _component_exe,
    _history_call_names,
    _is_ip,
    _name_in,
    allowed_hosts_of,
    anthropic_error_type,
    api_key_of,
    apply_saved_tuning,
    arm_faulthandler,
    asks_greedy,
    check_request_sampling,
    clean_shared_defaults,
    default_model_name,
    desktop_vram_note,
    explicit_no_key,
    host_allowed,
    host_name,
    key_list,
    key_matches,
    matched_key,
    lift_reasoning_effort_arg,
    linux_desktop,
    load_shared_settings,
    model_card_sampling,
    open_server_warning,
    origin_allowed,
    origins_of,
    parse_config_text,
    read_config,
    request_asks_sampling,
    sampling_defaults_from_config,
    split_joined_args,
    tool_call_recovery_setting,
    undeclared_tool_calls_setting,
    warn_budget_over_ram,
    warn_tight_ram,
    with_component_engine,
    with_current_reader,
    with_history_tools,
)
from serve.server_parts.openai_api import (  # noqa: F401  (moved unchanged, file split)
    LPLines,
    LogprobEntries,
    PP_DONE_TAIL,
    _is_json,
    _same_json,
    _structured_logprobs,
    _with_reasoning,
    logprob_span,
    openai_collect,
)
from serve.frontend import (CALL_END, CALL_START, FUNC_END, FUNC_START, THINK_END, ChatTemplate, Event, OutputParser,  # noqa: E402
                            parser_for_template, anthropic_to_messages, anthropic_tools,
                            forced_call, images_of, literal_tags, mark_think_literals, openai_to_messages, require_messages,
                            tool_choice_of, unmark_think_literals)
from serve.mcp import McpCancelled, hub_from_config, resolve_image_results  # noqa: E402
from serve import appendlog  # noqa: E402
from serve import runconfig  # noqa: E402
from serve import expert_heat  # noqa: E402
from serve import commit_limit  # noqa: E402
from serve import oomd_pacer  # noqa: E402
from serve import lastrun  # noqa: E402
from serve import hy4_kind  # noqa: E402
from serve import hy4_dialect  # noqa: E402
from serve import inkling_kind  # noqa: E402
from serve import inkling_dialect  # noqa: E402
from serve.webfiles import subtree_file as web_subtree_file  # noqa: E402
from strata_control import engine_component  # noqa: E402
from strata_control import tls  # noqa: E402
from strata_control.childenv import arena_env, limit_own_arenas, quiet_crash_env  # noqa: E402
from strata_control import reader_room  # noqa: E402

tls.install()       # #3722: image URLs and tool servers over HTTPS are checked against the operating system's certificates
from strata_control import hipblaslt_tuning  # noqa: E402
from strata_control import speed_switches  # noqa: E402
from strata_control import vision_reader  # noqa: E402
from strata_control import draft_layer  # noqa: E402
from strata_control import drafting  # noqa: E402
from strata_control import vram_fit  # noqa: E402
from strata_control import user_args  # noqa: E402
from strata_control import gpu_roles  # noqa: E402
from serve import ready as ready_file  # noqa: E402
from serve import owner_guard  # noqa: E402
from serve import launch_identity  # noqa: E402
from serve import logit_bias as bias_api  # noqa: E402
from serve import lifeline  # noqa: E402
from serve.structured import StructuredOutputError, prepare_format, validated_json, extract_json  # noqa: E402
from serve import responses as responses_api  # noqa: E402
from serve.responses import ResponsesError, error_body as responses_error_body  # noqa: E402
from serve import images as images_api  # noqa: E402


def popen(what: str, args: list, role: str = "engine", **kw):
    """#735: subprocess.Popen, but Windows' refusal to start a program (WinError 4551, or 1260 when a policy blocks it:
    Smart App Control on a clean Windows 11 blocks an exe that is not signed) is said in words instead of a bare
    OSError.  On Windows the program gets no console window of its own (as serve/mcp.py's servers): a server started
    without a console (pythonw, a service, a launcher that hides it) would otherwise open one for the engine and one
    for the image encoder; their output goes through pipes and the log either way."""
    if os.name == "nt":
        kw.setdefault("creationflags", getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        return lifeline.popen(args, role, **kw)       # a child that cannot outlive this server (serve/lifeline.py)
    except OSError as e:
        if getattr(e, "winerror", None) in (4551, 1260):
            raise OSError(f"Windows blocked {what} ({args[0]}): Smart App Control, or an application-control policy, does "
                          "not let a program that is not signed run. Turn Smart App Control off (Windows Security > App "
                          "& browser control > Smart App Control settings), or - for the image encoder only - run setup "
                          "with --vision no (see docs/TROUBLESHOOTING.md)") from e
        raise


# How long close() gives the engine at each step of ending it (seconds). Module values so that a test can shorten them.
ENGINE_QUIT_WAIT_S, ENGINE_TERM_WAIT_S, ENGINE_KILL_WAIT_S = 20, 5, 20


def end_leftover_engines(exe: str) -> None:
    """Before an engine starts: end the ones a lost server left running (strata_control/procs.py). Only a process
    this install recorded whose starter is gone, or an orphan of exactly this executable running --serve, and only
    after its PID, start time and command line are checked again at the moment of the signal. Never raises.
    #4051: the new engine is not started while the old one still holds its memory (tens of GB take longer than the
    10 s this used to wait): a 40 GiB arena loaded beside 50 GiB the old engine had not returned put a 60 GiB PC into
    memory reclaim for good.  One still there after LEFTOVER_SETTLE_S is said in plain words, and the start goes on
    (the engine waits for the RAM itself and warns; nothing is refused)."""
    if os.environ.get("STRATA_KEEP_LEFTOVERS") == "1":
        return
    try:
        from strata_control import procs
        say = lambda text: print(text, flush=True)        # noqa: E731
        stuck = [r["pid"] for r in procs.end_leftovers(say=say, settle=LEFTOVER_SETTLE_S)["stuck"]]
        stuck += [r["pid"] for r in procs.end_orphans(exe, say=say, settle=LEFTOVER_SETTLE_S) if r["outcome"] == "stuck"]
        if stuck:
            print(f"[strata] WARNING: the engine from an earlier run (process {', '.join(map(str, stuck))}) was told to stop "
                  f"but still holds its memory after {LEFTOVER_SETTLE_S:.0f} s. Starting the next one now can leave this PC "
                  "short of RAM and very slow: wait until that process is gone (it is shown by `ps`/Task Manager), "
                  "or restart the PC if it does not go.", flush=True)
    except Exception as e:  # noqa: BLE001 - a failed check must not stop a start
        print(f"[strata] could not check for engines left by an earlier run: {type(e).__name__}", flush=True)


def combined_embeddings_path(vision_dir: Path, nbytes: int) -> Path:
    """#874: where one request's combined image-embeddings file goes.  Every request with images rewrites it (it is
    deleted after the request), which on a disk is steady heavy writing: on Linux it goes to /dev/shm when that has
    room (the engine reads it back at once), on Windows the file is opened short-lived (the cache is not forced out
    to the drive).  Anywhere else, or when /dev/shm is too small (Docker's default is 64 MB), it stays in the encoder's
    own directory, as before."""
    name = f"req-{uuid.uuid4().hex[:12]}.sve"
    shm = Path("/dev/shm")
    if os.name != "nt" and shm.is_dir() and os.access(shm, os.W_OK):
        try:
            if shutil.disk_usage(shm).free > 2 * nbytes + (16 << 20):
                return shm / f"strata-vision-{os.getpid()}-{name}"
        except OSError:
            pass
    return Path(vision_dir) / name


QUEUE_BEAT_S = 10.0         # a request waiting for the single turn sends a keep-alive this often (#1619)


# #4271: a streaming client that stops reading (a hung process, a network that dropped) leaves the thread that writes to
# it blocked once the socket buffers are full - and that thread holds the engine's turn.  A write that makes no progress
# for this long ends the request as a disconnect (the engine is stopped, the turn given back).  0: no limit, as before.
STREAM_WRITE_TIMEOUT_S = float(os.environ.get("STRATA_STREAM_WRITE_TIMEOUT_S", "60") or 0)
# Reads that had no timeout (#1317).  A blocking read of the image encoder's pipe or of the engine's READY line that
# never returned held the request turn for good: the engine then sat idle in its command loop, so no watchdog could
# fire and every later request queued behind it (the "lost step" class of #481).  Generous on purpose: a cold model
# load, a slow disk or a slow CPU encode is not a hang.  Only a process that says nothing at all for this long is
# ended; the next request starts it again.
# STRATA_ENGINE_READY_S / STRATA_VISION_READY_S / STRATA_VISION_ENCODE_S override them (0 = wait for ever, as before).


# #1317 part 2: the quick watchdog for a FROZEN engine.  engine_silence_s (above) is generous on purpose: a slow PC reads a
# prompt chunk for minutes without a line.  But an engine that prints nothing for ENGINE_STALL_S and also uses no CPU time and
# moves no bytes of I/O in that time is not slow, it is idle or frozen (a deadlock in a CUDA call, a driver stall, a lost
# step), and waiting longer cannot help: it is ended, and the next request starts it again.  An engine that is silent but
# still working is never ended by this, only reported once (the recommendation: engine_silence_s, or this variable).
# STRATA_ENGINE_STALL_S overrides it (0 = off).  Needs psutil (setup installs it); without it nothing is ended.
ENGINE_STALL_S = _timeout_env("STRATA_ENGINE_STALL_S", 90.0)
# --batch: new requests take the control lines (one prompt read at a time) short prompts first: a prompt of at most
# STRATA_ADMIT_SHORT_FIRST tokens goes before longer ones (opt-in; unset or 0: whoever gets the lock first), and a
# longer one that waited STRATA_ADMIT_MAX_WAIT_S (default 30) counts as short.  A chat question then waits for the one
# read in progress, not for every agent read that came before it.
try:
    ADMIT_SHORT_FIRST = max(0, int(os.environ.get("STRATA_ADMIT_SHORT_FIRST") or 0))
except ValueError:
    ADMIT_SHORT_FIRST = 0
ADMIT_MAX_WAIT_S = _timeout_env("STRATA_ADMIT_MAX_WAIT_S", 30.0) or 0.0


VISION_READY_S = _timeout_env("STRATA_VISION_READY_S", 300.0)
VISION_ENCODE_S = _timeout_env("STRATA_VISION_ENCODE_S", 300.0)
ENGINE_READY_S = _timeout_env("STRATA_ENGINE_READY_S", 900.0)

# A stalled peer must release the request turn; 0 preserves unbounded socket waits.
PEER_STALL_S = _timeout_env("STRATA_PEER_STALL_S", 120.0)


# ------------------------------------------------------------------------------------------------ engines


class MockEngine:
    """Replays a scripted completion (text) as token ids, one per step, then the end-of-turn token.  Given a list of
    scripts, each request gets the next one and the last one repeats (a tool call, then the answer after it)."""

    def __init__(self, tokenizer, script: str | list[str], max_context: int = 32768, delay_s: float = 0.0,
                 cycle: bool = False):
        self.tok, self.max_context, self.delay = tokenizer, max_context, delay_s
        end = tokenizer.encode(IM_END, parse_special=True)
        self.scripts = [tokenizer.encode(x, parse_special=True) + end for x in ([script] if isinstance(script, str)
                                                                                 else script)]
        self.script, self.turns = self.scripts[0], 0
        # `cycle` picks the script by request position instead of advancing to the last one and staying
        # there. A harness adds requests the script author did not plan for -- a session title, a re-ask
        # after a tool result -- and with the default behaviour one extra request shifts every later
        # reply, so a run looks like a model that simply chose differently.
        self.cycle = bool(cycle)
        self.last_prompt: list[int] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        self.last_embeddings = embeddings
        if len(self.scripts) > 1:
            if self.cycle:
                self.script = self.scripts[self.turns % len(self.scripts)]
            else:
                self.script = self.scripts[min(self.turns, len(self.scripts) - 1)]
            self.turns += 1
        for t in self.script[:max_new]:
            if cancel.is_set():
                return
            if self.delay:
                time.sleep(self.delay)
            yield t


START_RETRIES = max(0, int(os.environ.get("STRATA_START_RETRIES", "3") or 0))


def transient_start_failure(log: str | None, offset: int, elapsed_s: float, rc: int | None = None) -> bool:
    """True when the start that wrote the log from `offset` failed the transient way above: within 30 s, and its lines
    are only the launcher's own "engine started" line plus (optionally) a GPU runtime failure.  An exit with no line
    at all counts only when it was not the engine's own error exit (1: it said why on stdout)."""
    if not log or elapsed_s > 30.0:
        return False
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            text = f.read(64 * 1024).decode("utf-8", "replace")
    except (OSError, ValueError):
        return False
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    engine_lines = [x for x in lines if not x.startswith("[strata]")]
    if any(m in text for m in TRANSIENT_START_MARKS):
        return all(x.startswith("terminate called") or x.startswith("what()") or
                   any(m in x for m in TRANSIENT_START_MARKS) for x in engine_lines)
    # it exited without writing a single line of its own - but not when the system ended it (an illegal instruction, a
    # bad memory access, the OOM killer, a missing DLL): that is a crash a second try repeats, and the retry notice
    # must not call it a GPU runtime start (#2324: an old CPU's illegal instruction was retried 3 times as "GPU
    # runtime" and the real reason never shown)
    return not engine_lines and rc != 1 and not engine_crashed(rc)


def echo_requests(log_path: str, offset: int) -> None:
    """STRATA_REQUEST_LINES=1: one stdout line per finished request, from the engine's own summary in its log.

    The engine's stderr goes to the log file (the start narrator reads it), so a supervisor that only sees this
    process's output - a tray, llama-swap - has no per-request numbers. This re-states the engine's line with the
    total the two times make: `request prompt P cached C output O prompt_read R ms total S ms prefill X tok/s decode Y
    tok/s` (prompt_read: the time the engine spent reading the prompt's new tokens, not a time to first token).
    """
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            m = ENGINE_REQUEST.search(line)
            if m:
                read_ms, gen_ms = float(m["read"]), float(m["gen_ms"])
                print("[strata] request prompt %s cached %s output %s prompt_read %.0f ms total %.0f ms prefill %s "
                      "tok/s decode %s tok/s" % (m["prompt"], m["reused"], m["gen"], read_ms, read_ms + gen_ms, m["pp"],
                                                 m["tg"]), flush=True)


OPENS_BROWSER = False        # set in main() when the server was started with --open: only then may a message say a browser opens


def narrate_start(log_path: str, offset: int, args: list, done: threading.Event, heartbeat=20.0) -> None:
    """While the engine starts, say in the server window what it is doing, from its log: the start reads tens of GB
    into RAM and locks part of it for the GPU, and on many PCs everything is slow or frozen for a minute or more -
    people closed the window thinking it had hung.  The warning comes at that step, not after it."""
    gb = 0.0
    if "--native" in args:                              # about the size of the experts it will read
        try:
            gb = os.path.getsize(args[args.index("--native") + 1]) / 1e9
        except (OSError, IndexError):
            pass
    size = f"about {gb:.0f} GB" if gb >= 1 else "tens of GB"
    loading = experts_loading_words(args, size)
    t0 = last = time.time()
    said = set()

    def say(key, text):
        nonlocal last
        if key not in said:
            said.add(key)
            last = time.time()
            print(text, flush=True)

    say("weights", "[strata] starting the engine: reading the model's weights ...")
    pos = offset
    while not done.wait(0.5):
        try:
            with open(log_path, "rb") as f:
                f.seek(pos)
                chunk = f.read()
        except OSError:
            chunk = b""
        if chunk.count(b"\n"):
            cut = chunk.rfind(b"\n") + 1
            pos += cut
            for line in chunk[:cut].decode("utf-8", "replace").splitlines():
                if "PLE on" in line or "expert arena:" in line or "experts via mmap" in line:   # #505: mapped
                    say("arena", f"[strata] {loading}\n"
                                 "         YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES NOW - this is normal.\n"
                                 "         Please wait and don't close this window"
                                 + ("; the browser opens when it is ready." if OPENS_BROWSER else "; the server says when it is ready."))
                elif " loaded " in line and "GiB at" in line:
                    say("loaded", "[strata] experts loaded: " + line.split(" loaded ", 1)[1].strip() +
                        f" ({time.time() - t0:.0f} s so far)")
                elif "expert cache " in line and " slots, " in line and "auto" not in line:
                    n = line.split("expert cache ", 1)[1].split(";")[0].replace(" slots,", " experts,").strip()
                    say("cache", f"[strata] filling the GPU's expert cache ({n}) ...")
                elif "session is up" in line:
                    say("up", "[strata] almost ready ...")
        if time.time() - last > heartbeat:
            last = time.time()
            print(f"[strata] still starting ({time.time() - t0:.0f} s) - please wait ...", flush=True)


class ConvCacheLog:
    """#596: the engine's conversation cache as its log tells it (the engine writes "strata serve: conversation
    cache: parked N tokens ...; parked=P bytes=B evictions=E" and "restored N tokens ...; parked=P bytes=B" to stderr,
    which is the log): read on from where it was last read, from the start of the engine's current run."""
    # #879: these two completed recovery messages follow ConversationCache::clear(), which removes every parked
    # entry and its retained K/V.  Keep history counters: the clear is not a cache park or restore event.
    CLEARED = re.compile(
        r"strata serve: after non-finite logits \(#879\): (?:the restored session replaces the live one; dropped \d+ parked "
        r"conversations?|dropped the live session, \d+ checkpoints?, \d+ parked conversations? and \d+ idle slots? "
        r"\(\d+ running slots? not kept after their reply\); reading this prompt from token 0)$")
    EVENT = re.compile(r"conversation cache: (parked|skipped|restored) (\d+) tokens.*?parked=(\d+) bytes=(\d+)"
                       r"(?: evictions=(\d+))?")
    DROPPED = re.compile(r"conversation cache: dropped \d+ superseded .*?parked=(\d+)")
    # #6081: a park the cache refused, by reason (the engine's "skip parking (...)" / "allocation failed" lines; a
    # "skipped N tokens" line is put()'s refusal).  The first match names the reason; any other text is the size
    # estimate's own error ("estimate").
    SKIP = re.compile(r"conversation cache: (?:skip parking \((.*)\)|allocation failed; skip parking)")
    REASONS = (("gave no memory back", "ram_no_progress"), ("physical RAM admission", "ram_floor"),
               ("RAM floor after capture", "ram_after_capture"), ("exceeds available budget", "budget"),
               ("below the", "short"), ("stage CUDA", "estimate"))
    READ_MAX = 1 << 20                                  # at most the last MiB of new lines per read

    def __init__(self):
        self.key, self.pos = None, 0
        self.lock = threading.Lock()                    # polls come from several request / monitor threads at once
        self.reset()

    def reset(self):
        self.state = {"parked": 0, "bytes": 0, "evictions": 0, "parks": 0, "restores": 0, "last_event": None,
                      "last_tokens": None, "last_at": None, "refused": {}, "refused_total": 0, "last_refused": None}

    def poll(self, path, start) -> dict:
        """The state after the log's new lines; `start` is where the engine's current run began in it.  One poll at a
        time: a second one moved `pos` between the size check and the seek and raised ValueError (negative seek)."""
        with self.lock:
            st = self._poll(path, start)
            st["refused"] = dict(st["refused"])         # a copy: the caller's view does not move with later polls
            return st

    def _refused(self, reason: str) -> None:
        st = self.state
        st["refused"][reason] = st["refused"].get(reason, 0) + 1
        st["refused_total"] += 1
        st["last_refused"] = reason

    def _poll(self, path, start) -> dict:
        if not path or start is None:
            return dict(self.state)
        if self.key != (path, start):                   # another start of the engine: its cache starts empty
            self.key, self.pos = (path, start), start
            self.reset()
        try:
            size = os.path.getsize(path)
            if size < self.pos:                         # the log was emptied or replaced
                self.pos = 0
            if size - self.pos > self.READ_MAX:
                self.pos = size - self.READ_MAX
            with open(path, "rb") as f:
                f.seek(self.pos)
                data = f.read(size - self.pos)
        except OSError:
            return dict(self.state)
        end = data.rfind(b"\n") + 1                     # whole lines only: the rest is read next time
        self.pos += end
        now = time.time()
        for line in data[:end].decode("utf-8", "replace").splitlines():
            if "conversation cache:" not in line:
                if self.CLEARED.fullmatch(line):
                    self.state["parked"] = self.state["bytes"] = 0
                continue
            m = self.EVENT.search(line)
            if m:
                st = self.state
                st["parked"], st["bytes"] = int(m.group(3)), int(m.group(4))
                if m.group(5) is not None:
                    st["evictions"] = int(m.group(5))
                if m.group(1) != "skipped":
                    st["parks" if m.group(1) == "parked" else "restores"] += 1
                    st["last_event"], st["last_tokens"], st["last_at"] = m.group(1), int(m.group(2)), now
                else:
                    self._refused("put")
                continue
            m = self.DROPPED.search(line)
            if m:
                self.state["parked"] = int(m.group(1))
                continue
            m = self.SKIP.search(line)
            if m:
                text = m.group(1)
                if text is None:
                    self._refused("alloc")
                else:
                    self._refused(next((r for k, r in self.REASONS if k in text), "estimate" if text else "other"))
        return dict(self.state)


EOS_IDS = {248044, 248046}   # <|endoftext|>, <|im_end|>: the engine's default --eos-ids


def configure_batch_telemetry_env(env: dict, cfg: dict, override: bool | None) -> None:
    """Only an explicit setting changes the inherited opt-in engine environment."""
    choice = override if override is not None else cfg.get("batch_telemetry")
    if choice is True:
        env["STRATA_BATCH_TELEMETRY"] = "1"
    elif choice is False:
        env.pop("STRATA_BATCH_TELEMETRY", None)


class StrataEngine:
    """The resident engine: `strata --serve` reads `GEN <max_new> <ids>` lines and streams `T <id>` lines, then
    `DONE ...`.  Requests are serialized by the service's FIFO, so one pipe is enough.

    Per-request sampling rides the same line as engine-side keys between max_new and the ids
    (`temperature=F top_p=F top_k=N seed=N`, the engine's own spelling).  An absent temperature keeps the
    engine's default, which is greedy; `temperature=0` means the same thing, so it is not forwarded.
    """
    silence_s = ENGINE_SILENCE_S         # #481: main() sets the config's engine_silence_s (survives restart())
    silent_note = None                   # #481: why the server ended a silent engine (death_note says it)
    gpu_busy = None                      # #1317: () -> bool, "the GPU is working" (the telemetry's reading); a GPU at work is not frozen
    last_err = None                      # #997: an ERR that was its last stdout line (death_note says it)
    batch = 0                            # --batch: the engine's batch slots (0: one request at a time)
    _ctl_mode = _RequestOwned()          # the request that has the control lines: solo / batch (see _RequestOwned)
    _ctl_result = _RequestOwned()
    _yielded = _RequestOwned()
    _last_done = _RequestOwned()
    _ctl_erred = _RequestOwned(False)
    _ctl_held = _RequestOwned(False)     # this thread already holds the control lines (Service.vram takes them itself)
    CONTROL_WAIT_S = 300.0               # how long vram() / session_file() wait for the control lines of a batch engine
    DRAIN_TIMEOUT_S = 300.0              # how long an interrupted request waits, with no line from its engine, for DONE / BADM
    QUARANTINE_S = 600.0                 # ... and how long a slot or the control lines stay held after that with no line

    # `last`: the figures of the engine's last DONE line.  With batch slots several requests run at once, each on its
    # own thread, so each one reads its own request's figures (Service.run records them); one at a time, the one dict.
    @property
    def last(self):
        if self.batch:
            tl = self.__dict__.setdefault("_tl", threading.local())
            if not hasattr(tl, "last"):
                tl.last = {}
            return tl.last
        return self.__dict__.get("_last", {})

    @last.setter
    def last(self, value):
        self.__dict__["_last"] = value
        if self.batch:
            tl = self.__dict__.setdefault("_tl", threading.local())
            tl.last = value

    def __init__(self, exe: str, args: list[str], cwd: str | None = None, log: str | None = None,
                 env: dict | None = None, lazy: bool = False, cpus: list[int] | None = None, _attempt: int = 0,
                 retry_args: list[str] | None = None, retry_vram_args: list[str] | None = None):
        # retry_args: the arguments to start again with, once, when this start ends before the engine is ready (an Auto start that picked DFlash2: mtp_retry_args)
        # retry_vram_args: the same start with no drafter (nodraft_retry_args), used instead when the engine ended because a card ran out of memory: swapping DFlash2 for
        # the MTP takes MORE memory from a card that is already short, so that direction is never taken then
        self.spawn = (exe, list(args), cwd, log, env, False, cpus)   # to start it again after it died (issue #27)
        # the all-cards split engine (sFNX, strata-split-N[-arch]): its GEN parser accepts fewer keys than Strata's engine (tests/split/test_split_gen_keys.py pins them)
        self.split_engine = os.path.basename(str(exe)).startswith("strata-split")
        self.hy4_engine = hy4_kind.is_hy4_exe(exe)     # the Hy4-Flash engine (serve/hy4_kind.py): one request at a time, no logprobs, no logit bias
        self.session_save_reclaim = '--session-save-reclaim' in args
        paths = {k: v for k, v in zip(args, args[1:]) if k in ("--native", "--pack")}
        self.model_path = paths.get("--native") or paths.get("--pack", "pack/full")
        self.log_path = log
        self.proc, self.pump, self.log = None, None, None
        self.ended, self.unloaded = True, True
        self.max_context = int(args[args.index("--max-context") + 1]) if "--max-context" in args else 4096
        self.can_stop = False            # the engine honours a STOP line mid-request (READY <ctx> stop)
        self.can_hold = False            # the engine takes GEN hold=N:ID (READY <ctx> ... hold)
        self.can_hold_batch = False      # ... and its --batch slots end at the hold too, BDONE hold (READY ... holdb)
        self.last = {}
        self.info = {}                   # INFO key=value facts (engine 0.1.8+): kv, expert slots, ... (Monitor tab)
        self.terminal_stages: list[dict] = []  # final layer ranges from this engine start's bounded stderr log
        self.terminal_batch_sample: dict | None = None
        self.terminal_stage_sample: dict | None = None
        self.prefill_tok_s_mean = None
        self.progress = None             # (read, total) prompt tokens while a prompt is read, from PP lines
        self.progress_ms = 0             # PP's own milliseconds since the prompt started (its third field)
        self.reused = 0                  # RESUME: prompt tokens not read again (a client takes them out of the work)
        self.silent_note = None
        self.last_err = None
        self._last_exit_code = None
        try:                            # a ready-made engine's BUILD.json says its version
            self.info["version"] = json.loads((Path(exe).parent / "BUILD.json").read_text()).get("version")
        except (OSError, ValueError):
            self.info["version"] = None
        if lazy:
            return
        self.unloaded = False            # `ended` stays True until READY (below): not alive while starting (#344)
        self.log = appendlog.open_append(log) if log else subprocess.DEVNULL      # #5175: the engine writes through this handle
        loading = threading.Event()                     # set once READY: the narrator below stops
        log_start = os.path.getsize(log) if log else 0  # where this start's lines begin (start_failure_hint)
        self.log_start = log_start                      # #596: the Monitor's conversation cache reads from here
        if log:
            threading.Thread(target=narrate_start, args=(log, os.path.getsize(log), args, loading),
                             daemon=True).start()
            # once per log: the follower keeps reading the same (appended) log across restarts and reloads
            if os.environ.get("STRATA_REQUEST_LINES") and os.path.abspath(log) not in _echoing:
                _echoing.add(os.path.abspath(log))
                threading.Thread(target=echo_requests, args=(log, os.path.getsize(log)), daemon=True).start()
        if log:                                          # #779: when an engine started (a crash report has the time)
            try:
                self.log.write(f"[strata] {time.strftime('%Y-%m-%d %H:%M:%S')} engine started: {exe} --serve "
                               f"{' '.join(args)}\n")
                self.log.flush()
            except OSError:
                pass
        self.max_context = 0                             # before the new process is visible: never its predecessor's
        pin = {}
        if cpus and hasattr(os, "sched_setaffinity"):    # replicas: this engine's own physical cores (serve/replicas.py)
            pin["preexec_fn"] = lambda: os.sched_setaffinity(0, cpus)
        nice = engine_nice(env)                          # Linux with built-in Radeon graphics: the desktop keeps its share of the CPU
        if nice:
            say_nice = pin.get("preexec_fn")
            pin["preexec_fn"] = lambda: (say_nice() if say_nice else None, os.nice(nice))[0]
            print(f"[strata] engine priority: nice +{nice} (this PC has built-in Radeon graphics that usually draw the desktop; "
                  "STRATA_ENGINE_NICE=0 turns it off)", flush=True)
        end_leftover_engines(exe)                        # one an earlier run lost still holds the GPU and the memory
        oomd_pacer.make_room(args)                       # Linux desktop: the file cache an earlier start left is dropped first, so the engine's RAM is free (serve/oomd_pacer.py)
        spawned_at = time.monotonic()
        try:
            self.proc = popen("the Strata engine", [exe, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8", errors="replace",
                              bufsize=1, env=env, **pin)    # (#5044: a byte that is not UTF-8 killed the reader thread)
        except OSError:                                  # nothing started: the narrator stops, the log handle goes back
            loading.set()
            if log:
                self.log.close()
            raise
        proc = self.proc                                 # THIS start's process: a close() from another thread may set self.proc to None meanwhile (#4792)
        oomd_pacer.start(proc, loading, args=args)                # Linux desktop: keeps systemd-oomd from ending the load (serve/oomd_pacer.py); not on other systems
        # the READY read has a timeout (#1317): the lines come from a thread, up to READY, so the rest of the stream
        # stays for _pump (one reader at a time)
        ready_q: queue.Queue = queue.Queue()
        threading.Thread(target=self._ready_pump, args=(self.proc, ready_q), daemon=True).start()
        ready_deadline = None if ENGINE_READY_S is None else time.monotonic() + ENGINE_READY_S
        timed_out = False
        while True:
            left = None if ready_deadline is None else ready_deadline - time.monotonic()
            if left is not None and left <= 0:
                timed_out = True
                break
            try:
                line = ready_q.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                break
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = int(v) if v.lstrip("-").isdigit() else v
            if line.startswith("READY"):
                f = line.split()
                self.max_context = int(f[1])
                self.can_stop = "stop" in f[2:]
                self.can_hold = "hold" in f[2:]
                self.can_hold_batch = "holdb" in f[2:]
                break
        loading.set()
        if self.max_context <= 0:
            if timed_out:                               # still loading, or hung: end it, so the turn is not held for good
                try:
                    proc.kill()
                except OSError:
                    pass
            try:                                        # its pipes and our handle on its log (the log stays)
                proc.wait(timeout=5)
                proc.stdin.close()
                proc.stdout.close()
                if log:
                    self.log.close()
            except (OSError, subprocess.TimeoutExpired):
                pass
            if not timed_out and _attempt < START_RETRIES and \
                    transient_start_failure(log, log_start, time.monotonic() - spawned_at, proc.returncode):
                wait_s = 2.0 * (2 ** _attempt)              # 2, 4, 8 s: the previous process's GPU queues go away
                exit_words = engine_exit_words(proc.returncode)
                if start_log_marks(log, log_start):         # the runtime said so itself (hsa_exception ...)
                    print(f"[strata] the engine stopped while the GPU runtime was starting (most often right after "
                          f"another engine exited, whose GPU queues were still being released"
                          f"{'; ' + exit_words if exit_words else ''}): starting it again in {wait_s:.0f} s "
                          f"(try {_attempt + 2} of {START_RETRIES + 1})", flush=True)
                else:                                       # an exit with nothing in the log: say only what is known
                    print(f"[strata] the engine exited before it wrote anything to its log"
                          f"{' (' + exit_words + ')' if exit_words else ''}; it may be a GPU runtime that was not "
                          f"ready yet: starting it again in {wait_s:.0f} s (try {_attempt + 2} of {START_RETRIES + 1})",
                          flush=True)
                time.sleep(wait_s)
                StrataEngine.__init__(self, exe, args, cwd, log, env, lazy, cpus, _attempt + 1, retry_args, retry_vram_args)
                return
            short_reason = None if timed_out else vram_fit.engine_stop_reason(start_log_text(log, log_start).splitlines())
            if short_reason and retry_vram_args and retry_vram_args != list(args):
                # a card ran out of memory (the engine's own reason is in its log): one more start with no drafter, in one quiet line
                print(f"[strata] {drafting.line('off', drafting.SHORT_OF_VRAM)}: starting again without a drafter ({short_reason})", flush=True)
                try:
                    with open(log, "a", encoding="utf-8") as handle:
                        handle.write(f"[strata] {time.strftime('%Y-%m-%d %H:%M:%S')} the engine ended before it was ready because a card was short of memory "
                                     f"(exit code {proc.returncode}); starting it again without a drafter\n")
                except (OSError, TypeError):
                    pass
                drafting.remember_note(drafting.SHORT_OF_VRAM)
                StrataEngine.__init__(self, exe, retry_vram_args, cwd, log, env, lazy, cpus, 0, None, None)
                return
            if not timed_out and not short_reason and retry_args and retry_args != list(args):
                # Auto picked DFlash2 and the engine ended before it was ready (an engine that does not take one of its flags, a drafter file it cannot read...):
                # one more start with the MTP, said in one quiet line. The person named DFlash2 themselves: no retry, the start fails with its reason below.
                print(f"[strata] {drafting.line('mtp', drafting.DIDNT_START)}: starting again with the MTP", flush=True)
                try:
                    with open(log, "a", encoding="utf-8") as handle:
                        handle.write(f"[strata] {time.strftime('%Y-%m-%d %H:%M:%S')} the engine ended before it was ready with the DFlash2 drafter "
                                     f"(exit code {proc.returncode}); starting it again with the MTP\n")
                except (OSError, TypeError):
                    pass
                drafting.remember_note(drafting.DIDNT_START)
                StrataEngine.__init__(self, exe, retry_args, cwd, log, env, lazy, cpus, 0, None, None)
                return
            why = (f"the engine did not report READY within {ENGINE_READY_S:.0f} s" if timed_out else
                   "the engine exited before it was ready")
            # EngineDied, not bare RuntimeError: a request waiting on this load must end with a 503
            # through the request path's handler - a RuntimeError there left the turn with no answer (#1527)
            note = "" if timed_out else start_exit_note(proc.returncode, exe)
            tail = start_log_tail(log, log_start)
            lines = [x for x in tail.split("\n")[2:] if x.strip()]
            if tail and lines and all(x.strip().startswith("[strata]") for x in lines):
                tail += "\n  (those are the launcher's own lines: the engine itself wrote nothing)"
            raise EngineDied(why + (f" (see {log})" if log else "") + note +
                             start_failure_hint(log, log_start) + tail)
        self.known_ctx = self.max_context   # survives a failed restart: requests keep their limit and restart it
        # (from PR #41, midhatn) a locally built engine can sit next to another release's BUILD.json: engines that
        # report their own version (INFO engine=, 0.1.8+) win, the manifest stays the fallback for older ones
        if self.info.get("engine"):
            self.info["version"] = str(self.info["engine"])
        self.terminal_stages = read_engine_stage_log(log, log_start) if log else []
        self.info.pop("commit_warning", None)           # this start only: a restart that no longer warns clears it
        try:                                            # #1607: the engine's Windows commit-limit warning, for the Monitor and the app
            warned = commit_limit.commit_warning(start_log_text(log, log_start)) if log else None
            if warned:
                self.info["commit_warning"] = warned
        except Exception:                               # noqa: BLE001 - a note must never stop a start
            pass
        try:                                            # which drafter runs, and in plain words why not the one that was asked for
            if self.inkling_engine:
                self.info.update(inkling_kind.drafting_info(args, self.info))
            else:
                self.info.update(drafting.engine_info(args, start_log_text(log, log_start), drafting.take_note()))
        except Exception:                               # noqa: BLE001 - a note must never stop a start
            pass
        self.ended = False                              # READY: alive from here (restart() set it True, #344)
        self.closing = False                            # #6055: close() is ending this engine (set before its STOP)
        # the engine's stdout on a thread, so a request can wait with a timeout (heartbeats, cancel checks)
        self.gen = self.__dict__.get("gen", 0) + 1      # which engine process this is (a request notes its own); first, so
        #                                                 a request that sees the new number sees the new queues (#4451)
        self.lines: queue.Queue = queue.Queue()
        # --batch: concurrent requests in the engine's batch slots (see generate_batched).  The engine says how many
        # it runs (INFO batch_slots=N: it may have fewer than asked, or none, when they do not fit)
        asked = next((int(args[args.index(k) + 1]) for k in ("--batch", "--slots", "--parallel") if k in args), 0)
        self.batch = int(self.info.get("batch_slots") or 0)
        if asked and self.batch != asked:
            print(f"[strata] parallel requests: {asked} asked, the engine runs {self.batch or 'one at a time'} "
                  "(its log says why)", flush=True)
        _bg = args[args.index("--batch-groups") + 1] if "--batch-groups" in args else "default"
        # no `--batch-groups` (the default on a layer split) or `auto`: the engine picks the groups and says so
        # (INFO batch_groups=G); a number is the group count asked for (1 = off)
        groups = int(self.info.get("batch_groups") or 1) if _bg in ("auto", "default") else int(_bg)
        groups = groups if self.batch and groups > 0 and self.batch % groups == 0 else 1
        gs = self.batch // groups if self.batch else 0
        # slots in the order that spreads requests over the pipeline's groups first: 0, gs, 2gs, .., 1, gs+1, ..
        self.slot_order = [g * gs + t for t in range(gs) for g in range(groups)]
        self.slot_group = [b // gs if gs else 0 for b in range(self.batch)]   # each slot's pipeline group
        self.slot_groups = groups
        # pipelined groups (the engine says how many it runs: INFO batch_groups; older engines: the args).  A pipelined
        # group's window spans its slots up to its last busy one, and the engine keeps no pipelined slot as a
        # conversation cache - see pick_slot.  Not an engine that reads a conversation's next turn in its slot (INFO
        # slot_read=1): its windows leave an idle slot alone, so every slot keeps its conversation as elsewhere.
        groups_run = int(self.info.get("batch_groups") or groups)
        self.slot_gs = gs if groups > 1 and groups_run > 1 and not self.info.get("slot_read") else 0
        self.slot_q = [queue.Queue() for _ in range(self.batch)]
        self.slot_busy = [False] * self.batch
        # #1603: who owns a busy flag.  slot_claims: slots a request of this server picked or reserved and has not
        # given back; slot_releasing: slots handed to the background BDONE wait.  A busy slot in neither has no owner
        # (its flag leaked) and would keep every later request waiting for a slot that nothing will free.
        self.slot_claims: set[int] = set()
        self.slot_releasing: set[int] = set()
        self._orphan_seen: dict[int, float] = {}
        # what each slot's sessions hold (prompt + every token a window fed), so a conversation's next turn goes to
        # the slot that has its start (the engine checks it again); when the slot was last used
        self.slot_held: list[list[int]] = [[] for _ in range(self.batch)]
        # Inkling-2-Mini: the prompt of the request a slot ran last (valid while slot_held[b] is not empty).  Its engine resumes the
        # next turn at the end of that prompt (the prompt-end checkpoint), not only from the whole held stream: the reply it
        # decoded (hidden thinking, a call's own spacing) is never sent back token for token (#6733)
        self.slot_prompt: list[list[int]] = [[] for _ in range(self.batch)]
        self.slot_used = [0.0] * self.batch
        self.slot_live: list[dict | None] = [None] * self.batch   # /metrics: the request in each slot
        # #1012: the admission state belongs to the server, not to one engine process.  restart() runs this again
        # while requests still wait on it (they hold these very objects), so it is made once and kept: new ones
        # would leave the waiters on a lock and a condition nobody notifies, with their counts missing from the
        # new lists.
        if "slot_cv" not in self.__dict__:
            self.slot_cv = threading.Condition()
            self.waiting = 0                            # requests waiting for the control lines (ctl)
            self.admitting = 0                          # #1603: ... and those that gave the lines back to wait for a slot
            self.par_counters = {"queued": 0, "queue_wait_s": 0.0}
            self.wait_lens: list[list[int]] = []        # ... their prompt lengths (a long read gives way to short ones)
            self.arrivals = collections.deque(maxlen=64)   # monotonic times of the last requests that came for the control lines (STRATA_COALESCE_*)
            self.ctl_epoch = 0                          # how often the control lines were taken
            self.ctl = threading.Lock()                 # one admission or solo request on the control lines at a time
        self._ctl_erred = False                         # #1059: the control lines ended on an ERR
        self._yielded = None                            # (slot, tokens read): the last request on them gave way
        self.wlock = threading.Lock()                   # stdin writes from several request threads
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()

    def _pump(self):
        proc, lines = self.proc, self.lines             # this process's: a restart replaces both (#344)
        slot_q = self.slot_q
        slot_busy = self.slot_busy                      # this process's (a restart makes new ones)
        log = getattr(self, "log", None)                # retain this process's log across a restart
        line = None
        for line in proc.stdout:
            # stderr already reaches the log; native ERR diagnostics use the protocol pipe.
            if line.startswith("ERR ") and log not in (None, subprocess.DEVNULL):
                try:
                    log.write(f"[strata] {time.strftime('%Y-%m-%d %H:%M:%S')} native {line.rstrip()}\n")
                    log.flush()
                except (OSError, ValueError):          # a closed/full log must not prevent EOF publication
                    pass
            # checked before batch routing: a fatal line is never a slot's own
            if line.startswith(FATAL_PREFIXES):
                # release_gpu_waits invalidates the verifier, even if the native
                # process stays alive and prints DONE afterwards.
                if self.proc is proc:
                    self.silent_note = "Unrecoverable native verification failure: " + line[4:].strip()
                    self.ended = True
                try:
                    proc.terminate()
                except OSError:
                    pass
                break                           # publish EOF, never the trailing DONE
            if line.startswith("BSTAT"):
                sample = parse_batch_window_sample(line)
                if sample is not None and self.proc is proc:
                    self.terminal_batch_sample = {**sample, "observed_monotonic": time.monotonic()}
                continue                        # control telemetry never enters a request's token stream
            if line.startswith("PSTAT"):
                sample = parse_stage_in_flight_sample(line)
                if sample is not None and self.proc is proc:
                    self.terminal_stage_sample = {**sample, "observed_monotonic": time.monotonic()}
                continue
            if line.startswith(("BT ", "BDONE ")) and slot_q:   # --batch: a batch slot's own lines
                try:
                    b = int(line.split()[1])
                    if 0 <= b < len(slot_q) and not slot_busy[b]:
                        # #4451: a frame for a slot that no request has claimed belongs to a request that is over (its
                        # slot was given back too early, or never claimed): the next request on the slot must not read it
                        self.stale_frames = getattr(self, "stale_frames", 0) + 1
                        btrace("stale frame dropped", line.strip()[:60])
                        continue
                    slot_q[b].put(line)
                    continue
                except (IndexError, ValueError):
                    pass
            lines.put(line)
        if self.proc is proc:                           # a killed engine's pump must not mark its successor dead
            self.ended = True                           # its output closed: it is gone, even before the OS says so
            rc = proc.poll() if hasattr(proc, "poll") else getattr(proc, "returncode", None)
            if rc is not None:
                self._last_exit_code = rc
            if line and line.startswith("ERR"):         # #997 #890: why it exited, though no request may read it
                self.last_err = line[4:].strip()        # (a failed batch window: every request reads its slot)
        lines.put(None)
        for q in slot_q:
            q.put(None)

    def death_note(self) -> str:
        """Why the engine most likely ended, from the end of its log: its own watchdog (issue #29), else RAM."""
        if getattr(self, "silent_note", None):          # #481: the server ended it, not the OS or the engine itself
            return self.silent_note
        tail = wide = ""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 262144))
                wide = f.read().decode("utf-8", "replace")
            tail = wide[-4096:]
        except (OSError, TypeError):
            pass
        # #1705: STRATA_VERIFY_TRACE dumps are ~30 KB and push the watchdog's line out of a 4 KB tail
        for line in reversed(wide.splitlines()):
            if "issue #29" in line:
                return ("The engine stopped itself because it had stopped making progress - a hang it caught. Its log "
                        "line: " + line.strip() + " - please report it at github.com/Niko1221/Strata/issues.")
        if (self.last_err or "").startswith("fatal GPU error"):    # #4350: the engine ended itself; the next request starts it again
            return ("The graphics card reported a fatal error (" + self.last_err[len("fatal GPU error:"):].strip()[:160] + "), so the engine "
                    "ended. The next request starts it again. If this repeats, please report it at github.com/Niko1221/Strata/issues "
                    "with the log.")
        rc = self.exit_code()
        if rc is not None and rc >= 0 and self.last_err:  # its own last words on stdout: they say why
            return (f"The engine exited (code {rc}) after it reported: {self.last_err} - please report it at "
                    "github.com/Niko1221/Strata/issues with the log.")
        last = next((x.strip() for x in reversed(tail.splitlines()) if x.strip().startswith(("strata", "ERR"))), "")
        if rc is not None and rc >= 0 and last:          # it ended by itself: its own last words say why (#215)
            return (f"The engine exited (code {rc}). Its last log line: {last} - if that does not explain it, please "
                    "report it at github.com/Niko1221/Strata/issues with the log.")
        card = next((x.strip() for x in reversed(tail.splitlines())
                     if re.search(r"out of memory", x, re.I) and re.search(r"cuda|hip|vram|gpu|graphics", x, re.I)), "")
        if card and rc != -9:                             # #5078: the engine aborted on a card that was full: not a RAM story, and no dmesg to read
            return ("The engine stopped because the graphics card ran out of memory (its log says: " + card[:200] + "). Close other programs "
                    "that use the graphics card and try again. If it keeps happening, keep more graphics memory free: "
                    "strata launch --set vram-reserve-mib=2048.")
        rt_note = rttime_limit_note() if rc == -9 else ""
        if rt_note:                                       # #3238: the kernel killed it, but not for RAM
            return ("The engine was ended by signal 9. " + rt_note + " If dmesg (sudo dmesg | grep -i -E 'killed "
                    "process|out of memory') shows an out-of-memory kill instead, close other programs or use a smaller "
                    "model (Q2_0 / IQ2_XS).")
        return ("The usual cause is running out of RAM: Linux then ends the biggest program (check: sudo dmesg | "
                "grep -i -E 'killed process|out of memory'); Windows slows down instead. Close other programs or use a "
                "smaller model (Q2_0 / IQ2_XS).")

    def alive(self) -> bool:
        proc = self.proc                                 # read once: restart() / close() may set it to None meanwhile
        return proc is not None and not getattr(self, "ended", False) and proc.poll() is None

    def exit_code(self, proc=None, timeout: float = 0.25):
        """Return the engine's exit code/signal, waiting briefly when poll() is still None right as its pipes close."""
        p = self.proc if proc is None else proc
        if p is None:
            return getattr(self, "_last_exit_code", None)
        rc = p.poll() if hasattr(p, "poll") else getattr(p, "returncode", None)
        if rc is None and hasattr(p, "wait"):
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                rc = p.poll() if hasattr(p, "poll") else None
        if rc is not None:
            self._last_exit_code = rc
            return rc
        return getattr(self, "_last_exit_code", None)

    def _crash_tail(self, max_lines: int = 3) -> str:
        """Trailing diagnostic reason from stdout ERR or the engine's stderr log since this start."""
        if getattr(self, "last_err", None):
            return str(self.last_err).strip()[:280]
        log_path = getattr(self, "log_path", None)
        if not log_path:
            return ""
        try:
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                start = max(getattr(self, "log_start", 0) or 0, f.tell() - 8192)
                f.seek(start)
                text = f.read().decode("utf-8", "replace")
        except (OSError, TypeError, ValueError):
            return ""
        raw = [x.strip() for x in text.splitlines() if x.strip()]
        if not raw:
            return ""
        non_launcher = [x for x in raw if not x.startswith("[strata]")]
        picked = (non_launcher or raw)[-max_lines:]
        return " | ".join(picked)[:280]

    def _engine_died(self, proc=None) -> EngineDied:
        """Build an actionable EngineDied error with the real reaped exit code/signal and trailing stderr crash reason."""
        try:
            rc = self.exit_code(proc=proc) if proc is not None else self.exit_code()
        except TypeError:
            rc = self.exit_code()
        status = engine_exit_words(rc) if rc is not None else "process exited; status unavailable"
        tail = self._crash_tail()
        return EngineDied(f"the engine stopped unexpectedly ({status})" + (f": {tail}" if tail else ""))

    def unload(self):
        """Release GPU/RAM between requests, retaining the spawn config for automatic reloading."""
        self.close()
        self.unloaded = True

    RESTART_RETRY_S = 15.0   # between the tries of restart(): a dying engine's VRAM may take a while to come back

    @staticmethod
    def _ready_pump(proc, out: queue.Queue) -> None:
        """The engine's first lines up to READY, for a read that can time out; None when its output closed first."""
        done = False
        try:
            for line in proc.stdout:
                out.put(line)
                if line.startswith("READY"):
                    done = True
                    return
        except (OSError, ValueError):                   # the pipe closed under us (the engine was killed)
            pass
        finally:
            if not done:
                out.put(None)

    def restart(self, tries: int = 3):
        """Start the engine again (the same command) after it died; the new process has its own line queue.
        close() ends and waits for the old process first (EngineStuck when it cannot be ended).  A start that still
        exits before READY - a dead engine's VRAM can take a while to come back, notably on ROCm - is retried
        (PR #637)."""
        info = dict(self.info)
        self.starting = True                     # prepare() answers 503 "starting" meanwhile (#344)
        try:
            self.close()
            for i in range(tries):
                # #344: not alive until READY - __init__ sets max_context to 0 and blocks until the engine says
                # READY, and a request that saw alive() in that window skipped load() and failed with "context
                # (0)".  __init__ clears `ended` itself once READY (before its pump thread can set it again).
                self.ended = True
                try:
                    self.__init__(*self.spawn)
                    break
                except RuntimeError:
                    try:
                        self.proc.wait(timeout=60)
                    except (subprocess.TimeoutExpired, OSError):
                        pass
                    if i == tries - 1:
                        raise
                    print(f"[strata] the engine did not start (try {i + 1} of {tries}); again in "
                          f"{self.RESTART_RETRY_S:g} s", flush=True)
                    time.sleep(self.RESTART_RETRY_S)
        finally:
            self.starting = False
            # a lazy first start never reached the post-READY __init__ where slot_cv is made (#1527)
            if "slot_cv" in self.__dict__:
                with self.slot_cv:               # #1012: wake the requests that waited through it: they go on with the
                    self.slot_cv.notify_all()    # new engine, or (it did not start) end with a clean EngineDied
        self.info = {**info, **self.info}

    def _parse_done(self, line):
        f = line.split()
        self.last = {"generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                     "decode_ms": float(f[4]), "finish": f[5]}
        if len(f) >= 9:                                   # the conversation cache's fields (engine 0.1.3+)
            self.last.update(drafts_accepted=int(f[6]), drafts_offered=int(f[7]), reused=int(f[8]))
        if len(f) >= 11:                                  # decode hit rate fields
            self.last.update(hits=int(f[9]), lookups=int(f[10]))
        if len(f) >= 14:                                  # the expert tiers (engine 0.1.31+): RAM / file blobs, file MB
            self.last.update(ram_blobs=int(f[11]), file_blobs=int(f[12]), file_mb=float(f[13]))
        if len(f) >= 15:                                  # #471 (engine 0.1.36+): the prompt tokens actually read
            self.last.update(prompt_read=int(f[14]))
        if len(f) >= 16:                                  # #588 (engine 0.1.39+): routed experts read over PCIe
            self.last.update(offloaded=int(f[15]))
        for tok in f[16:]:                                # key=value fields after the positional ones (the prompt's
            key, eq, value = tok.partition("=")           # first chunk and its streamed expert blobs)
            if eq and key in ("chunk", "experts_streamed"):
                try:
                    self.last[key] = int(value)
                except ValueError:
                    pass

    @contextlib.contextmanager
    def _exclusive_control(self):
        """#4451 (same class): on a batch engine the control lines carry one exchange at a time, and every request takes
        them (`ctl`) before it sends or reads anything there.  VRAM and SAVE / RESTORE read the same lines for their
        answer; run beside an admission they took its PP / T / BADM lines, and the admission its answer.  They take the
        lines like a request does.  One request at a time: nothing to take."""
        ctl = self.__dict__.get("ctl")
        if not self.batch or ctl is None or self._ctl_held:
            yield
            return
        if not ctl.acquire(timeout=self.CONTROL_WAIT_S):
            raise ValueError("a request is still running; try again when it has finished")
        try:
            yield
        finally:
            ctl.release()

    def vram(self, reserve_mib: int | None, timeout: float = 120.0, device: int | None = None) -> dict:
        with self._exclusive_control():
            return self._vram(reserve_mib, timeout, device)

    def _vram(self, reserve_mib: int | None, timeout: float = 120.0, device: int | None = None) -> dict:
        """#533: `VRAM <reserve_mib>` between requests (the caller holds the service's FIFO): the engine shrinks its
        expert cache until that much VRAM is free, or grows it back when more is free; None: back to the reserve it
        started with.  -> the engine's figures (expert_slots, expert_cache_mib, vram_free_mib, ...); raises ValueError
        with the engine's reason (e.g. it was not started with --vram-elastic), EngineDied when it ended."""
        if not self.alive():
            raise EngineDied("the engine is not running")
        if device is not None and reserve_mib is None:   # per-GPU (#533): a device only makes sense with an explicit reserve.
            raise ValueError("VRAM with a device needs an explicit reserve_mib")
        cmd = "VRAM" + ("" if reserve_mib is None else f" {int(reserve_mib)}" + (f" {int(device)}" if device is not None else ""))
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                raise EngineDied("the engine did not answer the VRAM command") from None
            if line is None:
                raise EngineDied("the engine ended")
            line = line.strip()
            if line.startswith("ERR"):
                raise ValueError(line[4:].strip() or "the engine refused the VRAM command")
            if line.startswith("VRAM "):
                out = {}
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    out[k] = int(v) if v.lstrip("-").isdigit() else v
                self.info.update({k: out[k] for k in ("expert_slots", "vram_free_mib") if k in out})
                cards = out.get("cards")                     # #533 per-GPU free, keyed by CUDA device id (a card the command did not act on is reported too).
                if isinstance(cards, str):
                    per = {}
                    for part in cards.split("|"):
                        bits = part.split(":")
                        if len(bits) == 3 and all(b.isdigit() for b in bits):   # reject any sign: a malformed engine line cannot inject a negative device or figure.
                            dev, mapped_mib, free_mib = (int(b) for b in bits)
                            per[str(dev)] = {"mapped_mib": mapped_mib, "free_mib": free_mib}
                    if per:
                        out["per_device"] = per
                self.info["vram"] = out
                return out
            if time.time() > deadline:
                raise EngineDied("the engine did not answer the VRAM command")

    def pen_from_key(self, orig_len: int, prompt_len: int) -> str:
        """` pen_from=N` for a leg that continues a request (prompt + what it generated): where the reply began, Strata's penalty window start. The all-cards split engine
        (sFNX) and Hy4-Flash (`strata-hy4`) refuse the key (GEN: unknown/unsupported key pen_from): it is not sent."""
        if getattr(self, "split_engine", False) or getattr(self, "hy4_engine", False) or prompt_len <= orig_len:
            return ""
        return f" pen_from={orig_len - 1}"

    @staticmethod
    def sampling_keys(sampling: dict, *, hy4: bool = False) -> str:
        keys = ""
        t = sampling.get("temperature")
        if isinstance(t, (int, float)) and float(t) > 0.0:
            keys += f" temperature={float(t)!r}"
        tp = sampling.get("top_p")
        if isinstance(tp, (int, float)) and float(tp) < 1.0:
            keys += f" top_p={float(tp)!r}"
        tk = sampling.get("top_k")
        if isinstance(tk, int) and not isinstance(tk, bool) and tk >= 0:
            # the engine's sampled path keeps at most 64 candidates: 0 ("off") and wider lists get all 64
            keys += f" top_k={tk if 1 <= tk <= 64 else 64}"
        mp = sampling.get("min_p")
        if isinstance(mp, (int, float)) and 0.0 < float(mp) <= 1.0:
            keys += f" min_p={float(mp)!r}"
        rp = sampling.get("repetition_penalty")
        rp_on = isinstance(rp, (int, float)) and float(rp) != 1.0
        pf = sampling.get("frequency_penalty")
        pf_on = isinstance(pf, (int, float)) and float(pf) != 0.0
        pp = sampling.get("presence_penalty")
        pp_on = isinstance(pp, (int, float)) and float(pp) != 0.0
        if rp_on:
            keys += f" penalty_repeat={float(rp)!r}"
        if pf_on:
            keys += f" penalty_freq={float(pf)!r}"
        if pp_on:
            keys += f" penalty_present={float(pp)!r}"
        if rp_on or pf_on or pp_on:
            # a penalty without a window counts over nothing: the engine's default is the last 64 tokens
            pln = sampling.get("penalty_last_n")
            if isinstance(pln, int) and not isinstance(pln, bool) and pln > 0:
                keys += f" penalty_last_n={pln}"
            else:
                keys += " penalty_last_n=64"
        seed = sampling.get("seed")
        if isinstance(seed, int) and seed > 0:
            keys += f" seed={seed}"
        if hy4:
            # Hy4-Flash (serve/hy4_kind.py): its GEN parser strictly rejects Flash-Next-only keys
            # (cvec, ckpt, pin, hold, pipe, aux_cpus, img_keys, logprobs, allowed, bias).
            for p in sampling.get("strata_hy4_images") or []:
                keys += " image=" + hy4_kind.encode_path(p)
            for p in sampling.get("strata_inkling_images") or []:
                keys += " image=" + inkling_kind.encode_path(p)
            if inkling:
                prefix = sampling.get("strata_prefix")
                if isinstance(prefix, dict):
                    pn = prefix.get("tokens")
                    if isinstance(pn, int) and not isinstance(pn, bool) and pn > 0:
                        keys += f" pin={pn}"
                if sampling.get(KEEP_CKPT_KEY) is True:     # (the server sets it, only for an engine that says ckpt_key=1)
                    keys += " ckpt=0"
        effort = kwargs.get("reasoning_effort")
        off = kwargs.get("enable_thinking") is False
        if not self.effort_end or (not off and effort in (None, "", "xhigh", "high")):
            return self.template.render(messages, tools=tools, **kwargs)
        base = {k: v for k, v in kwargs.items() if k not in ("reasoning_effort", "enable_thinking")}
        history = self.template.render(messages, tools=tools, add_generation_prompt=False, **base)
        full = self.template.render(messages, tools=tools, add_generation_prompt=True, **kwargs)
        mine = self.template.render(messages, tools=tools, add_generation_prompt=False, **kwargs)
        if not full.startswith(mine):                  # a template this cannot take apart: rendered as asked
            return full
        tail = "" if off else EFFORT_TURN.format(EFFORT_TEXT.get(effort, EFFORT_TEXT["medium"]))
        return history + tail + full[len(mine):]

    def encode_prompt(self, messages, tools, kwargs, store: bool = True) -> list[int]:
        """The request's prompt: the template rendered and tokenized.  #537: a <think> / </think> written inside a
        message's text is encoded as the text it is, not as the model's reasoning markers (the template's own are).
        A control token's text (<|im_start|>, <|im_end|>, <|endoftext|>, ...) inside a message is text as well: only
        the control tokens the template writes are control tokens.  `store=False` (a token count) reuses the kept
        prompts without adding or replacing one."""
        marked, marked_tools, changed = mark_think_literals(messages, tools, self.literals)
        prompt = self.render_prompt(marked, marked_tools, kwargs)
        plain = ()
        if changed:
            prompt, plain = unmark_think_literals(prompt, self.literals)
        if self.prompts is not None:            # #567: only what follows the last special shared with a recent prompt
            return self.prompts.encode(prompt, plain, store)
        return self.tok.encode(prompt, parse_special=True, plain=plain) if plain else \
            self.tok.encode(prompt, parse_special=True)

    @contextlib.contextmanager
    def _turn(self, cancel=None):
        """The engine's single turn (self.fifo) for an image encode.  #1999: a client that leaves while this request
        waits for it gives its place up (ClientGone), as a queued generation does (#1619)."""
        if cancel is None:
            self.fifo.acquire()
        else:
            while not self.fifo.acquire(timeout=0.25):
                if cancel.is_set():
                    raise ClientGone()
        try:
            yield
        finally:
            self.fifo.release()

    def _note_unreadable_tool_images(self, messages, cancel=None):
        """A picture a tool returned (Claude Code's Read of an image file) that this server cannot read - it has no
        image encoder, or the encoder refuses that picture - becomes a short note in its place instead of a 400.
        Clients resend the whole history every turn, so one unreadable tool picture would fail every later request of
        the conversation; a picture the user sends keeps the 400.  A picture that encodes stays in the cache, so the
        prompt's own encode below finds it."""
        fetched = {}                     # a URL's bytes, so prepare() does not download the same picture again
        for m in messages:
            content = m.get("content")
            if m.get("role") != "tool" or not isinstance(content, list):
                continue
            for n, item in enumerate(content):
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                why = None if self.has_vision() else "this server has no image encoder"
                if why is None:
                    src = item["source"]
                    try:
                        if cancel is not None and cancel.is_set():
                            raise ClientGone()
                        if self.offline and src.startswith(("http://", "https://")):
                            raise ValueError("offline mode cannot download an image URL")
                        if src.startswith(("http://", "https://")):
                            src = fetched[item["source"]] = Vision.download(src)   # outside the FIFO, as in prepare()
                        if self.vision is not None:
                            with self._turn(cancel):  # the encoder takes its turn with the requests (see below)
                                self._vision_up()         # an elastic encoder is not running until a picture needs it
                                self.vision.encode(src)
                        else:
                            # Hy4-Flash in-process vision: validate readability without invoking a sidecar encoder
                            Vision.normalize(src if isinstance(src, bytes) else Vision.load(src))
                    except (ValueError, OSError, RuntimeError) as e:    # RuntimeError: the encoder did not start
                        why = str(e)
                if why is not None:
                    print(f"[strata] a picture a tool returned was left out of the prompt: {why}", flush=True)
                    content[n] = {"type": "text", "text": f"[image omitted: {why}]"}
        return fetched

    def forget_prompt(self, ids):
        if self.prompts is not None:
            self.prompts.forget(ids)


    def prepare(self, messages, tools, kwargs, max_new=None, force=None, req=None, cancel=None):
        """-> (ids, thinking, max_new). An unset or non-positive max_new (some clients send -1) means "unlimited":
        the rest of the context.  `force` (forced_call): without thinking the reply starts with it, so it ends the
        prompt; with thinking, Service.run writes it once the thinking is over."""
        # `cancel` (#1999): set when the client hangs up; the downloads, the wait for the turn and the encodes stop
        # between pictures and raise ClientGone
        # Cached tool images are references in browser history, never arbitrary file paths or huge base64 strings.
        for message in messages:
            for part in message.get("content", []) if isinstance(message.get("content"), list) else []:
                source = part.get("source", "") if isinstance(part, dict) else ""
                if isinstance(part, dict) and part.get("type") == "image" and isinstance(source, str) and source.startswith("strata-tool-media:"):
                    name = source[len("strata-tool-media:"):]
                    if not re.fullmatch(r"[0-9a-f]{64}\.(png|jpg|webp)", name):
                        raise ValueError("invalid tool image reference")
                    # Clients resend old tool results. A removed cache or disabled feature must not poison
                    # every later turn; keep the other text and replace only the unavailable picture.
                    media_dir = self.mcp.media_dir if self.mcp is not None else None
                    image = Path(media_dir) / name if media_dir is not None else None
                    if image is None or not image.is_file():
                        why = "tool image cache is disabled" if image is None else "cached tool image is no longer available"
                        part.clear()
                        part.update(type="text", text=f"[image omitted: {why}]")
                    else:
                        part["source"] = str(image)
        fetched = self._note_unreadable_tool_images(messages, cancel)
        ids = encoded_ids = self.encode_prompt(messages, tools, kwargs)
        if force and (kwargs.get("enable_thinking", True) is False or self.inkling):   # Inkling: the call opens the reply (no thinking pass to wait for)
            ids = ids + self.tok.encode(force, parse_special=True)
        self.embeddings.path = None
        self.embeddings.keys = None
        images = images_of(messages)
        had_images = bool(images)
        self.embeddings.hy4 = self._drop_hy4_images(getattr(self.embeddings, "hy4", None))
        self.embeddings.inkling = self._drop_hy4_images(getattr(self.embeddings, "inkling", None))      # an earlier pass of this request left its files: gone first
        if images and (self.hy4 or self.inkling):                     # Hy4-Flash / Inkling-2-Mini: in-process engine reads pictures directly (INFO images=1)
            kind_mod = inkling_kind if self.inkling else hy4_kind
            if (getattr(self.engine, "info", None) or {}).get("images") != 1:
                raise ValueError(hy4_kind.NO_IMAGES_REPLY)
            if self.offline and any(src.startswith(("http://", "https://")) for src in images):
                raise ValueError("offline mode cannot download an image URL; send a data: image instead")
            self.embeddings.hy4 = hy4_kind.write_images(
                images, lambda s: fetched[s] if s in fetched else Vision.load(s), Vision.normalize, cancel)
            images = []
        if images:
            if self.offline and any(src.startswith(("http://", "https://")) for src in images):
                raise ValueError("offline mode cannot download an image URL; send a data: image instead")
            if self.vision is None:
                raise ValueError("this server was started without the vision encoder (run setup again and choose "
                                 "'vision'), so it cannot read images")
            # An image URL is downloaded first, outside the FIFO: under it, a slow server held every other request
            # for as long as urlopen waited.
            def fetch(src):
                if cancel is not None and cancel.is_set():
                    raise ClientGone()
                return fetched[src] if src in fetched else Vision.download(src) \
                    if src.startswith(("http://", "https://")) else src
            images = [fetch(src) for src in images]
        if images:
            pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
            start = self.tok.encode(VISION_START, parse_special=True)[0]
            # Encode only while the engine is idle: the engine and the image encoder (a separate process) must not
            # run on the GPU at the same time - an encode during a running request left that request stuck at
            # "reading the prompt" with CPU and GPU busy, for good (reproduced).  So encoding takes its turn in the
            # same FIFO as the requests.
        with (self._turn(cancel) if images else contextlib.nullcontext()), \
                (self.vision.batch() if images and hasattr(self.vision, "batch") else contextlib.nullcontext()):
            if images:
                if not (hasattr(self.vision, "all_cached") and self.vision.all_cached(images)):
                    self._vision_up()
                try:
                    encoded = (self.vision.encode_all(images, cancel) if hasattr(self.vision, "encode_all")
                               else [self.vision.encode(src) for src in images])
                finally:
                    self.last_image_at = time.time()
                # one <|image_pad|> per image -> one per image token.  Only the markers the template writes for an image
                # (right after <|vision_start|>) are images: the same text inside a message (an agent reading these docs,
                # #150) is kept as plain text, or it took an image's place and the counts no longer matched.
                literal = self.tok.encode(IMAGE_PAD, parse_special=False)
                out, k = [], 0
                for j, t in enumerate(ids):
                    if t == pad and j > 0 and ids[j - 1] == start:
                        # more pairs than images: text parts that cut both markers apart (the literal marks keep whole ones text)
                        if k == len(encoded):
                            raise ValueError("the prompt and its images do not match")
                        out += [pad] * encoded[k][1]
                        k += 1
                    elif t == pad:
                        out += literal
                    else:
                        out.append(t)
                if k != len(encoded):
                    raise ValueError("the prompt and its images do not match")
                ids = out
            ctx = self.engine.max_context
            if ctx <= 0:
                if getattr(self.engine, "starting", False):   # #344: (re)starting, not a prompt that is too long
                    raise EngineStarting("the engine is starting (a minute or two); try again shortly")
                # a failed restart leaves max_context 0: check against the last known context, so the request reaches
                # run() - which starts the engine again - instead of failing with "exceeds the context (0)" forever
                ctx = getattr(self.engine, "known_ctx", 0)
                if ctx <= 0:
                    raise EngineStarting("the engine is starting (a minute or two); try again shortly")
            room = ctx - CTX_SLACK - len(ids)
            if max_new is None or max_new <= 0 or (self.fit_max_tokens and room < 1):
                if room < 1:
                    # A prompt refused for the context is not kept (#567): the client cannot resend it unchanged.  The
                    # other refusals (engine starting, images) keep it: the same prompt comes back and its encode is reused.
                    self.forget_prompt(encoded_ids)
                    raise ContextOverflow("request exceeds the context window: "
                                          f"prompt ({len(ids)} tokens) leaves no room to answer in the context "
                                          f"({ctx}); requests are never truncated", prompt=len(ids), context=ctx)
                max_new = room
            elif max_new > room:
                floor = fit_floor(ctx)
                if self.fit_max_tokens and room < floor:
                    # #5924: a clamp to a few hundred tokens left a thinking model no room to answer (or to write a tool
                    # call); agents resent the turn and wrote the same truncated sentence again and again.  The context is
                    # all but full: a 400 that says so makes the client compact (Claude Code reads "prompt is too long").
                    self.forget_prompt(encoded_ids)
                    raise ContextOverflow("request exceeds the context window: "
                                          f"the prompt ({len(ids)} tokens) leaves {max(0, room)} of the {ctx}-token "
                                          f"context, fewer than the {floor} an answer needs; Strata never truncates "
                                          f"requests. Shorten the conversation, or send max_tokens {max(0, room)} or less",
                                          prompt=len(ids), context=ctx, max_new=max_new, min_room=floor)
                if not self.fit_max_tokens:
                    self.forget_prompt(encoded_ids)
                    raise ContextOverflow("request exceeds the context window: "
                                          f"prompt ({len(ids)} tokens) + max tokens ({max_new}) exceeds the context "
                                          f"({ctx}); requests are never truncated. Send a smaller "
                                          f"max_tokens (at most {max(0, room)} here), or add \"fit_max_tokens\": true to the "
                                          "model's strata-<model>.json to shorten it to the room left (#545)",
                                          prompt=len(ids), context=ctx, max_new=max_new)
                print(f"[strata] max_tokens {max_new} shortened to {max(1, room)}: the prompt ({len(ids)} tokens) leaves that much "
                      f"of the context ({ctx}); \"fit_max_tokens\": false answers 400 instead", flush=True)
                max_new = max(1, room)          # a shorter completion beats a 400 (the default)
            if images:
                # The request's images in one file for GENI (~10 MB a picture), written once nothing above refuses the
                # request: one refused after it (the engine starting, no room) left it in the vision directory for good,
                # one more for every retry of a 503.  run() deletes it; drop_embeddings() if run() never starts.
                combined = combined_embeddings_path(self.vision.dir, sum(Path(p).stat().st_size for p, _ in encoded))
                self.embeddings.path = combined             # first, so a half-written one is found as well
                # #2228: each picture's key is its normalized bytes' hash (the encoder's file name), so the engine's
                # conversation cache matches a picture by what it is, not by its embedding floats: a picture encoded again
                # (an encoder restart, the cache's eviction) with other bits still matches its checkpoints
                self.embeddings.keys = [Path(p).stem[:16] for p, _ in encoded]
                write_temporary(combined, [p for p, _ in encoded])
        auto_prefix = False
        inkling_shared_ok = str((getattr(self.engine, "info", None) or {}).get("shared_prefix")) == "1"
        if (req is not None and "strata_prefix" not in req and self.inkling and not had_images and
                os.environ.get("STRATA_INKLING_SHARED_PREFIX") == "1" and inkling_shared_ok):
            # Pin only leading instructions; never include conversation-specific user text.
            count = 0
            for message in messages:
                if message.get("role") not in ("system", "developer"):
                    break
                count += 1
            if 0 < count < len(messages):
                req["strata_prefix"] = {"messages": count}
                auto_prefix = True
        if req is not None and req.get("strata_prefix") is not None:
            req["strata_prefix"] = self.resolve_prefix(req["strata_prefix"], messages, tools, kwargs, ids, had_images)
            if self.inkling and not inkling_shared_ok:
                if req["strata_prefix"]:
                    print("[strata] strata_prefix ignored: this Inkling engine does not advertise shared_prefix=1", flush=True)
                req["strata_prefix"] = {}
            elif auto_prefix:
                # Preserve uncontended cold-prefill batch shapes: an arbitrary split changes CUDA arithmetic.
                try:
                    chunk = int((getattr(self.engine, "info", None) or {}).get("chunk", 0))
                except (TypeError, ValueError):
                    chunk = 0
                n = req["strata_prefix"].get("tokens", 0)
                n = (n // chunk) * chunk if chunk > 0 else 0
                req["strata_prefix"] = {"tokens": n} if n > 0 else {}
        if self.inkling:                       # the thinking level decides (a forced call skips the thinking)
            word, level = inkling_dialect.effort_of(kwargs)
            if isinstance(req, dict):
                req["strata_effort"] = word      # (the default budget of this level: reasoning_budget)
            return ids, level > 0.0 and not force, max_new
        return ids, kwargs.get("enable_thinking", True) is not False, max_new

    PREFIX_END = "\u0001strata-prefix-end\u0001"          # marks where a message's shared part ends in the rendered prompt

    def resolve_prefix(self, spec, messages, tools, kwargs, ids, has_images=False) -> dict:
        """R1: `"strata_prefix"` marks the start of the prompt as a shared read-only prefix: a long document that many
        short questions follow.  Three ways to say where it ends: `{"messages": K}` (the first K messages), `{"message":
        I, "chars": N}` (the first N characters of message I's text: a document and its question in one message) or
        `{"tokens": N}`.  The engine reads the prompt in two parts there, keeps the checkpoint at the boundary pinned
        (never evicted) and does not park the branch it leaves, so N questions cost one prefix.  Returns
        {"tokens": N} (what the engine's `pin=N` takes: always a prefix of the prompt's ids, the tokenizer's merge across
        the boundary included), or {} when the prefix cannot be marked (said in the log: the request is never refused for
        it).  A malformed field is a 400."""
        keys = ("messages", "message", "chars", "tokens")
        if (not isinstance(spec, dict) or not spec or any(k not in keys for k in spec) or
                set(spec) not in ({"messages"}, {"tokens"}, {"message", "chars"})):
            raise ValueError('strata_prefix: {"messages": K} (the first K messages), {"message": I, "chars": N} (the first '
                             'N characters of message I) or {"tokens": N}')
        for k, v in spec.items():
            if not isinstance(v, int) or isinstance(v, bool) or v < (0 if k == "message" else 1):
                raise ValueError(f"strata_prefix.{k}: a whole number ({'0' if k == 'message' else '1'} or more)")
        if has_images:
            print("[strata] strata_prefix ignored: the request has images", flush=True)
            return {}
        what = ",".join(f"{k}={v}" for k, v in spec.items())

        def lcp(a, b):
            n = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                n += 1
            return n

        def head_ids(msgs):
            """The ids the rendered `msgs` start with (a message's trailing marker cut at PREFIX_END when it is there)."""
            marked, marked_tools, changed = mark_think_literals(msgs, tools, self.literals)
            effort = kwargs.get("reasoning_effort")
            off = kwargs.get("enable_thinking") is False        # render_prompt's rule for #458's effort_end
            kw = ({k: v for k, v in kwargs.items() if k not in ("reasoning_effort", "enable_thinking")}
                  if self.effort_end and (off or effort not in (None, "", "xhigh", "high")) else dict(kwargs))
            kw["add_generation_prompt"] = False
            if self.inkling:
                marked = inkling_dialect.prepare_messages(marked)
                kw = inkling_dialect.template_kwargs(kw)
            text = self.template.render(marked, tools=marked_tools, **kw)
            cut = text.find(self.PREFIX_END)
            if cut >= 0:
                text = text[:cut]
            if changed:
                text, plain = unmark_think_literals(text, self.literals)
                return self.tok.encode(text, parse_special=True, plain=plain)
            return self.tok.encode(text, parse_special=True)

        if "tokens" in spec:
            n = spec["tokens"]
        elif "messages" in spec:
            k = spec["messages"]
            if k >= len(messages):
                print(f"[strata] strata_prefix.messages={k} ignored: the request has {len(messages)} message(s) and the "
                      "prefix must leave at least the last one", flush=True)
                return {}
            head = head_ids(messages[:k])
            n = lcp(head, ids)
        else:
            i, chars = spec["message"], spec["chars"]
            text = messages[i].get("content") if 0 <= i < len(messages) else None
            if not isinstance(text, str) or chars > len(text):
                print(f"[strata] strata_prefix ignored: message {i} has no text of {chars} characters", flush=True)
                return {}
            cut = messages[:i] + [{**messages[i], "content": text[:chars] + self.PREFIX_END}]
            head = head_ids(cut)
            n = lcp(head, ids)
        if n < 1 or n >= len(ids) - 1:
            print(f"[strata] strata_prefix {what} ignored: it leaves no usable prefix of the {len(ids)}-token prompt "
                  f"(common start {n})", flush=True)
            return {}
        return {"tokens": n}

    def drop_embeddings(self) -> None:
        """Delete the combined image file prepare() wrote when no run() took it over (a run deletes its own as it
        ends): a request can end before its answer starts - its client gone, anything raised after prepare() - and
        then nothing else would.  The handler calls this when a request is done."""
        path, self.embeddings.path = getattr(self.embeddings, "path", None), None
        if path is not None:
            Path(path).unlink(missing_ok=True)
        self.embeddings.hy4 = self._drop_hy4_images(getattr(self.embeddings, "hy4", None))

    @staticmethod
    def _drop_hy4_images(paths):
        """Delete the picture files of a Hy4-Flash request (the engine has read them by the end of the request)."""
        for p in paths or []:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
        return None

    def _note(self, n, evs, st=None, rate=None):
        with self.status_lock:
            s = self.status if st is None else st
            s["generated"] = n
            if s.get("first_token") is None:
                s["first_token"] = time.time()
            (self.rate if rate is None else rate).append((time.time(), n))   # the live rate's window (RATE_WINDOW_S)
            for ev in evs:
                if ev.kind == "reasoning":
                    s["phase"] = "thinking"
                elif ev.kind == "content":
                    s["phase"] = "answering"
                elif ev.kind == "tool_start":
                    s["phase"], s["tool"] = f"writing a tool call: {ev.call.name}", ev.call.name
                elif ev.kind == "tool_call":
                    s["phase"] = "tool call complete"
                s["tail"] = ((s.get("tail") or "") + (ev.text or ""))[-600:]

    def _progress(self, last_print, every=1.0, st=None):
        """A progress line in the server window every `every` seconds while a request runs."""
        now = time.time()
        if now - last_print < every:
            return last_print
        with self.status_lock:
            s = dict(self.status if st is None else st)
        el = now - s.get("started", now)
        if s.get("first_token") is None:
            pr = getattr(self.engine, "progress", None)   # (position reached, prompt tokens): a reused prefix counts
            done = f"{pr[0]:,} of {pr[1]:,}" if pr and pr[1] else f"{s.get('prompt_tokens', 0):,}"   # as read (#29)
            print(f"[strata] reading the prompt: {done} tokens, {el:.0f} s so far", flush=True)
        else:
            rate = s["generated"] / max(RATE_MIN_SPAN_S, now - s["first_token"])
            # #984: while a thinking budget applies, the thinking line shows it, not the request's max_tokens
            limit = (f"budget {s['thinking_budget']} (max {s.get('max_tokens')})"
                     if s["phase"] == "thinking" and s.get("thinking_budget") else f"max {s.get('max_tokens')}")
            print(f"[strata] {s['phase']}: {s['generated']} of {limit} tokens, {rate:.1f} tok/s, "
                  f"{el:.0f} s", flush=True)
        return now

    def run(self, ids, thinking, tools, max_new, sampling, cancel, force=None,
            declared_only=False, deferred=False) -> Iterator[tuple[str, object]]:
        """The request's events (see _run); with "prevent_sleep" in the config, the PC stays awake while it runs or waits."""
        ka = self.keep_awake
        if ka is not None:
            ka.acquire()
        gate = self.key_limits
        counted = False
        try:
            if gate is not None:
                # #5004 (the config's "api_keys"): hold this request within the limits of the key it arrived with
                # before the engine queue.  The wait is the queue's own kind of wait: queue_timeout bounds it (the
                # same 429 when it expires) and a client that hangs up ends it (ClientGone, like a prepared request
                # whose client left).  The cancel Event carries the key's identity, stamped when it was watched.
                gate.acquire(getattr(cancel, "_strata_key", None), cancel=cancel,
                             deadline=None if self.queue_timeout_s is None else time.monotonic() + self.queue_timeout_s)
                counted = True
            yield from self._run(ids, thinking, tools, max_new, sampling, cancel, force, declared_only, deferred)
        finally:
            if counted:
                gate.release(getattr(cancel, "_strata_key", None))
            if ka is not None:
                ka.release()

    def _run(self, ids, thinking, tools, max_new, sampling, cancel, force=None,
             declared_only=False, deferred=False) -> Iterator[tuple[str, object]]:
        """Yields ("event", Event) as text arrives, then ("done", {"finish": .., "completion_tokens": ..}).
        `force` (forced_call): the opening of the call the reply must make - see prepare()."""
        # the model card's sampling for this thinking state, then the config's block, then the Chat settings shared with
        # apps; the request's own fields win over all of them
        # #3354: when MTP drafting is active (has_mtp) and the request does not explicitly ask for sampling,
        # omitted sampling parameters remain greedy so MTP drafts are not silently disabled by model card defaults.
        mtp_greedy = self.has_mtp and not request_asks_sampling(sampling, self.shared, self.sampling_defaults)
        card = {} if mtp_greedy else model_card_sampling(
            self.model_arch, thinking,
            greedy=asks_greedy(sampling, self.shared, self.sampling_defaults),
        )
        defaults = {**card, **self.sampling_defaults, **self.shared}
        if defaults:                   # the request's own fields win (explicit 0 stays greedy)
            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}
            sampling = {**defaults, **req_values}
        # #123: read after the merge, so a budget shared through POST /settings is seen like the other keys
        # #6734: Inkling-2-Mini writes hidden thinking whatever the request asks for (even with its effort at "none"), so its budget
        # (the request's or the answer room's) applies to every request: the parser reports the hidden block (`reasoning_now`)
        think_budget = thinking or self.inkling
        budget = self.reasoning_budget(sampling) if think_budget else None
        room_cap = INKLING_ROOM_MIN_CAP if self.inkling else ANSWER_ROOM_MIN_CAP     # Inkling: whatever max_tokens, the answer keeps its share
        if think_budget and not budget and self.answer_room and max_new and max_new >= room_cap:
            # 0.1.45: max_tokens counts the thinking too.  A client that caps the reply at 8192 (the desktop agent's
            # harness sends max_completion_tokens: 8192 on every request) and a task the model thinks long about got
            # the cut-off in the middle of the thinking and NO answer.  So thinking is closed early enough to leave
            # the answer its quarter of the cap (the budget's wrap-up: a short closing sentence, </think>, the answer).
            # A request with a budget of its own, or max_tokens below ANSWER_ROOM_MIN_CAP, is left as it was.
            budget = max_new - min(max(ANSWER_RESERVE_MIN, max_new // 4), max_new // 2)
            if max_new <= 32768 and thinking:
                print(f"[strata] thinking limited to {budget} of max_tokens {max_new} so the answer has room "
                      "(\"answer_room\": false in the config turns this off)", flush=True)
        if budget and max_new and max_new > 0:
            # #984: a budget at or near max_tokens never fires before the reply is cut off while still thinking, so
            # it has no answer.  Close the thinking early enough to leave room for one.
            reserve = min(max(ANSWER_RESERVE_MIN, max_new // 4), max_new // 2)
            if budget > max_new - reserve:
                if not self.inkling:                  # (Inkling: the default budget meets a small max_tokens all the time; the room is the rule)
                    print(f"[strata] thinking capped at {max_new - reserve} of max_tokens {max_new} to leave room for "
                          f"the answer (budget {budget})", flush=True)
                budget = max(1, max_new - reserve)
        parser = self.parser_cls(thinking=thinking, tools=with_history_tools(tools, sampling), stream_tools=True,
                                 recover=self.tool_call_recovery, declared_only=declared_only,
                                 deferred=deferred or self.undeclared_tool_calls == "call")   # #4622: opt-in, as for a deferred tool
        inkling_special_ids = None
        if self.inkling and hasattr(parser, "feed_token"):
            types = getattr(self.tok, "token_types", None)
            toks = getattr(self.tok, "tokens", None)
            if types and toks:
                inkling_special_ids = {
                    i for i, ty in enumerate(types)
                    if ty == 3 and i < len(toks) and toks[i] in inkling_dialect.ALL_CHANNEL_TAGS
                }
        detok, n, finish = Detokenizer(self.tok), 0, "length"
        run_tok, run_len, repeated = None, 0, False     # #606: the current run of one repeated token
        thinking_n = 0                                  # tokens written while thinking (Responses' reasoning_tokens)
        stop_list = stop_strings(sampling)
        stops = StopMatcher(stop_list) if stop_list else None

        # the names the parser takes as declared: the request's tools and those its history called (#4663)
        declared = [t.get("name") for t in with_history_tools(tools, sampling) or []
                    if isinstance(t, dict) and isinstance(t.get("name"), str)]
        # (a): armed for a request with tools, on the template's own call form (the Qwen form; a JSON-form parser or another
        # family keeps sending the text as it is)
        hold = ToolMarkupHold(bool(self.tool_markup_retry and declared and self.family is None
                                   and self.parser_cls is OutputParser))
        parser_done, markup_retried = False, None

        def cut(evs):
            """The events with the answer's text cut at a stop string (OpenAI stop, Anthropic stop_sequences), then
            through the tool-markup hold (a)."""
            return hold.push(_cut(evs))

        def _cut(evs):
            if stops is None:
                return evs
            out = []
            for ev in evs:
                if stops.hit is not None:
                    break                               # nothing after the stop string is sent
                if ev.kind == "content":
                    text = stops.push(ev.text or "")
                    if text:
                        out.append(Event("content", text))
                    continue
                held = stops.flush()                    # a stop string does not run across a tool call or thinking
                if held:
                    out.append(Event("content", held))
                out.append(ev)
            return out
        forced_text = force                             # (a): the retry opens a forced call again (tool_choice)
        opening = []                                    # without thinking the prompt ends with the forced opening:
        if force and not thinking:                      # the parser reads it as if the model had written it
            opening, force = parser.feed(force), None
        tail = ""                                       # the last characters written (the newlines before a call)
        answered, close_retried = False, False          # #1053: content or a call came out; the thinking was closed
        nf_retried = False                              # #879: the request was sent again after non-finite logits
        timings, before = None, None                    # this request's timings; the engine's `last` before it
        recovery_count, reasoning_text, repeat_coverage = 0, "", 0.0
        looped, next_loop_check = False, LOOP_CHECK_EVERY       # #728: reasoning that repeats whole passages
        raw_ids = []                                    # every generated id (STRATA_DEBUG: dump raw model text)
        admission_rejected = False                      # #3040: the engine's batch slots refused it before it read anything
        emb, self.embeddings.path = getattr(self.embeddings, "path", None), None   # this run's to delete now
        img_keys, self.embeddings.keys = getattr(self.embeddings, "keys", None), None
        if emb and img_keys:
            sampling = {**(sampling or {}), "strata_img_keys": img_keys}   # #2228: sent beside the embeddings file
        hy4_images = getattr(self.embeddings, "hy4", None)                 # Hy4-Flash: the picture files of this request (kept until drop_embeddings)
        if hy4_images:
            sampling = {**(sampling or {}), "strata_hy4_images": list(hy4_images)}
        # Only a DONE line replaces engine.last, so a request that died, errored or was disconnected must not have
        # the PREVIOUS request's decode figures recorded as its own: each pass keeps its own DONE (a segment) if
        # one arrived.  A reasoning-budget continuation is another native generation, not another API request,
        # so the generated reasoning must not become this request's cached input (#615).
        segments = []
        image_prompt_tokens = 0                         # Inkling pictures: the prompt the engine read, with the pictures' tokens
        timed_out = False
        trace = getattr(self.request_trace, "record", None)
        waiting = time.perf_counter()
        first_engine_token_s = None  # this request, from entering the engine queue; includes queue/prefill
        # #465: with "parallel" (the engine's batch slots) requests run at once: each keeps its own status and rate
        # window (self.status says busy while any runs); one at a time they are self.status / self.rate, as before
        par = bool(getattr(self.engine, "batch", 0))
        st = {} if par else self.status
        rate = collections.deque(maxlen=32) if par else self.rate
        request_id = uuid.uuid4().hex[:12]
        setattr(cancel, "_strata_request_id", request_id)
        deadline = time.monotonic() + self.queue_timeout_s if self.queue_timeout_s is not None else None
        if deadline is not None:
            setattr(cancel, "_strata_queue_deadline", deadline)
        def queue_started():
            with self.status_lock:
                row = self.terminal_requests.get(request_id)
                if row is not None and row["state"] != "reading":
                    row["state"] = "reading"
                    self._terminal_event_locked(request_id, "reading")
        setattr(cancel, "_strata_queue_started", queue_started)
        def replica_selected(index: int) -> None:
            with self.status_lock:
                row = self.terminal_requests.get(request_id)
                if row is not None:
                    row["replica"] = index
        setattr(cancel, "_strata_replica_selected", replica_selected)
        with self.status_lock:
            self.status["queued"] += 1
            self.terminal_requests[request_id] = {"id": request_id, "state": "queued",
                                                    "queued_at": time.time(), "admitted_at": None,
                                                    "prompt_tokens": len(ids)}
            self._terminal_event_locked(request_id, "queued")
        queued_counted = True
        try:
            # --batch: the engine runs several requests at once (StrataEngine.generate_batched orders them)
            turn = contextlib.ExitStack()
            if not par:
                # #1619: a request waiting for the single turn sends a keep-alive every QUEUE_BEAT_S, so a client with
                # a stream idle timeout does not abort a healthy queued request (and its retry read the prompt again);
                # a client that left while queued gives its place up instead of taking the turn only to cancel there
                got = False
                while not got:
                    if cancel.is_set():
                        raise QueueCancelled("the client disconnected while waiting for the engine")
                    left = deadline - time.monotonic() if deadline is not None else QUEUE_BEAT_S
                    if left <= 0:
                        raise QueueWaitExpired("the inference queue wait deadline expired; retry later")
                    got = self.fifo.acquire(timeout=min(QUEUE_BEAT_S, left))
                    if got:
                        if cancel.is_set():
                            self.fifo.release()
                            raise QueueCancelled("the client disconnected while waiting for the engine")
                        if deadline is not None and time.monotonic() >= deadline:
                            self.fifo.release()
                            raise QueueWaitExpired("the inference queue wait deadline expired; retry later")
                        turn.callback(self.fifo.release)
                    else:
                        yield "ping", None
            with turn:
                try:
                    with self.status_lock:
                        if trace is not None:
                            trace["queue_s"] += round(time.perf_counter() - waiting, 3)
                            trace["state"] = "generating"
                        if not par:
                            self.status["queued"] -= 1
                            queued_counted = False
                            self.terminal_requests[request_id].update(state="admitted", admitted_at=time.time())
                            self._terminal_event_locked(request_id, "admitted")
                    # issue #27: it died in an earlier request (or was unloaded) - start it again instead of failing
                    # (parallel requests do not hold the fifo: one of them starts it, the others wait for that).
                    # #1603: a parallel request stays counted as queued until it is registered as running below -
                    # waiting here for the fifo (an image encode, a /v1/vram resize) left it neither - and one on a
                    # loaded engine has nothing to start, so it does not wait for the fifo at all
                    if not par:
                        self.ensure_loaded()
                    elif not (self.loaded() and not self._vision_down()):
                        with self._inference_fifo(deadline, cancel):
                            self.ensure_loaded()
                    with self.status_lock:
                        if par:
                            self.status["queued"] -= 1
                            queued_counted = False
                            self.terminal_requests[request_id].update(state="waiting", admitted_at=time.time())
                            self._terminal_event_locked(request_id, "admitted")
                        st.update(busy=True, phase="reading the prompt", prompt_tokens=len(ids),
                                  generated=0, started=time.time(), first_token=None, tool=None, tail="",
                                  max_tokens=max_new, reasoning_recoveries=0, thinking_budget=budget)
                        if par:
                            self.live_reqs[id(st)] = (st, rate)
                            self.status.update(st)
                        self.last_request_at = time.time()
                        rate.clear()                    # the previous request's samples must not leak into this one
                    before = getattr(self.engine, "last", None)
                    last_print = time.time()
                    prompt, thought = ids, 0            # thought: the reasoning tokens so far (the budget's count)
                    # the engine stops at the budget itself (HOLD_KEY), when it can and </think> is one token: a --batch engine
                    # only when its slots say they hold too (READY holdb)
                    hold_end = None
                    if (budget and getattr(self.engine, "can_hold", False) is True
                            and (not getattr(self.engine, "batch", 0) or getattr(self.engine, "can_hold_batch", False) is True)
                            and getattr(self.engine, "split_engine", False) is not True
                            and getattr(self.engine, "hy4_engine", False) is not True):   # (the split & Hy4 parsers refuse hold=)
                        te = self.tok.encode(THINK_END, parse_special=True)
                        hold_end = te[0] if len(te) == 1 else None
                    for ev in opening:
                        yield "event", ev
                    pass_n = 0      # (the passes of this request sent so far)
                    while True:
                        segment_before = getattr(self.engine, "last", None)
                        segments_before = len(segments)     # (#879: a pass sent again leaves no segment)
                        held = hold_end is not None and budget is not None
                        if held:   # the reasoning tokens left before the budget (at least one: a cut not yet clean)
                            pass_sampling = {k: v for k, v in (sampling or {}).items() if k != HOLD_KEY}   # (a copy: the hold is this pass's own)
                            pass_sampling[HOLD_KEY] = (max(1, int(budget) - thought), int(hold_end))
                        elif sampling and HOLD_KEY in sampling:   # (never a client's: strip it from a copy)
                            pass_sampling = {k: v for k, v in sampling.items() if k != HOLD_KEY}
                        else:
                            pass_sampling = sampling   # no hold: the engine gets the caller's own sampler object, as before P2
                        if (pass_n or os.environ.get("STRATA_INKLING_PROMPT_CKPT") == "0") and self.inkling and str((getattr(self.engine, "info", None) or {}).get("ckpt_key")) == "1":
                            # a later pass of this request (the thinking budget's wrap-up, a reply written again): the engine
                            # keeps the checkpoint the first pass saved at the end of the prompt the client sends back next
                            # (STRATA_INKLING_PROMPT_CKPT=0: no pass saves one: for an exactness A/B against an engine that has none)
                            pass_sampling = {**(pass_sampling or {}), KEEP_CKPT_KEY: True}
                        pass_n += 1
                        gen = self.engine.generate(prompt, max_new - n, pass_sampling, cancel, embeddings=emb) if emb \
                            else self.engine.generate(prompt, max_new - n, pass_sampling, cancel)
                        recover_prompt = None
                        seg, wrap, leaving = [], False, False   # this pass's tokens; the budget is reached; closed
                        nonfinite = False               # #879: the engine ended this pass on non-finite logits
                        opens = False                   # the thinking is over: write the forced call's opening
                        try:
                            # A value the frontend holds back (a tool call's array or object argument is sent
                            # whole, once complete) can keep the stream silent for minutes while tokens are
                            # generated; a client's idle timeout then ends the request.  Send the same keep-alive
                            # the engine's heartbeat sends when nothing has gone out for KEEPALIVE_S.
                            last_out = time.perf_counter()
                            for t in gen:
                                if t is None:               # heartbeat while the engine is quiet
                                    last_print = self._progress(last_print, st=st)
                                    last_out = time.perf_counter()
                                    yield "ping", None
                                    continue
                                n += 1
                                if first_engine_token_s is None:
                                    first_engine_token_s = round(time.perf_counter() - waiting, 3)
                                    with self.status_lock:
                                        row = self.terminal_requests.get(request_id)
                                        if row is not None:
                                            row["first_engine_token_s"] = first_engine_token_s
                                            self._terminal_event_locked(request_id, "first_token")
                                if trace is not None and trace["first_token_s"] is None:
                                    with self.status_lock:
                                        trace["first_token_s"] = round(time.perf_counter() - trace["_clock"], 3)
                                if t in self.stop_ids:
                                    if force and parser.state == "reasoning" and parser.buf == THINK_END:
                                        opens = True    # #537: the held </think> was the end: the call opens there
                                        break
                                    finish = "stop"
                                    raw_ids.append(t)
                                    break
                                raw_ids.append(t)
                                seg.append(t)
                                thinking_n += (parser.state in ("reasoning", "rcall") or getattr(parser, "reasoning_now", False))
                                run_len = run_len + 1 if t == run_tok else 1
                                run_tok = t
                                if self.repeat_stop_tokens and run_len >= self.repeat_stop_tokens:
                                    repeated = True     # #606: a degenerate output, not an answer: end it here
                                    break
                                piece = detok.push(t)
                                tail = (tail + piece)[-2:]
                                if inkling_special_ids is not None:
                                    evs = cut(parser.feed_token(piece, is_special=(t in inkling_special_ids)))
                                else:
                                    evs = cut(parser.feed(piece))
                                self._note(n, evs, st, rate)
                                last_print = self._progress(last_print, st=st)
                                for ev in evs:
                                    if self.reasoning_loop_recovery and ev.kind == "reasoning":
                                        reasoning_text += ev.text or ""
                                    if ev.kind in ("content", "tool_start", "tool_call"):
                                        answered = True
                                    yield "event", ev
                                if evs:
                                    last_out = time.perf_counter()
                                elif time.perf_counter() - last_out >= KEEPALIVE_S:
                                    last_out = time.perf_counter()
                                    yield "ping", None
                                if stops is not None and stops.hit is not None:
                                    finish = "stop"         # gen.close() below STOPs the engine, as for a stop token
                                    break
                                if (self.reasoning_loop_recovery and not recovery_count and parser.state == "reasoning"
                                        and n >= next_loop_check and not parser.buf and not parser.pending
                                        and not detok.pending()):
                                    next_loop_check = n + LOOP_CHECK_EVERY
                                    repeat_coverage = reasoning_repeat_coverage(reasoning_text)
                                    if repeat_coverage >= LOOP_COVERAGE:
                                        if self.reasoning_loop_recovery == "stop":
                                            looped = True        # #728: end the reply here, as for a repeated token
                                            break
                                        if not emb:              # "recover": not for a request with pictures
                                            recover_prompt = focused_recovery_prompt(self.tok, ids, raw_ids)
                                            if recover_prompt is not None:
                                                break
                                if budget and (parser.state == "reasoning" or getattr(parser, "reasoning_now", False)):
                                    thought += 1
                                    # at a clean point: no tag held back, no character split across tokens
                                    if thought >= budget and parser.clean and not detok.pending():
                                        wrap = True
                                        break
                                if force:
                                    if any(ev.kind in ("tool_start", "tool_call") for ev in evs):
                                        force = None    # the model is writing a call of its own
                                    elif parser.state == "content" and parser.clean and not detok.pending():
                                        opens = True
                                        break
                        except EngineNonFinite:
                            nonfinite = True
                        except EngineDied as e:
                            finish = "error"
                            self._say_died(e)
                            raise
                        except QueueWaitExpired:
                            finish = "error"
                            timed_out = True
                            raise
                        except ValueError as e:             # the engine's ERR line (it may have ended after it)
                            finish = "error"
                            print(f"[strata] the engine reported an error: {e}", flush=True)
                            raise
                        except GeneratorExit:               # the client went away: an engine that never acknowledges
                            leaving = True                  # the STOP below is ended, but no error replaces this
                            raise
                        finally:
                            try:
                                gen.close()             # STOP+drain to THIS request's DONE while still holding the
                                #                         fifo, so a stop-token break can't leave the shared engine
                                #                         queue mid-drain for the next request to read as its own DONE
                            except EngineSilent as e:   # #481: the STOP was never acknowledged: the engine is ended
                                finish = "error"
                                self._say_died(e)
                                if not leaving and not cancel.is_set():
                                    raise
                            segment_done = getattr(self.engine, "last", None)
                            if segment_done is not None and segment_done is not segment_before:
                                segments.append(dict(segment_done))
                        if nonfinite and not cancel.is_set():
                            # #879: the engine emitted none of the poisoned window's tokens and reads its next prompt
                            # from token 0.  Nothing sent yet: the same request once more, unseen by the client
                            if n == 0 and not nf_retried:
                                nf_retried = True
                                # the retry reads from token 0: its DONE, not this one's, says what was reused
                                del segments[segments_before:]
                                print("[strata] the engine's logits became non-finite before the first token (#879); it "
                                      "dropped its cached state: sending the request again once", flush=True)
                                continue
                            if n == 0:
                                finish = "error"
                                print("[strata] the engine's logits became non-finite again on the retry (#879): the "
                                      "request ends with an error", flush=True)
                                raise EngineNonFinite("the model's output became non-finite (NaN) twice in a row "
                                                      "(#879); the engine has dropped its cached state - send the "
                                                      "request again")
                            print(f"[strata] the engine's logits became non-finite after {n} tokens (#879): the reply "
                                  "ends here as \"length\"; the engine dropped its cached state, so the next request "
                                  "reads its prompt from the start", flush=True)
                            finish = "length"
                            break
                        if recover_prompt is not None and not cancel.is_set() and n < max_new:
                            recovery_count += 1
                            # Only the two settings that keep the same words coming are raised (temperature to at
                            # least 1.0, presence penalty to at least 1.5): a client's top_p and top_k are never
                            # touched, and its own request dict is copied, not changed.
                            want = (max(1.0, float((sampling or {}).get("temperature") or 0)),
                                    max(1.5, float((sampling or {}).get("presence_penalty") or 0)))
                            if want != tuple(float((sampling or {}).get(k) or 0)
                                             for k in ("temperature", "presence_penalty")):
                                sampling = {**(sampling or {}), "temperature": want[0], "presence_penalty": want[1]}
                            prompt = recover_prompt
                            with self.status_lock:
                                st["reasoning_recoveries"] = recovery_count
                            if trace is not None:
                                trace["reasoning_recoveries"] = recovery_count
                                trace["reasoning_repeat_coverage"] = round(repeat_coverage, 3)
                            print(f"[strata] repeated reasoning detected at {n} tokens "
                                  f"(coverage={repeat_coverage:.3f}); resuming the same output with the low-effort "
                                  "instruction (reasoning_loop_recovery)", flush=True)
                            continue
                        if (self.inkling and finish == "stop" and not close_retried and not answered and not wrap and not opens
                                and not repeated and not looped and (stops is None or stops.hit is None) and not cancel.is_set()
                                and max_new - n > 8):
                            # #6612 / #6620: the reply ended (its stop token) with no text and no call: the model wrote its hidden thinking and
                            # stopped, or stopped at once.  Never hand the client an empty reply: close the thinking if it is open, open the
                            # model's text turn, and let it answer, once.
                            close_retried = True
                            in_think = getattr(parser, "reasoning_now", False) and parser._dialect.state == "think"
                            extra = self.tok.encode(("<|end_message|>" if in_think else "") + "<|message_model|><|content_text|>", parse_special=True)
                            if max_new - n - len(extra) >= 1:
                                print("[strata] the reply ended with no answer" + (" inside its thinking" if in_think else "")
                                      + ": opening the answer and continuing once", flush=True)
                                for t in extra:
                                    n += 1
                                    raw_ids.append(t)
                                    thinking_n += (parser.state in ("reasoning", "rcall") or getattr(parser, "reasoning_now", False))
                                    evs = cut(parser.feed(detok.push(t)))
                                    self._note(n, evs, st, rate)
                                    for ev in evs:
                                        yield "event", ev
                                prompt = prompt + seg + extra
                                finish = "length"
                                continue
                        if (self.reasoning_close_retry and thinking and finish == "stop" and not close_retried
                                and not answered and not wrap and not opens and parser.state == "reasoning"
                                and parser.buf != THINK_END         # #537: not after a held </think>, the end
                                and (stops is None or stops.hit is None) and not cancel.is_set()):
                            # #1053: the model wrote its reasoning and stopped before </think>: the client would get
                            # an empty answer.  Close the thinking once and let it answer.
                            close_retried = True
                            extra = self.tok.encode(REASONING_CLOSE, parse_special=True)
                            if max_new - n - len(extra) >= 1:
                                print("[strata] the reply ended inside its thinking with no answer: closing the "
                                      "thinking once and continuing (reasoning_close_retry)", flush=True)
                                for t in extra:
                                    n += 1
                                    raw_ids.append(t)
                                    thinking_n += parser.state in ("reasoning", "rcall")
                                    evs = cut(parser.feed(detok.push(t)))
                                    self._note(n, evs, st, rate)
                                    for ev in evs:
                                        yield "event", ev
                                prompt = prompt + seg + extra
                                finish = "length"
                                continue
                        if (held and not (wrap or opens) and not cancel.is_set() and finish != "stop" and not repeated
                                and not looped and n < max_new and segments and segments[-1].get("finish") == "hold"
                                and len(segments) > segments_before):
                            # the engine stopped at the hold but the cut is not clean yet (a character split across
                            # tokens, a tag held back) or the reasoning count is short: on from the live session
                            prompt = prompt + seg
                            continue
                        if (hold.armed and not hold.calls and finish == "stop" and not (wrap or opens)
                                and not cancel.is_set() and not repeated and not looped and parser.state != "reasoning"
                                and (stops is None or (stops.hit is None and not stops.held))):
                            # (a): the model stopped (a request with tools).  Read the rest of its output here, as the
                            # end would: a call that is whole there goes out as one; when the reply ends in tool-call
                            # markup that is no call (held back, not sent), it is written again once
                            for ev in cut(parser.finish("stop")):
                                yield "event", ev
                            parser_done = True
                            dead = dead_markup_tail(hold.text) if hold.held and not hold.calls else None
                            if dead is not None:
                                names = [x for x in markup_names(hold.text) if x not in declared]
                                kind = "rejected" if names else dead[1]
                                near = closest_declared(names[0], declared) if names else None
                                note = tool_retry_note(kind, names[:3], declared, near)
                                extra = self.tok.encode(IM_END + "\n<|im_start|>user\n" + note + IM_END
                                                        + "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                                                        + (forced_text or ""), parse_special=True)
                                if max_new - n - len(extra) >= TOOL_RETRY_MIN_ROOM:
                                    dropped = hold.drop()
                                    hold.armed = False          # one retry: its own reply goes out as it is written
                                    markup_retried = {"reason": kind, "chars": len(dropped), "names": names[:3]}
                                    with self.status_lock:
                                        t = self.totals
                                        t["tool_markup_retries"] = t.get("tool_markup_retries", 0) + 1
                                        t["tool_calls_unread"] = t.get("tool_calls_unread", 0) + parser.unread
                                    print(f"[strata] the reply ended in tool-call markup that is no call ({kind}"
                                          + (f": {', '.join(names[:3])}" if names else "") + (f", closest declared "
                                          f"{near}" if near else "") + f"; {len(dropped)} characters held back): writing "
                                          "the reply's end again once (STRATA_TOOL_MARKUP_RETRY=0 turns this off)",
                                          flush=True)
                                    max_new -= len(extra)       # the note is prompt: the reply keeps inside the context
                                    prompt = prompt + seg + extra
                                    parser = self.parser_cls(thinking=False, tools=with_history_tools(tools, sampling),
                                                             stream_tools=True, recover=self.tool_call_recovery,
                                                             declared_only=declared_only,
                                                             deferred=deferred or self.undeclared_tool_calls == "call")
                                    detok, tail, parser_done, force, budget = Detokenizer(self.tok), "", False, None, None
                                    if forced_text:     # a forced call (tool_choice): its opening again, read as
                                        for ev in cut(parser.feed(forced_text)):     # if the model had written it
                                            yield "event", ev
                                    finish = "length"
                                    continue
                            break
                        if not (wrap or opens) or cancel.is_set():
                            break
                        # #123: the thinking reached reasoning_budget_tokens.  Close it the way the model would (a
                        # short wrap-up and </think>) and let it answer: the next pass's prompt is this one plus what
                        # was generated plus the wrap-up, so the engine continues from the prefix it already holds.
                        # A forced call (tool_choice) is opened the same way: after the wrap-up, or where the
                        # thinking ended, after the blank line the template puts before a call.
                        if wrap:
                            budget = st["thinking_budget"] = None
                            text = (getattr(parser, "wrap_up", None) or REASONING_WRAP_UP) + (force or "")
                        else:
                            text = "\n" * (2 - (len(tail) - len(tail.rstrip("\n")))) + force
                        force = None
                        extra = self.tok.encode(text, parse_special=True)
                        short = getattr(parser, "wrap_up_short", None)
                        if wrap and short and max_new - n - len(extra) < max(8, (max_new - n) // 2):
                            extra = self.tok.encode(short + (force or ""), parse_special=True)   # (Inkling: a bare <|end_message|> when room is short)
                        if max_new - n - len(extra) < 1:
                            break                       # no room left to answer: "length", as without a budget
                        if wrap:
                            print(f"[strata] thinking budget reached ({thought} tokens): wrapping up the thinking",
                                  flush=True)
                        for t in extra:
                            n += 1
                            raw_ids.append(t)
                            thinking_n += (parser.state in ("reasoning", "rcall") or getattr(parser, "reasoning_now", False))
                            evs = cut(parser.feed(detok.push(t)))
                            self._note(n, evs, st, rate)
                            for ev in evs:
                                yield "event", ev
                        if stops is not None and stops.hit is not None:
                            finish = "stop"
                            break
                        prompt = prompt + seg + extra
                    if cancel.is_set():
                        finish = "cancel"
                    elif looped:
                        print(f"[strata] the thinking repeated the same passages (coverage={repeat_coverage:.3f}) at "
                              f"{n} tokens: ended as \"length\" (reasoning_loop_recovery: \"stop\" in "
                              "strata-<model>.json; remove it to turn this off)", flush=True)
                    elif repeated:
                        print(f"[strata] the reply repeated one token ({self.tok.decode([run_tok])!r}) "
                              f"{run_len} times in a row: ended as \"length\" (repeat_stop_tokens in "
                              "strata-<model>.json; 0 turns this off). If a new request with a short prompt does the "
                              "same, restart the server and report it (#606)", flush=True)
                except GpuBusy:
                    # #3040 (PR #5044's approach): a batch engine can use up its BGEN retries before it has admitted this
                    # request.  The monitor keeps an error row, but it is not a completed generation: no prompt tokens
                    # and no part in the totals or latencies.  A continuation after native work keeps the old accounting.
                    if not segments and n == 0:
                        admission_rejected = True
                        finish = "error"
                    raise
                except GeneratorExit:                   # the client disconnected mid-stream
                    finish = "disconnect"
                    raise
                finally:
                    # #266: settle this request's status, history and totals while still holding the fifo: once
                    # it is released the next request sets its own status, which this must not record or clear
                    with self.status_lock:
                        if st.get("busy") and not timed_out:
                            last = request_stats(segments)          # only this request's own DONEs count
                            started = st.get("started", time.time())
                            cvec = (getattr(self.engine, "info", {}) or {}).get("cvec", 0)
                            loaded = str(cvec) not in ("0", "", "None")
                            hit_rate = round(last["hits"] / last["lookups"], 3) if last.get("lookups") else None
                            # #588: the share the GPU read over PCIe (--pcie-frac) or another GPU computed - not in
                            # the hit rate's lookups, which is the VRAM share of the rest (None: an older engine)
                            routed = (last.get("lookups") or 0) + (last.get("offloaded") or 0)
                            pcie_share = round(last["offloaded"] / routed, 3) \
                                if last.get("offloaded") is not None and routed else None
                            seen = 0 if admission_rejected else prompt_tokens_seen(len(ids), last)   # #471: < len(ids) when cancelled mid-read; #3040
                            if inkling_images and not admission_rejected and finish != "cancel":
                                # Inkling's pictures are ONE placeholder id each in `ids`; the engine expands them to the picture's tokens and its DONE line
                                # counts those (the prompt it really read): that is the prompt_tokens a client is told
                                image_prompt_tokens = int(last.get("prompt_tokens") or 0)
                            self.history.append({
                                "id": request_id,
                                "projection": (sampling or {}).get("experimental_speed_projection") is not False
                                if loaded else None,
                                "time": started, "duration_s": round(time.time() - started, 1), "finish": finish,
                                "gen": getattr(self.engine, "gen", None),       # #4636: the engine process this record is from
                                "prompt_tokens": seen, "reused": last.get("reused"), "output_tokens": n,
                                "first_engine_token_s": first_engine_token_s,
                                # the request's whole prompt, and the tokens read of it (None: an older engine)
                                "prompt_total": len(ids), "prompt_read": last.get("prompt_read"),
                                "reasoning_recoveries": recovery_count,
                                "reasoning_repeat_coverage": round(repeat_coverage, 3),
                                "engine_generated": last.get("generated"),
                                "prompt_ms": last.get("prompt_ms"), "decode_ms": last.get("decode_ms"),
                                "decode_tok_s": round(last["generated"] / (last["decode_ms"] / 1000), 1)
                                if n and (last.get("generated") or 0) >= 2 and last.get("decode_ms") else None,
                                "hit_rate": hit_rate, "pcie_share": pcie_share, "ram_blobs": last.get("ram_blobs"),
                                "file_blobs": last.get("file_blobs"), "file_mb": last.get("file_mb"),
                                # #457: the speculative drafts from the DONE line (None: the engine did not say)
                                "drafts_offered": last.get("drafts_offered"),
                                "drafts_accepted": last.get("drafts_accepted")})
                            if not admission_rejected:                  # #3040: a refused admission did no work
                                t = self.totals
                                t["requests"] += 1
                                t["prompt_tokens"] += seen
                                t["reused"] += last.get("reused") or 0
                                t["output_tokens"] += n
                                t["prompt_ms"] += last.get("prompt_ms") or 0.0
                                t["decode_ms"] += last.get("decode_ms") or 0.0
                                t["drafts_offered"] += last.get("drafts_offered") or 0
                                t["drafts_accepted"] += last.get("drafts_accepted") or 0
                                ft0 = st.get("first_token")             # this request's own (#465: one per request)
                                self.latencies.observe(ft0 - started if ft0 else None, n,
                                                       (last.get("decode_ms") or 0.0) / 1000, time.time() - started)
                                fresh = getattr(self.engine, "last", None)
                                if fresh is not None and fresh is not before:      # the engine's clock for THIS request
                                    timings = request_timings(seen, n, last)
                                    self.last_timings = dict(timings, at=int(time.time())) if timings else None
                            self.last_request_at = time.time()
                            now = time.time()
                            el = now - st.get("started", now)
                            ft = st.get("first_token")
                            rate = answer_speed(last, n, now - ft if ft else None)
                            hit_msg = f", expert cache {hit_rate*100:.1f}% hit" if hit_rate is not None else ""
                            if hit_msg and pcie_share:
                                hit_msg += f" (+{pcie_share*100:.1f}% of the routed experts over PCIe)"
                            rate_txt = f"{rate:.1f} tok/s" if rate is not None else "no speed"
                            print(f"[strata] done: {n} tokens in {el:.0f} s ({rate_txt}) "
                                  f"({finish}, cancel={cancel.is_set()}){hit_msg}", flush=True)
                            if finish == "length" and parser.state in ("reasoning", "rcall"):   # #530
                                print("[strata] the reply reached max tokens while still thinking, so it has no "
                                      "answer: a thinking budget (reasoning_budget_tokens, in the request or in "
                                      "strata-<model>.json for every request) leaves room to answer", flush=True)
                            if os.environ.get("STRATA_DEBUG") and raw_ids:
                                print(f"[strata] raw: {self.tok.decode(raw_ids)!r}", flush=True)
                                print(f"[strata] raw ids: {list(raw_ids)}", flush=True)
                        st["busy"] = False
                        st.pop("tail", None)                     # #212: the answer's end is not kept once it is done
                        st.pop("tool", None)
                        if par:
                            self.live_reqs.pop(id(st), None)
                            if self.live_reqs:              # the newest request still running
                                self.status.update(list(self.live_reqs.values())[-1][0])
                            else:
                                self.status.update(busy=False)
                                self.status.pop("tail", None)
                                self.status.pop("tool", None)
        except QueueCancelled:
            finish = "cancel"
            yield "done", {"finish": "cancel", "completion_tokens": 0, "reused": 0, "timings": None,
                           "reasoning_tokens": 0, "reasoning_recoveries": 0}
            return
        except QueueWaitExpired:
            finish = "timeout"
            with self.status_lock:
                self.queue_timeouts += 1
            raise
        except Exception:
            finish = "error"
            raise
        finally:
            with self.status_lock:
                if queued_counted:
                    self.status["queued"] -= 1
                self.terminal_requests.pop(request_id, None)
                event = {"stop": "finished", "length": "finished", "cancel": "cancelled",
                         "disconnect": "disconnected", "timeout": "timeout", "error": "error"}.get(finish, "error")
                self._terminal_event_locked(request_id, event, finish)
            if emb:
                Path(emb).unlink(missing_ok=True)
        if not parser_done:
            for ev in cut(parser.finish(finish)):
                answered = answered or ev.kind in ("content", "tool_start", "tool_call") and bool((ev.text or "").strip() or ev.kind != "content")
                yield "event", ev
        for ev in hold.release():                   # no retry (or its reply): the held text goes out as it is
            answered = answered or ev.kind in ("content", "tool_start", "tool_call") and bool((ev.text or "").strip() or ev.kind != "content")
            yield "event", ev
        hold.armed = False
        if (self.inkling and not answered and finish in ("stop", "length") and max_new and max_new > INKLING_ROOM_MIN_CAP
                and not cancel.is_set()):
            # Never a silent empty reply: nothing visible could be produced (the thinking took the whole token limit, or the model stopped with
            # nothing to say even after the retry above).  Say so in the answer; finish_reason stays as it is.
            note = (NO_ANSWER_LENGTH if finish == "length" else NO_ANSWER_STOP)
            print(f"[strata] no answer could be produced ({finish}, {n} tokens, {thinking_n} of them thinking): the reply says so", flush=True)
            yield "event", Event("content", note)
        if parser.rescued or parser.refused:
            self.totals["tool_calls_from_reasoning"] = self.totals.get("tool_calls_from_reasoning", 0) + parser.rescued
            print(f"[strata] tool calls inside the thinking: {parser.rescued} read as calls, {parser.refused} kept as "
                  f"reasoning (quoted, or the turn did not end on a stop)", flush=True)
        if getattr(parser, "stray_think", 0):
            # #4077: the model wrote a closing think tag while thinking; the reasoning after it was sent as the answer
            self.totals["stray_think_closers"] = self.totals.get("stray_think_closers", 0) + parser.stray_think
            print(f"[strata] {parser.stray_think} extra closing think tag(s) in the answer were dropped (the model wrote one "
                  "while thinking, so the rest of its reasoning went out as the answer; a call after it still counts)", flush=True)
        if getattr(parser, "notes", None):
            self.totals["tool_args_repaired"] = self.totals.get("tool_args_repaired", 0) + len(parser.notes)
            print(f"[strata] {len(parser.notes)} tool call argument(s) repaired: {'; '.join(sorted(set(parser.notes)))}", flush=True)
        rejected = []
        if parser.unread:
            named = getattr(parser, "unread_names", None)
            print(f"[strata] {parser.unread} <tool_call> block(s) in the answer did not read as a call of a declared "
                  "tool and went back as text (a name the request did not declare, or a form the template does "
                  "not use)" + (f"; the names the model wrote: {', '.join(named)}" if named else "")
                  + "; the response carries them as strata_rejected_tool_calls, /metrics counts them", flush=True)
            with self.status_lock:
                self.totals["tool_calls_unread"] = self.totals.get("tool_calls_unread", 0) + parser.unread
            # #3796: the same names for the client (a machine-readable marker next to the text, which stays as the model wrote it)
            # #5904 / #5933 / #4622: plus the declared tool the name comes closest to, as a hint for the client or the
            # model's next turn.  Only a hint: the call is not run under that name (a guessed tool is a safety risk;
            # "undeclared_tool_calls": "call" stays the only way to run an undeclared call)
            rejected = []
            for nm in named or []:
                entry = {"name": nm, "reason": "undeclared"}
                near = closest_declared(nm, declared)
                if near:
                    entry["closest_declared"] = near
                rejected.append(entry)
        if stops is not None and stops.hit is None and stops.held:
            # the held tail was not a stop string after all
            for ev in hold.push([Event("content", stops.flush())]):
                yield "event", ev
        if image_prompt_tokens > len(ids):
            prompt_total = image_prompt_tokens
        else:
            prompt_total = None
        done = {"finish": finish, "completion_tokens": n, "reused": (timings or {}).get("cache_n", 0),
                "timings": timings, "reasoning_tokens": thinking_n,
                "reasoning_recoveries": recovery_count}
        if stops is not None and stops.hit is not None:
            done["stop_sequence"] = stops.hit
            if finish == "length":
                # #4035: the stop string was found and cut from the answer, so the answer ended at it: "stop", not the token cap that fell on the
                # same token (the Anthropic route already says stop_sequence)
                done["finish"] = "stop"
        if prompt_total:
            done["prompt_tokens"] = prompt_total          # (a request with pictures: with their tokens; the other APIs read this)
        if rejected:
            done["rejected_tool_calls"] = rejected
        if markup_retried is not None:
            done["tool_markup_retry"] = markup_retried
        if declared:
            # strata_unread_tool_markup (GUI-C-1116 reads it): the reply the client got ends in tool-call markup that is
            # no call and is not one of strata_rejected_tool_calls' (a declared name's block that does not read, a cut one)
            text = hold.reply_text()
            tail_markup = dead_markup_tail(text)
            if tail_markup is not None:
                said = {e["name"] for e in rejected}
                if not said.intersection(markup_names(text[tail_markup[0]:])):
                    done["unread_tool_markup"] = {"reason": tail_markup[1], "chars": len(text) - tail_markup[0]}
            if "unread_tool_markup" not in done and hold.started - hold.finished and finish != "cancel" and not cancel.is_set():
                # a call the model had started (tool_start and argument pieces were sent) and the output ended inside - max tokens, or a
                # stop in the middle of it: no tool_call follows and no markup is in the answer's text to find above.  The client still
                # hears that the call was cut (chars 0: it is not in the text); the server's own retry (finish "stop", no call started)
                # never ran for it, and a cut by max tokens would only meet the same cap again.
                done["unread_tool_markup"] = {"reason": "cut_off", "chars": 0}
        yield "done", done


def _debug_req(api, req, messages, tools, max_new, thinking, prompt_tokens):
    """One compact line per request while diagnosing blank/empty turns. Set STRATA_DEBUG=1 to enable."""
    if not os.environ.get("STRATA_DEBUG"):
        return
    last = messages[-1] if messages else {}
    body = last.get("content")
    if isinstance(body, list):
        body = " ".join(p.get("text", "") for p in body if isinstance(p, dict))
    preview = (str(body or "")[:80]).replace("\n", " ")
    print(f"[strata] req {api}: msgs={len(messages)} tools={len(tools or [])} "
          f"max_tokens_raw={req.get('max_tokens')!r}/{req.get('max_completion_tokens')!r} "
          f"max_new={max_new} thinking={thinking} stream={bool(req.get('stream'))} "
          f"prompt_tokens={prompt_tokens} last={last.get('role')!r}:{preview!r}", flush=True)


# ------------------------------------------------------------------------------------------------ MCP tool loop
def run_with_mcp(svc: Service, hub, messages, tools, kw, ids, thinking, max_new, max_req, sampling, cancel,
                 mcp_names):
    """Service.run with the MCP tools executed here: the model writes a call to an MCP tool, the server runs it, adds
    the call and its result to the conversation and lets the model continue - up to `max_rounds` times.  Yields what
    Service.run yields (text, thinking, the request's own tool calls) plus ("mcp", {...}) for the tool activity, and
    one ("done", ...) at the very end with the output tokens of every round. After the last tool round, one final
    reply uses the results already obtained, with no tools offered or executed.

    `mcp_names`: the MCP tools this request offered; any other call is one of the request's own tools and ends the
    turn as always (the client answers it).  MCP calls written in the same answer are then not run (their results
    could not reach the model before the client's)."""
    max_rounds = int(hub.settings["max_rounds"])
    total, rounds, done = 0, 0, None
    finalizing = False
    messages = list(messages)
    while True:
        text, reasoning, calls, own_calls = [], [], [], 0
        for kind, x in svc.run(ids, thinking, tools, max_new, sampling, cancel, declared_only=True):
            if kind == "done":
                done = x
                continue
            if kind == "event":
                ev: Event = x
                if ev.call is not None and ev.call.name in mcp_names:
                    if ev.kind == "tool_start":          # the model has started writing a call: say so at once
                        yield "mcp", {"event": "start", "id": ev.call.id, "name": ev.call.name}
                    elif ev.kind == "tool_call":
                        calls.append(ev.call)
                    continue                             # its argument pieces are not streamed to the client
                if ev.kind == "tool_call":
                    own_calls += 1
                elif ev.kind == "content":
                    text.append(ev.text)
                elif ev.kind == "reasoning":
                    reasoning.append(ev.text)
            yield kind, x
        total += done["completion_tokens"]
        run_them = calls and not own_calls and not finalizing and done["finish"] == "stop" and not cancel.is_set()
        if not run_them:
            for c in calls:                              # announced, never run: close them in the client's view
                yield "mcp", {"event": "result", "id": c.id, "ok": False, "skipped": True, "text": "not run",
                              "chars": 0, "truncated": False, "ms": 0}
            break
        rounds += 1
        results = []
        for c in calls:
            s, tool = hub.routes().get(c.name, (None, c.name))
            yield "mcp", {"event": "call", "id": c.id, "name": c.name, "server": s.name if s else None,
                          "tool": tool, "arguments": c.arguments, "round": rounds}
            # The call runs on a thread while this generator keeps yielding heartbeats: they reach the client as
            # keep-alives, which is how a client that went away (the web app's Stop) is noticed during a slow tool.
            box = {}

            def work(c=c, box=box):
                try:
                    box["r"] = hub.call(c.name, c.arguments, cancel)
                except McpCancelled:
                    box["cancelled"] = True
            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            try:
                while worker.is_alive():
                    worker.join(1.0)
                    if worker.is_alive():
                        yield "ping", None
            except GeneratorExit:
                cancel.set()                             # the client is gone: stop the tool too
                raise
            if "r" not in box:
                break
            r = box["r"]
            print(f"[strata] tool {c.name}: {'ok' if r['ok'] else 'error'}, {r['chars']:,} characters in "
                  f"{r['ms'] / 1000:.1f} s{' (truncated for the model)' if r['truncated'] else ''}", flush=True)
            results.append(r.get("model_content", r["text"]) if svc.has_vision() else r["text"])
            yield "mcp", {"event": "result", "id": c.id,
                          **({"images": r["images"], "vision": svc.has_vision()} if r.get("images") else {}),
                          **{k: r[k] for k in ("ok", "text", "chars", "truncated", "ms")}}
        if cancel.is_set() or len(results) < len(calls):
            done = {**done, "finish": "cancel"}
            break
        messages.append({"role": "assistant", "content": "".join(text).strip(),
                         **({"reasoning_content": "".join(reasoning).strip()} if reasoning else {}),
                         "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in calls]})
        messages += [{"role": "tool", "content": r} for r in results]
        if rounds >= max_rounds:
            finalizing = True
            tools = None
            system_end = next((index for index, message in enumerate(messages)
                               if message.get("role") not in ("system", "developer")), len(messages))
            messages.insert(system_end, {"role": "system", "content":
                f"The tool-use limit of {max_rounds} rounds has been reached. No tools are available for this final "
                "reply. Answer the user's original request in their language using the results already obtained. "
                "Explain any gaps in the available evidence. Do not request or emit further tool calls."})
            yield "mcp", {"event": "limit", "max_rounds": max_rounds}
        try:
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_req, cancel=cancel)
        except ClientGone:                                      # #1999: the client left while a picture was prepared
            done = {**done, "finish": "cancel"}
            break
    yield "done", {**done, "completion_tokens": total, "prompt_tokens": len(ids)}


# ------------------------------------------------------------------------------------------------ OpenAI


def prompt_progress(svc: Service):
    """The last PP line as llama.cpp's `prompt_progress`, so a client written for llama.cpp reads both servers: the
    same four fields, `time_ms` elapsed since prompt reading started, and `processed` counting from token 0 so that a
    client takes the reused prefix out with `cache` (llama.cpp's rule: the timed progress is
    `(processed-cache)/(total-cache)`).  None when there is nothing to say: no PP yet, or the line that says the prompt
    is read.  An engine too old to print RESUME gives `cache` 0, which only makes a reused prefix count as work."""
    engine = svc.engine
    if getattr(engine, "batch", 0):      # --batch: the engine's progress is one global line, not this request's
        return None
    progress = getattr(engine, "progress", None)
    if not progress:
        return None
    read, total = progress
    if total <= 0 or total - read <= PP_DONE_TAIL:
        return None
    return {"total": int(total), "cache": int(getattr(engine, "reused", 0) or 0), "processed": int(read),
            "time_ms": int(getattr(engine, "progress_ms", 0) or 0)}


def openai_chunks(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, run=None, force=None):
    """`run`: the events to send instead of Service.run's (run_with_mcp); its ("mcp", {...}) items become chunks with
    an empty delta and a `strata_mcp` field, which only the web app reads.  `force`: see Service.run."""
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
    model = svc.model_for(req)
    # llama.cpp's flag, and its default: a client that wants prompt progress asks for it, and nothing changes for one
    # that does not.  Strata already sends a keep-alive comment on every PP line, so this only fills that line in.
    want_progress = bool(req.get("return_progress"))

    def chunk(delta, finish=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    # #2391: the role-only chunk goes out right before the first data chunk, not before the prompt is read: a client
    # that times the first data chunk (llama-benchy's TTFR / est_ppt) would otherwise see a prefill of ~0 ms.  vLLM
    # and llama.cpp send it with the engine's first output too.  The keep-alive comments are not data and go first.
    opened = False

    def first():
        nonlocal opened
        if opened:
            return ()
        opened = True
        return (chunk({"role": "assistant", "content": ""}),)

    lp = LogprobEntries(svc, allowed_ids(req), cancel) if top_logprobs(req) is not None else None

    def with_lp(c, key):
        got = lp.spell(svc.engine, c["choices"][0]["delta"][key]) if lp is not None else None
        if got:
            c["choices"][0]["logprobs"] = {key: got}
        return c
    calls = 0
    streamed = {}                                  # tool call id -> index, for calls sent piece by piece
    finished = set()                               # ... and the ones whose final tool_call came (#211)
    answered_done = False                          # the run's own end (its "done" event) arrived
    born = getattr(svc.engine, "gen", None)        # the engine process this request is on (read again with every event: a start inside the run is its own)
    for kind, x in run if run is not None else svc.run(ids, thinking, tools, max_new, req, cancel, force=force, declared_only=True):
        born = getattr(svc.engine, "gen", born)
        if kind == "ping":
            # a PP line: the prompt is being read and there is nothing to say yet.  It used to be only the SSE comment;
            # now it carries the progress when the client asked, and stays the comment when there is nothing to report.
            pp = prompt_progress(svc) if want_progress else None
            if pp is not None:
                yield from first()
            yield None if pp is None else {**chunk({}), "prompt_progress": pp}
        elif kind == "mcp":
            yield from first()
            c = chunk({})
            c["strata_mcp"] = x
            yield c
        elif kind == "event":
            ev: Event = x
            if ev.kind == "reasoning" and ev.text:
                yield from first()
                yield with_lp(chunk({"reasoning_content": ev.text}), "reasoning_content")
            elif ev.kind == "content" and ev.text:
                yield from first()
                yield with_lp(chunk({"content": ev.text}), "content")
            elif ev.kind == "tool_start":
                streamed[ev.call.id] = calls
                calls += 1
                yield from first()
                yield chunk({"tool_calls": [{"index": streamed[ev.call.id], "id": ev.call.id, "type": "function",
                                             "function": {"name": ev.call.name, "arguments": ""}}]})
            elif ev.kind == "tool_args":
                yield from first()
                yield chunk({"tool_calls": [{"index": streamed[ev.call.id], "function": {"arguments": ev.text}}]})
            elif ev.kind == "tool_call" and ev.call.id in streamed:
                finished.add(ev.call.id)
            elif ev.kind == "tool_call":
                yield from first()
                yield chunk({"tool_calls": [{"index": calls, "id": ev.call.id, "type": "function",
                                             "function": {"name": ev.call.name,
                                                          "arguments": json.dumps(ev.call.arguments, ensure_ascii=False)}}]})
                calls += 1
        else:
            answered_done = True
            # a streamed call without its final tool_call is one the output ended inside: not "tool_calls" (#211)
            yield from first()                     # an empty answer still opens with the role
            whole = calls and streamed.keys() <= finished
            finish = "tool_calls" if whole and x["finish"] == "stop" else {"cancel": "stop"}.get(x["finish"], x["finish"])
            last = chunk({}, finish)
            pt = x.get("prompt_tokens", len(ids))     # after MCP rounds: the last round's prompt
            last["usage"] = {"prompt_tokens": pt, "completion_tokens": x["completion_tokens"],
                             "total_tokens": pt + x["completion_tokens"],
                             # the part of the prompt the conversation cache already held (OpenAI's field)
                             "prompt_tokens_details": {"cached_tokens": x.get("reused") or 0},
                             # the tokens written while thinking (OpenAI's field): Inkling's hidden thinking counts too
                             "completion_tokens_details": {"reasoning_tokens": x.get("reasoning_tokens") or 0}}
            if x.get("timings"):
                last["timings"] = x["timings"]          # llama.cpp's field: the speed its clients show
            if x.get("rejected_tool_calls"):
                last["strata_rejected_tool_calls"] = x["rejected_tool_calls"]     # #3796: only when a call was refused as undeclared
            if x.get("unread_tool_markup"):
                last["strata_unread_tool_markup"] = x["unread_tool_markup"]
            yield last
    if not answered_done and not cancel.is_set():
        # Every admitted request ends with a "done" event that carries its finish reason. A run that simply stopped (a 200 with one chunk and no
        # finish_reason, in 0.2 s, while the engine never logged the request: #3961) used to read to the client as an empty answer that worked, and
        # /health stayed ok. It is an error now, counted in the log, so a client's retry can tell it from an answer.
        print("[strata] a request ended without an answer and without a finish reason; it is answered as an error", flush=True)
        # the engine is ended so "the next request restarts it" is true - but only that engine process, and only while nobody is on it (the turn is taken
        # without waiting): this runs after the run released its turn, and a request admitted since must not be killed by it
        ended = svc.end_engine_if_unchanged(born)
        lost = EngineDied("the engine ended the request without an answer")
        lost.restarts = bool(ended)                 # "the next request restarts it" only where the engine really was ended (not with batch slots, replicas, another request on it)
        raise lost


def structured_chunks(chunks, validator):
    """Buffer structured streams so a client never receives unvalidated content.  A turn that calls a tool is not the
    final answer: from its first call on, what was held goes out as it is and the rest streams unchecked, as
    /v1/responses does."""
    buffered = []
    heartbeat = time.monotonic()
    try:
        for chunk in chunks:
            if buffered is None:
                yield chunk
                continue
            progress = chunk is not None and "prompt_progress" in chunk
            if chunk is not None and not progress and (chunk.get("choices") or [{}])[0].get("delta", {}).get("tool_calls"):
                yield from buffered
                yield chunk
                buffered = None
                continue
            if chunk is not None and not progress:
                buffered.append(chunk)      # a progress chunk carries no content: there is nothing in it to validate
            if chunk is None or progress or time.monotonic() - heartbeat >= 1:
                heartbeat = time.monotonic()
                yield chunk if progress else None
        if buffered is None:
            return
        result = openai_collect(buffered)
        choice = result["choices"][0]
        content = validated_json(choice["message"]["content"], validator, choice["finish_reason"])
        # logprobs: the entries spell what the model wrote, which this function replaces with the
        # canonical JSON.  `_structured_logprobs` gives back the form the tokens spell, or None.
        content, logprobs = _structured_logprobs(choice, content)
        yield buffered[0]
        delta = {"content": content}
        if choice["message"].get("reasoning_content"):
            delta["reasoning_content"] = choice["message"]["reasoning_content"]
        piece = {"index": 0, "delta": delta, "finish_reason": None}
        if logprobs:
            piece["logprobs"] = logprobs
        yield {**buffered[0], "choices": [piece]}
        end = buffered[-1]
        if end["choices"][0].get("logprobs"):
            # those entries went out in the piece above (or were dropped with it): the closing
            # chunk must not carry a second copy
            end = {**end, "choices": [{k: v for k, v in end["choices"][0].items() if k != "logprobs"}]}
        yield end
    finally:
        chunks.close()


# ------------------------------------------------------------------------------------------------ Anthropic
def anthropic_events(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, force=None, deferred=False):
    mid = "msg_" + uuid.uuid4().hex[:24]
    yield "message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": svc.model_for(req), "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": len(ids), "output_tokens": 0}}}
    index, open_kind, used_tool = -1, None, False

    def close():
        return ("content_block_stop", {"type": "content_block_stop", "index": index})

    streamed, finished = set(), set()              # calls sent piece by piece; those whose final tool_call came (#211)
    for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel, force=force, declared_only=True,
                           deferred=deferred):
        if kind == "ping":
            yield None
            continue
        if kind == "event":
            ev: Event = x
            if ev.kind == "tool_args":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "input_json_delta", "partial_json": ev.text}}
                continue
            if ev.kind == "tool_call" and ev.call.id in streamed:
                finished.add(ev.call.id)
                continue
            want = {"reasoning": "thinking", "content": "text", "tool_call": "tool_use", "tool_start": "tool_use"}[ev.kind]
            if ev.kind not in ("tool_call", "tool_start") and not ev.text:
                continue
            if ev.kind == "tool_start":
                streamed.add(ev.call.id)
                used_tool = True
            if open_kind != want or want == "tool_use":
                if open_kind is not None:
                    yield close()
                index += 1
                open_kind = want
                block = {"thinking": {"type": "thinking", "thinking": "", "signature": ""},
                         "text": {"type": "text", "text": ""},
                         "tool_use": {"type": "tool_use", "id": ev.call.id if ev.call else "", "name":
                                      ev.call.name if ev.call else "", "input": {}}}[want]
                yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": block}
            if want == "thinking":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "thinking_delta", "thinking": ev.text}}
            elif want == "text":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "text_delta", "text": ev.text}}
            elif ev.kind == "tool_start":
                pass                                # its input follows as tool_args pieces
            else:
                used_tool = True
                yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
                    "type": "input_json_delta", "partial_json": json.dumps(ev.call.arguments, ensure_ascii=False)}}
        else:
            if open_kind is not None:
                yield close()
            stop = "stop_sequence" if x.get("stop_sequence") is not None else \
                "tool_use" if used_tool and streamed <= finished and x["finish"] == "stop" else \
                {"stop": "end_turn", "length": "max_tokens", "cancel": "end_turn"}[x["finish"]]
            # the final counts (message_start could only say the whole prompt): what was read now stays for the next
            # request, a cache write; input_tokens keeps the last token, as 0 makes Claude Code count the prompt twice (#1013)
            whole = max(len(ids), int(x.get("prompt_tokens") or 0))     # (pictures: their tokens count)
            reused = max(0, min(x.get("reused") or 0, whole - 1))
            read_now = whole - reused
            delta_event = {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": x.get("stop_sequence")},
                           "usage": {"input_tokens": 1, "cache_creation_input_tokens": read_now - 1,
                                     "cache_read_input_tokens": reused, "output_tokens": x["completion_tokens"]}}
            if x.get("rejected_tool_calls"):
                delta_event["strata_rejected_tool_calls"] = x["rejected_tool_calls"]     # #3796
            if x.get("unread_tool_markup"):
                delta_event["strata_unread_tool_markup"] = x["unread_tool_markup"]
            yield "message_delta", delta_event
            yield "message_stop", {"type": "message_stop"}


# ------------------------------------------------------------------------------------------------ HTTP
CHUNKED_BODY_MAX = 256 << 20                # #893: the most a Transfer-Encoding: chunked body may hold (read into memory)


def body_limit() -> int:
    """The most a request body may hold, in bytes (read into memory): 256 MiB, which is a million-token conversation
    with room to spare; STRATA_MAX_BODY_MIB changes it (0 or an unreadable value: the default).  A larger one is
    answered 413 before it is read (a Content-Length of 10 TB used to be read until the client gave up)."""
    try:
        mib = int(os.environ.get("STRATA_MAX_BODY_MIB", "0"))
    except ValueError:
        mib = 0
    return mib << 20 if mib > 0 else CHUNKED_BODY_MAX


def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        timeout = PEER_STALL_S
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection

        record = None                                       # #332: this request's monitor record, if kept
        watch_done = None                                   # #430 #431: stops this request's disconnect watcher
        cancel_event = None                                 # active generation owned by this handler
        body_read = False                                   # whether a handler took this request's body
        DRAIN_SECONDS = 5                                   # the longest an unread body is read and dropped

        def log_message(self, fmt, *args):
            pass

        def handle_one_request(self):
            self.answer_started = False
            super().handle_one_request()
            self._drain_body()

        def send_response(self, code, message=None):
            self.answer_started = True                 # a malformed-request answer can only replace one not yet begun
            super().send_response(code, message)

        def _chunked(self) -> bool:
            """#893: a body sent as Transfer-Encoding: chunked (a relay or proxy that does not buffer it).  By RFC 9112
            it wins over a Content-Length."""
            te = self.headers.get("Transfer-Encoding", "") or ""
            return te.split(",")[-1].strip().lower() == "chunked"

        def _read_chunked(self, limit: int, keep: bool = True, deadline: float | None = None) -> bytes:
            """Decode a chunked body: at most `limit` bytes (BadBody 413 beyond it - the size is checked before a chunk
            is read, so a huge announced chunk is never allocated), BadBody 400 for a malformed one.  keep=False reads
            and drops it (the drain); `deadline` (monotonic) ends the read early."""
            parts, total = [], 0
            while True:
                if deadline is not None:
                    wait = deadline - time.monotonic()
                    if wait <= 0:
                        return b"".join(parts)
                    self.connection.settimeout(wait)
                line = self.rfile.readline(1025)
                if not line.endswith(b"\n"):
                    raise BadBody(400, "malformed chunked request body")
                try:
                    size = int(line.split(b";", 1)[0].strip(), 16)
                except ValueError:
                    raise BadBody(400, "malformed chunk size in the request body") from None
                if size < 0:
                    raise BadBody(400, "malformed chunk size in the request body")
                if size == 0:
                    for _ in range(64):                       # the trailers, including the final blank line
                        trailer = self.rfile.readline(8193)
                        if trailer == b"\r\n":
                            return b"".join(parts)
                        if not trailer.endswith(b"\r\n"):
                            raise BadBody(400, "malformed or incomplete chunked request trailers")
                    raise BadBody(400, "too many chunked request trailers")
                total += size
                if total > limit:
                    self.close_connection = True
                    raise BadBody(413, f"the request body is larger than {limit >> 20} MiB")
                left = size
                while left > 0:
                    piece = self.rfile.read(min(left, 1 << 20))
                    if not piece:
                        raise BadBody(400, "the chunked request body ended early")
                    left -= len(piece)
                    if keep:
                        parts.append(piece)
                if self.rfile.read(2) != b"\r\n":
                    raise BadBody(400, "malformed chunked request body")

        def _body(self) -> bytes:
            self.body_read = True
            limit = body_limit()
            if self._chunked():
                return self._read_chunked(limit)
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                raise BadBody(400, "invalid Content-Length") from None
            if length < 0:
                raise BadBody(400, "invalid Content-Length")
            if length > limit:
                self.close_connection = True               # the body is not read: the connection ends with the answer
                raise BadBody(413, f"the request body is larger than {limit >> 20} MiB "
                                   "(STRATA_MAX_BODY_MIB raises the limit)")
            return self.rfile.read(length)

        def _drain_body(self):
            """An answer sent before the body was read (a 401, a 403, /load, a method with no handler) must not close
            the connection on unread bytes: the close then sends a reset, and a client that sends its body after the
            headers (http.client, urllib, requests) gets a connection error instead of the answer.  So the body is
            read and dropped here, once, after an answer, in pieces so that its size is never held in memory.  What
            has not arrived DRAIN_SECONDS later is left unread: the limit is time, so that a conversation of many
            megabytes, at any speed the client has, still gets its answer."""
            headers = getattr(self, "headers", None)
            if self.body_read or headers is None:
                return
            if self._chunked():
                try:
                    self._read_chunked(body_limit(), keep=False, deadline=time.monotonic() + self.DRAIN_SECONDS)
                except (BadBody, OSError):
                    self.close_connection = True
                return
            try:
                left = int(headers.get("Content-Length", 0))
            except ValueError:
                return
            if left <= 0:
                return
            deadline = time.monotonic() + self.DRAIN_SECONDS
            try:
                while left > 0 and (wait := deadline - time.monotonic()) > 0:
                    self.connection.settimeout(wait)         # a client that never sends what it announced
                    piece = self.rfile.read1(min(left, 1 << 20))   # one socket read: read() would wait for it all
                    if not piece:
                        break
                    left -= len(piece)
            except OSError:
                pass

        def parse_request(self):
            """Without an API key, every request (any method) first passes the Host check: DNS rebinding protection
            (host_allowed).  With a key a rebinding page cannot authenticate, so the check is skipped: tunnels and
            proxies that pass their own name on keep working."""
            if not super().parse_request():
                return False
            host = self.headers.get("Host")
            if svc.api_key or host_allowed(host, svc.host_names, "*" in svc.allowed_hosts):
                return True
            print(f"[strata] refused a request for Host {host!r} from {self.client_address[0]}: not a name this server "
                  f"answers to (add it to \"allowed_hosts\" in the config or STRATA_ALLOWED_HOSTS, or set an API key)",
                  flush=True)
            self._json(403, {"error": {"type": "forbidden", "message":
                             f"Host {host!r} is not allowed (DNS rebinding protection). Reaching Strata under this "
                             f"name on purpose? Add it to \"allowed_hosts\" in the config (strata-<model>.json) or to "
                             f"the STRATA_ALLOWED_HOSTS environment variable, or set an API key (\"api_key\"), which "
                             f"turns this check off"}})
            return False

        def _foreign_page(self, body: bool = True) -> bool:
            """Without an API key, a /v1 POST from a browser page of another site (any site can POST text/plain
            there without a CORS preflight) would burn GPU time: an Origin header must name an allowed page, and
            then the body must be JSON.  No Origin (curl, the SDKs, other servers): any content type, as before.
            The /v1 foreign-Origin rule (owner 11-16, the same contract as the app host's /v1): an Origin header
            that is present but empty is another site too, and GET /v1 goes through the same check (body=False:
            no content type to check)."""
            origin = self.headers.get("Origin")
            if svc.api_key or origin is None:
                return False
            if not origin_allowed(origin, self.headers.get("Host"), svc.host_names,
                                  [*svc.trusted_origins, *svc.cors_origins]):
                print(f"[strata] refused an API request from the web page {origin!r} (no API key; add its host to "
                      f"\"allowed_hosts\" or its origin to \"cors_origins\" in the config)", flush=True)
                self._json(403, {"error": {"type": "forbidden", "message":
                                 f"web pages of {origin} may not use this server without an API key; set \"api_key\", "
                                 f"or add the page's host to \"allowed_hosts\" (or its origin to \"cors_origins\") "
                                 f"in the config"}})
                return True
            if body and not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return True
            return False

        def end_headers(self):
            # HTTP/1.0 without keep-alive: the server closes the connection after every response.  Say so, or a
            # client that pools connections (.NET's SocketsHttpHandler) can put its next request on this socket
            # before the close reaches it, and that request fails with "response ended prematurely" without the
            # server ever reading it (a .NET agent app, 2026-10-03: 25 of ~1000 requests, 1-24 ms after the previous answer).
            self.send_header("Connection", "close")
            super().end_headers()

        def _watch_client(self, cancel: threading.Event) -> None:
            """#430 #431: cancel the request as soon as its client hangs up.  A non-streamed request writes nothing
            until it ends, and a streamed one only a keep-alive per prompt chunk (and the first write after a hang-up
            usually still succeeds), so a dropped request kept the engine busy until its answer or the whole prompt
            was done.  Every 0.5 s: the socket readable with nothing to read (EOF) means the client closed it.  A
            request is HTTP/1.0 and fully read here, so no later bytes are expected - data is not a hang-up."""
            done = self.watch_done = threading.Event()
            self.cancel_event = cancel
            # #5004 (the config's "api_keys"): carry the key this request arrived on to Service.run(), where the
            # per-key limits count it.  Every inference path watches its cancel Event right after creating it.
            cancel._strata_key = getattr(self, "key_id", None)
            with svc.status_lock:
                svc.active_cancels.add(cancel)
            sock = self.connection

            def watch():
                while not done.wait(0.5) and not cancel.is_set():
                    try:
                        readable, _, _ = select.select([sock], [], [], 0)
                        gone = bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
                    except (ConnectionError, TimeoutError):
                        gone = True
                    except (OSError, ValueError):            # the socket was closed here: the request has ended
                        return
                    if gone and not done.is_set():
                        self._note(outcome="disconnected")
                        svc.note_client_disconnect(cancel)
                        return

            threading.Thread(target=watch, daemon=True, name="strata-client-watch").start()

        def _note(self, **values):
            """#332: what the monitor shows about this request (nothing when the monitor is off)."""
            if self.record is not None:
                with svc.status_lock:
                    self.record.update(values)

        def _cors(self):
            """#321: CORS headers for an API path (/v1/*) and an origin the config lists in cors_origins - nothing
            otherwise, so a browser keeps every other page away from the API, /settings, /unload and the MCP tools."""
            if not svc.cors_origins or not self.path.split("?")[0].startswith("/v1/"):
                return
            origin = (self.headers.get("Origin") or "").rstrip("/")
            if "*" in svc.cors_origins:
                allow = "*"
            elif origin and origin in svc.cors_origins:
                allow = origin
                self.send_header("Vary", "Origin")
            else:
                return
            self.send_header("Access-Control-Allow-Origin", allow)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            # the headers the preflight asks for (SDKs add their own; "*" does not cover Authorization)
            asked = self.headers.get("Access-Control-Request-Headers")
            self.send_header("Access-Control-Allow-Headers",
                             asked or "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta")
            self.send_header("Access-Control-Max-Age", "600")

        def do_OPTIONS(self):
            # a CORS preflight: no credentials come with it, so no API key; the headers only for cors_origins
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _json(self, code, obj, headers=None, retry_after=None):
            path = self.path.split("?")[0].rstrip("/")
            if code >= 400 and isinstance(obj.get("error"), dict) and \
                    V1_V1_ALIASES.get(path, path) in ("/v1/messages", "/v1/messages/count_tokens"):
                # Anthropic's error shape: @ai-sdk/anthropic drops the message of a body without the top-level type
                request_id = "req_" + (self.record["id"] if self.record is not None else uuid.uuid4().hex[:12])
                error = {**obj["error"], "type": anthropic_error_type(code, obj["error"].get("type"))}
                obj = {"type": "error", **obj, "error": error, "request_id": request_id}
                headers = {**(headers or {}), "request-id": request_id}
            if self.record is not None:
                with svc.status_lock:
                    self.record["http_status"] = code
                    raw = json.dumps(obj, ensure_ascii=False, indent=2)
                    self.record.update(response=raw[:262144], response_truncated=len(raw) > 262144)
                    for key in ("error", "usage", "timings"):
                        if key in obj:
                            self.record[key] = obj[key]
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            if retry_after is not None:
                self.send_header("Retry-After", str(retry_after))
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            self.key_id = None                             # #5004: which configured key this request came with (None = keyless)
            if not svc.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            hit = matched_key(given, svc.api_key)
            if hit is not None:
                self.key_id = hit                          # #5004: the per-key limits count this request against it
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False

        def _authorized_telemetry(self) -> bool:
            if not svc.api_key and not svc.telemetry_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if (svc.api_key and key_matches(given, svc.api_key)) or \
                    (svc.telemetry_key and key_matches(given, svc.telemetry_key)):
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong telemetry key"}})
            return False

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path.startswith("/v1/") and self._foreign_page(body=False):
                return
            if path.startswith("/tool-media/"):
                if not self._authorized():
                    return
                name = path[len("/tool-media/"):]
                hub = svc.mcp
                if not re.fullmatch(r"[0-9a-f]{64}\.(png|jpg|webp)", name) or not hub or not hub.media_dir:
                    self._json(404, {"error": {"message": "not found"}})
                    return
                file = Path(hub.media_dir) / name
                if not file.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}[file.suffix])
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "private, max-age=86400")
                self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/fonts/"):
                # the web app's font (Outfit, OFL: serve/web/fonts); the page falls back to the system font
                name = path[len("/fonts/"):]
                f = ROOT / "serve" / "web" / "fonts" / name
                if "/" in name or "\\" in name or not name.endswith(".woff2") or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "font/woff2")
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            sub = web_subtree_file(path)
            if path.startswith(("/web/locales/", "/web/vendor/")):
                # the page's translations (serve/web/locales/<lang>.json) and the vendored KaTeX (serve/web/vendor):
                # only whitelisted names and types, never a path out of those folders
                if sub is None:
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = sub[0].read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", sub[1])
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/web/"):
                # the web app's own files (serve/web): styles, script, icon sprite - same origin, no CDN
                name = path[len("/web/"):]
                types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                         ".svg": "image/svg+xml"}
                f = ROOT / "serve" / "web" / name
                ext = os.path.splitext(name)[1]
                if "/" in name or "\\" in name or ext not in types or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", types[ext])
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/metrics/prometheus":
                if self._authorized():
                    body = svc.prometheus().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                return
            if path == "/metrics":
                from serve import prometheus
                if prometheus.wants_prometheus(self.headers.get("Accept", ""), urlsplit(self.path).query):
                    if self._authorized():   # a scraper: the text format, vLLM's names (serve/prometheus.py)
                        body = prometheus.render(svc.metrics(), svc.latencies.snapshot()).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", prometheus.CONTENT_TYPE)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    return
                if self._authorized():
                    # the last 12 requests; `?requests=all` every one kept (the Monitor's "Show all", issue #35)
                    self._json(200, svc.metrics(all_requests="requests=all" in self.path))
                return
            if path == "/v1/telemetry":
                if self._authorized_telemetry():
                    from strata_cli.client import normalize_legacy
                    snapshot = normalize_legacy(svc.metrics(), svc.v1_status())
                    snapshot["source"] = "server-v1"
                    snapshot["work"]["slots_requested"] = getattr(svc, "requested_slots", None)
                    # Keep /metrics' established slot shape intact; enrich only this versioned contract.
                    replica_engines = (getattr(svc.engine, "engines", None)
                                       if hasattr(svc.engine, "replica_view") else None)
                    if replica_engines:
                        groups, live_slots, replica_by_slot = [], [], []
                        group_base = 0
                        all_batched = True
                        for replica, child in enumerate(replica_engines):
                            count = int(getattr(child, "batch", 0) or 0)
                            all_batched = all_batched and count > 0
                            count = max(1, count)
                            child_groups = getattr(child, "slot_group", [])
                            child_live = getattr(child, "slot_live", [])
                            for local in range(count):
                                group = child_groups[local] if local < len(child_groups) else None
                                groups.append(group_base + group if type(group) is int and group >= 0 else None)
                                live_slots.append(child_live[local] if local < len(child_live) else None)
                                replica_by_slot.append(replica)
                            group_base += int(getattr(child, "slot_groups", 0) or 0)
                        snapshot["work"]["groups"] = group_base if all_batched and group_base > 0 else None
                    else:
                        groups = getattr(svc.engine, "slot_group", [])
                        live_slots = getattr(svc.engine, "slot_live", [])
                        replica_by_slot = []
                    for slot in snapshot["work"]["slots"]:
                        i = slot.get("slot")
                        if type(i) is int and 0 <= i < len(groups):
                            slot["group"] = groups[i]
                        if type(i) is int and 0 <= i < len(replica_by_slot):
                            slot["replica"] = replica_by_slot[i]
                        if type(i) is int and 0 <= i < len(live_slots) and live_slots[i]:
                            slot["request_id"] = live_slots[i].get("request_id")
                    slot_by_id = {slot.get("request_id"): slot for slot in snapshot["work"]["slots"]
                                  if slot.get("request_id")}
                    with svc.status_lock:
                        active = list(svc.terminal_requests.values())
                        snapshot["activity"] = {"last_seq": svc.terminal_event_seq,
                                                "events": list(svc.terminal_events)[-32:]}
                        draining = svc.draining
                        admission_reserved = svc.admission_reserved
                        admission_rejected = svc.admission_rejected
                        queue_timeouts = svc.queue_timeouts
                        client_disconnect_cancels = svc.client_disconnect_cancels
                        in_flight_tokens = (sum(max(0, int(st.get("generated") or 0))
                                                for st, _ in svc.live_reqs.values())
                                            if svc.live_reqs else
                                            max(0, int(svc.status.get("generated") or 0))
                                            if svc.status.get("busy") else 0)
                        output_counter = int(svc.totals.get("output_tokens") or 0) + in_flight_tokens
                    snapshot["instance"]["id"] = svc.instance_id
                    snapshot["instance"]["state"] = "draining" if draining else "ready"
                    if getattr(svc, "launch_attestation", None) is not None:
                        snapshot["instance"]["launch_attestation"] = svc.launch_attestation
                    snapshot["rates"]["output_tokens_server_counter"] = output_counter
                    snapshot["work"]["requests_omitted"] = max(0, len(active) - 50)
                    snapshot["work"].update(max_queue=svc.max_queue, queue_timeout_s=svc.queue_timeout_s,
                                            queue_timeouts=queue_timeouts,
                                            client_disconnect_cancels=client_disconnect_cancels,
                                            admission_limit=svc.admission_limit(),
                                            admission_reserved=admission_reserved,
                                            admission_rejected=admission_rejected,
                                            oldest_queue_s=max((max(0, time.time() - row["queued_at"])
                                                                for row in active if row["state"] in ("queued", "waiting")),
                                                               default=0))
                    snapshot["work"]["requests"] = [
                        {**row, "state": slot_by_id[row["id"]]["state"] if row["id"] in slot_by_id else row["state"],
                         "slot": slot_by_id[row["id"]]["slot"] if row["id"] in slot_by_id else None,
                         "group": slot_by_id[row["id"]]["group"] if row["id"] in slot_by_id else None,
                         "replica": slot_by_id[row["id"]]["replica"] if row["id"] in slot_by_id else row.get("replica"),
                         "generated": slot_by_id[row["id"]].get("generated") if row["id"] in slot_by_id else None,
                         "prompt_read": slot_by_id[row["id"]].get("prompt_read") if row["id"] in slot_by_id else None}
                        for row in active[:50]]
                    snapshot["quality"]["request_correlation"] = "reported"
                    batch_sample = (getattr(svc.engine, "terminal_batch_sample", None)
                                    if svc.loaded() and not replica_engines else None)
                    if replica_engines:
                        replica_batches = []
                        for replica, child in enumerate(replica_engines[:16]):
                            row = {"replica": replica}
                            sample = (getattr(child, "terminal_batch_sample", None)
                                      if child.alive() else None)
                            if sample:
                                row.update(batch_sample_windows=sample["windows"],
                                           batch_sample_rows=sample["processed_rows"],
                                           batch_sample_active_rows=sample["active_rows"],
                                           batch_sample_span_s=sample["elapsed_ms"] / 1000.0,
                                           batch_sample_age_s=max(0.0, time.monotonic() -
                                                                  sample["observed_monotonic"]))
                            replica_batches.append(row)
                        snapshot["work"]["replica_batches"] = replica_batches
                    if batch_sample:
                        windows = batch_sample["windows"]
                        snapshot["work"].update(effective_rows_window=batch_sample["processed_rows"] / windows,
                                                batch_active_rows_window=batch_sample["active_rows"] / windows,
                                                batch_sample_windows=windows,
                                                batch_sample_rows=batch_sample["processed_rows"],
                                                batch_sample_active_rows=batch_sample["active_rows"],
                                                batch_sample_span_s=batch_sample["elapsed_ms"] / 1000.0,
                                                batch_sample_age_s=max(0.0, time.monotonic() -
                                                                       batch_sample["observed_monotonic"]))
                        snapshot["quality"]["batch_window_rows"] = "measured"
                        snapshot["quality"]["batch_window_source"] = "engine_bstat"
                    stage_sample = (getattr(svc.engine, "terminal_stage_sample", None)
                                    if svc.loaded() and not replica_engines else None)
                    if replica_engines:
                        replica_stage_samples = []
                        for replica, child in enumerate(replica_engines[:16]):
                            row = {"replica": replica}
                            sample = (getattr(child, "terminal_stage_sample", None)
                                      if child.alive() else None)
                            if sample:
                                row.update(stage_sample_elapsed_us=sample["elapsed_us"],
                                           stage_in_flight_us=sample["in_flight_us"],
                                           stage_sample_age_s=max(0.0, time.monotonic() -
                                                                  sample["observed_monotonic"]))
                            replica_stage_samples.append(row)
                        snapshot["work"]["replica_stage_samples"] = replica_stage_samples
                    if stage_sample:
                        snapshot["work"].update(stage_sample_elapsed_us=stage_sample["elapsed_us"],
                                                stage_in_flight_us=stage_sample["in_flight_us"],
                                                stage_sample_age_s=max(0.0, time.monotonic() -
                                                                       stage_sample["observed_monotonic"]))
                        snapshot["quality"]["stage_in_flight_source"] = "engine_pstat_host_span"
                    indices = list(getattr(svc, "gpu_indices", []) or [])
                    visibility = list(getattr(svc, "terminal_visibility_order", []) or [])
                    reported = (list(getattr(svc.engine, "terminal_stages", []) or [])
                                if svc.loaded() and not replica_engines else [])
                    stages = [{**stage, "device_index":
                               visibility[stage["engine_ordinal"]]
                               if stage["engine_ordinal"] < len(visibility) else None}
                              for stage in reported]
                    snapshot["topology"]["selected_devices"] = indices if replica_engines else visibility or indices
                    replica_devices = getattr(svc, "replica_gpu_groups", []) if replica_engines else []
                    snapshot["topology"]["replica_devices"] = [
                        {"replica": i, "devices": list(devices)}
                        for i, devices in enumerate(replica_devices[:16])]
                    replica_stages = []
                    for replica, child in enumerate((replica_engines or [])[:16]):
                        child_reported = (list(getattr(child, "terminal_stages", []) or [])
                                          if child.alive() else [])
                        spawn = getattr(child, "spawn", None)
                        child_env = spawn[4] if isinstance(spawn, tuple) and len(spawn) > 4 else None
                        child_visibility = visible_device_order(child_env, getattr(svc, "backend", None))
                        configured_cards = (set(replica_devices[replica])
                                            if replica < len(replica_devices) else set())
                        child_stages = []
                        for stage in child_reported:
                            ordinal = stage["engine_ordinal"]
                            card = (child_visibility[ordinal]
                                    if type(ordinal) is int and 0 <= ordinal < len(child_visibility) else None)
                            child_stages.append({**stage, "device_index": card if card in configured_cards else None})
                        replica_stages.append({"replica": replica, "stages": child_stages})
                    snapshot["topology"]["replica_stages"] = replica_stages
                    snapshot["topology"]["mode"] = ("layer split (engine reported)" if stages else
                                                     getattr(svc, "terminal_topology_mode", None))
                    snapshot["topology"]["configured_split"] = getattr(svc, "terminal_configured_split", None)
                    snapshot["topology"]["stages"] = stages
                    snapshot["topology"]["resolved"] = bool(stages) and all(
                        stage["device_index"] is not None for stage in stages)
                    snapshot["topology"]["source"] = "engine_start_log" if stages else None
                    snapshot["quality"]["placement"] = ("reported" if snapshot["topology"]["resolved"] else
                                                         "partial" if stages else "unavailable")
                    present = {d.get("index") for d in snapshot["devices"]}
                    for index in indices:
                        if index not in present:
                            snapshot["devices"].append({"index": index, "name": None, "mem_used": None,
                                                        "mem_total": None, "util": None, "temp": None, "power": None})
                    self._json(200, snapshot)
                return
            if path == "/api/requests" and svc.api_monitor:
                if self._authorized():
                    request_id = parse_qs(urlsplit(self.path).query).get("id", [None])[0]
                    records = svc.request_records(request_id)
                    self._json(404 if records is None else 200,
                               {"error": {"message": "request no longer retained"}} if records is None else
                               records if request_id else {"requests": records, "retention": 100, "persistent": False,
                                                          "loaded": svc.loaded(), "auto_load": hasattr(svc.engine, "restart")})
                return
            if path == "/api/expert-heat":                   # #5723: the web page's Experts view (serve/expert_heat.py)
                if self._authorized():
                    try:
                        cfg = runconfig.load(svc.config_path) if svc.config_path else None
                    except (OSError, ValueError):
                        cfg = None
                    self._json(*expert_heat.view(cfg))
                return
            if path == "/settings":
                if self._authorized():
                    self._json(200, {"shared": bool(svc.shared), "defaults": svc.shared})
                return
            if path == "/config":                            # #564: the Settings view's keys of the run config
                if self._authorized():
                    self._config_get()
                return
            if path == "/mcp":
                # the MCP servers, their state and tools (the web app's switch and Monitor card)
                if self._authorized():
                    self._json(200, svc.mcp.status() if svc.mcp else {"servers": [], "tools": 0})
                return
            if path == "" or (path == "/api-monitor" and svc.api_monitor):
                body = (ROOT / "serve" / "web" / ("monitor.html" if path else "index.html")).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                if not path:
                    # the chat's svg/html preview is a sandboxed frame; this keeps it from navigating away to a web page
                    # (a link or <meta refresh> inside a model's html), which no sandbox flag does (serve/web/preview.js)
                    self.send_header("Content-Security-Policy", PAGE_CSP)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path in ("/health", "/api/health"):
                self._json(200, {"status": "ok", "max_context": svc.reported_ctx(), "model": svc.model,
                                 "family": svc.family,
                                 "images": svc.has_vision(), "api_key": bool(svc.api_key),
                                 "image_generation": svc.images is not None,
                                 "loaded": svc.loaded(), "service": "strata"})
            elif path.startswith("/v1/images/files/"):
                # response_format url: the name (32 random hex digits) is the access, as an OpenAI image URL is
                f = svc.images.file(path[len("/v1/images/files/"):]) if svc.images is not None else None
                if f is None:
                    self._json(404, {"error": {"message": "not found (the server keeps the newest images only)"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "private, max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/status":
                if not self._authorized():                  # #212: it shows the end of the last answer
                    return
                with svc.status_lock:
                    s = dict(svc.status)
                now = time.time()
                if s.get("busy"):
                    s["elapsed_s"] = round(now - s["started"], 1)
                    if s.get("first_token"):
                        s["tokens_per_s"] = round(svc._tok_s(), 1)
                        s["tokens_per_s_mean"] = round(svc._tok_s_mean(), 1)
                for k in ("started", "first_token"):
                    s.pop(k, None)
                if svc.key_limits is not None:      # #5004: what each api_keys entry holds right now (names, never keys)
                    s["api_keys"] = svc.key_limits.snapshot()
                self._json(200, s)
            elif path in ("/v1/models", "/models") or path.startswith(("/v1/models/", "/models/")):
                if self._authorized():
                    loaded = svc.loaded()
                    ctx = svc.reported_ctx()
                    created = int(getattr(svc, "started_at", 0) or time.time())
                    cap = svc.output_cap()
                    # #6402: the answer cap under the names clients already look for: OpenAI-shaped tools read
                    # `limit.output` (with `limit.context` beside it), llama-swap's readers read
                    # meta.llamaswap.max_output_tokens.  Without them a client guesses (1,024, 32k, or its own default)
                    # the context size under the names other APIs use too (vLLM's max_model_len, OpenRouter's
                    # context_length), so a client that sizes its window from /v1/models reads it
                    model = {"id": svc.model, "object": "model", "created": created, "owned_by": "strata",
                             "status": {"value": "loaded"},
                             "max_model_len": ctx, "context_length": ctx, "max_context_length": ctx,
                             "limit": {"context": ctx, "output": cap},
                             "meta": {"n_ctx": ctx, "llamaswap": {"max_output_tokens": cap}},
                             "architecture": {"input_modalities": ["text", "image"] if svc.has_vision() else ["text"],
                                              "output_modalities": ["text"]}}
                    if not loaded and (svc.idle_unload_s or getattr(svc.engine, "unloaded", False)):
                        model["status"] = {"value": "unloaded"}   # like llama-server's router: listed, loads on use
                        loaded = True
                    if svc.aliases:                       # #297: the aliases, and each one listed under its own id
                        model["aliases"] = list(svc.aliases)
                    data = [model, *({**model, "id": x, "alias_of": svc.model} for x in svc.aliases)]
                    data = data if loaded else []
                    if svc.images is not None:            # the image model: POST /v1/images/generations
                        data.append({"id": svc.images.model, "object": "model", "created": created, "owned_by": "strata",
                                     "status": {"value": "loaded"},
                                     "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]}})
                    if path in ("/v1/models", "/models"):
                        self._json(200, {"object": "list", "data": data})
                    else:                                 # #5460: GET /v1/models/{model_id} (OpenAI model retrieve)
                        head = "/v1/models/" if path.startswith("/v1/models/") else "/models/"
                        asked = unquote(path[len(head):])
                        item = next((m for m in data if m["id"] == asked), None)
                        if item is None:
                            self._json(404, {"error": {"message": f"The model '{asked}' does not exist",
                                                       "type": "invalid_request_error", "param": "model",
                                                       "code": "model_not_found"}})
                        else:
                            self._json(200, item)
            elif path == "/props":
                if self._authorized():
                    self._props()
            elif path == "/slots":
                if self._authorized():
                    loaded = not hasattr(svc.engine, "alive") or svc.engine.alive()
                    with svc.status_lock:
                        busy = bool(svc.status.get("busy"))
                        # llama.cpp's name for what the slot's context holds: the running request's prompt size,
                        # kept after it ends (the conversation cache carries it on).  A front-end's context meter
                        # divides it by n_ctx; without the key it reads 0 % against Strata forever (#562).
                        in_use = int(svc.status.get("prompt_tokens") or 0)
                    if getattr(svc.engine, "batch", 0) and hasattr(svc.engine, "slots_view"):
                        # --batch: one entry per slot (the same view /metrics has), not one for the whole engine
                        n_ctx = svc.reported_ctx()
                        slots = [{"id": s["slot"], "n_ctx": n_ctx, "is_processing": s["state"] != "idle",
                                  "n_prompt_tokens": int(s.get("prompt_tokens") or s.get("held_tokens") or 0)}
                                 for s in svc.engine.slots_view()]
                    else:
                        slots = [{"id": 0, "n_ctx": svc.reported_ctx(), "is_processing": busy,
                                  "n_prompt_tokens": in_use}]
                    self._json(200, slots if loaded else [])
            elif path == "/v1/status":
                if self._authorized():
                    self._json(200, svc.v1_status())
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            try:
                self._post()
            except BadBody as e:                              # #893: a malformed or oversized chunked body
                self._json(e.status, {"error": {"type": "invalid_request_error", "message": str(e)}})

        def _post(self):
            if not self._authorized():
                return
            path = self.path.split("?")[0].rstrip("/")   # issue #55: Claude Code posts /v1/messages?beta=true
            path = V1_V1_ALIASES.get(path, path)         # an Anthropic client given ".../v1" as its base posts /v1/v1/messages
            if path.startswith("/v1/") and self._foreign_page():
                return
            if path == "/settings":
                self._settings()
                return
            if path == "/config":
                self._config_post()
                return
            if path in ("/unload", "/load") and not self._control_body():
                return
            # JSON from Strata's own page only, as /settings: else a plain form POST from any site unloads the model
            if path in ("/unload", "/load") and not self._own_page("the model can be loaded or unloaded"):
                return
            # the same for the slot files: else any site could overwrite a saved conversation or replace the live one
            if path.startswith("/slots/") and not self._own_page("conversations can be saved or restored"):
                return
            if path == "/unload":                            # give the GPU back now (between requests)
                try:
                    wait_s = unload_wait(parse_qs(urlsplit(self.path).query).get("wait", [None])[0],
                                         svc.UNLOAD_WAIT_S)
                    r = svc.unload_waiting(wait_s) if wait_s else svc.unload()
                except EngineStuck as e:
                    self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                    return
                self._json(409 if r == "busy" else 200, {"status": r})
                return
            if path == "/load":                              # load now, e.g. ahead of a request
                try:
                    svc.load()
                    self._json(200, {"status": "loaded"})
                except (GpuBusy, EngineStarting) as e:
                    ra = getattr(e, "retry_after", None)
                    self._json(503, {"error": {"type": "server_error", "message": str(e)}},
                               headers={"Retry-After": int(ra)} if ra else None)
                except EngineDied as e:                      # #2497: a program that cannot start (or died at READY)
                    self._json(503, {"error": {"type": "server_error", "message": died_note(e)}})
                except EngineStuck as e:                     # is a 503 with the reason, as on the request routes,
                    self._json(503, {"error": {"type": "server_error", "message": str(e)}})   # not a dropped connection
                return
            reserved = False
            try:
                if path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
                    reserved = svc.reserve_inference()
                    if reserved is None:
                        error = {"type": "server_error", "code": "server_draining",
                                 "message": "this instance is draining; retry against the next instance"}
                        body = responses_error_body(error["message"], "server_error", code=error["code"]) \
                            if path == "/v1/responses" else {"error": error}
                        self._json(503, body, retry_after=1)
                        return
                    if not reserved:
                        error = {"type": "rate_limit_error", "code": "admission_full",
                                 "message": "the inference admission limit is full; retry later"}
                        body = responses_error_body(error["message"], "server_error", code=error["code"]) \
                            if path == "/v1/responses" else {"error": error}
                        self._json(429, body, retry_after=1)
                        return
                req = json.loads(self._body() or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("send a JSON object")
                if path.startswith("/v1/responses/"):        # retrieve/delete/cancel/compact: nothing is stored
                    self._json(404, responses_error_body("this server keeps no responses (stateless); send the "
                                                         "whole conversation to POST /v1/responses", code="not_found"))
                    return
                if path in ("/v1/load", "/v1/unload"):
                    if not self._own_page("the model can be loaded or unloaded"):
                        return
                    if req.get("model") not in (None, svc.model):
                        self._json(404, {"error": {"message": "model not found"}})
                        return
                    if path == "/v1/unload":
                        wait_s = unload_wait(parse_qs(urlsplit(self.path).query).get("wait", [None])[0],
                                             svc.UNLOAD_WAIT_S)
                        result = svc.unload_waiting(wait_s) if wait_s else svc.unload()
                        if result == "busy":
                            raise ModelBusy("a request is running or queued")
                    else:
                        if not svc.fifo.acquire(blocking=False):
                            raise ModelBusy("a request is running or queued")
                        try:
                            with svc.status_lock:
                                if svc.status.get("busy") or svc.status.get("queued"):
                                    raise ModelBusy("a request is running or queued")
                            svc.ensure_loaded()
                            svc.last_request_at = time.time()
                        finally:
                            svc.fifo.release()
                        result = "loaded"
                    self._json(200, {"status": result, **svc.v1_status()})
                    return
                if path == "/v1/vram":                       # #533: give VRAM back to other programs, or take it
                    if not self._own_page("the VRAM reserve can be changed"):
                        return
                    r = req.get("reserve_mib")
                    d = req.get("device")                     # per-GPU (#533): free only this CUDA card; null/absent: every card.
                    if r is not None and (isinstance(r, bool) or not isinstance(r, int) or r < 0):
                        raise ValueError("reserve_mib: a whole number of MiB (0 or more), or null for the start's")
                    if d is not None and (isinstance(d, bool) or not isinstance(d, int) or d < 0):
                        raise ValueError("device: a CUDA device index (0 or more), or null for every card")
                    if d is not None and r is None:
                        raise ValueError("device needs an explicit reserve_mib to free")
                    try:
                        self._json(200, svc.vram(r, d))
                    except EngineDied as e:
                        self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                    return
                if path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
                    check_model = getattr(svc, "check_model", None)
                    if callable(check_model):
                        model_err = check_model(req)
                        if model_err is not None:
                            self._json(404, responses_error_body(model_err["message"], "invalid_request_error",
                                                                 code="model_not_found")
                                       if path == "/v1/responses" else {"error": model_err})
                            return
                    self.record = svc.begin_request(path, req)
                if path == "/v1/responses":
                    self._responses(req)
                elif path.startswith("/slots/"):
                    query = parse_qs(self.path.partition("?")[2])
                    try:
                        code, body = svc.slot_action(path[len("/slots/"):], (query.get("action") or [""])[0],
                                                     req.get("filename"))
                    except EngineDied as e:                  # EngineSilent included: the engine was ended
                        code, body = 500, {"error": {"code": 500, "message": str(e) + "; the next request starts "
                                                     "the engine again", "type": "server_error"}}
                    self._json(code, body)
                elif path == "/v1/chat/completions":
                    self._openai(req)
                elif path == "/v1/messages":
                    self._anthropic(req)
                elif path == "/v1/messages/count_tokens":
                    self._count_tokens(req)
                elif path == "/v1/images/generations":
                    self._images(req)
                else:
                    self._json(404, {"error": {"message": "not found"}})
            except EngineNonFinite as e:                    # #879, twice: not the request's fault
                print(f"[strata] 503: {e}", flush=True)
                self._json(503, responses_error_body(str(e), "server_error") if path == "/v1/responses"
                           else {"error": {"type": "server_error", "message": str(e)}})
            except ValueError as e:
                print(f"[strata] 400 invalid request: {e}", flush=True)
                if path == "/v1/responses":
                    self._json(400, responses_error_body(str(e)))
                else:
                    self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
            except ModelBusy as e:
                self._json(409, {"error": {"type": "model_busy", "message": str(e)}})
            except QueueWaitExpired as e:
                error = {"type": "rate_limit_error", "code": "queue_wait_timeout", "message": str(e)}
                self._note(error=error)
                body = responses_error_body(str(e), "server_error", code="queue_wait_timeout") \
                    if path == "/v1/responses" else {"error": error}
                self._json(429, body, retry_after=1)
            except StructuredOutputError as e:
                self._json(502, {"error": {"type": "structured_output_failed", "code": "structured_output_failed",
                                          "message": str(e)}})
            except (GpuBusy, EngineStarting) as e:
                ra = getattr(e, "retry_after", None)       # #1997 / #1994: a refusal that is worth retrying
                self._json(503, {"error": {"type": "server_error", "message": str(e)}},
                           headers={"Retry-After": int(ra)} if ra else None)
            except (TypeError, KeyError, AttributeError, IndexError, UnicodeError) as e:
                # a body that is JSON but the wrong shape ("messages": 5, a content part that is a number): a 400 that
                # says so, with the traceback in the log, instead of a dropped connection - unless the answer began
                if getattr(self, "answer_started", False):
                    raise
                import traceback
                print(f"[strata] 400 malformed request ({type(e).__name__}: {e}):\n" + traceback.format_exc(),
                      flush=True)
                body = {"error": {"type": "invalid_request_error",
                                  "message": f"the request is malformed ({type(e).__name__}: {e})"}}
                self._json(400, responses_error_body(body["error"]["message"]) if path == "/v1/responses" else body)
            except EngineDied as e:                          # before the answer started (not streamed)
                self._json(503, {"error": {"type": "server_error", "message": died_note(e)}})
            except EngineStuck as e:                         # an unload or restart that could not end the engine
                self._json(503, {"error": {"type": "server_error", "message": str(e)}})

            except ClientGone:                               # #1999: the client left while the request was prepared
                self._note(outcome="disconnected")
            except (ConnectionError, TimeoutError):
                self._note(outcome="disconnected")
                raise                                        # as before #332: the server's own handling
            except BadBody:
                raise
            except Exception as e:
                if getattr(self, "answer_started", False):
                    raise
                import traceback
                print(f"[strata] 500 internal error ({type(e).__name__}: {e}):\n" + traceback.format_exc(), flush=True)
                self._note(error=f"{type(e).__name__}: {e}")
                message = f"internal server error ({type(e).__name__}: {e})"
                self._json(500, responses_error_body(message, "server_error") if path == "/v1/responses"
                           else {"error": {"type": "server_error", "message": message}})
            finally:
                if reserved:
                    svc.release_inference()
                svc.drop_embeddings()                        # the images' file of a request that never got to run()
                if self.watch_done is not None:
                    self.watch_done.set()
                if self.cancel_event is not None:
                    with svc.status_lock:
                        svc.active_cancels.discard(self.cancel_event)
                record = self.record
                if record is not None:
                    with svc.status_lock:
                        record["wallclock_s"] = round(time.perf_counter() - record["_clock"], 3)
                        record["finished_at"] = time.time()
                        record["state"] = "error" if record.get("error") else record.get("outcome", "completed")
                    svc.request_trace.record = None

        def _images(self, req):
            """POST /v1/images/generations (serve/images.py): one sd-cli job in the GPU's turn."""
            gen = svc.images
            if gen is None:
                self._json(404, {"error": {"type": "invalid_request_error", "code": "images_not_set_up",
                                           "message": "image generation is not set up on this server (run setup "
                                                      "with --images qwen-image)"}})
                return
            job = images_api.parse_request(req, gen.cfg)        # ImageRequestError (a ValueError): 400
            svc.last_request_at = time.time()
            cancel = threading.Event()
            self._watch_client(cancel)
            print(f"[strata] image: {job['width']}x{job['height']}, n={job['n']}, {job['steps']} steps, "
                  f"seed {job['seed']}", flush=True)
            try:
                with images_api.gpu_turn(svc, gen, job, cancel) as policy:
                    images, figures = gen.run(job, cancel)
            except images_api.ImageJobFailed as e:
                self._json(500, {"error": {"type": "server_error", "code": "image_generation_failed",
                                           "message": str(e)}})
                return
            figures["gpu_policy"] = policy
            print(f"[strata] image done in {figures['seconds']:.1f} s ({policy})", flush=True)
            host = self.headers.get("Host") or f"{self.server.server_address[0]}:{self.server.server_address[1]}"
            scheme = "https" if (self.headers.get("X-Forwarded-Proto") or "").lower() == "https" else "http"
            self._json(200, gen.response(images, job, figures, f"{scheme}://{host}"))

        def _props(self):
            model = parse_qs(urlsplit(self.path).query).get("model", [svc.model])[0]
            if model not in svc.model_names():
                self._json(404, {"error": {"message": "model not found"}})
                return
            if not svc.loaded() and not getattr(svc.engine, "unloaded", False):
                self._json(503, {"error": {"message": "the engine is not running"}})
                return
            card_defaults = {} if (svc.has_mtp and not request_asks_sampling(svc.shared, svc.sampling_defaults)) else \
                model_card_sampling(svc.model_arch, True, greedy=asks_greedy(svc.shared, svc.sampling_defaults))
            defaults = {**card_defaults,
                        **svc.sampling_defaults, **svc.shared}
            names = {"repetition_penalty": "repeat_penalty", "penalty_last_n": "repeat_last_n"}
            params = {names.get(k, k): v for k, v in defaults.items()
                      if k in ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
                               "presence_penalty", "frequency_penalty", "penalty_last_n")}
            params["n_predict"] = svc.shared.get("max_tokens", -1)
            props = {"default_generation_settings": {"n_ctx": svc.reported_ctx(), "params": params},
                     "total_slots": max(1, int(getattr(svc.engine, "batch", 0) or 0)),   # #1004: the batch slots
                     "model_alias": svc.model, "chat_template": svc.template.source,
                     "chat_template_caps": svc.template.caps,
                     "modalities": {"vision": svc.has_vision()}, "models_autoload": hasattr(svc.engine, "restart"),
                     "is_sleeping": not svc.loaded()}
            if getattr(svc.engine, "model_path", None):
                props["model_path"] = svc.engine.model_path
            version = getattr(svc.engine, "info", {}).get("version")
            if version:
                props["build_info"] = "Strata " + str(version)
            self._json(200, props)

        def _control_body(self) -> bool:
            """Consume the unused control body before replying/closing (Windows otherwise sends a TCP reset)."""
            if self._chunked():
                self.body_read = True
                try:
                    self.connection.settimeout(2.0)
                    self._read_chunked(65536, keep=False)
                except BadBody as e:
                    self._json(e.status, {"error": {"message": f"control request: {e}"}})
                    return False
                except OSError:
                    self._json(400, {"error": {"message": "incomplete control request body"}})
                    return False
                return True
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"error": {"message": "invalid Content-Length"}})
                return False
            if not 0 <= length <= 65536:
                self._json(413, {"error": {"message": "control request body is limited to 64 KiB"}})
                return False
            timeout = self.connection.gettimeout()
            try:
                self.connection.settimeout(2.0)
                complete = len(self.rfile.read(length)) == length
            except OSError:
                complete = False
            finally:
                self.connection.settimeout(timeout)
                self.body_read = True                    # nothing left for the drain to wait for (#594)
            if not complete:
                self._json(400, {"error": {"message": "incomplete control request body"}})
            return complete

        def _own_page(self, what) -> bool:
            """Only JSON (a form or a "simple" cross-site request can't send it without a CORS preflight, which this
            server never grants) and no foreign Origin: a web page elsewhere must not change settings or run tools."""
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return False
            if self._foreign_origin():
                self._json(403, {"error": {"message": f"{what} only from Strata's own page (or an origin in the "
                                                      f"config's trusted_origins)"}})
                return False
            return True

        def _config_get(self):
            if not svc.config_path:
                self._json(404, {"error": {"message": "this server was started without a run config"}})
                return
            try:
                cfg = runconfig.load(svc.config_path)
            except (OSError, ValueError) as e:
                self._json(500, {"error": {"type": "server_error", "message": f"the run config cannot be read: {e}"}})
                return
            self._json(200, runconfig.view(cfg, svc.config_path))

        def _config_post(self):
            """#564: change a few documented keys of the run config - JSON from Strata's own page only, as
            /settings (the key is checked before); every other key of the file stays as it is."""
            body = self._body()
            if not self._own_page("the run config can be changed"):
                return
            if not svc.config_path:
                self._json(404, {"error": {"message": "this server was started without a run config"}})
                return
            try:
                req = json.loads(body or b"{}")
                with svc.config_lock:
                    cfg = runconfig.load(svc.config_path)
                    new, changed = runconfig.apply(cfg, req.get("set") if isinstance(req, dict) else None)
                    if changed:
                        bak = runconfig.save(svc.config_path, new)
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            except OSError as e:
                self._json(500, {"error": {"type": "server_error", "message": f"the run config cannot be written: {e}"}})
                return
            if changed:
                print(f"[strata] the Settings view changed {', '.join(changed)} in {Path(svc.config_path).name} "
                      f"(the earlier file: {bak.name}); used from the next start", flush=True)
            self._json(200, {**runconfig.view(new, svc.config_path), "changed": changed})
        def _foreign_origin(self) -> bool:
            """A web page of another origin sent this.  The Origin must be this server's own address (host and port),
            or an origin the config trusts (trusted_origins: the web app behind a reverse proxy or tunnel); no Origin
            is a client that is no browser.  Headers a proxy adds (X-Forwarded-*, CF-Ray, CF-Connecting-IP) prove
            nothing about the page that sent the request, so they open nothing."""
            origin = (self.headers.get("Origin") or "").rstrip("/")
            return bool(origin) and origin.split("://", 1)[-1] != self.headers.get("Host", "") and \
                origin not in svc.trusted_origins

        def _no_local_images(self, messages):
            """A web page of another origin must not have a file on this computer read as an image.  Without an
            api_key, _foreign_page refuses such a page unless cors_origins lets it in - and a page cors_origins lets
            in (["*"]: every page) could name any file and, CORS allowing, read what the model says about it.  data:
            and http(s) images are the page's own to send; files come from Strata's own page, a trusted origin or a
            client that is no browser.  ValueError (a 400)."""
            if self._foreign_origin() and any(not src.startswith(("data:", "http://", "https://"))
                                              for src in images_of(messages)):
                raise ValueError("a web page of another origin cannot have a file on this computer read as an "
                                 "image: send it as a data: URL (or add the page's origin to the config's "
                                 "trusted_origins)")

        def _settings(self):
            # They change what every client gets, so only the app's own page may set them
            body = self._body()
            if not self._own_page("settings can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                if not isinstance(req, dict) or "defaults" not in req:
                    self._json(400, {"error": {"type": "invalid_request_error",
                                              "message": 'the body must be {"defaults": {...}} to set them, '
                                                         'or {"defaults": null} to clear them'}})
                    return
                shared = svc.set_shared(req["defaults"])
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            print("[strata] other apps now use the Chat settings: " + ", ".join(f"{k}={v}" for k, v in shared.items())
                  if shared else "[strata] other apps use their own settings again", flush=True)
            self._json(200, {"shared": bool(shared), "defaults": shared})

        def _sse(self):
            self._note(http_status=200)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")   # nginx and similar proxies pass each event at once
            self._cors()
            self.end_headers()
            # (#4271) a stuck write ends as OSError: disconnected.  The connection's own PEER_STALL_S deadline stays the
            # bound when it is the shorter one: one stall rule for every socket operation.
            limits = [x for x in (STREAM_WRITE_TIMEOUT_S, self.timeout) if x]     # self.timeout: PEER_STALL_S when the class was made
            if limits:
                try:
                    self.connection.settimeout(min(limits))
                except OSError:
                    pass

        def _capture(self, items, api):
            """Retain bounded input/output for the monitor without changing the API response (the items as they
            are when the monitor is off)."""
            if self.record is None:
                return items
            return self._captured(items, api)

        def _captured(self, items, api):
            try:
                for item in items:
                    if item is not None:
                        if api == "openai":
                            delta = item["choices"][0]["delta"]
                            content, reasoning = delta.get("content", ""), delta.get("reasoning_content", "")
                            usage, timings = item.get("usage"), item.get("timings")
                        elif api == "responses":
                            kind = item["type"]
                            content = item["delta"] if kind == "response.output_text.delta" else ""
                            reasoning = item["delta"] if kind == "response.reasoning_text.delta" else ""
                            usage, timings = (item.get("response") or {}).get("usage"), None
                        else:
                            _, event = item
                            delta = event.get("delta", {})
                            content, reasoning = delta.get("text", ""), delta.get("thinking", "")
                            usage, timings = event.get("usage"), None
                        with svc.status_lock:
                            for name, value in (("output", content), ("reasoning", reasoning)):
                                if value:
                                    combined = self.record[name] + value
                                    self.record[name] = combined[:262144]
                                    self.record[name + "_truncated"] |= len(combined) > 262144
                            if usage:
                                self.record["usage"] = usage
                            if timings:
                                self.record["timings"] = timings
                    yield item
            finally:
                items.close()

        def _openai(self, req):
            req = svc.with_shared(req, "openai")
            bias_api.normalize(req.get("logit_bias"), len(getattr(svc.tok, "tokens", ())) or None)
            messages, tools, kw = openai_to_messages(req)
            require_messages(messages)                # an empty conversation is a 400, as OpenAI answers it, not a reply to nothing
            self._no_local_images(messages)
            if tool_choice_of(req.get("tool_choice"))[0] == "none":   # as the Responses route: no tools are offered
                tools = None
            force = svc.force_text(forced_call(req.get("tool_choice"), tools))      # a bad value is a 400 before anything is sent
            messages, validator = prepare_format(req.get("response_format"), messages, with_tools=bool(tools))
            if validator is not None and req.get("strata_mcp"):
                raise ValueError("structured response_format with MCP is not supported")
            svc.load()
            limit_new = token_limit(req.get("max_completion_tokens"), "max_completion_tokens")
            limit_old = token_limit(req.get("max_tokens"), "max_tokens")      # (both checked, whichever is used)
            max_req = max_new = limit_new or limit_old or 0                   # 0/-1: the rest
            # No cap of our own on the reply's length when logprobs are asked for: the reply is already bounded by
            # max_tokens (by the rest of the context when it is not sent), which is the only bound OpenAI and vLLM
            # put on it either.  A reply that is not streamed does hold every entry (its token, bytes and up to 20
            # alternatives) until it ends, so a long one holds a lot: streaming hands each entry out with its chunk.
            refuse_for_split(svc.engine, req)                 # (before the batch-slot rule below: its message would say to run without --parallel)
            if top_logprobs(req) is not None and batch_slots_on(svc.engine) and max_req != 1:
                # --parallel/--batch: a slot's "BT <slot> <id>" lines carry no LP line, and the batched loop never
                # calls emit_lp_line (src/program/generate.cpp); an admission -- a max_tokens 1 request -- is read
                # as GEN 1, which does write them.  Past that one token a reply would hold entries for a prefix
                # only, which a client cannot tell from the model's own answer, so the longer one is refused.
                raise ValueError('logprobs is available with "parallel" (batch slots) only for max_tokens 1; '
                                 'run the server without "--parallel" for longer replies')
            n_vocab = len(getattr(svc.tok, "tokens", ())) or None
            if top_logprobs(req) is not None and n_vocab:
                # emit_lp_line indexes the logits row with each named id as given, so an id past the vocabulary
                # would read out of bounds: refused here, before anything is sent
                bad = [i for i in allowed_ids(req) if i >= n_vocab]
                if bad:
                    raise ValueError(f"allowed_token_ids: {bad[0]} is outside the vocabulary "
                                     f"(0..{n_vocab - 1})")
            bias_api.validate_request(req, svc.engine, len(getattr(svc.tok, "tokens", ())) or None)
            use_mcp = req.get("strata_mcp") is True and svc.mcp is not None      # the web app's opt-in (serve/mcp.py)
            own = {t.get("name") for t in tools or [] if isinstance(t, dict)}   # #592: a second line of defence
            if use_mcp:
                if not self._own_page("MCP tools can be used"):   # tools run with the user's rights on this PC
                    return
                svc.mcp.wait(10)                                  # servers still starting (only right after start)
                extra = svc.mcp.template_tools(exclude=own)       # the request's own tools win a name clash
                use_mcp = bool(extra)
                tools = (tools or []) + extra or None
            if force and use_mcp:
                raise ValueError("a forced tool_choice with MCP tools is not supported")
            svc.reasoning_budget(req)                         # a bad value is a 400 before anything is sent
            stop_strings(req)                                 # ... and so is a bad stop / stop_sequences
            check_request_sampling(req)                       # ... and a sampling field of the wrong type
            cancel = threading.Event()
            self._watch_client(cancel)                       # #430 #431 (#1999: before prepare, which can take long)
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, force=force, req=req, cancel=cancel)
            _debug_req("openai", req, messages, tools, max_new, thinking, len(ids))
            run = run_with_mcp(svc, svc.mcp, messages, tools, kw, ids, thinking, max_new, max_req, req, cancel,
                               {t["name"] for t in extra}) if use_mcp else None
            chunks = openai_chunks(svc, req, ids, thinking, tools, max_new, cancel, run=run, force=force)
            if validator is not None:
                chunks = structured_chunks(chunks, validator)
            chunks = self._capture(chunks, "openai")
            if not req.get("stream"):
                return self._json(200, openai_collect(chunks))
            self._sse()
            # OpenAI's streaming contract: with "stream_options": {"include_usage": true} the usage rides in its own
            # last chunk, "choices": []. Clients and gateways read it there (LiteLLM ignores a usage that sits on the
            # finish_reason chunk and recounts the tokens itself, too low). Without the option the chunk is unchanged.
            # A value that is not an object is ignored (the spec allows an object or null): an exception here, after the
            # headers are out, would end the stream with an empty 200.
            stream_options = req.get("stream_options")
            split_usage = isinstance(stream_options, dict) and bool(stream_options.get("include_usage"))
            try:
                for c in chunks:
                    if c is None:
                        self.wfile.write(b": keep-alive\n\n")      # an SSE comment: clients ignore it
                    elif split_usage and c.get("usage") is not None:
                        head = {k: v for k, v in c.items() if k != "usage"}
                        tail = {"id": c["id"], "object": c["object"], "created": c["created"], "model": c["model"],
                                "choices": [], "usage": c["usage"]}
                        for part in (head, tail):
                            self.wfile.write(b"data: " + json.dumps(part, ensure_ascii=False).encode() + b"\n\n")
                    else:
                        self.wfile.write(b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            except OSError:
                self._note(outcome="disconnected")
                svc.note_client_disconnect(cancel)           # client went away: stop the engine
                chunks.close()
            except EngineDied as e:                          # mid-stream: say so, then end the stream properly
                self._chat_failed(died_note(e))
            except QueueWaitExpired as e:
                self._chat_failed(str(e), "rate_limit_error", "queue_wait_timeout")
            except StructuredOutputError as e:
                self._chat_failed(str(e), "structured_output_failed", "structured_output_failed")
            except ValueError as e:                          # the engine's ERR after the stream started: the
                self._chat_failed(str(e))   # headers are sent, so no 400 now
            except (GpuBusy, EngineStarting, EngineStuck, ModelBusy) as e:
                self._chat_failed(str(e))   # _post's handler would write HTTP into the stream's body
            except ClientGone:
                raise
            except Exception:                                # no silent end: the client would take it for a whole answer
                self._chat_failed(self._stream_crashed())

        def _stream_crashed(self):
            """An unexpected error in a started stream: the traceback for the log, a plain sentence for the client."""
            import traceback
            print("[strata] error in a started stream:\n" + traceback.format_exc(), flush=True)
            return "the server failed while answering; the server's log has the details"

        def _chat_failed(self, message, kind="server_error", code=None):
            """A chat stream that cannot finish: the error object, then [DONE] (#5633).  No chunk with a made-up
            finish_reason: "error" is outside OpenAI's list and strict clients reject the chunk; the OpenAI SDKs raise
            on the error object itself, and the web app reads it."""
            error = {"type": kind, **({"code": code} if code else {}), "message": message}
            self._note(error=error)
            try:
                self.wfile.write(b"data: " + json.dumps({"error": error}).encode() + b"\n\ndata: [DONE]\n\n")
                self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")

        def _responses_ready(self, req, events):
            """Send a finished Responses result that did not run on the engine."""
            if not req.get("stream"):
                return self._json(200, responses_api.collect(events))
            self._sse()
            try:
                for e in events:
                    self.wfile.write(f"event: {e['type']}\n".encode() + b"data: " +
                                     json.dumps(e, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")

        def _responses(self, req):
            """#451: POST /v1/responses - OpenAI's Responses API, stateless (serve/responses.py), on the chat path.
            Errors use the Responses format; once the stream has started they arrive as a response.failed event."""
            try:
                title = (responses_api.thread_title_events(req, svc.model_for(req))
                         if svc.codex_thread_titles else None)
            except ResponsesError as e:
                return self._json(e.status, e.body())
            if title is not None:
                return self._responses_ready(req, title)
            cancel = threading.Event()
            self._watch_client(cancel)                       # #430 #431 (#1999: before prepare, which can take long)
            try:
                ids, thinking, tools, max_new, asm, validator, req, force = self._responses_prepare(req, cancel)
            except ClientGone:
                self._note(outcome="disconnected")
                return
            except ResponsesError as e:
                return self._json(e.status, e.body())
            except ModelBusy as e:
                return self._json(409, responses_error_body(str(e), "server_error", code="model_busy"))
            except (GpuBusy, EngineStarting, EngineStuck, EngineDied) as e:
                ra = getattr(e, "retry_after", None)         # #1997: as chat and messages (only GpuBusy carries one)
                return self._json(503, responses_error_body(str(e), "server_error", code="server_error"),
                                  headers={"Retry-After": int(ra)} if ra else None)

            def check(text, finish):
                return validated_json(text, validator, finish)

            def events():
                yield from asm.start()
                done = None
                for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel, force):
                    if kind == "ping":
                        yield None
                    elif kind == "event":
                        yield from asm.feed(x)
                    elif kind == "done":
                        done = x
                if done is not None and done["finish"] != "cancel" and not cancel.is_set():
                    yield from asm.finish(done, check)
                elif cancel.is_set() or (done or {}).get("finish") == "cancel":
                    # no final event exists: collect() must not read the last delta as the answer, a stream must not
                    # end without response.failed.  Cancelled by cancel_owned_requests, or by a client that left
                    if svc.draining:
                        raise ServerStopping("this instance is draining; retry against the next instance")
                    raise ClientGone()
            items = self._capture(events(), "responses")
            if not req.get("stream"):
                try:
                    result = responses_api.collect(items)
                except StructuredOutputError as e:
                    return self._json(502, responses_error_body(str(e), "server_error",
                                                                code="structured_output_failed"))
                except QueueWaitExpired as e:
                    return self._json(429, responses_error_body(str(e), "server_error",
                                                                code="queue_wait_timeout"), retry_after=1)
                except ServerStopping as e:                  # as a request refused while draining
                    return self._json(503, responses_error_body(str(e), "server_error", code="server_draining"),
                                      retry_after=1)
                except EngineDied as e:
                    return self._json(503, responses_error_body(died_note(e), "server_error",
                                                                code="server_error"))
                except EngineNonFinite as e:                 # #879, twice: not the request's fault
                    return self._json(503, responses_error_body(str(e), "server_error", code="server_error"))
                except ValueError as e:                      # the engine's ERR line
                    return self._json(500, responses_error_body(str(e), "server_error", code="server_error"))
                return self._json(200, result)
            self._sse()
            last = time.monotonic()

            def send(e):
                self.wfile.write(f"event: {e['type']}\n".encode() + b"data: " +
                                 json.dumps(e, ensure_ascii=False).encode() + b"\n\n")

            try:
                for e in items:
                    if e is None:
                        # Codex's idle timeout (5 minutes by default) counts events, not SSE comments: while a long
                        # prompt is read, a response.in_progress every 15 s tells it the server is still working
                        if time.monotonic() - last >= 15:
                            send(asm.in_progress())
                            last = time.monotonic()
                        else:
                            self.wfile.write(b": keep-alive\n\n")
                    else:
                        send(e)
                        last = time.monotonic()
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")
                svc.note_client_disconnect(cancel)           # client went away: stop the engine
                items.close()
            except EngineDied as e:                          # mid-stream: response.failed, then the stream ends
                self._responses_failed(asm, died_note(e), "server_error", send)
            except ServerStopping as e:                      # server_error: ResponseErrorCode is a closed list
                self._responses_failed(asm, str(e), "server_error", send)
            except QueueWaitExpired as e:
                self._responses_failed(asm, str(e), "queue_wait_timeout", send)
            except StructuredOutputError as e:
                self._responses_failed(asm, str(e), "structured_output_failed", send)
            except ValueError as e:                          # the engine's ERR after the stream started
                self._responses_failed(asm, str(e), "server_error", send)
            except (GpuBusy, EngineStarting, EngineStuck, ModelBusy) as e:   # _post would write HTTP into the stream's body
                self._responses_failed(asm, str(e), "server_error", send)
            except ClientGone:
                raise
            except Exception:                                # no silent end: the client would take it for a whole answer
                self._responses_failed(asm, self._stream_crashed(), "server_error", send)

        def _responses_failed(self, asm, message, code, send):
            e = asm.failed(message, code)
            self._note(error=e["response"]["error"])
            try:
                send(e)
                self.wfile.flush()
            except OSError:
                pass

        def _responses_prepare(self, req, cancel=None):
            responses_api.check_request(req)
            messages = responses_api.input_messages(req)
            tools, names, skipped = responses_api.request_tools(req)
            force = svc.force_text(forced_call(req.get("tool_choice"), tools))   # "required" / {"type": "function", "name": N}
            kw = responses_api.template_kwargs(req, {"reasoning_effort": svc.effort_default()})
            try:
                self._no_local_images(messages)              # as on the chat route: no file read for a foreign page
            except ValueError as e:
                raise ResponsesError(str(e), "input") from None
            try:
                messages, validator = prepare_format(responses_api.text_format(req), messages, with_tools=bool(tools))
            except ValueError as e:
                raise ResponsesError(str(e).replace("response_format", "text.format"), "text.format") from None
            noted = svc.__dict__.setdefault("hosted_tools_noted", set())   # said once per tool, not per request
            if set(skipped) - noted:
                print(f"[strata] /v1/responses: left out the hosted tools {', '.join(sorted(set(skipped) - noted))} "
                      f"(this server cannot run them)", flush=True)
                noted.update(skipped)
            if req.get("max_output_tokens") is None and "max_tokens" in svc.shared:   # the shared Chat settings
                req = {**req, "max_output_tokens": svc.shared["max_tokens"]}
            try:
                svc.reasoning_budget(req)                    # a bad value is a 400 before anything is sent
            except ValueError as e:
                raise ResponsesError(str(e), "reasoning_budget_tokens") from None
            try:
                stop_strings(req)
            except ValueError as e:
                raise ResponsesError(str(e), "stop") from None
            try:
                check_request_sampling(req)                   # the request is also the sampling dict (see below)
            except ValueError as e:                           # its message starts with the field's name: the param
                raise ResponsesError(str(e), str(e).partition(":")[0]) from None
            svc.load()
            # #924 (opt-in): a Codex compaction request is rendered with its conversation's tools, so its prompt
            # starts as the cached one did; the parser and the response keep the request's own tools
            shown = responses_api.prompt_tools(req, tools) if svc.codex_compaction_cache else tools
            try:
                try:
                    ids, thinking, max_new = svc.prepare(messages, shown, kw, int(req.get("max_output_tokens") or 0), force=force,
                                                          req=req, cancel=cancel)
                except ValueError:
                    if shown is tools:
                        raise
                    shown = tools                            # too long with the kept tools: the request as sent
                    ids, thinking, max_new = svc.prepare(messages, shown, kw, int(req.get("max_output_tokens") or 0), force=force,
                                                          req=req, cancel=cancel)
            except ResponsesError:
                raise
            except ValueError as e:                          # too long for the context, an image without vision
                raise ResponsesError(str(e), "input", "context_length_exceeded" if "context" in str(e) else None) \
                    from None
            if svc.codex_compaction_cache:
                responses_api.prompt_made(req, shown)
            _debug_req("responses", req, messages, tools, max_new, thinking, len(ids))
            # #4621 / #1917: what the client's own tool_search returned in THIS request's input is declared in the
            # conversation where the search answered; after the prompt is made (so its start, and the conversation
            # cache, stay as they were) the parser and the response accept exactly those names as declared - never a
            # guessed one.  tool_choice "none" (tools None) hides the found tools too.
            found, found_names = responses_api.client_found_tools(req)
            if tools is not None and found:
                tools = tools + [t for t in found if t["name"] not in names]
                names = {**found_names, **names}
            include = req.get("include") if isinstance(req.get("include"), list) else []
            asm = responses_api.Assembler(req, svc.model_for(req), len(ids), names,
                                          "reasoning.encrypted_content" in include, validator is not None)
            # the request is also the sampling dict, as on the chat path (temperature, top_p, top_k, seed, ...)
            return ids, thinking, tools, max_new, asm, validator, req, force

        def _count_tokens(self, req):
            """Anthropic's token count, which Claude Code asks for its context figures: the prompt this server would
            read for the same request, rendered and tokenized - the model does not run."""
            req = svc.with_shared(req, "anthropic")
            messages, tools, kw = anthropic_to_messages(req, svc.anthropic_think_unasked)
            stop_strings(req)                                 # the same 400 as the request itself would get
            check_request_sampling(req)                       # ... likewise a sampling field of the wrong type
            self._json(200, {"input_tokens": len(svc.encode_prompt(messages, tools, kw, store=False))})

        def _anthropic(self, req):
            svc.load()
            req = svc.with_shared(req, "anthropic")
            messages, tools, kw = anthropic_to_messages(req, svc.anthropic_think_unasked)
            self._no_local_images(messages)
            callable_tools, deferred = anthropic_tools(req)     # #4663: the deferred tools are callable, unlisted
            if tool_choice_of(req.get("tool_choice"))[0] == "none":   # Anthropic's {"type": "none"}: no tools offered
                tools = callable_tools = None
            force = svc.force_text(forced_call(req.get("tool_choice"), tools, callable_tools))   # "any" / {"type": "tool", "name": N}
            max_new = token_limit(req.get("max_tokens"), "max_tokens") or 0   # 0/-1: the rest of the context
            svc.reasoning_budget(req)                         # a bad value is a 400 before anything is sent
            stop_strings(req)                                 # ... and so is a bad stop / stop_sequences
            check_request_sampling(req)                       # ... and a sampling field of the wrong type
            cancel = threading.Event()
            self._watch_client(cancel)                       # #430 #431 (#1999: before prepare, which can take long)
            try:
                ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, force=force, req=req, cancel=cancel)
            except ContextOverflow as e:
                raise ValueError(e.anthropic_text()) from None       # #4838: Anthropic's own wording first, Strata's sentence after it
            # #6460: Claude Code's auto-compaction summary omits `thinking` and runs near the context limit. Under the
            # default `"anthropic_thinking": "model"`, unasked thinking can burn the little remaining output room and
            # produce an empty summary ("summarization produced empty response"). Disable unasked thinking when the
            # request did not ask for thinking/effort/budget and the prompt is near the context ceiling.
            req_effort = (req.get("output_config") or {}).get("effort") if isinstance(req.get("output_config"), dict) else None
            asked_thinking = (req.get("thinking") is not None or bool(req_effort)
                              or req.get("reasoning_budget_tokens") is not None)
            ctx_limit = int(getattr(svc.engine, "max_context", 0) or 0)
            req_max = token_limit(req.get("max_tokens"), "max_tokens") or 0
            near_limit = bool(ctx_limit >= 32768 and len(ids) >= int(ctx_limit * 0.85))
            if thinking and not asked_thinking and near_limit and kw.get("enable_thinking") is not False:
                kw_no_think = {**kw, "enable_thinking": False}
                try:
                    ids, thinking, max_new = svc.prepare(messages, tools, kw_no_think, req_max, force=force, req=req, cancel=cancel)
                except ContextOverflow as e:
                    raise ValueError(e.anthropic_text()) from None
            _debug_req("anthropic", req, messages, tools, max_new, thinking, len(ids))
            events = anthropic_events(svc, req, ids, thinking, callable_tools, max_new, cancel, force=force,
                                      deferred=deferred)
            events = self._capture(events, "anthropic")
            if not req.get("stream"):
                return self._json(200, anthropic_collect(events))
            self._sse()
            try:
                for item in events:
                    if item is None:
                        self.wfile.write(b": keep-alive\n\n")
                    else:
                        name, e = item
                        self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                         json.dumps(e, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")
                svc.note_client_disconnect(cancel)
                events.close()
            except EngineDied as e:                          # mid-stream: Anthropic's error event
                err = {"type": "error", "error": {"type": "api_error", "message": died_note(e)}}
                self._note(error=err["error"])
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except QueueWaitExpired as e:
                err = {"type": "error", "error": {"type": "rate_limit_error", "code": "queue_wait_timeout",
                                                   "message": str(e)}}
                self._note(error=err["error"])
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                self._note(error=err["error"])
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except (GpuBusy, EngineStarting, EngineStuck, ModelBusy) as e:   # _post would write HTTP into the stream's body
                self._anthropic_failed(str(e), "overloaded_error" if isinstance(e, GpuBusy) else "api_error")
            except ClientGone:
                raise
            except Exception:                                # no silent end: the client would take it for a whole answer
                self._anthropic_failed(self._stream_crashed(), "api_error")

        def _anthropic_failed(self, message, kind):
            err = {"type": "error", "error": {"type": kind, "message": message}}
            self._note(error=err["error"])
            try:
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except OSError:
                self._note(outcome="disconnected")

    return Handler


def lan_addresses() -> list[str]:
    """This PC's IPv4 addresses on its networks (what another device types in), without loopback/link-local."""
    import socket
    first, ips = None, set()
    try:                                                # the address of the default route; sends nothing (UDP)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            first = s.getsockname()[0]
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ok = lambda ip: ip and not ip.startswith(("127.", "169.254.", "0."))
    return ([first] if ok(first) else []) + sorted(ip for ip in ips if ok(ip) and ip != first)


def host_names_for(bind_host: str, allowed_hosts=(), trusted_origins=()) -> set[str]:
    """Every name this server answers to besides an IP address: localhost, the address it listens on, the config's
    allowed_hosts and trusted_origins' hosts, and - listening beyond this PC (0.0.0.0 or a LAN address) - this PC's
    name and LAN addresses (the LAN addresses matter for the Origin check, which takes no IP on trust)."""
    names = set(LOOPBACK_NAMES)
    bind = host_name(bind_host)
    if bind and bind not in ("0.0.0.0", "::"):
        names.add(bind)
    if bind not in LOOPBACK_NAMES:
        try:
            pc = socket.gethostname().lower()
            names.update((pc, pc + ".local"))
        except OSError:
            pass
        names.update(("host.docker.internal", *lan_addresses()))
    names.update(x for x in allowed_hosts if x != "*")
    names.update(filter(None, (host_name(o.split("://", 1)[-1]) for o in trusted_origins)))
    return names


class KeyLimits:
    """#5004 (the config's "api_keys"): what each key may have in flight, and what that leaves the others.

    A request counts against the key it arrived with from the moment it enters `Service.run()` to the moment it
    ends: the queue wait, the prompt read and the generation all count.  While its key is at a limit the request
    waits - the queue's own kind of wait: `queue_timeout` bounds it and the same 429 answers its expiry, and a
    client that hangs up mid-wait ends it (ClientGone, as for a prepared request whose client left).

    * "max_concurrent": how many requests of that key the server holds at once.
    * "reserved_capacity": slots of a "parallel" server that only this key may sit on.  Every other key's reach
      shrinks by it - the others may hold at most `slots - reserved` at once, whether this key is using its
      reservation or not: a reserved slot waits empty rather than being borrowed.  Nothing is ever moved off a
      slot a request already holds (there is no preemption): the reservation holds at the next admission.

    A key with no entry (a plain "api_key") shares `slots - every reservation`, and so does a request whose key
    the server accepts but the block does not name.  Names are what GET /status and the logs show; the keys
    themselves never leave the config.  The class exists only when the config lists "api_keys" - without it
    nothing is counted and admission is exactly as before."""

    OTHER = "(other keys)"                 # the shared bucket; api_key_limits_of refuses it as a name

    def __init__(self, policies, slots):
        self.policies = policies                      # api_key_limits_of()'s rows, in config order
        self.by_key = {p["key"]: p for p in policies}
        self.reserved_all = sum(p["reserved_capacity"] for p in policies)
        self.slots = slots                            # callable: the engine's slots right now (its batch)
        self.held = collections.Counter()             # name -> requests inside (self.OTHER for unlisted keys)
        self.cv = threading.Condition()

    def _row(self, key):
        pol = self.by_key.get(key) if key is not None else None
        return pol, self.OTHER if pol is None else pol["name"]

    def _reach(self, pol) -> int:
        # the slots this bucket may sit on: its own key's reservation counts for it, no other key's does
        return max(0, self.slots() - (self.reserved_all - (pol["reserved_capacity"] if pol else 0)))

    def _room(self, pol, name) -> bool:
        reach = self._reach(pol)
        if pol is not None and pol["max_concurrent"] is not None:
            reach = min(reach, pol["max_concurrent"])
        return self.held[name] < reach

    def acquire(self, key, cancel=None, deadline=None) -> None:
        """Wait until this key's figures leave room, then count the request.  Raises QueueWaitExpired when
        `deadline` runs out (the handler turns it into the same 429 the engine queue's wait gets) and ClientGone
        when `cancel` says the client already left."""
        pol, name = self._row(key)
        with self.cv:
            while not self._room(pol, name):
                if cancel is not None and cancel.is_set():
                    raise ClientGone(f"the client disconnected while {name} waited for its api_keys limit to free")
                left = deadline - time.monotonic() if deadline is not None else None
                if left is not None and left <= 0:
                    raise QueueWaitExpired(f"the inference queue wait deadline expired while {name} was at its "
                                           "api_keys limit; retry later")
                self.cv.wait(0.25 if left is None else min(0.25, left))
            self.held[name] += 1

    def release(self, key) -> None:
        pol, name = self._row(key)
        with self.cv:
            self.held[name] -= 1
            self.cv.notify_all()

    def snapshot(self):
        """What GET /status shows: names, counts and limits - never a key."""
        return {"slots": self.slots(),
                "keys": [{"name": p["name"], "inflight": self.held[p["name"]],
                          "max_concurrent": p["max_concurrent"], "reserved_capacity": p["reserved_capacity"]}
                         for p in self.policies],
                "unlisted": {"inflight": self.held[self.OTHER], "max": self._reach(None)}}


def api_key_limits_of(value, slots: int):
    """#5004 (the config's "api_keys"): per-key admission limits, keyed by the key itself.  Every entry is a key
    the server accepts, like the ones in "api_key"; its object names what that key may do:

        "api_keys": {
            "hermes-key":       {"name": "hermes", "max_concurrent": 1},
            "home-assistant-key": {"name": "home assistant", "max_concurrent": 1, "reserved_capacity": 1}
        }

    "max_concurrent": requests of that key at once (default: as many as its reach).  "reserved_capacity": slots
    of a "parallel" server kept for this key alone (default 0).  "name": what GET /status and the logs call it
    (default: "key 1", "key 2", ... in order).  Like origins_of, a wrong entry stops the start rather than
    limiting less or more than meant: a value that is not a whole number, a key that is empty or "none", a field
    the server does not know, two entries sharing a name, or reservations that leave some key no slot at all
    (`slots` is the "parallel" the server starts with; KeyLimits follows the engine's own batch from then on).
    A wrong entry is named by its place in the block ("entry 2"), never by its key: a start that fails is read
    off the console and pasted into reports, and the keys themselves stay inside the config.
    Returns [{"key", "name", "max_concurrent", "reserved_capacity"}] in config order."""
    if value is None or value == "" or value == {}:
        return []
    if not isinstance(value, dict):
        raise ValueError('api_keys must be an object: {"the-key": {"max_concurrent": ..., "reserved_capacity": ...}}')
    out = []
    for i, (key, pol) in enumerate(value.items(), 1):
        key = str(key).strip()
        if not key or key.lower() == "none":
            raise ValueError(f'api_keys entry {i} is not a usable key (empty, or "none")')
        pol = {} if pol is None else pol
        if not isinstance(pol, dict):
            raise ValueError(f"api_keys entry {i} must be an object (or null)")
        unknown = sorted(set(pol) - {"name", "max_concurrent", "reserved_capacity"})
        if unknown:
            raise ValueError(f'api_keys entry {i} says {", ".join(map(repr, unknown))}, '
                             "which is not a limit (the fields are name, max_concurrent, reserved_capacity)")
        name = str(pol.get("name") or f"key {i}").strip() or f"key {i}"
        if name == KeyLimits.OTHER:
            raise ValueError(f'api_keys entry {i} may not be named "{KeyLimits.OTHER}"')
        mc = pol.get("max_concurrent")
        if mc is not None and (isinstance(mc, bool) or not isinstance(mc, int) or mc < 1):
            raise ValueError(f"api_keys entry {i}: max_concurrent must be a whole number of requests, at least 1")
        rc = pol.get("reserved_capacity")
        if rc is None:
            rc = 0
        if isinstance(rc, bool) or not isinstance(rc, int) or rc < 0:
            raise ValueError(f"api_keys entry {i}: reserved_capacity must be a whole number of slots, 0 or more")
        out.append({"key": key, "name": name, "max_concurrent": mc, "reserved_capacity": rc})
    if len({p["name"] for p in out}) != len(out):
        raise ValueError("api_keys: two entries share a name; names must differ (GET /status shows them)")
    reserved_all = sum(p["reserved_capacity"] for p in out)
    for p in out:
        if slots - (reserved_all - p["reserved_capacity"]) < 1:
            raise ValueError(f'api_keys: the reservations of the other keys leave "{p["name"]}" no slot of a '
                             f"{slots}-slot server (raise \"parallel\" or lower \"reserved_capacity\")")
        if p["max_concurrent"] and p["max_concurrent"] > slots - (reserved_all - p["reserved_capacity"]):
            print(f'[strata] api_keys: "{p["name"]}" may run {p["max_concurrent"]} at once, but only '
                  f"{slots - (reserved_all - p['reserved_capacity'])} of the {slots} slots are its reach (the rest "
                  "are reserved): the limit that holds is the reach", flush=True)
    if reserved_all >= slots:
        print("[strata] api_keys: the reservations cover every slot: a request with a key that has no "
              "reserved_capacity of its own never gets one", flush=True)
    return out


def serve(svc: Service, host="127.0.0.1", port=8095) -> ThreadingHTTPServer:
    svc.host_names = host_names_for(host, svc.allowed_hosts, svc.trusted_origins)
    svc.start_telemetry()
    httpd = Server((host, port), make_handler(svc))
    httpd.svc = svc
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def start_engine_or_fallback(cfg: dict, exe: str, effort_end: list[str] | None, env: dict, lazy: bool, tok, *, offline: bool = False,
                             batch_telemetry=None):
    """Start the run config's engine. A config written for the all-cards split engine (strata_control/split_engine.py) carries the
    normal layer-split config under "split_fallback": if that engine refuses or fails to come up, say so in one plain line and start the
    normal one instead, so a start never fails because of the option. Any other config starts exactly as before."""
    fb = cfg.get("split_fallback")
    try:
        return StrataEngine(exe, engine_args(cfg, exe) + (effort_end or []), cwd=cfg.get("cwd"), log=cfg.get("log"), env=env, lazy=lazy,
                            retry_args=mtp_retry_args(cfg, exe, effort_end), retry_vram_args=nodraft_retry_args(cfg, exe, effort_end))
    except Exception as e:
        if not isinstance(fb, dict) or not fb.get("exe"):
            raise
        msg = str(e).strip()
        why = msg.splitlines()[0][:200] if msg else type(e).__name__
        # the engine's own reason ("sfnx: failed: ..." in its log tail, which the exception carries) is what a person needs to read
        own = next((ln.split("sfnx: failed: ", 1)[1].strip() for ln in reversed(msg.splitlines()) if "sfnx: failed: " in ln), "")
        if own:
            why = own[:300]
        log = cfg.get("log")
        if log:                                     # the normal engine writes the same log file: keep the split engine's
            try:
                os.replace(log, str(log) + ".split-engine")
            except OSError:
                pass
        print(f"[strata] The all-cards split engine could not start ({why}). Starting the normal layer split instead.", flush=True)
    cfg2 = {k: v for k, v in cfg.items() if k not in SPLIT_KEYS + ("exe", "args", "env", "layer_split", "replicas", "user_args", "drafting")}
    cfg2.update(fb)
    cfg2 = with_component_engine(cfg2)
    if engine_component.is_reference(cfg2["exe"]):
        exe2 = _component_exe(cfg2["exe"], cfg2)
    else:
        exe2 = cfg2["exe"] if os.path.isabs(cfg2["exe"]) else os.path.abspath(os.path.join(cfg2.get("cwd") or ".", cfg2["exe"]))
    env2 = child_env(cfg2, offline=offline)
    configure_batch_telemetry_env(env2, cfg2, batch_telemetry)
    effort2 = effort_end_args(cfg2, exe2, tok)
    engine = StrataEngine(exe2, engine_args(cfg2, exe2) + (effort2 or []), cwd=cfg2.get("cwd"), log=cfg2.get("log"), env=env2, lazy=lazy,
                          retry_args=mtp_retry_args(cfg2, exe2, effort2), retry_vram_args=nodraft_retry_args(cfg2, exe2, effort2))
    engine.info["split_fallback"] = True
    return engine


def start_replicas(cfg: dict, groups: list[list[int]], exe: str, effort_end: list[str], lazy: bool,
                   *, offline: bool = False):
    """Opt-in data-parallel replicas (serve/replicas.py): one StrataEngine per card group, each with its own cards, log
    and CPU cores, started together; the ReplicaEngine in front spreads the requests."""
    from serve import replicas as rp
    n = len(groups)
    cpus = rp.cpu_sets(n, cfg.get("replica_cpus") is not False)
    print(f"[strata] {n} replicas on the cards {' | '.join(','.join(map(str, g)) for g in groups)}"
          f"{' (own CPU cores each)' if cpus else ''}: requests go to the least busy one, a conversation back to its own",
          flush=True)
    engines: list = [None] * n
    errors: list = []
    arena, arena_note = rp.shared_arena_plan(cfg, n)         # opt-in: one copy of the experts in RAM for all replicas
    if arena_note:
        print(f"[strata] {arena_note}", flush=True)

    def one(i: int):
        rc = rp.replica_cfg(cfg, i, groups[i])
        if arena:
            rc["args"] = list(rc["args"]) + ["--shared-expert-arena", arena]
        try:
            engines[i] = StrataEngine(exe, engine_args(rc, exe) + effort_end, cwd=rc.get("cwd"), log=rc.get("log"),
                                      env=child_env(rc, offline=offline), lazy=lazy, cpus=cpus[i] if cpus else None,
                                      retry_args=mtp_retry_args(rc, exe, effort_end), retry_vram_args=nodraft_retry_args(rc, exe, effort_end))
        except BaseException as e:        # noqa: BLE001
            errors.append((i, e))
    rc0 = rp.replica_cfg(cfg, 0, groups[0])
    warn_budget_over_ram(engine_args(rc0, announce=False) if "args" in rc0 else [])
    if os.environ.get("STRATA_REPLICA_START", "") == "serial":
        for i in range(n):
            one(i)
    else:
        ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    if errors and arena and len(errors) < n:
        # a replica that died while it was reading the experts into the shared arena (or any other one that failed) while
        # another one is up: the living engine took the read over (the state word in the arena), so start the failed ones
        # once more on the same arena instead of giving up the sharing
        failed = sorted(i for i, _ in errors)
        errors.clear()
        print(f"[strata] replica(s) {', '.join(map(str, failed))} did not start; starting again on the shared expert arena",
              flush=True)
        for i in failed:
            one(i)
    if errors:
        for e in engines:
            if e is not None:
                try:
                    e.close()
                except Exception:        # noqa: BLE001
                    pass
        rp.remove_arena(arena)
        i, e = errors[0]
        if arena:       # a shared arena that could not be made (no room in /dev/shm, a file of another pack): as before
            print(f"[strata] replica {i} did not start with the shared expert arena ({e}); starting every replica with "
                  "its own copy of the experts instead", flush=True)
            return start_replicas({**cfg, "replica_shared_arena": False}, groups, exe, effort_end, lazy, offline=offline)
        raise SystemExit(f"[strata] replica {i} did not start: {e}")
    load = int(cfg.get("replica_load_tokens") or rp.LOAD_TOKENS)
    eng = rp.ReplicaEngine(engines, load_tokens=load)
    if arena:
        eng.__dict__["arena_file"] = arena
        import atexit
        atexit.register(rp.remove_arena, arena)             # (a killed server leaves the file: the next start sweeps it)
    return eng


def start_split(cfg: dict, spec: dict, exe: str, effort_end: list[str], lazy: bool, *, offline: bool = False):
    """Opt-in split serving (serve/split.py): a prefill engine and a decode engine, one card each, the same model and the
    same session settings.  Started one after the other, so a shared expert arena is filled once; the SplitEngine in
    front hands a long prompt from the first to the second."""
    from serve import split as sp
    sp.check_cfg(cfg, spec)
    arena, arena_note = sp.arena_path(cfg, spec)
    work = sp.session_dir(spec)
    print(f"[strata] split serving: the prompt is read on card {spec['prefill_gpu']}, the answer is made on card "
          f"{spec['decode_gpu']} (prompts with {spec['min_prompt_tokens']}+ new tokens are handed over); {arena_note}",
          flush=True)
    engines = {}
    try:
        for role in ("prefill", "decode"):
            rc = sp.role_cfg(cfg, spec, role)
            if arena:
                rc["args"] = list(rc["args"]) + ["--shared-expert-arena", arena]
            if role == "prefill":
                warn_budget_over_ram(engine_args(rc, announce=False) if "args" in rc else [])
            engines[role] = StrataEngine(exe, engine_args(rc, exe) + effort_end, cwd=rc.get("cwd"), log=rc.get("log"),
                                         env=child_env(rc, offline=offline), lazy=lazy, retry_args=mtp_retry_args(rc, exe, effort_end), retry_vram_args=nodraft_retry_args(rc, exe, effort_end))
    except BaseException:
        for e in engines.values():
            try:
                e.close()
            except Exception:        # noqa: BLE001
                pass
        shutil.rmtree(work, ignore_errors=True)
        raise
    return sp.SplitEngine(engines["prefill"], engines["decode"], min_tokens=spec["min_prompt_tokens"], work_dir=work,
                          arena_file=arena)


#: AMD (HIP): the rule works there (the HIP reader starts on the first picture, +5.4 s on an R9700, and stops when idle) but is OFF by default: with no shrinking
#: expert cache on HIP the room has to stay in the reserve, so the engine gets no more slots (R9700, Qwen UD-Q4_K_XL: 7,959 against 7,943 with the reader on the card).
#: "elastic": true in the vision section asks for it.  Set True if the expert cache can give its room on HIP.
VISION_ON_DEMAND_AMD_BY_DEFAULT = False
#: Owner 11-11: on NVIDIA the on-demand reader holds no room back before the first picture (apply_vision_defaults); the shrinking expert cache
#: makes it.  In the resident RAM mode, where the cache cannot resize, the engine keeps the reader's room in its reserve itself
#: (include/strata/core/reader_room.hpp).
VISION_NO_RESERVE_BY_DEFAULT = True


def vision_on_demand_by_default(cfg: dict) -> bool:
    """True when this config's GPU reader should start on demand without the config saying so: a GPU reader (vision.gpu) of the NVIDIA
    kind (or AMD's, with its CPU reader) on one GPU, no batch slots, no --kv-grow, no helper caches, and nothing in the vision section that says otherwise.  Not Intel or a Mac (not measured there), not a reader pinned to another card (vision.cuda_device)."""
    vis = cfg.get("vision")
    if not VISION_ON_DEMAND_BY_DEFAULT or not isinstance(vis, dict) or vis.get("gpu") is not True or "elastic" in vis:
        return False
    if vis.get("cuda_device") is not None or sys.platform == "darwin":
        return False
    backend = cfg.get("backend")
    if backend not in (None, "cuda", "hip"):
        return False
    args = [str(a) for a in (cfg.get("args") or [])]
    if len(gpu_list(cfg)) > 1 or parallel_slots(cfg) or any(a in args for a in _ON_DEMAND_NOT_WITH):
        return False
    name = re.split(r"[\\/:]", str(vis.get("exe") or ""))[-1].lower()      # a path or a "component:..." reference
    if backend == "hip":       # AMD: the HIP reader, with the CPU reader setup wrote beside it to read when the card has no room
        return VISION_ON_DEMAND_AMD_BY_DEFAULT and name in (vision_reader.GPU_READER, vision_reader.GPU_READER + ".exe") and bool(vis.get("cpu_exe"))
    if vision_reader.shared_memory_nvidia_host():      # a DGX Spark: the reader stays loaded (owner 2026-11-11; vision_reader.shared_memory_nvidia_host)
        return False
    return name in (vision_reader.CPU_READER, vision_reader.CPU_READER + ".exe")


def apply_vision_defaults(cfg: dict) -> dict:
    """cfg with the on-demand reader switched on (vision.elastic) where vision_on_demand_by_default says so; the same dict.
    NVIDIA (owner 11-11, "picture reader loaded on demand with no memory reserve": Ok): no room is held back for the reader before the first
    picture.  The expert cache is started in segments ("vram_elastic": the engine's --vram-elastic) and makes the reader's room when a picture
    arrives, then takes it back after vision.elastic_idle_s; the reserve setup wrote (700 MiB, or none) stays as it is.  Measured on an RTX A6000
    (UD-Q4_K_XL partly streamed): 13,526 expert slots before any picture, as with the processor reader, against 13,077 with the room held back.
    A config that says "vram_elastic" itself, and AMD (no shrinking expert cache on HIP), keep the room in the reserve as before: the reserve
    setup writes for a GPU reader that stays on the card (700 MiB, or none) becomes VISION_RESERVE_MIB; a reserve the person or the app chose
    (any other number) is kept."""
    if not vision_on_demand_by_default(cfg):
        return cfg
    cfg["vision"] = {**cfg["vision"], "elastic": True}
    if VISION_NO_RESERVE_BY_DEFAULT and cfg.get("backend") != "hip" and "vram_elastic" not in cfg:
        cfg["vram_elastic"] = True                     # the cache gives the reader its room when a picture comes; nothing is held back before
        return cfg
    args = list(cfg.get("args") or [])
    if "--vram-reserve-mib" not in args[:-1]:
        args += ["--vram-reserve-mib", str(VISION_RESERVE_MIB)]
    elif args[args.index("--vram-reserve-mib") + 1] == "700":
        args[args.index("--vram-reserve-mib") + 1] = str(VISION_RESERVE_MIB)
    cfg["args"] = args
    return cfg


def main(argv=None, *, lifecycle_event=None, interrupt_exit_code: int = 0) -> int:
    """Serve until stopped. For as long as it runs this server holds the shared activity lease of the runtime folder (the same one `strata serve` and
    a runtime host take), so `strata update` and a calibration run see an engine that was started by running this file directly - the desktop app's
    source install does - and wait for it or refuse, instead of test-starting a second copy of the model beside it (#4463). A lease that cannot be
    taken (an exclusive holder, a runtime folder that is not usable) never stops the server.

    Its console output goes through a bounded queue (serve/nonblocking_out.py, #4479 / #2481): a console that stops taking output must not hold a
    request's turn, a cancel or an admission.  STRATA_BLOCKING_CONSOLE=1 turns it off (testers)."""
    quiet_crash_env()                                  # every child inherits it: no gdb on a user's machine (strata_control/childenv.py)
    limit_own_arenas()                                 # a server started by hand: glibc arenas capped as the managed one is (childenv.arena_env)
    arm_faulthandler()                                 # #5787: a native death names its Python line in this server's own output
    lease = None
    try:
        from strata_control import runtime as _runtime
        lease = _runtime.acquire_activity_lease(shared=True)
    except Exception:                                  # noqa: BLE001 - serving never depends on the lease
        lease = None
    last_run = lastrun.begin(os.environ.get("STRATA_RUNTIME_PROFILE")) if lastrun.wants(argv) else None   # P4b: says what happened to the last run, if it did not end normally
    try:
        if os.environ.get("STRATA_BLOCKING_CONSOLE") == "1":
            return _main(argv, lifecycle_event=lifecycle_event, interrupt_exit_code=interrupt_exit_code)
        from serve.nonblocking_out import decoupled_console
        with decoupled_console():
            return _main(argv, lifecycle_event=lifecycle_event, interrupt_exit_code=interrupt_exit_code)
    finally:
        lastrun.end(last_run)
        if lease is not None:
            try:
                os.close(lease)
            except OSError:
                pass


def _autocorrect_shape_config(cfg: dict) -> None:
    """Auto-correct misconfigured Hy4-Flash or Inkling-2-Mini run configs whose `--native` / `--model` points to a
    shape-engine `.strata`/`.gguf` file while `arch` / `exe` were left as the generic `strata-engine`."""
    if not isinstance(cfg, dict):
        return
    args = cfg.get("args")
    if not isinstance(args, list):
        return
    model_val = None
    for i, tok in enumerate(args):
        if tok in ("--native", "--model") and i + 1 < len(args):
            model_val = str(args[i + 1])
            break
    if not model_val:
        return
    base = model_val.replace("\\", "/").rsplit("/", 1)[-1].lower()
    exe = str(cfg.get("exe") or "")
    exe_base = exe.replace("\\", "/").rsplit("/", 1)[-1].lower()
    generic_exe = (not exe) or exe == "component:strata-engine" or exe_base in ("strata", "strata.exe")
    if hy4_kind.is_hy4_model_path(model_val) or "--engram" in args:
        if cfg.get("arch") != hy4_kind.ARCH:
            cfg["arch"] = hy4_kind.ARCH
        if generic_exe:
            cfg["exe"] = hy4_kind.REFERENCE
    elif ("inkling" in base and (base.endswith(".strata") or base.endswith(".gguf"))) or "--dspark" in args:
        if cfg.get("arch") != inkling_kind.ARCH:
            cfg["arch"] = inkling_kind.ARCH
        if generic_exe:
            cfg["exe"] = inkling_kind.REFERENCE


def _ensure_shape_tokenizer(cfg: dict, tpath: Path) -> bool:
    """Auto-generate missing `vocab.json` on the fly locally for Hy4-Flash or Inkling-2-Mini (`.strata`/`.gguf`) models
    before failing with 'the model's tokenizer is missing'. Local-only (zero network calls on the start path)."""
    if (tpath / "vocab.json").exists():
        return True
    if not isinstance(cfg, dict):
        return False
    is_hy4 = hy4_kind.is_hy4(cfg)
    is_ink = inkling_kind.is_inkling(cfg)
    args = cfg.get("args") if isinstance(cfg.get("args"), list) else []
    model_raw = next((str(args[i + 1]) for i, tok in enumerate(args) if tok in ("--native", "--model") and i + 1 < len(args)), None)
    if not is_hy4 and not is_ink:
        if model_raw and hy4_kind.is_hy4_model_path(model_raw):
            is_hy4 = True
        elif model_raw and "inkling" in model_raw.lower():
            is_ink = True
        else:
            return False
    cwd = Path(str(cfg.get("cwd") or "."))
    model_path = None
    if model_raw:
        p = Path(model_raw)
        model_path = p if p.is_absolute() else (cwd / p)
    tools_dir = str(Path(__file__).resolve().parents[1] / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        tok_mod = __import__("hy4_tokenizer" if is_hy4 else "inkling_tokenizer")
    except Exception:                                   # noqa: BLE001
        return False
    # 1. Direct Python Section 0 read from `.strata`
    if model_path is not None and model_path.is_file() and model_path.suffix.lower() == ".strata":
        try:
            tok_mod.convert_manifest(tok_mod.read_manifest(model_path), tpath)
            if (tpath / "vocab.json").exists():
                print(f"[strata] generated missing tokenizer from {model_path.name} -> {tpath}", flush=True)
                return True
        except Exception:                               # noqa: BLE001
            pass
    # 2. Companion tokenizer.json / tokenizer.hf.json + chat_template.jinja
    search_dirs: list[Path] = []
    if model_path is not None:
        search_dirs.append(model_path.parent)
    for d in (tpath, tpath.parent):
        if d not in search_dirs:
            search_dirs.append(d)
    for base in search_dirs:
        if not base.is_dir():
            continue
        for tok_name in ("tokenizer.json", "tokenizer.hf.json"):
            tj = base / tok_name
            ct = base / "chat_template.jinja"
            if tj.is_file() and ct.is_file():
                try:
                    tok_mod.convert_files(tj, ct, tpath)
                    if (tpath / "vocab.json").exists():
                        print(f"[strata] generated missing tokenizer from {tj} -> {tpath}", flush=True)
                        return True
                except Exception:                       # noqa: BLE001
                    pass
    # 3. Export via the shape engine binary (`--export-tokenizer`)
    if model_path is not None and model_path.is_file() and model_path.suffix.lower() == ".strata":
        try:
            resolved = with_component_engine(dict(cfg))
            exe_raw = resolved.get("exe")
            exe = Path(exe_raw) if exe_raw else None
            if exe is not None and not exe.is_absolute():
                exe = cwd / exe
            if exe is not None and exe.is_file():
                raw = tpath.with_name(tpath.name + "-export")
                raw.mkdir(parents=True, exist_ok=True)
                env = quiet_crash_env(dict(os.environ))
                var = "PATH" if os.name == "nt" else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
                dirs = [str(exe.resolve().parent)] + [str(d) for d in (resolved.get("lib_dirs") or []) if Path(d).is_dir()]
                env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
                done = subprocess.run([str(exe), "--export-tokenizer", str(model_path), str(raw)],
                                      capture_output=True, text=True, errors="replace", timeout=60, env=env)
                if done.returncode == 0 and (raw / "tokenizer.hf.json").is_file() and (raw / "chat_template.jinja").is_file():
                    tok_mod.convert_files(raw / "tokenizer.hf.json", raw / "chat_template.jinja", tpath)
                    if (tpath / "vocab.json").exists():
                        print(f"[strata] exported missing tokenizer via {exe.name} -> {tpath}", flush=True)
                        return True
        except Exception:                               # noqa: BLE001
            pass
    return (tpath / "vocab.json").exists()


def _main(argv=None, *, lifecycle_event=None, interrupt_exit_code: int = 0) -> int:
    try:                                               # files made by a run as administrator still belong to the user
        from strata_control import private_files as _pf
        _pf.adopt_user_defaults()
    except Exception:
        pass
    event_sink = lifecycle_event

    def notify_lifecycle(kind: str, payload: dict) -> None:
        nonlocal event_sink
        if event_sink is None:
            return
        try:
            event_sink(kind, payload)
        except (OSError, ValueError):
            # A closed event consumer must not strand the owned engine or API.
            event_sink = None

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["mock", "strata"], default="mock")
    ap.add_argument("--config", help="strata engine config (JSON: exe, args, cwd, tokenizer, model_name), "
                                     "written by setup.py")
    ap.add_argument("--host", default=None,
                    help="the address to listen on: 127.0.0.1 = this PC only (the default), 0.0.0.0 = also other devices "
                         "on your network, :: = all IPv6 interfaces (set an API key); also \"host\" in the config")
    ap.add_argument("--script-cycle", action="store_true",
                    help="with --script given more than once: repeat the scripts by position instead of advancing "
                         "to the last one and staying there, so extra requests a harness makes (a session title, "
                         "a re-ask after a tool result) cannot shift the sequence")
    ap.add_argument("--script-delay", type=float, default=0.0,
                    help="seconds to wait between the mock engine's tokens, so a turn stays live long enough for a "
                         "client to cancel it; the mock honours cancellation between tokens")
    ap.add_argument("--script", action="append",
                    help="the mock engine's answer (default: a short greeting); given more than once, requests get "
                         "them in turn and the last one repeats")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--ready-file", help="write a JSON instance record once the API is bound and model state is known")
    ap.add_argument("--owner-fd", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--shutdown-timeout", type=float, default=None,
                    help="opt-in graceful drain deadline in seconds (0..300) before owned requests are cancelled")
    ap.add_argument("--parallel", type=int, default=None,
                    help="request this many simultaneous engine slots without changing the saved config")
    ap.add_argument("--batch-groups", default=None,
                    help="pipeline groups: auto or a positive integer, without changing the saved config")
    ap.add_argument("--max-queue", type=int, default=None,
                    help="maximum inference calls waiting beyond effective engine slots; 0 refuses overload")
    ap.add_argument("--queue-timeout", type=float, default=None,
                    help="maximum seconds waiting for the first engine turn (0.1..600; off by default)")
    ap.add_argument("--batch-telemetry", action=argparse.BooleanOptionalAction, default=None,
                    help="opt in to low-rate engine decode-window samples for /v1/telemetry")
    ap.add_argument("--offline", action="store_true",
                    help="do not fetch remote image URLs or start MCP/tool subprocesses")
    ap.add_argument("--gpu", help="the GPU to run on, as nvidia-smi numbers them, or several for a layer split "
                                  "(\"0,2\"; also \"gpu\" in the config)")
    ap.add_argument("--tokenizer", default=str(ROOT / "pack/full/tokenizer"),
                    help="pack tokenizer directory (falls back to a byte tokenizer if absent)")
    ap.add_argument("--open", action="store_true", help="open the local page in the browser once the model is ready")
    ap.add_argument("--fit-max-tokens", action="store_true",
                    help="clamp max_tokens to the remaining context instead of rejecting the request "
                         "(the default; kept so older scripts keep working)")
    ap.add_argument("--no-fit-max-tokens", action="store_true",
                    help="answer 400 when prompt + max_tokens does not fit the context, like llama.cpp "
                         "(also \"fit_max_tokens\": false in the config). A prompt that alone does not fit is always 400")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""),
                    help="require this key on /v1/* (Authorization: Bearer ... or x-api-key); also $STRATA_API_KEY. "
                         "Several keys: separate them with commas (key1,key2), as llama.cpp does")
    ap.add_argument("--no-api-key", action="store_true",
                    help="serve on the network without an API key (anyone who can reach the port can use the model; "
                         "also \"api_key\": \"none\" in the config). Warns once, never refuses")
    ap.add_argument("--mcp-config", help="a JSON file with MCP servers in Claude Desktop's format ({\"mcpServers\": "
                                         "{...}}); the web app's chat can use their tools (also \"mcp_servers\" in "
                                         "the config)")
    ap.add_argument("--lazy", action="store_true", help="start the API unloaded (the engine and the vision encoder start with the first request)")
    ap.add_argument("--api-monitor", action="store_true",
                    help="the API request monitor at /api-monitor: keeps the last 100 requests' prompts and answers in "
                         "memory (also \"api_monitor\": true in the config; off by default)")
    ap.add_argument("--strict-model", action="store_true",
                    help="reject requests whose 'model' is not the loaded model or an alias "
                         "(also \"strict_model\": true in the config or $STRATA_STRICT_MODEL=1)")
    ap.add_argument("--idle-unload", type=float, default=None, metavar="SECONDS",
                    help="unload the model after this many seconds without requests, so other programs (games, other "
                         "model servers) can use the VRAM; the next request loads it again (also \"idle_unload_s\" "
                         "in the config; default: never)")
    ap.add_argument("--min-free-vram-mib", type=int, default=None,
                    help="load an unloaded model only when this much VRAM is free, else answer 503 (also "
                         "\"min_free_vram_mib\" in the config; default: always load)")
    ap.add_argument("--before-load", help="a command run before the model is loaded again (e.g. to unload another "
                                          "server's model; also \"before_load\" in the config, a string or a list)")
    ap.add_argument("--slot-save-path", default=None, metavar="DIR",
                    help="enable POST /slots/0?action=save|restore {\"filename\": NAME} (llama-server's API): the "
                         "conversation the engine holds, to or from DIR/NAME (also \"slot_save_path\" in the config)")
    ap.add_argument("--replicas", type=int, default=None, metavar="N",
                    help="data-parallel replicas (opt-in): cut the \"gpu\" list into N equal groups and run one engine on "
                         "each, requests spread over them (also \"replicas\" in the config: a number, or "
                         "[{\"gpus\": [0, 1]}, {\"gpus\": [2, 3]}])")
    ap.add_argument("--split-serve", default=None, metavar="PREFILL,DECODE",
                    help="split serving (opt-in): one engine on card PREFILL reads the prompt, a second on card DECODE makes "
                         "the answer, the session handed over in RAM (also \"split_serve\" in the config: "
                         "{\"prefill_gpu\": 0, \"decode_gpu\": 1}; docs/SPLIT_SERVE.md)")
    launch_args = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(launch_args)
    if a.engine == "strata" and (_rt := rttime_limit_note()):   # #3238: say it before the first unexplained -9
        print("[strata] WARNING: " + _rt, flush=True)
    global OPENS_BROWSER
    OPENS_BROWSER = bool(a.open)
    # Every ordinary way of being asked to stop takes Ctrl+C's path (QUIT to the engine, then exit): SIGTERM (#96 docker
    # stop; systemd), SIGHUP (the terminal closed, ssh dropped), and on Windows Ctrl+Break and the console's close,
    # logoff and shutdown events. They are installed before anything slow starts, so one that arrives while the model
    # loads is not the default "die at once". SIGINT keeps Python's own handler (a second Ctrl+C stops waiting).
    termination = "sigint"
    finished = threading.Event()                        # set when the clean stop is over (a Windows console handler waits for it)

    def request_stop(reason: str):
        nonlocal termination
        termination = reason
        if threading.current_thread() is threading.main_thread():
            raise KeyboardInterrupt
        import _thread
        _thread.interrupt_main()
    lifeline.install_stop_handlers(request_stop, finished)
    if a.owner_fd is not None:
        try:
            owner_guard.start(a.owner_fd, grace=(a.shutdown_timeout or 0) + 40)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
    if a.shutdown_timeout is not None and not 0 <= a.shutdown_timeout <= 300:
        ap.error("--shutdown-timeout must be in 0..300 seconds")
    config_bytes = Path(a.config).read_bytes() if a.config else b""
    cfg = parse_config_text(config_bytes.decode("utf-8-sig"), a.config) if a.config else {}   # Notepad adds a BOM
    split_joined_args(cfg)
    from strata_control import retired
    _ds = retired.config_reason(cfg) if a.config else None
    if _ds:                                             # DeepSeek: this Strata no longer runs it; stop before the engine starts (retired.py)
        print(retired.server_line(_ds), flush=True)
        return 2
    if a.config and a.engine == "strata" and not hy4_kind.is_hy4(cfg):     # (the Hy4-Flash engine has no tuning knobs)
        cfg = apply_saved_tuning(cfg)
    attestation_key = os.environ.pop("STRATA_OWNER_ATTEST_KEY", None)
    launch_attestation = None
    if a.owner_fd is not None:
        if attestation_key is None or not re.fullmatch(r"[0-9a-f]{64}", attestation_key):
            ap.error("owned server needs a private launch attestation key")
        launch_attestation = launch_identity.attestation(bytes.fromhex(attestation_key),
                                                          config_bytes, launch_args)
    if "batch_telemetry" in cfg and type(cfg["batch_telemetry"]) is not bool:
        ap.error("batch_telemetry must be true or false")
    try:
        ready_path = ready_file.preflight(a.ready_file) if a.ready_file else None
    except ValueError as e:
        ap.error(str(e))
    max_queue = a.max_queue if a.max_queue is not None else cfg.get("max_queue")
    if max_queue is not None and (isinstance(max_queue, bool) or not isinstance(max_queue, int)
                                  or not 0 <= max_queue <= 4096):
        ap.error("--max-queue / max_queue must be a whole number in 0..4096")
    queue_timeout = a.queue_timeout if a.queue_timeout is not None else cfg.get("queue_timeout")
    if queue_timeout is not None and (isinstance(queue_timeout, bool) or not isinstance(queue_timeout, (int, float))
                                      or not math.isfinite(queue_timeout) or not 0.1 <= queue_timeout <= 600):
        ap.error("--queue-timeout / queue_timeout must be finite seconds in 0.1..600")
    if queue_timeout is not None and max_queue is None:
        ap.error("--queue-timeout requires --max-queue / max_queue to bound pending calls")
    if a.parallel is not None:
        if a.parallel < 1 or a.parallel > 1024:
            ap.error("--parallel must be in 1..1024; the engine may reduce it to the supported count")
        if parallel_flag_conflicts(cfg):
            ap.error("--parallel conflicts with an explicit --batch/--slots in the saved engine args")
        cfg["parallel"] = a.parallel
    if a.batch_groups is not None:
        if a.batch_groups != "auto" and (not a.batch_groups.isdecimal() or int(a.batch_groups) < 1):
            ap.error("--batch-groups must be auto or a positive integer")
        if not cfg or "args" not in cfg:
            ap.error("--batch-groups needs an engine config")
        if "--batch-groups" in cfg["args"]:
            ap.error("--batch-groups conflicts with the saved engine args")
        cfg["args"] = [*cfg["args"], "--batch-groups", a.batch_groups]
    lift_reasoning_effort_arg(cfg)
    if a.replicas is not None:
        cfg["replicas"] = a.replicas
    if a.split_serve is not None:
        pair = [x.strip() for x in a.split_serve.split(",")]
        if len(pair) != 2 or not all(x.isdecimal() for x in pair):
            ap.error("--split-serve takes two card numbers: PREFILL,DECODE (for example 0,1)")
        cfg["split_serve"] = {**(cfg.get("split_serve") if isinstance(cfg.get("split_serve"), dict) else {}),
                              "prefill_gpu": int(pair[0]), "decode_gpu": int(pair[1])}
    if a.gpu is not None:
        cfg["gpu"] = int(a.gpu) if a.gpu.strip().isdigit() else a.gpu
    a.host = a.host or cfg.get("host") or "127.0.0.1"   # issue #26: the run scripts pass no --host, the config can
    supplied_args = argv if argv is not None else sys.argv[1:]
    if ("STRATA_API_KEY" in os.environ and not os.environ["STRATA_API_KEY"].strip()) or \
            any((x == "--api-key" and i + 1 < len(supplied_args) and not supplied_args[i + 1].strip()) or
                x.strip() == "--api-key=" for i, x in enumerate(supplied_args)):
        print("[strata] an API key was given but it is empty: set a key, or leave it out", file=sys.stderr)
        return 2
    raw_key = a.api_key or cfg.get("api_key", "")
    no_key_chosen = bool(getattr(a, "no_api_key", False)) or explicit_no_key(raw_key)
    if explicit_no_key(raw_key):
        raw_key = ""
    try:
        preflight_api_key = api_key_of(raw_key)
    except ValueError as e:
        print(f"[strata] {e}: set a key, or leave it out", file=sys.stderr)
        return 2
    # A key is recommended on a network, never required (owner decision 2026-10-20): the banner below warns once.
    try:                                                # before the minutes of loading: is the port free?
        Server((a.host, a.port), BaseHTTPRequestHandler).server_close()
    except OSError as e:
        ap.error(listen_problem(a.host, a.port, e))
    if cfg:
        _autocorrect_shape_config(cfg)
    if cfg.get("tokenizer"):
        a.tokenizer = cfg["tokenizer"]
    tok = ByteTokenizer()
    tcfg = {}
    tpath = Path(a.tokenizer)
    if a.engine == "strata" and not (tpath / "vocab.json").exists():
        _ensure_shape_tokenizer(cfg, tpath)
    if a.engine == "strata" and not (tpath / "vocab.json").exists():
        ap.error(f"the model's tokenizer is missing ({tpath / 'vocab.json'}); run setup again")
    if (tpath / "vocab.json").exists():
        import strata_tokenizer as ST
        vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tpath / "token_type.json").read_text())
        try:                                            # the pack's pre-tokenizer name and special ids
            tcfg = json.loads((tpath / "tokenizer.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        try:
            tok = ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)
        except ST.UnknownPreTokenizer as e:             # a model with its own splitting rule: stop here, never split with another's
            ap.error(f"the model's tokenizer ({tpath}): {e}")
    hub = hub_from_config(cfg, a.mcp_config)            # before the minutes of loading: a bad entry stops here
    if a.offline and (hub is not None or cfg.get("before_load") or a.before_load):
        ap.error("--offline requires MCP servers and before_load commands to be disabled in this profile")
    if a.offline:
        # The same decision, reported through the policy module so the two cannot drift: a profile
        # that permits a network source is refused here rather than on its first request.
        try:
            from strata_control import offline as offline_policy
            blocked = offline_policy.ensure_strictly_offline(cfg)
        except ImportError:
            blocked = []
        if blocked:
            ap.error("--offline cannot be strict for this profile: " + "; ".join(blocked))
    lazy = False
    if a.engine == "strata":
        if not cfg:
            ap.error("--engine strata needs --config")
        cfg = apply_vision_defaults(with_component_engine(cfg))
        vision = None
        # The engine's own environment. It carries the offline policy too: an engine that fetched a
        # missing shard or a plugin would break the promise the flag makes.
        env = child_env(cfg, offline=a.offline)
        configure_batch_telemetry_env(env, cfg, a.batch_telemetry)
        sampling_defaults = sampling_defaults_from_config(cfg)
        if sampling_defaults:
            pretty = ", ".join(f"{k}={v}" for k, v in sampling_defaults.items())
            print(f"[strata] sampling defaults from the config: {pretty}", flush=True)
        lazy = a.lazy or cfg.get("lazy_load") is True
        for line in card_roles_lines(cfg):             # the card roles: said first; every card Off stops here, before anything touches a card
            print(line, flush=True)
        if cfg.get("vision"):
            elastic = cfg["vision"].get("elastic") is True
            print("the vision encoder starts with the first picture (elastic) ..." if elastic else
                  "loading the vision encoder ..." if not lazy else
                  "vision encoder unloaded; it starts with the model ...", flush=True)
            # relative paths are the config's cwd's, as for the engine below
            vcfg = {k: (os.path.abspath(os.path.join(cfg.get("cwd") or ".", v))
                        if k in ("exe", "mmproj", "model") and isinstance(v, str) and not os.path.isabs(v) else v)
                    for k, v in cfg["vision"].items() if not (k == "exe" and engine_component.is_reference(v))}
            if engine_component.is_reference(cfg["vision"].get("exe")):     # the engine from the component folder
                vcfg["exe"] = _component_exe(cfg["vision"]["exe"], cfg)
            pw_cap = _configured_pool_workers(cfg)
            if pw_cap is not None and not vcfg.get("threads"):              # #6497: inherit explicit CPU thread cap
                vcfg["threads"] = pw_cap
            if vcfg.get("elastic") is True and layer_split_of(cfg):
                # the engine turns --vram-elastic off on a layer split (a cache per GPU): the encoder still starts
                # with the first picture and stops when idle, but its VRAM has to be free already
                print('[strata] images: on a layer split the expert cache cannot make room for the image encoder (the '
                      'engine turns --vram-elastic off); keep its VRAM free with "--vram-reserve-mib" on the '
                      "encoder's GPU, or the first picture may not get it", flush=True)
            vision = Vision(vcfg, log=appendlog.open_append(cfg["log"]) if cfg.get("log") else None,
                            env=vision_env(cfg, env), lazy=lazy)
        print("model unloaded; the first request loads it ..." if lazy else
              "loading the model (the first start takes a minute or two) ...", flush=True)
        warn_budget_over_ram(engine_args(cfg, announce=False) if "args" in cfg else [])      # #1080
        note = card_note(cfg, env)
        if note:
            print(note, flush=True)
        arch_line = engine_arch_note(cfg)
        if arch_line:
            print(arch_line, flush=True)
        igpu_line = igpu_split_warning(cfg)
        if igpu_line:
            print(igpu_line, flush=True)
        if layer_split_of(cfg):
            try:
                split = layer_split_value(cfg)          # #644: before the (minutes-long) start
            except ValueError as e:
                raise SystemExit(f"[strata] config {e}")
            print(f"[strata] layer split across GPUs {gpu_list(cfg)} ({split})", flush=True)
        elif len(gpu_list(cfg)) > 1:
            print(f"[strata] GPUs {gpu_list(cfg)}: the later card(s) serve as the peer expert tier (--peer-device)",
                  flush=True)
        # a relative "exe" is the config's cwd's: Windows' CreateProcess resolves "engine/strata.exe" against nothing
        # it is told about (WinError 2), so it is made absolute here
        if engine_component.is_reference(cfg["exe"]):      # "component:strata-engine": the engine package for this PC
            exe = _component_exe(cfg["exe"], cfg)
        else:
            exe = cfg["exe"] if os.path.isabs(cfg["exe"]) else os.path.abspath(os.path.join(cfg.get("cwd") or ".", cfg["exe"]))
        try:
            silence = engine_silence_s(cfg)             # #481: checked before the (minutes-long) start
            apply_ready_timeout(cfg)
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        try:
            effort_end = effort_end_args(cfg, exe, tok)  # #458
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        from serve import replicas as rp
        try:
            groups = rp.gpu_groups(cfg)
        except rp.ReplicaConfigError as e:
            raise SystemExit(f"[strata] config {e}")
        from serve import split as split_mod
        try:
            split = split_mod.split_spec(cfg)
            if split:
                split_mod.check_cfg(cfg, split)
        except split_mod.SplitConfigError as e:
            raise SystemExit(f"[strata] config {e}")
        if groups:
            engine = start_replicas(cfg, groups, exe, effort_end or [], lazy, offline=a.offline)
        elif split:
            engine = start_split(cfg, split, exe, effort_end or [], lazy, offline=a.offline)
        else:
            engine = start_engine_or_fallback(cfg, exe, effort_end, env, lazy, tok, offline=a.offline, batch_telemetry=a.batch_telemetry)
        engine.silence_s = silence                      # an attribute of its own: restart() keeps it
        warn_tight_ram(engine.info.get("arena_mib"))
        note = desktop_vram_note(cfg.get("backend"), engine.info.get("vram_free_mib"), engine.spawn[1],
                                 linux_desktop()) if not groups and not split else None
        if note:                                        # #560 #516: before --open starts a browser on that card
            print(note, flush=True)
    else:
        effort_end = None
        engine, vision, sampling_defaults = MockEngine(tok, a.script or [
            "Thinking about it.</think>\n\nHello from the mock engine."],
            cycle=getattr(a, "script_cycle", False),
            delay_s=getattr(a, "script_delay", 0.0) or 0.0), None, {}
    # the model's own chat template (exported with its tokenizer), else the original model's
    tpl = tpath / "chat_template.jinja"
    svc = Service(engine, tok, ChatTemplate(tpl if tpl.exists() else ROOT / "serve/chat_template.jinja"),
                  model_name=default_model_name(cfg), vision=vision,
                  sampling_defaults=sampling_defaults,
                  fit_max_tokens=a.fit_max_tokens or (not a.no_fit_max_tokens and cfg.get("fit_max_tokens") is not False))
    svc.launch_attestation = launch_attestation
    svc.offline = a.offline
    svc.max_queue = max_queue
    svc.queue_timeout_s = float(queue_timeout) if queue_timeout is not None else None
    if a.engine == "strata":                            # the model card's sampling when nothing else is said (see MODEL_CARD_SAMPLING)
        svc.model_arch = cfg.get("arch") or "qwen4exp"
        svc.set_family("hy4" if svc.hy4 else None)
        if svc.hy4:
            svc._force_text = hy4_dialect.force_text
        requested = cfg.get("parallel")
        if requested is None:
            args = cfg.get("args", [])
            for flag in ("--batch", "--slots"):
                if flag in args and args.index(flag) + 1 < len(args):
                    try:
                        requested = int(args[args.index(flag) + 1])
                    except ValueError:
                        pass
        per_replica = requested if isinstance(requested, int) and not isinstance(requested, bool) and requested > 0 else 1
        svc.requested_slots = per_replica * len(groups) if groups else per_replica
    svc.answer_room = cfg.get("answer_room") is not False
    if cfg.get("inkling_thinking_budget") is not None:     # Inkling-2-Mini's default thinking budget (tokens, 0 = none); else STRATA_INKLING_THINKING_BUDGET
        v = cfg["inkling_thinking_budget"]
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise SystemExit(f"[strata] config \"inkling_thinking_budget\" must be a whole number of tokens (0: none), not {v!r}")
        svc.inkling_thinking_budget = v
    svc.reasoning_close_retry = cfg.get("reasoning_close_retry") is True    # #1053: opt-in, off by default
    if cfg.get("images"):                               # image generation (serve/images.py), set up by setup --images
        icfg = {k: (os.path.abspath(os.path.join(cfg.get("cwd") or ".", v))
                    if k in ("exe", "diffusion_model", "vae", "llm", "out_dir") and isinstance(v, str)
                    and not os.path.isabs(v) else v)
                for k, v in cfg["images"].items()}
        try:
            svc.images = images_api.ImageGen(icfg, log=appendlog.open_append(cfg["log"]) if cfg.get("log") else None)
        except ValueError as e:
            raise SystemExit(f"[strata] config: {e}")
        gone = svc.images.missing()
        if gone:
            print(f"[strata] image generation: missing {', '.join(gone)} - requests fail until setup --images has run "
                  "again", flush=True)
        noise_problem = svc.images.noise_problem()
        if noise_problem:
            print(f"[strata] image generation: {noise_problem}", flush=True)
        vae_problem = svc.images.vae_problem()
        if vae_problem:
            print(f"[strata] image generation: {vae_problem}", flush=True)
        print(f"[strata] image generation: {svc.images.model} {icfg.get('quant') or ''} on "
              f"{icfg.get('backend') or '?'} (POST /v1/images/generations; GPU policy {svc.images.policy})", flush=True)
    if cfg.get("prevent_sleep") is not None and not isinstance(cfg["prevent_sleep"], bool):
        raise SystemExit(f"[strata] config \"prevent_sleep\" must be true or false, not {cfg['prevent_sleep']!r}")
    from serve.keepawake import KeepAwake
    svc.keep_awake = KeepAwake.create(cfg.get("prevent_sleep") is True)      # #1727: opt-in, Windows only
    if cfg.get("reasoning_effort") is not None:         # #1641: the thinking level of requests that name none
        try:
            svc.config_effort = clean_shared_defaults({"reasoning_effort": cfg["reasoning_effort"]}, hy4=hy4_kind.is_hy4(cfg)).get("reasoning_effort")
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        if svc.config_effort:
            print(f"[strata] thinking level: {svc.config_effort} (reasoning_effort in the config; a request, or the "
                  "Chat settings shared with other apps, can set its own)", flush=True)
    svc.codex_thread_titles = cfg.get("codex_thread_titles") is True    # #923: opt-in, off by default
    svc.codex_compaction_cache = cfg.get("codex_compaction_cache") is True   # #924: opt-in, off by default
    try:
        svc.set_aliases(cfg.get("aliases"))             # #297: other names the model answers to
    except ValueError as e:
        raise SystemExit(f"[strata] config {e}")
    if svc.aliases:
        print(f"[strata] model aliases: {', '.join(svc.aliases)}", flush=True)
    if a.strict_model or cfg.get("strict_model") is True:
        svc.strict_model = True
    svc.api_key = preflight_api_key
    # #5004 (the config's "api_keys"): keys with their own admission limits.  They are keys the server accepts,
    # so they join the accepted set before the telemetry-key clash check below sees it.  The reservations are
    # sanity-checked against the "parallel" asked for here; KeyLimits follows the engine's own batch after that.
    par = a.parallel if getattr(a, "parallel", None) is not None else cfg.get("parallel")
    key_slots = par if isinstance(par, int) and not isinstance(par, bool) and par >= 1 else 1
    try:
        key_policies = api_key_limits_of(cfg.get("api_keys"), key_slots)
    except ValueError as e:
        raise SystemExit(f"[strata] config {e}")
    if key_policies:
        if no_key_chosen:
            raise SystemExit('[strata] config "api_keys" and the choice to serve without a key contradict each other')
        merged = key_list(svc.api_key)
        merged += [p["key"] for p in key_policies if p["key"] not in merged]
        svc.api_key = merged
        svc.key_limits = KeyLimits(key_policies, lambda: max(1, int(getattr(svc.engine, "batch", 0) or 0)))
        print(f"[strata] api_keys: {len(key_policies)} key(s) with their own limits "
              f"({', '.join(p['name'] for p in key_policies)}); GET /status shows what each holds", flush=True)
    try:
        svc.telemetry_key = api_key_of(os.environ.get("STRATA_TELEMETRY_KEY") or cfg.get("telemetry_read_key", ""))
    except ValueError as e:
        print(f"[strata] invalid telemetry_read_key: {e}", file=sys.stderr)
        return 2
    if set(key_list(svc.api_key)) & set(key_list(svc.telemetry_key)):
        print("[strata] telemetry_read_key must differ from every inference API key", file=sys.stderr)
        return 2
    if not svc.api_key and "api_key" in cfg and not no_key_chosen:             # #569: written, but empty: the server has none (a warning, not a stop)
        print("[strata] the api_key in the config is empty: this server has no API key (anyone who can reach it can "
              "use it); set one, or leave api_key out", file=sys.stderr, flush=True)
    svc.cors_origins = origins_of(cfg.get("cors_origins"), "cors_origins", wildcard=True)
    svc.trusted_origins = origins_of(cfg.get("trusted_origins"), "trusted_origins", wildcard=False)
    try:
        svc.allowed_hosts = allowed_hosts_of(cfg.get("allowed_hosts"), os.environ.get("STRATA_ALLOWED_HOSTS", ""))
    except ValueError as e:
        raise SystemExit(f"[strata] config {e}")
    if svc.allowed_hosts:
        print("[strata] Host check off: any name reaches this server (allowed_hosts \"*\")" if "*" in svc.allowed_hosts
              else f"[strata] also answers to the host names {', '.join(svc.allowed_hosts)} (allowed_hosts)", flush=True)
    svc.api_monitor = a.api_monitor or cfg.get("api_monitor") is True
    if svc.api_monitor:
        print("[strata] API request monitor on (/api-monitor): the last 100 requests' prompts and answers are kept in "
              "memory" + ("" if svc.api_key else "; anyone who can reach this server can read them (no API key)"),
              flush=True)
    if svc.cors_origins:
        print(f"[strata] CORS on /v1/* for {', '.join(svc.cors_origins)}"
              + ("" if svc.api_key or "*" not in svc.cors_origins else
                 " - WARNING: any web page may use the model (no API key)"), flush=True)
    svc.idle_unload_s = a.idle_unload if a.idle_unload is not None else float(cfg.get("idle_unload_s") or 0)
    svc.min_free_vram_mib = a.min_free_vram_mib if a.min_free_vram_mib is not None else \
        int(cfg.get("min_free_vram_mib") or 0)
    svc.before_load = a.before_load or cfg.get("before_load") or None
    try:
        svc.before_load_timeout_s = float(cfg.get("before_load_timeout_s") or svc.before_load_timeout_s)
    except (TypeError, ValueError):
        print("[strata] \"before_load_timeout_s\" in the config is not a number; using "
              f"{svc.before_load_timeout_s:g} s", flush=True)
    svc.before_load_ignore_failure = cfg.get("before_load_ignore_failure") is True
    svc.before_load_strict = cfg.get("before_load_strict") is True
    # --slot-save-path is relative to the working directory the server was started in; the config's slot_save_path
    # to the config's "cwd" (the engine's folder) when it has one.  Either way the engine gets an absolute path.
    if a.slot_save_path or cfg.get("slot_save_path"):
        try:
            svc.slot_save_path = slot_save_dir(a.slot_save_path) if a.slot_save_path else \
                slot_save_dir(cfg["slot_save_path"], cfg.get("cwd"))
        except (ValueError, OSError) as e:
            raise SystemExit(f"[strata] {e}")
        print(f"[strata] slot save/restore on: {svc.slot_save_path} (session files are kept until deleted: "
              "about 1.2 GB per 63K-token conversation; restore only files this server wrote)", flush=True)
    mode = str(cfg.get("anthropic_thinking") or "model")   # #278: "on_request" = only when the request asks
    if mode not in ("model", "on_request"):
        raise SystemExit(f"[strata] config anthropic_thinking must be \"model\" or \"on_request\", not {mode!r}")
    svc.anthropic_think_unasked = mode == "model"
    svc.tool_call_recovery = tool_call_recovery_setting(cfg)
    svc.undeclared_tool_calls = undeclared_tool_calls_setting(cfg)
    retry = cfg.get("tool_markup_retry", True)         # (a): false (or STRATA_TOOL_MARKUP_RETRY=0) sends the markup as text
    if not isinstance(retry, bool):
        raise SystemExit(f"[strata] config \"tool_markup_retry\" must be true or false, not {retry!r}")
    svc.tool_markup_retry = svc.tool_markup_retry and retry
    etu = cfg.get("env_to_user", False)                # opt-in: see frontend._env_to_user
    if not isinstance(etu, bool):
        raise SystemExit(f"[strata] config \"env_to_user\" must be true or false, not {etu!r}")
    import serve.frontend as _frontend
    _frontend.ENV_TO_USER = etu
    rs = cfg.get("repeat_stop_tokens", REPEAT_STOP_TOKENS)   # #606: opt-out with 0
    if isinstance(rs, bool) or not isinstance(rs, int) or rs < 0:
        raise SystemExit(f"[strata] config \"repeat_stop_tokens\" must be a whole number >= 0 (0 = off), not {rs!r}")
    svc.repeat_stop_tokens = rs
    svc.effort_end = bool(effort_end)                   # #458: "effort_position": "end" with an engine that has it
    if cfg.get("reasoning_budget_tokens") is not None:  # #123: a default thinking budget for every request
        svc.reasoning_budget_configured = True
        try:
            svc.reasoning_budget_tokens = cfg["reasoning_budget_tokens"]
            budget = svc.reasoning_budget({})
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        if budget:
            print(f"[strata] thinking budget: {budget} tokens (reasoning_budget_tokens; a request can set its own)",
                  flush=True)
    recovery = cfg.get("reasoning_loop_recovery", False)   # #728: false (default) | "stop" | "recover" (true)
    if recovery is True:
        recovery = "recover"
    if recovery is not False and recovery not in ("stop", "recover"):
        raise SystemExit("[strata] config \"reasoning_loop_recovery\" must be false, \"stop\" or \"recover\", "
                         f"not {recovery!r}")
    svc.reasoning_loop_recovery = recovery
    svc.gpu_index = monitored_gpu(gpu_list(cfg), env if a.engine == "strata" else None, cfg.get("backend"))   # the card the engine runs on (#51, #4640)
    svc.gpu_indices = gpu_list(cfg)                     # ... or every card of a layer split (issue #112)
    if a.engine == "strata" and groups:                 # replicas: every card of every group
        svc.gpu_indices = [c for g in groups for c in g]
        svc.gpu_index = svc.gpu_indices[0]
        svc.replica_gpu_groups = groups
    if a.engine == "strata" and split:                  # split serving: the two cards (the prefill card first)
        svc.gpu_indices = [split["prefill_gpu"], split["decode_gpu"]]
        svc.gpu_index = split["decode_gpu"]
    if a.engine == "strata" and not groups and not split:    # helper / peer cards too, so the Monitor samples them
        roles = {}
        base = list(svc.gpu_indices or [svc.gpu_index])
        for card, role in helper_gpus(cfg, env):
            if card not in base:
                base.append(card)
                roles[card] = role
        if len(base) > len(svc.gpu_indices or []):
            svc.gpu_indices = base
            print("[strata] Monitor: watching " + ", ".join(
                f"GPU {c} ({roles.get(c, 'main')})" for c in base) + " (as nvidia-smi numbers them)", flush=True)
    svc.terminal_visibility_order = visible_device_order(env, cfg.get("backend")) if a.engine == "strata" else []
    svc.terminal_topology_mode = ("replicas (configured)" if a.engine == "strata" and groups
                                  else "split serve (configured)" if a.engine == "strata" and split
                                  else "layer split (configured)" if a.engine == "strata" and layer_split_of(cfg)
                                  else "single GPU or CPU" if a.engine == "strata" else "mock")
    svc.terminal_configured_split = (layer_split_value(cfg) if a.engine == "strata" and layer_split_of(cfg)
                                     else None)
    svc.backend = cfg.get("backend")                    # "hip": the AMD cards' readings come from sysfs (#301)
    if a.config:
        svc.config_path = a.config                      # #564: the web page's Settings view
    if a.config:                                        # the Chat settings shared with other apps, from last time
        svc.shared_path = str(Path(a.config).with_suffix("")) + ".shared-settings.json"
        svc.shared = load_shared_settings(svc.shared_path, hy4=svc.hy4)   # #4933: one unknown key no longer clears the rest
        if svc.shared:
            print("[strata] other apps use the Chat settings: " +
                  ", ".join(f"{k}={v}" for k, v in svc.shared.items()), flush=True)
    if hub is not None:
        import atexit
        svc.mcp = hub
        # #2343: tool images reach a model that reads pictures unless the config says "image_results": false
        hub.settings["image_results"] = resolve_image_results(hub.settings, svc.has_vision())
        if hub.settings["image_results"]:
            hub.media_dir = Path(a.config).with_suffix(".tool-media") if a.config else Path.cwd() / ".tool-media"
        print(f"[strata] starting {len(hub.servers)} MCP server{'s' * (len(hub.servers) != 1)} for the web app's "
              f"chat: {', '.join(hub.servers)}", flush=True)
        hub.start()
        atexit.register(hub.close)                      # the servers Strata started end with it
    httpd = serve(svc, host=a.host, port=a.port)
    bound_port = httpd.server_address[1]               # --port 0: report the port the OS actually assigned
    svc.start_idle_unload()
    svc.start_vision_idle()
    here = "[::1]" if a.host == "::" else "127.0.0.1" if a.host in ("0.0.0.0", "") else f"[{a.host}]" if ":" in a.host else a.host
    print(f"ready: http://{here}:{bound_port}/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, "
          f"context {engine.max_context} tokens{', images on' if svc.has_vision() else ''}"
          f"{', API key required' if svc.api_key else ''})", flush=True)
    print(f"       open http://{here}:{bound_port}/ in a browser to chat; close this window to stop the model", flush=True)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        # issue #26: reachable from other devices - say at which address, and what can still block it
        ips = [f"[{a.host}]"] if ":" in a.host and a.host != "::" else [] if a.host == "::" else lan_addresses()
        for ip in ips:
            print(f"       from other devices: http://{ip}:{bound_port}/   (API: http://{ip}:{bound_port}/v1)", flush=True)
        if not ips:
            address = "[<this PC's IPv6 address>]" if a.host == "::" else "<this PC's IP address>"
            print(f"       from other devices: http://{address}:{bound_port}/", flush=True)
        if not svc.api_key:
            print("       WARNING: " + open_server_warning(bound_port, tools=svc.mcp is not None),
                  flush=True)
        if os.name == "nt":
            print("       nothing arrives? Windows Firewall blocks it until allowed: accept its prompt for Python, or run "
                  "in an admin PowerShell:\n         New-NetFirewallRule -DisplayName \"Strata " + str(bound_port) + "\" "
                  "-Direction Inbound -Protocol TCP -LocalPort " + str(bound_port) + " -Action Allow -Profile Private\n"
                  "       (and set this network to Private in Windows' network settings)", flush=True)
    if a.open and cfg.get("open_browser") is not False:   # #609: the config's "open_browser": false wins (an older
        import webbrowser                                  # run-<model>.bat still passes --open)
        # on its own thread: webbrowser.open waits for the browser launcher, which on Linux can run for as long as
        # the browser does, and the main thread must be free to take Ctrl+C and SIGTERM (#1872)
        threading.Thread(target=webbrowser.open, daemon=True, name="strata-open-browser",
                         args=(f"http://{here}:{bound_port}/",)).start()
    if ready_path is not None or event_sink is not None:
        ready_host = f"[{here}]" if ":" in here and not here.startswith("[") else here
        record = {"schema_version": 1, "instance_id": svc.instance_id, "pid": os.getpid(),
                  "api_url": f"http://{ready_host}:{bound_port}", "bound_host": a.host, "bound_port": bound_port,
                  "model_id": svc.model, "model_state": "unloaded" if lazy else "ready",
                  "api_state": "ready", "created_at": time.time()}
    interrupted = False
    drained = True
    try:
        # inside the try: a SIGTERM that arrives the moment the ready file is visible (a supervisor that reads it and
        # stops the server at once) takes the clean stop below, not an uncaught KeyboardInterrupt with the file left behind
        if ready_path is not None:
            try:
                ready_file.publish(ready_path, record)
            except OSError as e:
                print(f"[strata] could not write ready file ({type(e).__name__})", file=sys.stderr)
                return 6                                # the finally below closes everything once
        if event_sink is not None:
            notify_lifecycle("ready", record)
        while True:
            time.sleep(1)                               # Windows never delivers Ctrl+C to an untimed Event.wait()
    except KeyboardInterrupt:
        interrupted = True
        notify_lifecycle("stopping", {"reason": termination, "instance_id": svc.instance_id})
        print("\n[strata] stopping (Ctrl+C again to end the engine at once) ...", flush=True)
    finally:
        cancelled = 0
        if a.shutdown_timeout is not None:
            pending = svc.begin_drain()
            notify_lifecycle("draining", {"instance_id": svc.instance_id,
                                          "pending_requests": pending,
                                          "deadline_s": a.shutdown_timeout})
            if pending:
                print(f"[strata] draining {pending} inference call(s) for up to {a.shutdown_timeout:g} s ...",
                      flush=True)
            pending = svc.wait_for_drain(a.shutdown_timeout)
            if pending:
                cancelled = svc.cancel_owned_requests()
                print(f"[strata] drain deadline expired; cancelling {cancelled} owned request(s) ...", flush=True)
                svc.wait_for_drain(5)
                drained = False
        else:
            # #6055: without --shutdown-timeout there is no wait, but a request that arrives while the engine closes is
            # still refused (503 server_draining, retry elsewhere) instead of being accepted and cut off with 0 tokens
            svc.begin_drain()
        closers =[httpd.shutdown, getattr(engine, "close", None), vision.shutdown if vision else None,
                   hub.close if hub is not None else None]
        try:
            for close in filter(None, closers):
                try:
                    close()
                except KeyboardInterrupt:               # a second Ctrl+C: don't wait for the engine to free its memory
                    if getattr(engine, "proc", None):
                        engine.proc.kill()
                        try:
                            from serve import cache_drop
                            cache_drop.drop_after_exit(engine)       # opt-in (STRATA_DROP_CACHE_ON_EXIT=1), default unchanged
                        except Exception as e:  # noqa: BLE001 - never block the exit
                            print(f"[strata] could not release the model files' cache: {e}", flush=True)
        finally:
            if ready_path is not None:
                ready_file.remove_if_owned(ready_path, svc.instance_id)
            finished.set()
        if interrupted:
            print("[strata] stopped", flush=True)
    exit_code = (8 if not drained else
                 interrupt_exit_code if interrupted and termination == "sigint" else 0)
    notify_lifecycle("stopped", {"instance_id": svc.instance_id, "reason": termination if interrupted else "normal",
                                 "drained": drained, "cancelled_requests": cancelled,
                                 "exit_code": exit_code})
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
