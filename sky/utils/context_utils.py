"""Utilities for SkyPilot context."""
import asyncio
import contextvars
import functools
import io
import multiprocessing
import os
import subprocess
import sys
import typing
from typing import Any, Callable, IO, Optional, Tuple, TypeVar

from typing_extensions import ParamSpec

from sky import sky_logging
from sky.utils import context
from sky.utils import subprocess_utils

StreamHandler = Callable[[IO[Any], IO[Any]], str]


# TODO(aylei): call hijack_sys_attrs() proactivly in module init at server-side
# once we have context widely adopted.
def hijack_sys_attrs():
    """hijack system attributes to be context aware

    This function should be called at the very beginning of the processes
    that might use sky.utils.context.
    """
    # Modify stdout and stderr of unvicorn process to be contextually aware,
    # use setattr to bypass the TextIO type check.
    setattr(sys, 'stdout', context.Stdout())
    setattr(sys, 'stderr', context.Stderr())
    # Reload logger to apply latest stdout and stderr.
    sky_logging.reload_logger()
    # Hijack os.environ with ContextualEnviron to make env variables
    # contextually aware.
    setattr(os, 'environ', context.ContextualEnviron(os.environ))
    # Hijack subprocess.Popen to pass the contextual environ to subprocess
    # by default.
    setattr(subprocess, 'Popen', context.Popen)


def passthrough_stream_handler(in_stream: IO[Any], out_stream: IO[Any]) -> str:
    """Passthrough the stream from the process to the output stream"""
    from collections import deque
    import sys
    import time as time_module
    line_count = 0
    bytes_written = 0
    buffer = []
    buffer_size = 0
    MAX_BUFFER_SIZE = 64 * 1024  # 64KB buffer before flush
    start_time = time_module.time()

    # Keep track of last 20 lines read from kubectl for debugging
    last_lines_from_kubectl = deque(maxlen=20)

    print(
        f'[DEBUG passthrough_stream_handler] Starting at {start_time}, out_stream={out_stream}',
        file=sys.stderr,
        flush=True)

    wrapped = io.TextIOWrapper(in_stream,
                               encoding='utf-8',
                               newline='',
                               errors='replace',
                               write_through=True)

    def flush_buffer():
        """Helper to flush accumulated lines"""
        nonlocal buffer_size
        if buffer:
            try:
                out_stream.writelines(buffer)
                # Don't call flush() here - let OS buffer handle it for speed
                # Only flush at the very end
                buffer.clear()
                buffer_size = 0
            except Exception as e:
                print(
                    f'[DEBUG passthrough_stream_handler] flush_buffer failed: {e}',
                    file=sys.stderr,
                    flush=True)
                raise

    try:
        while True:
            try:
                line = wrapped.readline()
            except Exception as e:
                print(
                    f'[DEBUG passthrough_stream_handler] readline() failed after {line_count} lines: {e}',
                    file=sys.stderr,
                    flush=True)
                raise

            if line:
                line_count += 1
                line_size = len(line)
                bytes_written += line_size

                # Keep track of last lines from kubectl
                last_lines_from_kubectl.append(
                    line.rstrip('\n')[:100])  # Store first 100 chars

                # Add to buffer
                buffer.append(line)
                buffer_size += line_size

                # Flush when buffer is full
                if buffer_size >= MAX_BUFFER_SIZE:
                    flush_buffer()

                if line_count % 10000 == 0:
                    print(
                        f'[DEBUG passthrough_stream_handler] Processed {line_count} lines, {bytes_written} bytes',
                        file=sys.stderr,
                        flush=True)
            else:
                break
    except Exception as e:
        print(
            f'[DEBUG passthrough_stream_handler] Exception after {line_count} lines: {type(e).__name__}: {e}',
            file=sys.stderr,
            flush=True)
        raise
    finally:
        # Flush remaining buffer and force write to disk at the end
        flush_buffer()
        out_stream.flush()  # Final flush to ensure all data is written
        elapsed = time_module.time() - start_time
        print(
            f'[DEBUG passthrough_stream_handler] Finished in {elapsed:.2f}s. Total lines: {line_count}, bytes: {bytes_written}',
            file=sys.stderr,
            flush=True)

        # Print last 20 lines received from kubectl
        print(
            f'[DEBUG passthrough_stream_handler] Last 20 lines received from kubectl (in_stream):',
            file=sys.stderr,
            flush=True)
        for i, line in enumerate(last_lines_from_kubectl, 1):
            print(f'  kubectl[{i}] {line}', file=sys.stderr, flush=True)

        # Print last few lines that were written to help debug truncation
        if hasattr(out_stream, 'name'):
            try:
                import subprocess
                result = subprocess.run(['tail', '-10', out_stream.name],
                                        capture_output=True,
                                        text=True,
                                        timeout=2)
                if result.returncode == 0:
                    print(
                        f'[DEBUG passthrough_stream_handler] Last 10 lines written to {out_stream.name}:',
                        file=sys.stderr,
                        flush=True)
                    for i, line in enumerate(
                            result.stdout.strip().split('\n')[-10:], 1):
                        print(f'  file[{i}] {line[:100]}',
                              file=sys.stderr,
                              flush=True)
            except Exception:
                pass

    return ''


def pipe_and_wait_process(
        ctx: context.Context,
        proc: subprocess.Popen,
        poll_interval: float = 0.5,
        cancel_callback: Optional[Callable[[], None]] = None,
        stdout_stream_handler: Optional[StreamHandler] = None,
        stderr_stream_handler: Optional[StreamHandler] = None
) -> Tuple[str, str]:
    """Wait for the process to finish or cancel it if the context is cancelled.

    Args:
        proc: The process to wait for.
        poll_interval: The interval to poll the process.
        cancel_callback: The callback to call if the context is cancelled.
        stdout_stream_handler: An optional handler to handle the stdout stream,
            if None, the stdout stream will be passed through.
        stderr_stream_handler: An optional handler to handle the stderr stream,
            if None, the stderr stream will be passed through.
    """
    import sys as sys_module
    print(f'[DEBUG pipe_and_wait_process] Starting, proc.pid={proc.pid}',
          file=sys_module.stderr,
          flush=True)

    if stdout_stream_handler is None:
        stdout_stream_handler = passthrough_stream_handler
    if stderr_stream_handler is None:
        stderr_stream_handler = passthrough_stream_handler

    # Threads are lazily created, so no harm if stderr is None
    with multiprocessing.pool.ThreadPool(processes=2) as pool:
        # Context will be lost in the new thread, capture current output stream
        # and pass it to the new thread directly.
        print(f'[DEBUG pipe_and_wait_process] Launching stdout thread',
              file=sys_module.stderr,
              flush=True)
        stdout_fut = pool.apply_async(
            stdout_stream_handler, (proc.stdout, ctx.output_stream(sys.stdout)))
        stderr_fut = None
        if proc.stderr is not None:
            print(f'[DEBUG pipe_and_wait_process] Launching stderr thread',
                  file=sys_module.stderr,
                  flush=True)
            stderr_fut = pool.apply_async(
                stderr_stream_handler,
                (proc.stderr, ctx.output_stream(sys.stderr)))
        try:
            import time as time_module
            wait_start = time_module.time()
            print(
                f'[DEBUG pipe_and_wait_process] Waiting for process at {wait_start}',
                file=sys_module.stderr,
                flush=True)
            wait_process(ctx,
                         proc,
                         poll_interval=poll_interval,
                         cancel_callback=cancel_callback)
            wait_end = time_module.time()
            print(
                f'[DEBUG pipe_and_wait_process] Process wait returned at {wait_end} ({wait_end - wait_start:.2f}s), returncode={proc.returncode}',
                file=sys_module.stderr,
                flush=True)
        except Exception as e:
            print(
                f'[DEBUG pipe_and_wait_process] Exception in wait_process: {type(e).__name__}: {e}',
                file=sys_module.stderr,
                flush=True)
            raise
        finally:
            # Wait for the stream handler threads to exit when process is done
            # or cancelled
            thread_wait_start = time_module.time()
            print(
                f'[DEBUG pipe_and_wait_process] Waiting for stdout thread to finish at {thread_wait_start}',
                file=sys_module.stderr,
                flush=True)
            stdout_fut.wait()
            stdout_done = time_module.time()
            print(
                f'[DEBUG pipe_and_wait_process] Stdout thread finished at {stdout_done} (took {stdout_done - thread_wait_start:.2f}s)',
                file=sys_module.stderr,
                flush=True)
            if stderr_fut is not None:
                stderr_wait_start = time_module.time()
                print(
                    f'[DEBUG pipe_and_wait_process] Waiting for stderr thread to finish at {stderr_wait_start}',
                    file=sys_module.stderr,
                    flush=True)
                stderr_fut.wait()
                stderr_done = time_module.time()
                print(
                    f'[DEBUG pipe_and_wait_process] Stderr thread finished at {stderr_done} (took {stderr_done - stderr_wait_start:.2f}s)',
                    file=sys_module.stderr,
                    flush=True)
            all_done = time_module.time()
            print(
                f'[DEBUG pipe_and_wait_process] All threads finished at {all_done}',
                file=sys_module.stderr,
                flush=True)

        try:
            stdout = stdout_fut.get()
            stderr = ''
            if stderr_fut is not None:
                stderr = stderr_fut.get()
            print(
                f'[DEBUG pipe_and_wait_process] Successfully got results from threads',
                file=sys_module.stderr,
                flush=True)
        except Exception as e:
            print(
                f'[DEBUG pipe_and_wait_process] Exception getting thread results: {type(e).__name__}: {e}',
                file=sys_module.stderr,
                flush=True)
            raise

        return stdout, stderr


def wait_process(ctx: context.Context,
                 proc: subprocess.Popen,
                 poll_interval: float = 0.5,
                 cancel_callback: Optional[Callable[[], None]] = None):
    import sys as sys_module
    import time as time_module
    """Wait for the process to finish or cancel it if the context is cancelled.

    Args:
        proc: The process to wait for.
        poll_interval: The interval to poll the process.
        cancel_callback: The callback to call if the context is cancelled.
    """
    start_time = time_module.time()
    poll_count = 0

    while True:
        poll_count += 1
        if ctx.is_canceled():
            elapsed = time_module.time() - start_time
            print(
                f'[DEBUG wait_process] Context cancelled after {elapsed:.2f}s, {poll_count} polls',
                file=sys_module.stderr,
                flush=True)
            if cancel_callback is not None:
                cancel_callback()
            # Kill the process despite the caller's callback, the utility
            # function gracefully handles the case where the process is
            # already terminated.
            # Bash script typically does not forward SIGTERM to childs, thus
            # cannot be killed gracefully, shorten the grace period for faster
            # termination.
            subprocess_utils.kill_process_with_grace_period(proc,
                                                            grace_period=1)
            raise asyncio.CancelledError()
        try:
            proc.wait(poll_interval)
        except subprocess.TimeoutExpired:
            pass
        else:
            # Process exited
            elapsed = time_module.time() - start_time
            print(
                f'[DEBUG wait_process] Process exited normally after {elapsed:.2f}s, returncode={proc.returncode}',
                file=sys_module.stderr,
                flush=True)
            break


F = TypeVar('F', bound=Callable[..., Any])


def cancellation_guard(func: F) -> F:
    """Decorator to make a synchronous function cancellable via context.

    Guards the function execution by checking context.is_canceled() before
    executing the function and raises asyncio.CancelledError if the context
    is already cancelled.

    This basically mimics the behavior of asyncio, which checks coroutine
    cancelled in await call.

    Args:
        func: The function to be decorated.

    Returns:
        The wrapped function that checks cancellation before execution.

    Raises:
        asyncio.CancelledError: If the context is cancelled before execution.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ctx = context.get()
        if ctx is not None and ctx.is_canceled():
            raise asyncio.CancelledError(
                f'Function {func.__name__} cancelled before execution')
        return func(*args, **kwargs)

    return typing.cast(F, wrapper)


P = ParamSpec('P')
T = TypeVar('T')


# TODO(aylei): replace this with asyncio.to_thread once we drop support for
# python 3.8
def to_thread(func: Callable[P, T], /, *args: P.args,
              **kwargs: P.kwargs) -> 'asyncio.Future[T]':
    """Asynchronously run function *func* in a separate thread.

    This is same as asyncio.to_thread added in python 3.9
    """
    loop = asyncio.get_running_loop()
    # This is critical to pass the current coroutine context to the new thread
    pyctx = contextvars.copy_context()
    func_call: Callable[..., T] = functools.partial(
        # partial deletes arguments type and thus can't figure out the return
        # type of pyctx.run
        pyctx.run,  # type: ignore
        func,
        *args,
        **kwargs)
    return loop.run_in_executor(None, func_call)
