from contextlib import contextmanager
from datetime import datetime
import hashlib
import random
import string
from typing import (
    Callable,
)

import subprocess
import threading


def nanoid(size=21) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(size))


@contextmanager
def timeout(seconds: int, callback: Callable = lambda: print()):
    """
    Context manager that raises a TimeoutError if the code inside the context takes longer than the specified time.

    This implementation uses threading.Timer which is thread-safe, unlike signal-based approaches.

    Args:
        seconds (int): Maximum number of seconds to allow the code to run

    Yields:
        None: The context to execute code within the timeout constraint

    Raises:
        TimeoutError: If the code execution exceeds the specified timeout

    Example:
        >>> with timeout(5):
        ...     # Code that should complete within 5 seconds
        ...     long_running_function()
    """
    timer = None
    exception = TimeoutError(f"Execution timed out after {seconds} seconds")

    def timeout_handler():
        nonlocal timer
        callback()
        raise exception

    timer = threading.Timer(seconds, timeout_handler)
    timer.start()

    try:
        yield
    finally:
        if timer:
            timer.cancel()


def int_to_ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def unflatten_toml_dict(d: dict) -> dict:
    result = {}
    for key, value in d.items():
        parts = key.split(".")
        current_level = result
        for i, part in enumerate(parts):
            if i == len(parts) - 1:  # Last part, assign value
                current_level[part] = value
            else:
                current_level = current_level.setdefault(part, {})

    return result


def generate_readable_run_id(
    random_length: int = 6, date_format_str: str = "%Y%m%d", separator: str = "-"
) -> str:
    current_date = datetime.now()

    date_part = current_date.strftime(date_format_str)

    characters = string.ascii_lowercase + string.digits
    random_part = "".join(random.choices(characters, k=random_length))

    # 4. Combine the parts
    run_id = f"{date_part}{separator}{random_part}"

    return run_id


def get_formatted_repo_info():
    """
    Gets current Git repository information formatted as:
    "{commit-hash-capitalized}-{branch-capitalized}-{number-of-uncommitted-files}"

    This version lets ANY failure from Git commands (including trying to get a
    commit hash when none exist) raise the original FileNotFoundError or
    subprocess.CalledProcessError, causing the script to terminate immediately.
    No special strings, no custom errors.
    """

    # 1. Preliminary check: Is this a Git repository?
    #    If 'git' command isn't found, FileNotFoundError propagates.
    #    If 'git rev-parse --git-dir' fails (e.g., not a repo), CalledProcessError propagates.
    #    stderr from THIS specific check is suppressed as the exception itself is enough indication.
    subprocess.check_output(
        ["git", "rev-parse", "--git-dir"],
        stderr=subprocess.DEVNULL,  # Suppress "fatal: not a git repository" for this check only
        text=True,
    )

    # 2. Get current commit hash.
    #    If 'git rev-parse HEAD' fails for ANY reason (e.g., no commits yet, corrupted HEAD),
    #    the CalledProcessError will propagate, and Git's error message will appear on stderr.
    commit_hash_raw = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,  # Decodes stdout to string
    ).strip()
    commit_hash_lower = commit_hash_raw.lower()

    # 3. Get current branch name.
    #    If 'git rev-parse --abbrev-ref HEAD' fails, CalledProcessError propagates.
    #    Git's error message will appear on stderr.
    branch_name_raw = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()
    branch_name_lower = branch_name_raw.lower()

    # 4. Get hash of uncommitted files.
    #    If 'git status --porcelain' fails, CalledProcessError propagates.
    #    Git's error message will appear on stderr.
    status_output = subprocess.check_output(
        ["git", "diff", "--stat"], text=True
    ).strip()

    uncommitted_files_hash = "0"
    if status_output:  # Only splitlines if there's actual output.
        uncommitted_files_hash = string_hash(status_output)

    return f"{commit_hash_lower[:6]}-{branch_name_lower}-{uncommitted_files_hash[:6]}"


def string_hash(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()
