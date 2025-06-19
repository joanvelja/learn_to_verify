# pvg/utils/logging.py

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# Define custom level below INFO and above DEBUG
NOTICE_LEVEL_NUM = 25
logging.addLevelName(NOTICE_LEVEL_NUM, "NOTICE")


def notice(self, message, *args, **kws):
    if self.isEnabledFor(NOTICE_LEVEL_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(NOTICE_LEVEL_NUM, message, args, **kws)


logging.Logger.notice = notice


def setup_logger(
    name: str = "pvg",
    level: int = logging.INFO,
    rank: int = -1,
    world_size: int = 1,
    log_to_file: bool = True,
    log_dir: str = "logs",
    log_filename: str = "training.log",
    main_process_only_file: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 3,
) -> logging.Logger:
    """
    Sets up a logger for console and optional file logging in a distributed environment.

    Args:
        name: The name for the root logger of the project.
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
        rank: The process rank in the distributed setup (-1 if unknown or single process).
        world_size: The total number of processes.
        log_to_file: Whether to enable logging to a file.
        log_dir: The directory to save log files.
        log_filename: The base name for the log file.
        main_process_only_file: If True, only rank 0 writes to the file.
                                If False, each rank writes to a rank-specific file.
        max_bytes: Maximum size of the log file before rotation.
        backup_count: Number of backup log files to keep.

    Returns:
        The configured root logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set root logger to lowest level to allow handlers to control effective level
    logger.propagate = False  # Prevent root logger from propagating to parent (avoids duplicate messages if root logger is configured elsewhere)

    # --- Formatter ---
    # Include rank/world_size if in distributed mode
    if world_size > 1 and rank != -1:
        log_format = f"[%(asctime)s] [Rank {rank:02d}/{world_size:02d}] [%(name)s] [%(levelname)s] - %(message)s"
    else:
        log_format = "[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # --- Clear existing handlers (important for reconfiguration) ---
    # This prevents adding handlers multiple times if setup_logger is called again
    # (e.g., once initially, then again after accelerator init)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)  # Set level for this handler
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File Handler ---
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = None
        can_log_to_file = (
            (main_process_only_file and rank == 0) or (not main_process_only_file and rank != -1) or (rank == -1)
        )  # Single process case

        if can_log_to_file:
            if not main_process_only_file and rank != -1:
                # Rank-specific filename
                base, ext = os.path.splitext(log_filename)
                effective_log_filename = f"{base}_rank_{rank}{ext}"
            else:
                # Shared filename (used by rank 0 or single process)
                effective_log_filename = log_filename

            log_filepath = os.path.join(log_dir, effective_log_filename)

            # Use RotatingFileHandler for log rotation
            file_handler = RotatingFileHandler(
                log_filepath,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            # File handler should typically log more verbosely than console
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    # Add custom level method if not already present
    if not hasattr(logger, "notice"):
        logging.Logger.notice = notice

    # Log a confirmation message using the new setup
    init_message = f"Logger '{name}' configured. Level={logging.getLevelName(level)}, Rank={rank}, WorldSize={world_size}, FileLogging={log_to_file}"
    if log_to_file and can_log_to_file:
        init_message += f", LogFile='{log_filepath}'"
    logger.info(init_message)

    return logger


# Example of getting a logger elsewhere in the code
# import logging
# logger = logging.getLogger(f"pvg.{__name__}") # Get a child logger
