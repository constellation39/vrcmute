import logging
import logging.handlers
import os


class LoggerSetup:
    """
    A class to set up and configure logging for production environments.

    This class provides a centralized way to configure logging with
    appropriate handlers, formatters, and log rotation.

    Attributes:
        LOG_FORMAT (str): The format string for log messages.
        DATE_FORMAT (str): The format string for timestamps in log messages.
        LOG_FILE (str): The name of the log file.
        LOG_DIR (str): The directory where log files will be stored.
    """

    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_FILE: str = "vrcmute.log"
    LOG_DIR: str = "logs"

    @classmethod
    def setup_logger(
        cls,
        logger_name: str | None = None,
        log_level: int = logging.INFO,
        log_to_console: bool = True,
        log_to_file: bool = True,
        max_file_size: int = 1024 * 1024 * 5,  # 5 MB
        backup_count: int = 5,
    ) -> logging.Logger:
        """
        Set up and configure a logger.

        Args:
            logger_name (Optional[str]): Name of the logger. If None, the root logger is used.
            log_level (int): The logging level. Defaults to logging.INFO.
            log_to_console (bool): Whether to log to console. Defaults to True.
            log_to_file (bool): Whether to log to file. Defaults to True.
            max_file_size (int): Maximum size of each log file in bytes. Defaults to 5 MB.
            backup_count (int): Number of backup log files to keep. Defaults to 5.

        Returns:
            logging.Logger: Configured logger instance.

        Raises:
            OSError: If there's an error creating the log directory or file.
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.handlers = []  # Clear any existing handlers

        formatter = logging.Formatter(cls.LOG_FORMAT, cls.DATE_FORMAT)

        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if log_to_file:
            try:
                if not os.path.exists(cls.LOG_DIR):
                    os.makedirs(cls.LOG_DIR)

                file_handler = logging.handlers.RotatingFileHandler(
                    filename=os.path.join(cls.LOG_DIR, cls.LOG_FILE),
                    maxBytes=max_file_size,
                    backupCount=backup_count,
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except OSError as e:
                raise OSError(f"Error setting up file logging: {e}")

        return logger


# Usage
logger = LoggerSetup.setup_logger(__name__)
