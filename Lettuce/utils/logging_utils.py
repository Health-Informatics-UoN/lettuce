import logging
import time
from pathlib import Path


class Logger(object):
    """
    logger preparation


    Parameters
    ----------
    log_dir: string
        path to the log directory

    logging_level: string
        required Level of logging. INFO, WARNING or ERROR can be selected. Default to 'INFO'

    console_logger: bool
        flag if console_logger is required. Default to False

    Returns
    ----------
    logger: logging.Logger
        logger object
    """

    def __init__(
        self, logging_level="INFO", console_logger=True, multi_module=True
    ) -> None:
        """
        Initialises the logger

        Parameters
        ----------
        logging_level:
            required Level of logging. INFO, WARNING or ERROR can be selected. Default to 'INFO'
        console_logger: bool
            flag if console_logger is required. Default to False
        multi_module: bool
            Not yet implemented

        Returns
        -------
        None

        """
        super().__init__()
        self._log_dir = f"./log/"
        self.console_logger = console_logger
        self.logging_level = logging_level.lower()
        self.multi_module = multi_module
        self._make_level()

    def _make_level(self):
        """
        Sets the level of logging

        Uses the logging_level to set own _level property.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.logging_level == "info":
            self._level = logging.INFO
        elif self.logging_level == "warning":
            self._level = logging.WARNING
        elif self.logging_level == "error":
            self._level = logging.ERROR
        else:
            raise ValueError(
                "logging_level not specified correctly. INFO, WARNING or ERROR must be chosen"
            )

    def make_logger(self):
        """
        Constructs a Logger instance.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # logging configuration
        log_dir = Path(self._log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_name = log_dir.joinpath(f'{time.strftime("%Y%m%d")}.log')

        # Create a custom logger
        # if self.multi_module:
        #     logger = logging.getLogger()
        # else:
        #     logger = logging.getLogger(__name__)
        logger = logging.getLogger(__name__)
        logger.setLevel(self._level)

        # Create handlers
        f_handler = logging.FileHandler(filename=file_name)
        f_handler.setLevel(self._level)

        # Create formatters
        format = logging.Formatter("%(levelname)s - %(message)s - %(module)s")
        f_handler.setFormatter(format)

        # Add handlers to the logger
        logger.addHandler(f_handler)

        # Console handler creation
        if self.console_logger:
            c_handler = logging.StreamHandler()
            c_handler.setLevel(self._level)
            c_handler.setFormatter(format)
            logger.addHandler(c_handler)

        return logger
