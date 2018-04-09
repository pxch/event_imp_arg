import io
import logging

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


def get_console_logger(level='info'):
    # Prepare a logger
    logger = logging.getLogger()
    if level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif level == 'info':
        logger.setLevel(logging.INFO)
    elif level == 'warning':
        logger.setLevel(logging.WARNING)
    elif level == 'error':
        logger.setLevel(logging.ERROR)
    elif level == 'critical':
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.NOTSET)

    if not logger.handlers:
        # Just log to the console
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(log_formatter)

        logger.addHandler(sh)

    return logger


def add_file_handler(logger, log_path):
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)


class PBToLogger(io.StringIO):
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(PBToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


def get_file_logger(name, file_path, log_level='info', propagate=False):
    logger = logging.getLogger(name)
    logger.propagate = propagate
    logger.setLevel(getattr(logging, log_level.upper()))

    assert not logger.handlers
    file_handler = logging.FileHandler(file_path, mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    return logger


log = get_console_logger()

pb_log = PBToLogger(log)
