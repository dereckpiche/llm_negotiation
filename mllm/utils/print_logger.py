class PrintLogger:
    def __init__(self, logger):
        self.logger = logger

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
        print(msg)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
        print(msg)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
        print(msg)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
        print(msg)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
        print(msg)

    # Add other methods as needed (exception, log, etc.)