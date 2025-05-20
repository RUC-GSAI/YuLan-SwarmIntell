from logging import getLogger


class Logger:
    def __init__(self, name):
        self.name = name
        self.logger = getLogger(self.name)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def error(self, msg):
        self.logger.error(msg)

    def warning(self, msg):
        self.logger.warning(msg)
