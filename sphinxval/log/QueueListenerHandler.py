# examples/QueueListenerHandler.py
# https://rob-blackbourn.medium.com/how-to-use-python-logging-queuehandler-with-dictconfig-1e8b1284e27a
from logging.config import ConvertingList, ConvertingDict, valid_ident
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from atexit import register


def _resolve_handlers(l):
    if not isinstance(l, ConvertingList):
        return l

    # Indexing the list performs the evaluation.
    return [l[i] for i in range(len(l))]


class QueueListenerHandler(QueueHandler):

    def __init__(self, handlers, respect_handler_level=False, auto_run=True, queue=Queue(-1)):
        super().__init__(queue)
        handlers = _resolve_handlers(handlers)
        self._listener = QueueListener(
            self.queue,
            *handlers,
            respect_handler_level=respect_handler_level)
        if auto_run:
            self.start()
            register(self.stop)


    def start(self):
        self._listener.start()


    def stop(self):
        self._listener.stop()


    def emit(self, record):
        return super().emit(record)
