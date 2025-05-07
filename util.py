import os
from threading import Lock

class SafeOpenWrapper:
    dict_lock = Lock()
    file_locks = {}

    def __init__(self, path, *args, **kwargs):
        self.path = os.path.realpath(path)
        self.open_args = args
        self.open_kwargs = kwargs
        self.__file = None

    @classmethod
    def get_lock(cls, file):
        with cls.dict_lock:
            if file not in cls.file_locks:
                cls.file_locks[file] = Lock()
            lock = cls.file_locks[file]
        return lock

    def __enter__(self):
        lock = SafeOpenWrapper.get_lock(self.path)
        lock.acquire()
        try:
            self.__file = open(self.path, *self.open_args, **self.open_kwargs)
            res = self.__file.__enter__()
        except:
            lock.release()
            raise
        return res

    def __exit__(self, *args, **kwargs):
        lock = SafeOpenWrapper.get_lock(self.path)
        self.__file.__exit__(*args, **kwargs)
        lock.release()


def safe_open(path, *args, **kwargs):
    return SafeOpenWrapper(path, *args, **kwargs)
