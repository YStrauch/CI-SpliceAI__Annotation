import abc

from cispliceai import data

class BaseCache(abc.ABC):
    def __init__(self, cache_miss_fn=None):
        self.cache_miss_fn = cache_miss_fn

    def exists(self, key: any):
        return self._exists(key)

    def fetch(self, key: any)->any:
        if not self.exists(key):
            if self.cache_miss_fn is None:
                raise RuntimeError('Cache miss but no cache miss function provided')
            self._put(key, self.cache_miss_fn(key))

        return self._fetch(key)

    def put(self, key: any, value: any):
        return self._put(key, value)


    @abc.abstractmethod
    def _exists(self, key: any) -> bool:
        pass

    @abc.abstractmethod
    def _fetch(self, key: any) -> any:
        pass

    @abc.abstractmethod
    def _put(self, key: any, value: any):
        pass


class RAMCache(BaseCache):
    def __init__(self, cache_miss_fn=None):
        super().__init__(cache_miss_fn=cache_miss_fn)
        self._dict = {}

    def _exists(self, key: any):
        return str(key) in self._dict

    def _fetch(self, key: str) -> any:
        return self._dict[str(key)]

    def _put(self, key: str, value: any):
        self._dict[str(key)] = value