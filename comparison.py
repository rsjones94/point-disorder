import collections.abc


class BidirectionalDict(collections.abc.MutableMapping):
    """
    A dictionary-like object where d[(a,b)] is d[(b,a)]
    """

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        try:
            return self.store[key]
        except KeyError:
            return self.store[self.flip_key(key)]

    def __setitem__(self, key, value):
        if self.flip_key(key) in self:
            self.store[self.flip_key(key)] = value
        else:
            self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def flip_key(self, key):
        return key[1], key[0]
