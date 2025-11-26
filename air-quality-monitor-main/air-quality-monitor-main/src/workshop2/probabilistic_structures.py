import hashlib
import math


class BloomFilter:
    def __init__(self, size=2048, hash_count=3):
        #size = number of bits
        self.size = size
        #hash_count = number of hash functions
        self.hash_count = hash_count
        #bits array as bytes
        self.bits = bytearray(size)

    def _hashes(self, item):
        #generate k hash values in range [0, size)
        data = str(item).encode("utf-8")
        for i in range(self.hash_count):
            seed = i.to_bytes(1, "little")
            h = hashlib.sha256(data + seed).hexdigest()
            value = int(h, 16) % self.size
            yield value

    def add(self, item):
        #insert element into bloom filter
        for index in self._hashes(item):
            self.bits[index] = 1

    def contains(self, item):
        #check if element is possibly in the set
        for index in self._hashes(item):
            if self.bits[index] == 0:
                return False
        return True

    def clear(self):
        #reset all bits
        self.bits = bytearray(self.size)


class FMEstimator:
    def __init__(self, max_bits=64):
        #max_zero_run = longest run of trailing zeros
        self.max_zero_run = 0
        #max_bits = safety limit for bit length
        self.max_bits = max_bits

    def _trailing_zeros(self, value):
        #count trailing zeros in binary representation
        if value == 0:
            return self.max_bits
        count = 0
        while (value & 1) == 0 and count < self.max_bits:
            count += 1
            value >>= 1
        return count

    def add(self, item):
        #process a new element into the estimator
        data = str(item).encode("utf-8")
        h = hashlib.sha256(data).hexdigest()
        value = int(h, 16)
        r = self._trailing_zeros(value)
        if r > self.max_zero_run:
            self.max_zero_run = r

    def estimate(self):
        #return approximate number of distinct elements
        if self.max_zero_run == 0:
            return 0
        phi = 0.77351
        return int((2 ** self.max_zero_run) / phi)

    def merge(self, other):
        #merge two estimators
        if not isinstance(other, FMEstimator):
            return
        if other.max_zero_run > self.max_zero_run:
            self.max_zero_run = other.max_zero_run

    def clear(self):
        #reset estimator
        self.max_zero_run = 0
