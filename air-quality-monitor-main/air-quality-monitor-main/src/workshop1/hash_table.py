import re
import random
import pprint

class HashTable:
    def __init__(self, capacity=8):
        self._capacity = capacity if capacity >= 8 else 8
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0

    # --- Detección y hash para ObjectId (hex 24 chars) ---
    def _is_objectid(self, key):
        return isinstance(key, str) and len(key) == 24 and re.fullmatch(r"[0-9a-fA-F]{24}", key) is not None

    def _hash_objectid(self, key):
        try:
            return int(key, 16) % self._capacity
        except Exception:
            return self._hash_string(key)

    # --- FNV-1a simple para strings ---
    def _hash_string(self, key):
        if not isinstance(key, str):
            key = str(key)
        fnv_offset = 14695981039346656037
        fnv_prime = 1099511628211
        h = fnv_offset
        # iterar por bytes
        for b in key.encode('utf-8'):
            h ^= b
            h = (h * fnv_prime) & 0xFFFFFFFFFFFFFFFF
        return h % self._capacity

    def _bucket_index(self, key):
        if self._is_objectid(key):
            return self._hash_objectid(key)
        return self._hash_string(key)

    # --- Operaciones básicas ---
    def insert(self, key, value):
        idx = self._bucket_index(key)
        bucket = self._buckets[idx]
        for i, pair in enumerate(bucket):
            if pair[0] == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1
        if self.load_factor() > 0.75:
            self._resize(self._capacity * 2)

    def get(self, key, default=None):
        idx = self._bucket_index(key)
        bucket = self._buckets[idx]
        for k, v in bucket:
            if k == key:
                return v
        return default

    def delete(self, key):
        idx = self._bucket_index(key)
        bucket = self._buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self._size -= 1
                if self._capacity > 8 and self.load_factor() < 0.2:
                    self._resize(max(8, self._capacity // 2))
                return True
        return False

    def keys(self):
        result = []
        for bucket in self._buckets:
            for k, _ in bucket:
                result.append(k)
        return result

    def values(self):
        result = []
        for bucket in self._buckets:
            for _, v in bucket:
                result.append(v)
        return result

    def items(self):
        result = []
        for bucket in self._buckets:
            for k, v in bucket:
                result.append((k, v))
        return result

    def __len__(self):
        return self._size

    def capacity(self):
        return self._capacity

    def load_factor(self):
        return float(self._size) / float(self._capacity)

    def _resize(self, new_capacity):
        old_items = self.items()
        self._capacity = new_capacity if new_capacity >= 8 else 8
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0
        for k, v in old_items:
            self.insert(k, v)

    def __repr__(self):
        return "<HashTable size={} capacity={} load_factor={:.3f}>".format(self._size, self._capacity, self.load_factor())


# --------------------------
# Ejemplo de uso con la estructura dada (simulada).

def make_mock_doc(_id, city, pm25, no2, timestamp):
    return {
        "_id": _id,
        "sensorLocation": city,
        "airQualityData": {"PM25": pm25, "NO2": no2},
        "timestamp": timestamp
    }

# Crear índices
index_by_id = HashTable()
index_by_city = HashTable()

docs = [
    make_mock_doc("5f1d7f7a9e1b8b3a6f0a1c2d", "Madrid", 12.34, 20.5, "2025-11-24T08:00:00 +0000"),
    make_mock_doc("5f1d7f7a9e1b8b3a6f0a1c2e", "Bogota", 45.12, 30.01, "2025-11-24T09:00:00 +0000"),
    make_mock_doc("5f1d7f7a9e1b8b3a6f0a1c2f", "Madrid", 55.00, 10.00, "2025-11-24T09:05:00 +0000"),
]

# Insertar documentos en ambos índices
for d in docs:
    # Índice por _id (único)
    index_by_id.insert(d["_id"], d)
    # Índice por ciudad (almacenar lista de documentos)
    city = d["sensorLocation"]
    existing = index_by_city.get(city)
    if existing is None:
        index_by_city.insert(city, [d])
    else:
        existing.append(d)
        index_by_city.insert(city, existing)

print("Estado índice por _id:", index_by_id)
print("Estado índice por ciudad:", index_by_city)

print("\nBuscar por _id (primer doc):")
pprint.pprint(index_by_id.get("5f1d7f7a9e1b8b3a6f0a1c2d"))

print("\nBuscar por ciudad 'Madrid' (lista de docs):")
pprint.pprint(index_by_city.get("Madrid"))

# Forzar más inserciones para generar colisiones y redimensionamiento
for i in range(30):
    cid = "City-{}".format(i % 7)
    doc = make_mock_doc("{:024x}".format(random.randrange(1<<60))[:24], cid, random.random()*100, random.random()*50, "2025-11-24T10:00:00 +0000")
    lst = index_by_city.get(cid) or []
    lst.append(doc)
    index_by_city.insert(cid, lst)

print("\nDespués de inserciones adicionales:")
print(index_by_city)
print("Keys en índice por ciudad (muestra):", index_by_city.keys()[:12], "... (total", len(index_by_city.keys()), ")")
