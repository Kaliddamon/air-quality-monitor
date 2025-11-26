def euclidean(v1, v2):
    s = 0.0
    for a, b in zip(v1, v2):
        d = a - b
        s += d * d
    return math.sqrt(s)

def dist2d(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def mean_vector(v):
    return sum(v) / len(v)

def encode_source(src):
    # codificación simple para comparar fuentes
    if src == "traffic":
        return 0.0
    if src == "industrial":
        return 1.0
    return 0.5  # mixed
