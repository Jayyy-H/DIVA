def linear_warmup(step, total, max_v):
    ratio = min(step / total, 1.0)
    return max_v * ratio
