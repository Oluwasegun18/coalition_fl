import numpy as np
from collections import defaultdict

# Here’s a flexible splitter that:

# Uses different strategies per cluster (you can mix & match).

# Keeps a single global pool of samples and tops up if some classes run out.

# Lets you choose whether to balance sizes across devices or allow quantity skew.


# Included strategies
# "dirichlet_low": very low-α Dirichlet per device (e.g., α=0.05) → spiky label mixes.

# "label_k": each device is assigned K classes only (K=1 or 2) → extreme label skew.

# "majority": each device has a single majority class (90–95%) plus crumbs of others.

# "zipf_major": quantity skew via Zipf sizes; each device gets 1–2 dominant classes.


# Tuning non-IID level
# More skew: lower dirichlet_alpha (e.g., 0.01), set label_k=1, raise majority share to 0.95–0.99, and use balanced_sizes=False.

# Less skew: increase dirichlet_alpha (≥0.3), label_k=2–3, majority share ~0.7–0.8, and/or balanced_sizes=True.




def _take_from_pool(pools, lbl, n):
    take = min(n, len(pools[lbl]))
    picked = pools[lbl][:take]
    pools[lbl] = pools[lbl][take:]
    return picked, take

def split_non_iid_multi_cluster(
    x, y,
    num_clusters=3,
    devices_per_cluster=10,
    seed=42,
    strategies=None,               # list of length num_clusters; see below
    balanced_sizes=True,
    dirichlet_alpha=0.05,          # spiky by default
    label_k=1,                     # devices constrained to K classes in "label_k"
):
    """
    Returns:
        cluster_device_data[c][d] -> x for device d in cluster c
        cluster_device_labels[c][d] -> y for device d in cluster c
    """
    rng = np.random.default_rng(seed)
    total_devices = num_clusters * devices_per_cluster
    labels = np.unique(y)
    num_classes = len(labels)

    # Pools of indices per class (global), shuffled
    pools = {lbl: rng.permutation(np.where(y == lbl)[0]).tolist() for lbl in labels}

    # Choose strategies per cluster if not given
    default_cycle = ["dirichlet_low", "label_k", "majority", "zipf_major"]
    if strategies is None:
        strategies = [default_cycle[i % len(default_cycle)] for i in range(num_clusters)]
    assert len(strategies) == num_clusters, "Provide one strategy per cluster."

    # Target sizes per device
    N = len(y)
    if balanced_sizes:
        base = N // total_devices
        rem = N % total_devices
        target_sizes = np.array([base + (i < rem) for i in range(total_devices)], dtype=int)
    else:
        # Quantity skew: Zipf-ish then normalized to N
        raw = 1.0 / (np.arange(1, total_devices + 1) ** 1.1)
        raw = raw / raw.sum()
        target_sizes = np.floor(raw * N).astype(int)
        target_sizes[0] += N - target_sizes.sum()  # fix rounding

    device_indices = [[] for _ in range(total_devices)]

    def allocate_dirichlet_low(dev_id, target):
        if target <= 0: return
        p = rng.dirichlet([dirichlet_alpha] * num_classes)
        req = np.floor(p * target).astype(int)
        req[0] += target - req.sum()
        # take requested, then top-up if short
        taken = 0
        for k, lbl in enumerate(labels):
            picked, t = _take_from_pool(pools, lbl, req[k])
            device_indices[dev_id].extend(picked)
            taken += t
        short = target - taken
        if short > 0:
            for lbl in rng.permutation(labels):
                if short == 0: break
                picked, t = _take_from_pool(pools, lbl, short)
                device_indices[dev_id].extend(picked)
                short -= t

    def allocate_label_k(dev_id, target):
        if target <= 0: return
        K = min(label_k, num_classes)
        chosen = rng.choice(labels, size=K, replace=False)
        # heavy preference to chosen labels
        weights = np.array([10.0 if lbl in chosen else 1e-6 for lbl in labels], dtype=float)
        p = weights / weights.sum()
        req = np.floor(p * target).astype(int)
        req[0] += target - req.sum()
        taken = 0
        for k, lbl in enumerate(labels):
            picked, t = _take_from_pool(pools, lbl, req[k])
            device_indices[dev_id].extend(picked)
            taken += t
        short = target - taken
        if short > 0:
            for lbl in rng.permutation(chosen):  # top-up from chosen first
                if short == 0: break
                picked, t = _take_from_pool(pools, lbl, short)
                device_indices[dev_id].extend(picked)
                short -= t
        if short > 0:
            for lbl in rng.permutation(labels):  # then anywhere
                if short == 0: break
                picked, t = _take_from_pool(pools, lbl, short)
                device_indices[dev_id].extend(picked)
                short -= t

    def allocate_majority(dev_id, target):
        if target <= 0: return
        majority_lbl = rng.choice(labels)
        majority_p = rng.uniform(0.90, 0.95)  # strong majority
        rest = (1 - majority_p) / (num_classes - 1)
        p = np.array([majority_p if lbl == majority_lbl else rest for lbl in labels])
        req = np.floor(p * target).astype(int)
        req[labels.tolist().index(majority_lbl)] += target - req.sum()
        taken = 0
        # majority first
        idx_major = labels.tolist().index(majority_lbl)
        picked, t = _take_from_pool(pools, majority_lbl, req[idx_major])
        device_indices[dev_id].extend(picked)
        taken += t
        # then others
        for k, lbl in enumerate(labels):
            if k == idx_major: continue
            picked, t = _take_from_pool(pools, lbl, req[k])
            device_indices[dev_id].extend(picked)
            taken += t
        # top-up anywhere
        short = target - taken
        if short > 0:
            for lbl in rng.permutation(labels):
                if short == 0: break
                picked, t = _take_from_pool(pools, lbl, short)
                device_indices[dev_id].extend(picked)
                short -= t

    def allocate_zipf_major(dev_id, target):
        if target <= 0: return
        # pick 1–2 dominant classes
        Kdom = 1 if num_classes == 1 else rng.integers(1, 3)
        dom = rng.choice(labels, size=Kdom, replace=False)
        # dominant share 85–95% split among them, rest spread thinly
        dom_share = rng.uniform(0.85, 0.95)
        p = np.ones(num_classes) * ((1 - dom_share) / (num_classes - Kdom if num_classes > Kdom else 1))
        for d in dom:
            p[labels.tolist().index(d)] = dom_share / Kdom
        p = p / p.sum()
        req = np.floor(p * target).astype(int)
        req[0] += target - req.sum()
        taken = 0
        for k, lbl in enumerate(labels):
            picked, t = _take_from_pool(pools, lbl, req[k])
            device_indices[dev_id].extend(picked)
            taken += t
        short = target - taken
        if short > 0:
            for lbl in rng.permutation(labels):
                if short == 0: break
                picked, t = _take_from_pool(pools, lbl, short)
                device_indices[dev_id].extend(picked)
                short -= t

    allocators = {
        "dirichlet_low": allocate_dirichlet_low,
        "label_k":       allocate_label_k,
        "majority":      allocate_majority,
        "zipf_major":    allocate_zipf_major,
    }

    # Allocate per device, grouped by cluster with its strategy
    for c in range(num_clusters):
        strat = strategies[c]
        if strat not in allocators:
            raise ValueError(f"Unknown strategy '{strat}' for cluster {c}")
        alloc = allocators[strat]
        for d in range(devices_per_cluster):
            dev_id = c * devices_per_cluster + d
            alloc(dev_id, int(target_sizes[dev_id]))

    # Pack outputs
    cluster_device_data, cluster_device_labels = {}, {}
    for c in range(num_clusters):
        cluster_device_data[c] = {}
        cluster_device_labels[c] = {}
        for d in range(devices_per_cluster):
            dev_id = c * devices_per_cluster + d
            idx = device_indices[dev_id]
            rng.shuffle(idx)
            cluster_device_data[c][d] = x[idx]
            cluster_device_labels[c][d] = y[idx]
    return cluster_device_data, cluster_device_labels
