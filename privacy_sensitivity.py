import numpy as np
import itertools
import copy
from scipy.special import lambertw
from scipy.optimize import curve_fit, least_squares
import os
import logging
import pandas as pd

from multi_split import split_non_iid_multi_cluster

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42   # Embed TrueType fonts
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Times New Roman'
import matplotlib.pyplot as plt

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# Load MNIST
# -----------------------------
data = np.load('mnist.npz')
x_train, y_train = data['x_train'], data['y_train']

# -----------------------------
# Output path
# -----------------------------
path = 'new_results/emds_privacy3'
os.makedirs(path, exist_ok=True)

# -----------------------------
# Parameters
# -----------------------------
num_clusters = 1
devices_per_cluster = 10
total_devices = num_clusters * devices_per_cluster
communication_rounds = 10
privacy_sensitivities = [1.0, 2.0, 3.0, 4.0]
E_k = 6  # can be scalar or dict per device
gammaB_values = np.arange(100, 4501, 500)
# gammaB_values = np.arange(500, 1501, 500)

# -----------------------------
# Precompute split + device EMD once
# -----------------------------
np.random.seed(0)

strategies = ["dirichlet_low"]
cluster_device_data, cluster_device_labels = split_non_iid_multi_cluster(
    x_train, y_train,
    num_clusters=num_clusters,
    devices_per_cluster=devices_per_cluster,
    strategies=strategies,
    balanced_sizes=False,       # allow quantity skew
    dirichlet_alpha=0.03,       # spiky
    label_k=2
)

server_dist = np.bincount(y_train, minlength=10).astype(float)
server_dist /= server_dist.sum()

# device-level EMD from server distribution (L1)
device_emd = {}
for c in range(num_clusters):
    device_emd[c] = {}
    for d in range(devices_per_cluster):
        labels = cluster_device_labels[c][d]
        device_dist = np.bincount(labels, minlength=len(server_dist)).astype(float)
        device_dist /= max(device_dist.sum(), 1e-12)
        emd = np.sum(np.abs(device_dist - server_dist))  # in [0, 2]
        device_emd[c][d] = emd

GLOBAL_NUM_CLASSES = len(server_dist)

# -----------------------------
# Helpers
# -----------------------------
def coalition_signature_map(partition):
    """
    Map each device -> signature (sorted tuple of coalition members).
    """
    sig_map = {}
    for coalition in partition:
        sig = tuple(sorted(coalition))
        for d in coalition:
            sig_map[d] = sig
    return sig_map

def safe_acc_loss(M_prime, D_S, L, eta=0.01, phi=0.1, T_local=10, beta=0.1, G=1.0, epsilon=10.0):
    """
    Acc. bound (not used for plotting, but left intact for completeness)
    """
    den = (1 - (beta * eta / 2) - (L * G * D_S) / (epsilon ** 2))
    if M_prime <= 0 or T_local <= 1 or den <= 0:
        return float('inf')
    return 1.0 / (M_prime * eta * phi * (T_local - 1) * den)

# container to store curves for each (cluster, gammaB, privacy)
plot_data = {cid: {g: {} for g in gammaB_values} for cid in range(num_clusters)}

# -----------------------------
# Main loop with coalition local search
# -----------------------------
for privacy_sensitivity in privacy_sensitivities:
    logging.info(f"Running with privacy_sensitivity={privacy_sensitivity}")

    for cluster_id in range(num_clusters):
        N_k = devices_per_cluster
        L = communication_rounds

        # calc_utility captures cluster_id & num_classes
        def calc_utility(partition, r):
            """
            Returns per-device utility and reward (dicts).
            Reward sharing proportional to c_k; coalition cost scales with coalition size and privacy_sensitivity.
            """
            utilities, rewards = {}, {}
            for coalition in partition:
                # aggregate labels within the coalition
                total_labels = np.zeros(GLOBAL_NUM_CLASSES, dtype=float)
                total_samples = 0
                for d in coalition:
                    lbls = cluster_device_labels[cluster_id][d]
                    if len(lbls) == 0:
                        continue
                    total_labels += np.bincount(lbls, minlength=GLOBAL_NUM_CLASSES)
                    total_samples += len(lbls)
                if total_samples == 0:
                    continue

                label_dist = total_labels / total_samples
                d_bar = np.sum(np.abs(label_dist - server_dist))   # L1 distance in [0,2]
                c_p = 1.0 - d_bar / 2.0                           # coalition quality in [0,1]
                alpha = 3.0
                # non-linear boost for higher-quality coalitions
                new_c_p = (np.exp(alpha * c_p) - 1.0) / (np.exp(alpha) - 1.0)

                # per-device quality scores
                c_k_cache = {}
                score_sum = 0.0
                for di in coalition:
                    d_k_i = device_emd[cluster_id][di]  # in [0,2]
                    c_k_i = 1.0 - d_k_i / 2.0           # in [0,1]
                    c_k_cache[di] = c_k_i
                    score_sum += c_k_i
                score_sum = max(score_sum, 1e-12)

                # distribute reward; compute per-device utility
                for d in coalition:
                    c_k = c_k_cache[d]
                    R_k = (c_k / score_sum) * new_c_p * r
                    L_k = (len(coalition) - 1) * c_k * privacy_sensitivity
                    E_k_d = E_k[d] if isinstance(E_k, dict) else E_k
                    u_k = R_k - L_k - E_k_d
                    utilities[d] = u_k
                    # "sticky" reward baseline (min cost = E_k)
                    rewards[d] = R_k  #if R_k > E_k_d else E_k_d
            return utilities, rewards

        # sweep over gammaB
        for gammaB in gammaB_values:
            EMDs = []  # (r, GEMD) points

            # budget-implied r range
            r_start = gammaB_values[0] / (L * N_k)
            r_end   = gammaB / L
            if r_start > r_end:
                r_start, r_end = r_end, r_start

            # sticky rewards (update only if coalition membership changes)
            sticky_rewards = {d: 0.0 for d in range(N_k)}
            prev_sig_map = None
            prev_partition = None
            prev_r = None

            # sweep r; (use step=50 like your original)
            for r in np.arange(r_start, r_end + 1, 20):
                # -------- Local search: start from singletons --------
                partition_updated = [[i] for i in range(N_k)]
                partition_prev = []

                while partition_updated != partition_prev:
                    partition_prev = copy.deepcopy(partition_updated)

                    # ---- Merge ----
                    merged = False
                    for i in range(len(partition_updated)):
                        for j in range(i + 1, len(partition_updated)):
                            sx = partition_updated[i]
                            sx_ = partition_updated[j]
                            snew = sx + sx_
                            current_util, _ = calc_utility([sx, sx_], r)
                            merged_util,  _ = calc_utility([snew], r)
                            # Pareto-improve for all devices in snew
                            if all(merged_util.get(d, -np.inf) >= current_util.get(d, -np.inf) for d in snew):
                                partition_updated = [p for k, p in enumerate(partition_updated) if k not in (i, j)]
                                partition_updated.append(snew)
                                merged = True
                                break
                        if merged:
                            break
                    if merged:
                        continue

                    # ---- Split ----
                    splitted = False
                    for idx, coalition in enumerate(partition_updated):
                        if len(coalition) > 1:
                            # quick pass: try carve out 1 or leave 1
                            for split_size in (1, len(coalition) - 1):
                                if split_size <= 0:
                                    continue
                                for subset in itertools.combinations(coalition, split_size):
                                    subset = list(subset)
                                    remaining = list(set(coalition) - set(subset))
                                    if not remaining:
                                        continue
                                    current_util, _ = calc_utility([coalition], r)
                                    split_util,   _ = calc_utility([subset, remaining], r)
                                    if all(split_util.get(d, -np.inf) >= current_util.get(d, -np.inf) for d in coalition):
                                        partition_updated = [p for k, p in enumerate(partition_updated) if k != idx]
                                        partition_updated.append(subset)
                                        partition_updated.append(remaining)
                                        splitted = True
                                        break
                                if splitted:
                                    break
                        if splitted:
                            break
                    if splitted:
                        continue

                    # ---- Switch ----
                    switched = False
                    for i, coalition in enumerate(partition_updated):
                        if len(coalition) > 1:
                            for d in list(coalition):
                                for j, target in enumerate(partition_updated):
                                    if i == j:
                                        continue
                                    new_i = list(set(coalition) - {d})
                                    new_j = target + [d]
                                    current_util, _ = calc_utility([coalition, target], r)
                                    new_util,     _ = calc_utility([new_i, new_j], r)
                                    # Pareto-improve for members of new_i + new_j
                                    if all(new_util.get(x, -np.inf) >= current_util.get(x, -np.inf) for x in new_i + new_j):
                                        partition_updated[i] = new_i
                                        partition_updated[j] = new_j
                                        switched = True
                                        break
                                if switched:
                                    break
                        if switched:
                            break

                    # if no operation applied, stop
                    if not (merged or splitted or switched):
                        break
                # -------- end local search --------

                # compute rewards for current partition
                _, rewards_now = calc_utility(partition_updated, r)
                sig_map_now = coalition_signature_map(partition_updated)

                # sticky update
                device_rewards_tmp = sticky_rewards.copy()
                if prev_sig_map is None:
                    for d in range(N_k):
                        device_rewards_tmp[d] = rewards_now.get(d, 0.0)
                else:
                    for d in range(N_k):
                        if sig_map_now.get(d) != prev_sig_map.get(d):
                            device_rewards_tmp[d] = rewards_now.get(d, 0.0)
                        # else: keep sticky

                # enforce per-round budget
                total_payment = sum(device_rewards_tmp.values())
                if total_payment > (gammaB / L):
                    break  # stop increasing r for this gammaB

                # commit step
                sticky_rewards = device_rewards_tmp
                prev_sig_map = sig_map_now
                prev_partition = copy.deepcopy(partition_updated)
                prev_r = r

                # ---- Compute GEMD (dataset heterogeneity across coalitions) ----
                coalition_emd = []
                samples_per_coalition = []
                total_samples_in_all_coalitions = 0

                for coalition in partition_updated:
                    total_labels = np.zeros(GLOBAL_NUM_CLASSES, dtype=float)
                    total_samples = 0
                    for d in coalition:
                        lbls = cluster_device_labels[cluster_id][d]
                        if len(lbls) == 0:
                            continue
                        total_labels += np.bincount(lbls, minlength=GLOBAL_NUM_CLASSES)
                        total_samples += len(lbls)
                    if total_samples == 0:
                        continue
                    label_dist = total_labels / total_samples
                    d_bar = np.sum(np.abs(label_dist - server_dist))  # coalition EMD to server
                    coalition_emd.append(d_bar)
                    samples_per_coalition.append(total_samples)
                    total_samples_in_all_coalitions += total_samples

                if total_samples_in_all_coalitions == 0:
                    D_S = 0.0
                else:
                    # weighted average coalition EMD (GEMD)
                    D_S = sum((s / total_samples_in_all_coalitions) * d
                              for s, d in zip(samples_per_coalition, coalition_emd))

                # store (r, GEMD)
                EMDs.append([r, D_S])

            # after sweeping r for this gammaB, stash curves for plotting
            rewards_arr = [row[0] for row in EMDs]
            D_S_values = [row[1] for row in EMDs]
            plot_data[cluster_id][gammaB][privacy_sensitivity] = {
                "reward": rewards_arr,
                "gemd": D_S_values
            }

# -----------------------------
# Overlay plots per (cluster, gammaB)
# -----------------------------
for cluster_id in range(num_clusters):
    for gammaB in gammaB_values:
        curves = plot_data[cluster_id][gammaB]
        if not curves:
            continue

        plt.figure(figsize=(10, 6), dpi=600)
        for privacy in sorted(curves.keys()):
            arr = curves[privacy]
            if len(arr["reward"]) == 0:
                continue
            plt.plot(arr["reward"], arr["gemd"], marker='o', label=f"privacy={privacy}")

        plt.xlabel("Reward", fontsize=14)
        plt.ylabel("GEMD", fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12, title="Privacy sensitivity")
        out = os.path.join(path, f"GEMD_vs_Reward_cluster{cluster_id}_gammaB_{gammaB}_by_privacy.pdf")
        plt.savefig(out, dpi=600, bbox_inches='tight')
        plt.close()

logging.info("All plots saved.")
