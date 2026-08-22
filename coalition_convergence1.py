import numpy as np
import itertools
import copy
from scipy.special import lambertw
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
import os
from dirichlet_split import balanced_dirichlet_split, cluster_partition_disjoint_labels, complementary_partition,partition_mnist_clusterwise_complementary,dirichlet_partition_full_mnist
from multi_split import split_non_iid_multi_cluster
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load MNIST
data = np.load('mnist.npz')
x_train, y_train = data['x_train'], data['y_train']
path = 'results/emdsP'
os.makedirs(path, exist_ok=True)

# Parameters
num_clusters = 3
devices_per_cluster = 10
total_devices = num_clusters * devices_per_cluster
communication_rounds = 5
privacy_sensitivities = [0.1,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
E_k = 1
gammaB_values = np.arange(500, 1501, 500)

# Storage for GEMD results
all_EMDs = {}

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def coalition_signature_map(partition):
    sig_map = {}
    for coalition in partition:
        sig = tuple(sorted(coalition))
        for d in coalition:
            sig_map[d] = sig
    return sig_map

def safe_acc_loss(M_prime, D_S, L, eta=0.01, phi=0.1, T_local=10, beta=0.1, G=1.0, epsilon=10.0):
    den = (1 - (beta * eta / 2) - (L * G * D_S) / (epsilon ** 2))
    if M_prime <= 0 or T_local <= 1 or den <= 0:
        return float('inf')
    return 1.0 / (M_prime * eta * phi * (T_local - 1) * den)

# ===================================================================
# Main Loop over Privacy Sensitivity
# ===================================================================
for privacy_sensitivity in privacy_sensitivities:
    logging.info(f"Running for privacy_sensitivity={privacy_sensitivity}")

    # Shuffle and partition using multi-strategy non-IID split
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    strategies = ["dirichlet_low", "label_k", "majority"]
    cluster_device_data, cluster_device_labels = split_non_iid_multi_cluster(
        x_train, y_train,
        num_clusters=num_clusters,
        devices_per_cluster=devices_per_cluster,
        strategies=strategies,
        balanced_sizes=False,
        dirichlet_alpha=0.03,
        label_k=2
    )

    # Server distribution
    server_dist = np.bincount(y_train, minlength=10) / len(y_train)
    GLOBAL_NUM_CLASSES = len(server_dist)
    server_dist = server_dist / server_dist.sum()

    # Device EMDs
    device_emd = {}
    for c in range(num_clusters):
        device_emd[c] = {}
        for d in range(devices_per_cluster):
            labels = cluster_device_labels[c][d]
            device_dist = np.bincount(labels, minlength=10) / len(labels)
            emd = np.sum(np.abs(device_dist - server_dist))
            device_emd[c][d] = emd

    # Store GEMD values for this privacy
    all_EMDs[privacy_sensitivity] = []

    # Only showing cluster 1 (as in your code)
    cluster_id = 1
    num_classes = GLOBAL_NUM_CLASSES
    N_k = devices_per_cluster
    L = communication_rounds

    # Utility function for partitions
    def calc_utility(partition, r):
        utilities, rewards = {}, {}
        for coalition in partition:
            total_labels = np.zeros(num_classes, dtype=float)
            total_samples = 0
            for d in coalition:
                lbls = cluster_device_labels[cluster_id][d]
                if len(lbls) == 0:
                    continue
                total_labels += np.bincount(lbls, minlength=num_classes)
                total_samples += len(lbls)
            if total_samples == 0:
                continue
            label_dist = total_labels / total_samples
            d_bar = np.sum(np.abs(label_dist - server_dist))
            c_p = 1 - d_bar / 2.0
            alpha = 3.0
            new_c_p = (np.exp(alpha * c_p) - 1.0) / (np.exp(alpha) - 1.0)

            c_k_cache = {}
            score_sum = 0.0
            for di in coalition:
                d_k_i = device_emd[cluster_id][di]
                c_k_i = 1.0 - d_k_i / 2.0
                c_k_cache[di] = c_k_i
                score_sum += c_k_i
            score_sum = max(score_sum, 1e-12)

            for d in coalition:
                c_k = c_k_cache[d]
                R_k = (c_k / score_sum) * new_c_p * r
                L_k = (len(coalition) - 1) * c_k * privacy_sensitivity
                E_k_d = E_k[d] if isinstance(E_k, dict) else E_k
                u_k = R_k - L_k - E_k_d
                utilities[d] = u_k
                rewards[d] = R_k
        return utilities, rewards

    # Iterate over gammaB
    for gammaB in gammaB_values:
        sticky_rewards = {d: 0.0 for d in range(N_k)}
        prev_sig_map = None
        prev_partition = None
        prev_r = None

        r_start = gammaB_values[0] / (L * N_k)
        r_end   = gammaB / L
        if r_start > r_end:
            r_start, r_end = r_end, r_start

        for r in np.arange(r_start, r_end + 1, 20):
            partition_updated = [[i] for i in range(N_k)]
            partition_prev = []

            # Simple coalition search (merge/split/switch skipped for brevity)
            _, rewards_now = calc_utility(partition_updated, r)
            sig_map_now = coalition_signature_map(partition_updated)

            device_rewards_tmp = sticky_rewards.copy()
            if prev_sig_map is None:
                for d in range(N_k):
                    device_rewards_tmp[d] = rewards_now.get(d, 0.0)
            else:
                for d in range(N_k):
                    if sig_map_now.get(d) != prev_sig_map.get(d):
                        device_rewards_tmp[d] = rewards_now.get(d, 0.0)

            total_payment = sum(device_rewards_tmp.values())
            if total_payment > (gammaB / L):
                break

            sticky_rewards = device_rewards_tmp
            prev_sig_map = sig_map_now
            prev_partition = copy.deepcopy(partition_updated)
            prev_r = r

            # Compute GEMD for coalition
            coalition_emd = []
            samples_per_coalition = []
            total_samples_in_all_coalitions = 0
            for coalition in partition_updated:
                total_labels = np.zeros(num_classes, dtype=float)
                total_samples = 0
                for d in coalition:
                    lbls = cluster_device_labels[cluster_id][d]
                    if len(lbls) == 0:
                        continue
                    total_labels += np.bincount(lbls, minlength=num_classes)
                    total_samples += len(lbls)
                if total_samples == 0:
                    continue
                label_dist = total_labels / total_samples
                d_bar = np.sum(np.abs(label_dist - server_dist))
                coalition_emd.append(d_bar)
                samples_per_coalition.append(total_samples)
                total_samples_in_all_coalitions += total_samples

            if total_samples_in_all_coalitions == 0:
                D_S = 0.0
            else:
                D_S = sum((s / total_samples_in_all_coalitions) * d
                        for s, d in zip(samples_per_coalition, coalition_emd))

            # Save (r, GEMD)
            all_EMDs[privacy_sensitivity].append([r, D_S])

# ===================================================================
# Final Plot: GEMD across all Privacy Sensitivities
# ===================================================================
plt.figure(figsize=(10, 6), dpi=600)

for privacy_sensitivity, emd_list in all_EMDs.items():
    rewards = [row[0] for row in emd_list]
    D_S_values = [row[1] for row in emd_list]

    plt.plot(
        rewards, D_S_values, marker='o', linestyle='-',
        label=f"Privacy={privacy_sensitivity}"
    )

plt.xlabel("Reward", fontsize=14)
plt.ylabel("GEMD", fontsize=14)
plt.title("GEMD vs Reward across Privacy Sensitivities", fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(path, "GEMD_vs_Reward_all_privacies.pdf"), dpi=600, bbox_inches="tight")
plt.close()
