import numpy as np
# from google.colab import files
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


# Configure logging once at the start of your script
logging.basicConfig(
    level=logging.INFO,                      # Set minimum log level
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Upload and load MNIST
# uploaded = files.upload()
data = np.load('mnist.npz')
x_train, y_train = data['x_train'], data['y_train']
path='new_results/benchmark1'
os.makedirs(path,exist_ok=True)
# Parameters
num_clusters = 1
devices_per_cluster = 10
total_devices = num_clusters * devices_per_cluster
communication_rounds = 10
privacy_sensitivity = 1  #{0.1,0.5,0.75,1.0,1.25,1.5}
E_k = 6

# Shuffle and partition using Dirichlet (non-iid)
indices = np.arange(len(x_train))
np.random.shuffle(indices)

alpha = 0.1
# cluster_device_data,cluster_device_labels = balanced_dirichlet_split(x_train,y_train,num_clusters, devices_per_cluster,alpha)

# Example: 4 clusters, each with a different scheme
# strategies = ["dirichlet_low", "label_k", "majority", "zipf_major"]
strategies = ["dirichlet_low"]
cluster_device_data,cluster_device_labels = split_non_iid_multi_cluster(
    x_train,y_train,
    num_clusters=num_clusters,
    devices_per_cluster=devices_per_cluster,
    strategies=strategies,
    balanced_sizes=False,        # allow quantity skew (use True if you want equal sizes)
    dirichlet_alpha=0.03,        # make Dirichlet extra spiky
    label_k=2                   # single-class devices in that cluster
)

server_dist = np.bincount(y_train, minlength=10) / len(y_train)

device_emd = {}
for c in range(num_clusters):
    device_emd[c] = {}
    for d in range(devices_per_cluster):
        labels = cluster_device_labels[c][d]
        device_dist = np.bincount(labels, minlength=10) / len(labels)
        emd = np.sum(np.abs(device_dist - server_dist))
        device_emd[c][d] = emd

gammaB_values = np.arange(2000, 8000, 200) # np.arange(100, 3000, 200)

# --- ASSUMED GLOBALS (provide these) ---
# num_clusters, devices_per_cluster, communication_rounds
# cluster_device_labels: dict[cluster_id][device_id] -> 1D array of labels
# device_emd: dict[cluster_id][device_id] -> float in [0,2]
# server_dist: 1D array with length = GLOBAL_NUM_CLASSES (normalized to sum=1)
# privacy_sensitivity: float
# E_k: float OR dict[device_id] -> float
# gammaB_values: iterable of gammaB
# path: output folder for plots

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

def safe_acc_loss(M_prime, D_S, L=1, eta=0.01, phi=0.1, T_local=10, beta=0.1, G=1.0, epsilon=10.0):
    den = (1 - (beta * eta / 2) - (L * G * D_S) / (epsilon ** 2))
    if M_prime <= 0 or T_local <= 1 or den <= 0:
        return float('inf')
    return 1.0 / (M_prime * eta * phi * (T_local - 1) * den)

# -------------------------------------------------------------------
# Main run (Version A: sticky rewards update only on coalition change)
# -------------------------------------------------------------------
cluster_results = []
gammaB_lists = [[] for _ in range(num_clusters)]
acc_losses_list = [[] for _ in range(num_clusters)]
all_combinations_list = [[] for _ in range(num_clusters)]
all_best_combinations_list = [[] for _ in range(num_clusters)]

GLOBAL_NUM_CLASSES = int(len(server_dist))
server_dist = np.asarray(server_dist, dtype=float)
if server_dist.sum() <= 0:
    raise ValueError("server_dist must have positive mass.")
server_dist = server_dist / server_dist.sum()  # normalize defensively

for cluster_id in range(num_clusters):
    best_combinations = []
    all_combinations = []
    local_cost=[]

    # ---- Use GLOBAL class count (fixes your mismatch) ----
    num_classes = GLOBAL_NUM_CLASSES

    N_k = devices_per_cluster
    L = communication_rounds

    # Optional sanity: make sure labels are within range
    max_label_in_cluster = max(int(np.max(lbls)) if len(lbls) else -1
                               for lbls in cluster_device_labels[cluster_id].values())
    if max_label_in_cluster >= num_classes:
        raise ValueError(
            f"Found label {max_label_in_cluster} in cluster {cluster_id} "
            f"but server_dist expects classes [0..{num_classes-1}]."
        )

    # calc_utility captures cluster_id & num_classes
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
            d_bar = np.sum(np.abs(label_dist - server_dist))   # L1 distance
            c_p = 1 - d_bar / 2.0
            alpha = 3.0
            new_c_p = (np.exp(alpha * c_p) - 1.0) / (np.exp(alpha) - 1.0)

            c_k_cache = {}
            score_sum = 0.0
            for di in coalition:
                d_k_i = device_emd[cluster_id][di]  # in [0,2]
                c_k_i = 1.0 - d_k_i / 2.0
                c_k_cache[di] = c_k_i
                score_sum += c_k_i
            score_sum = max(score_sum, 1e-12)

            for d in coalition:
                c_k = c_k_cache[d]
                R_k = (c_k / score_sum) * new_c_p * r
                L_k = (len(coalition) - 1) * c_k * privacy_sensitivity
                # Support E_k as scalar or dict
                E_k_d = E_k[d] if isinstance(E_k, dict) else E_k
                u_k = R_k - L_k - E_k_d
                utilities[d] = u_k
                rewards[d] = R_k if R_k > E_k_d else E_k_d
        return utilities, rewards

    for gammaB in gammaB_values:
        # Sticky rewards per device (only update on coalition change)
        sticky_rewards = {d: 0.0 for d in range(N_k)}
        prev_sig_map = None
        prev_partition = None
        prev_r = None

        # r grid (ascending) — use this (you had it but then looped with arange)
        r_start = gammaB_values[0] / (L * N_k)
        r_end   = gammaB / L
        if r_start > r_end:
            r_start, r_end = r_end, r_start
        r_grid = np.linspace(r_start, r_end, num=40, endpoint=True)

        r_list = []
        gammaB_list_local = []
        acc_losses = []
        r_fixed=50

        for r in np.arange(r_fixed, r_fixed + 1, 50):
            # If you prefer continuity across r, reuse prev_partition:
            # partition_updated = copy.deepcopy(prev_partition) if prev_partition is not None else [[i] for i in range(N_k)]
            partition_updated = [[i] for i in range(N_k)]
            partition_prev = []

            # --------------- Local search: merge/split/switch ---------------
            while partition_updated != partition_prev:
                partition_prev = copy.deepcopy(partition_updated)

                # Merge
                merged = False
                for i in range(len(partition_updated)):
                    for j in range(i + 1, len(partition_updated)):
                        sx = partition_updated[i]
                        sx_ = partition_updated[j]
                        snew = sx + sx_
                        current_util, _ = calc_utility([sx, sx_], r)
                        merged_util,  _ = calc_utility([snew], r)
                        if all(merged_util.get(d, -np.inf) >= current_util.get(d, -np.inf) for d in snew):
                            partition_updated = [p for k, p in enumerate(partition_updated) if k not in (i, j)]
                            partition_updated.append(snew)
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    continue

                # Split
                splitted = False
                for idx, coalition in enumerate(partition_updated):
                    if len(coalition) > 1:
                        for split_size in (1, len(coalition) - 1):  # quick first pass
                            if split_size <= 0:
                                continue
                            for subset in itertools.combinations(coalition, split_size):
                                remaining = list(set(coalition) - set(subset))
                                if not remaining:
                                    continue
                                current_util, _ = calc_utility([coalition], r)
                                split_util,   _ = calc_utility([list(subset), remaining], r)
                                if all(split_util.get(d, -np.inf) >= current_util.get(d, -np.inf) for d in coalition):
                                    partition_updated = [p for k, p in enumerate(partition_updated) if k != idx]
                                    partition_updated.append(list(subset))
                                    partition_updated.append(remaining)
                                    splitted = True
                                    break
                            if splitted:
                                break
                    if splitted:
                        break
                if splitted:
                    continue

                # Switch
                switched = False
                for i, coalition in enumerate(partition_updated):
                    if len(coalition) > 1:
                        for d in coalition:
                            for j, target in enumerate(partition_updated):
                                if i == j:
                                    continue
                                new_i = list(set(coalition) - {d})
                                new_j = target + [d]
                                current_util, _ = calc_utility([coalition, target], r)
                                new_util,     _ = calc_utility([new_i, new_j], r)
                                if all(new_util.get(x, -np.inf) >= current_util.get(x, -np.inf) for x in new_i + new_j):
                                    partition_updated[i] = new_i
                                    partition_updated[j] = new_j
                                    switched = True
                                    break
                            if switched:
                                break
                    if switched:
                        break
                if not (merged or splitted or switched):
                    break
            # --------------- end local search ---------------

            # Current rewards & sticky update on coalition change
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
                    # else keep sticky

            total_payment = sum(device_rewards_tmp.values())
            if total_payment > (gammaB / L):
                break  # don’t commit, stop increasing r for this gammaB

            # Commit feasible step
            sticky_rewards = device_rewards_tmp
            prev_sig_map = sig_map_now
            prev_partition = copy.deepcopy(partition_updated)
            prev_r = r

            # M', D_S, acc loss4
            extra_round_cost_factor = 1
            numerator = ((gammaB / L) - total_payment) * L
            denominator = max(total_payment, 1e-12)
            M_prime = max(1, int(np.floor((numerator / denominator) * extra_round_cost_factor) + L))

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
            

            phi, epsilon, G, eta, beta, T_local = 0.1, 10.0, 1.0, 0.01, 0.1, 10
            acc_loss_bound = safe_acc_loss(M_prime, D_S, 1, eta, phi, T_local, beta, G, epsilon)
            # print(f'D_S={D_S} when r={r} and B={gammaB} and M`={M_prime} and loss={acc_loss_bound}')

            acc_losses.append(acc_loss_bound)
            gammaB_list_local.append(gammaB)
            r_list.append([r, acc_loss_bound, D_S,M_prime])

            all_combinations.append({
                "gammaB": gammaB,
                "acc_loss": acc_loss_bound,
                "partition": copy.deepcopy(partition_updated),
                "r": r,
            })

        # pick best for this gammaB (if any feasible r)
        if r_list:
            best_r = min(r_list, key=lambda x: x[1])[0]
            best_combination = next(e for e in all_combinations
                                    if (e["r"] == best_r and e["gammaB"] == gammaB))
            best_combinations.append(best_combination)

    # After sweeping all gammaB for this cluster
    if best_combinations:
        gamma_values = [e["gammaB"] for e in best_combinations]
        acc_losses_value = [e["acc_loss"] for e in best_combinations]
        gammaB_lists[cluster_id] = gamma_values
        acc_losses_list[cluster_id] = acc_losses_value
        all_best_combinations_list[cluster_id]=best_combinations
        all_combinations_list[cluster_id] = all_combinations
        # local_cost=float(acc_losses_value)+ 0.1*float(gamma_values)
        local_cost=[acc_loss+ 0.0005*gamma_value for (acc_loss,gamma_value) in zip(acc_losses_value,gamma_values)]
        best_idx = np.nanargmin(local_cost)
        optimal_B = gamma_values[best_idx]
        optimal_loss =acc_losses_value[best_idx]
        print(f'optimal budget is {optimal_B} and optimal loss is {optimal_loss}')
        best_comb = min(best_combinations, key=lambda x: abs(x["gammaB"] - optimal_B))
        optimal_info=[{
            "Optimal Budget": best_comb["gammaB"],
            "Optimal Loss": best_comb["acc_loss"],
            "Optimal R": best_comb["r"],
            "Optimal Partition": best_comb["partition"],
            "Optimal Cost": local_cost[best_idx]

        }]
        print(f'optimal information  {optimal_info}')


        # Save plotted data to CSV
        df = pd.DataFrame({
            "gammaB": gamma_values,
            "Accuracy Loss": acc_losses_value
        })
        df.to_csv(f"{path}/cluster{cluster_id}_gammaB_Loss.csv", index=False)
        df = pd.DataFrame({
            "gammaB": gamma_values,
            "Cost": local_cost
        })
        df.to_csv(f"{path}/cluster{cluster_id}_gammaB_cost.csv", index=False)
        # Save plotted data to CSV
        df = pd.DataFrame(optimal_info)
        df.to_csv(f"{path}/optimal_info.csv", index=False)

        plt.figure(figsize=(10, 8))
        plt.plot(gamma_values, acc_losses_value, marker='o')
        plt.xlabel("Budget")
        plt.ylabel("Accuracy Loss")
        plt.title(f"Budget vs. Accuracy Loss (cluster {cluster_id})")
        plt.grid(True)
        plt.savefig(f"{path}/cluster{cluster_id}_gammaB_Loss.png")
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.plot(gamma_values, local_cost, marker='o')
        plt.xlabel("Budget")
        plt.ylabel("Cost")
        plt.title(f"Budget vs. Cost (cluster {cluster_id})")
        plt.grid(True)
        plt.savefig(f"{path}/cluster{cluster_id}_cost.png")
        plt.close()



