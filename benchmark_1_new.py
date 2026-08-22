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
import pandas as pd

# Upload and load MNIST
# uploaded = files.upload()

from torchvision import datasets

# Load CIFAR-10
cifar_train = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)

x_train = cifar_train.data                    # shape: (50000, 32, 32, 3)
y_train = np.array(cifar_train.targets)       # shape: (50000,)


# data = np.load('mnist.npz')
# x_train, y_train = data['x_train'], data['y_train']
path = 'results_new1/cifar10/equal_b/result'
os.makedirs(path,exist_ok=True)

# Parameters
num_clusters = 10
devices_per_cluster = 20
total_devices = num_clusters * devices_per_cluster
communication_rounds = 10
privacy_sensitivity = 1
E_k = 3

# Shuffle and partition using Dirichlet (non-iid)
indices = np.arange(len(x_train))
np.random.shuffle(indices)

alpha = 0.1
# cluster_device_data,cluster_device_labels = balanced_dirichlet_split(x_train,y_train,num_clusters, devices_per_cluster,alpha)

strategies = ["dirichlet_low", "label_k", "majority", "zipf_major","dirichlet_low", "label_k", "majority","dirichlet_low", "label_k", "zipf_major"]
# strategies = ["dirichlet_low", "label_k", "majority"]
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

gammaB_values =np.arange(1000, 10000, 200) #(1000, 5000, 100) # np.arange(100, 3000, 200)

cluster_results = []
gammaB_lists=[[] for _ in range(num_clusters)]
acc_losses_list=[[] for _ in range(num_clusters)]
all_combinations_list = [[] for _ in range(num_clusters)]
all_best_combinations_list=[[] for _ in range(num_clusters)]

for cluster_id in range(num_clusters):
    acc_losses = []
    gammaB_list = []
    all_combinations = []
    best_combinations =[]
    for gammaB in gammaB_values:
        r_list=[]
        # r_list1=[]
        best_combination= []
        N_k = devices_per_cluster
        L = communication_rounds

        prev_partition = None
        prev_r = None

        r_start = gammaB_values[0] / (L * N_k)
        r_end = gammaB / L

        if r_start > r_end:
            r_start, r_end = r_end, r_start
        
        device_rewards = {d: r_start for d in range(N_k)}
        check_m = []
        for r in np.arange(r_start, r_end + 1, 100):
            partition_updated = [[i] for i in range(N_k)]
            partition_prev = []
            
            def get_device_coalition_map(partition):
                mapping = {}
                for idx, coalition in enumerate(partition):
                    for d in coalition:
                        mapping[d] = idx
                return mapping
            def calc_utility(partition):
                utilities = {}
                rewards = {}
                for coalition in partition:
                    total_labels = np.zeros(10)
                    total_samples = 0
                    for d in coalition:
                        lbls = cluster_device_labels[cluster_id][d]
                        total_labels += np.bincount(lbls, minlength=10)
                        total_samples += len(lbls)
                    if total_samples == 0:
                        continue
                    label_dist = total_labels / total_samples
                    d_bar = np.sum(np.abs(label_dist - server_dist))
                    c_p = 1 - d_bar / 2
                    alpha=3
                    new_c_p = (np.exp(alpha * c_p) - 1) / (np.exp(alpha) - 1)

                    for d in coalition:
                        d_k = device_emd[cluster_id][d]
                        c_k = 1 - d_k / 2
                        score_sum = sum(1 - device_emd[cluster_id][di] / 2 for di in coalition)
                        R_k = (c_k / score_sum) * new_c_p * r
                        # Support E_k as scalar or dict
                        E_k_d = E_k[d] if isinstance(E_k, dict) else E_k
                        L_k = (len(coalition) - 1) * c_k * privacy_sensitivity
                        u_k = R_k - L_k - E_k
                        utilities[d] = u_k
                        rewards[d] = R_k if R_k > E_k_d else E_k_d

                return utilities, rewards

            if r==r_start:
               _, device_rewards = calc_utility(partition_updated)

            old_mapping = get_device_coalition_map(partition_updated)
            while partition_updated != partition_prev:
                partition_prev = copy.deepcopy(partition_updated)

                # Merge
                merged = False
                for i in range(len(partition_updated)):
                    for j in range(i + 1, len(partition_updated)):
                        sx = partition_updated[i]
                        sx_ = partition_updated[j]
                        snew = sx + sx_

                        current_util, _ = calc_utility([sx, sx_])
                        merged_util, _ = calc_utility([snew])

                        if all(merged_util[d] >= current_util[d] for d in snew):
                            partition_updated = [p for k, p in enumerate(partition_updated) if k != i and k != j]
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
                        for split_size in range(1, len(coalition)):
                            for subset in itertools.combinations(coalition, split_size):
                                remaining = list(set(coalition) - set(subset))
                                if not remaining:
                                    continue

                                current_util, _ = calc_utility([coalition])
                                split_util, _ = calc_utility([list(subset), remaining])

                                if all(split_util[d] >= current_util[d] for d in coalition):
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
                            for j, target_coalition in enumerate(partition_updated):
                                if i != j:
                                    new_coalition_i = list(set(coalition) - {d})
                                    new_coalition_j = target_coalition + [d]

                                    current_util, _ = calc_utility([coalition, target_coalition])
                                    new_util, _ = calc_utility([new_coalition_i, new_coalition_j])

                                    if all(new_util[x] >= current_util[x] for x in new_coalition_i + new_coalition_j):
                                        partition_updated[i] = new_coalition_i
                                        partition_updated[j] = new_coalition_j
                                        switched = True
                                        break
                            if switched:
                                break
                    if switched:
                        break
                if not (merged or splitted or switched):
                    break

            new_mapping = get_device_coalition_map(partition_updated)
            _, new_rewards = calc_utility(partition_updated)

            # Update only if the device has migrated
            for d in range(N_k):
                if new_mapping[d] != old_mapping.get(d):
                    device_rewards[d] = new_rewards.get(d, device_rewards[d])
            # Then update old_mapping for next round
            old_mapping = new_mapping

            utilities, rewards = calc_utility(partition_updated)
            total_payment = sum(device_rewards.values()) #sum(rewards.values())

            if total_payment > gammaB / L:
                partition_updated = prev_partition
                r = prev_r
                break

            prev_partition = copy.deepcopy(partition_updated)
            prev_r = r

            numerator = ((gammaB / L) - total_payment) * L
            denominator = total_payment if total_payment != 0 else 1e-6
            M_prime = int(np.floor(numerator / denominator) + L)
            check_m.append([r,M_prime])

            coalition_emd = []
            total_samples_in_all_coalitions = 0
            samples_per_coalition = []

            for coalition in partition_updated:
                total_labels = np.zeros(10)
                total_samples = 0
                for d in coalition:
                    lbls = cluster_device_labels[cluster_id][d]
                    total_labels += np.bincount(lbls, minlength=10)
                    total_samples += len(lbls)
                if total_samples == 0:
                    continue
                label_dist = total_labels / total_samples
                d_bar = np.sum(np.abs(label_dist - server_dist))
                coalition_emd.append(d_bar)
                samples_per_coalition.append(total_samples)
                total_samples_in_all_coalitions += total_samples

            D_S = sum((samples / total_samples_in_all_coalitions) * d
                    for samples, d in zip(samples_per_coalition, coalition_emd))

            phi, epsilon, G, eta, beta, T_local = 0.1, 1.5, 1, 0.01, 0.1, 10
            acc_loss_bound = 1 / (M_prime * eta * phi * (T_local - 1) *
                                (1 - (beta * eta / 2) - (1 * G * D_S) / (epsilon ** 2)))

            acc_losses.append(acc_loss_bound)
            gammaB_list.append(gammaB)
            r_list.append([r,acc_loss_bound,D_S])

            all_combinations.append({
                "gammaB": gammaB,
                "acc_loss": acc_loss_bound,
                "partition": copy.deepcopy(partition_updated),
                "r": r,
                "M_prime":M_prime
            })


        best_r = min(r_list, key=lambda x: x[1])[0]
        best_combination = [entry for entry in all_combinations if (entry["r"] == best_r and entry["gammaB"]==gammaB)][0]
        best_combinations.append(best_combination)
    gamma_values = [entry["gammaB"] for entry in best_combinations]
    acc_losses_value = [entry["acc_loss"] for entry in best_combinations]
    m_prime_values= [entry["M_prime"] for entry in best_combinations]
    df1 = pd.DataFrame({
        "gammaB": gamma_values,
        "M_prime": m_prime_values
    })
    df1.to_csv(f"{path}/cluster{cluster_id}_gammaB_M_prime.csv", index=False)
    gammaB_lists[cluster_id]=gamma_values
    acc_losses_list[cluster_id]=acc_losses_value
    all_best_combinations_list[cluster_id]=best_combinations
    all_combinations_list[cluster_id]=all_combinations
     # Save plotted data to CSV
    df = pd.DataFrame({
        "gammaB": gamma_values,
        "Accuracy Loss": acc_losses_value
    })
    df.to_csv(f"{path}/cluster{cluster_id}_gammaB_Loss.csv", index=False)
    plt.figure(figsize=(10, 8))
    plt.plot(gamma_values, acc_losses_value, marker='o')
    plt.xlabel("gammaB")
    plt.ylabel("Accuracy Loss")
    plt.title(f"gammaB vs. Accuracy Loss for cluster {cluster_id}")
    plt.grid(True)
    plt.savefig(f"{path}/cluster{cluster_id}_gammaB_Loss.png")
    plt.close()




import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.special import lambertw


params = []

# 1️⃣ Fit exponential curve
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

for i in range(num_clusters):
    gammaB = gammaB_lists[i]
    loss = acc_losses_list[i]
    popt, _ = curve_fit(exp_func, gammaB, loss, p0=[1, 0.01, 0.1], maxfev=5000)
    params.append(popt)

# 2️⃣ Define full system of equations
def full_equations(vars, T, n_list, params):
    equations = []
    for k in range(num_clusters):
        delta_F = vars[3*k]
        gamma_m = vars[3*k + 1]
        Bm = vars[3*k + 2]

        # First equation: allocation rule
        denom = sum(1 / (n_list[j] * vars[3*j]) for j in range(num_clusters))
        eq1=(T/num_clusters)-Bm
        # eq1 = (1 / (n_list[k] * delta_F) / denom) * T - Bm

        # Second equation: exponential curve fit
        eq2 = exp_func(gamma_m * Bm, *params[k]) - delta_F

        # Third equation: Lambert function equation
        A, C, n_k = 1, 1, n_list[k]
        term1 = -np.exp(-C / (A * n_k))
        denom_l = A * n_k * (1 - gamma_m)
        lambert_arg = term1 / denom_l
        lambert_value = lambertw(lambert_arg).real
        L = lambert_value + (C / (A * n_k))
        eq3 = delta_F - (gamma_m * np.exp(L) - np.exp(L))

        equations.extend([eq1, eq2, eq3])

    return equations

# 3️⃣ T values
T_vals = np.arange(12000, 60000, 500) #np.arange(2000, 15000, 200)
avg_losses = []
utilities = []

for T in T_vals:
    n_list = [len(cluster_device_labels[k][0]) for k in range(num_clusters)]
    x0 = []
    for _ in range(num_clusters):
        x0.extend([0.1, 0.5, T / num_clusters])  # delta_F, gamma_m, B_m

    bounds_lower = [1e-6, 0.0, 1e-6] * num_clusters
    bounds_upper = [np.inf, 1.0, np.inf] * num_clusters

    try:
        res = least_squares(
            full_equations,
            x0,
            bounds=(bounds_lower, bounds_upper),
            args=(T, n_list, params),
            max_nfev=5000
        )
        if not res.success:
            raise RuntimeError(f"Solver failed at T={T}")

        vars_opt = res.x
        delta_F_list = [vars_opt[3*k] for k in range(num_clusters)]
        avg_loss = np.mean(delta_F_list)
        utility = avg_loss + 0.0002 * T # 0.00001  0.0000213359

        avg_losses.append(avg_loss)
        utilities.append(utility)
    except Exception as e:
        print(f"Failed at T={T}: {e}")
        avg_losses.append(np.nan)
        utilities.append(np.nan)

# Save plotted data to CSV
df = pd.DataFrame({
    "T_val": T_vals,
    "Average Loss": avg_losses
})
df.to_csv(f"{path}/average_loss_vs_T.csv", index=False)

# Save plotted data to CSV
df = pd.DataFrame({
    "T_vals": T_vals,
    "Utility": utilities
})
df.to_csv(f"{path}/utility_vs_T.csv", index=False)

# 4️⃣ Plot
plt.figure(figsize=(10, 8))
plt.plot(T_vals, avg_losses, marker='o', label='Average Loss')
plt.xlabel("T (Total Payment to Main Server)")
plt.ylabel("Average Loss")
plt.title("Average Loss vs T")
plt.grid(True)
plt.legend()
plt.savefig(f"{path}/average_loss_vs_T.png")
plt.close()

plt.figure(figsize=(10, 8))
plt.plot(T_vals, utilities, marker='s', color='blue', label='Utility')
plt.xlabel("T (Total Payment to Main Server)")
plt.ylabel("Utility")
plt.title("Utility vs T")
plt.grid(True)
plt.legend()
plt.savefig(f"{path}/utility_vs_T.png")
plt.close()

print("Plots saved as 'average_loss_vs_T_fixed.png' and 'utility_vs_T_fixed.png'.")



best_idx = np.nanargmin(utilities)
T_opt = T_vals[best_idx]

n_list = [sum(len(cluster_device_labels[k][d]) for d in range(devices_per_cluster)) for k in range(num_clusters)]
x0 = []
for _ in range(num_clusters):
    x0.extend([0.1, 0.5, T_opt / num_clusters])

bounds_lower = [1e-6, 0.0, 1e-6] * num_clusters
bounds_upper = [np.inf, 1.0, np.inf] * num_clusters

res = least_squares(
    full_equations,
    x0,
    bounds=(bounds_lower, bounds_upper),
    args=(T_opt, n_list, params),
    max_nfev=5000
)

vars_opt = res.x
delta_F_list = [vars_opt[3*k] for k in range(num_clusters)]
gamma_m_list = [vars_opt[3*k + 1] for k in range(num_clusters)]
B_m_list = [vars_opt[3*k + 2] for k in range(num_clusters)]

gammaB_opt_list = [gamma_m_list[k] * B_m_list[k] for k in range(num_clusters)]

optimal_info = []
for k in range(num_clusters):
    gammaB_opt = gammaB_opt_list[k]
    combinations=all_best_combinations_list[k]
    # combinations = all_combinations_list[k]
    best_comb = min(combinations, key=lambda x: abs(x["gammaB"] - gammaB_opt))
    optimal_info.append({
        "gammaB": best_comb["gammaB"],
        "loss": best_comb["acc_loss"],
        "r": best_comb["r"],
        "partition": best_comb["partition"],
        "M_prime":best_comb["M_prime"]
    })

print(f"Optimal T: {T_opt}")
print(f"Optimal Utility: {utilities[best_idx]}")
print(f"Optimal loss: {avg_losses[best_idx]}")

def append_row_to_csv(csv_path, row_dict):
    """
    Append a row to a CSV file. If the file does not exist, create it.
    
    :param csv_path: Path to the CSV file
    :param row_dict: Dictionary where keys are column names and values are row values
    """
    if os.path.exists(csv_path):
        # File exists: append without writing the header
        df = pd.DataFrame([row_dict])
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        # File does not exist: create with header
        df = pd.DataFrame([row_dict])
        df.to_csv(csv_path, mode='w', header=True, index=False)


# Save plotted data to CSV
csv_file = "results_new1/optimized_result.csv"
new_row = {
    "option":"EUBA", # "Equal_B",
    "T": T_opt,
     "Utility": utilities[best_idx],
     "loss": avg_losses[best_idx]
}
append_row_to_csv(csv_file, new_row)

save_optimal_info = []  # make it a list, not dict

for k, info in enumerate(optimal_info):
    entry = {
        "Cluster": k,
        "gammaB_combinations": round(info['gammaB'], 4),
        "Loss": round(info['loss'], 6),
        "r": round(info['r'], 4),
        "Partition": info['partition'],
        "M_prime":info["M_prime"],
        "gamma_m_optimization": round(gamma_m_list[k], 4),
        "B_m_optimization": round(B_m_list[k], 4),
        "delta_F_optimization": round(delta_F_list[k], 6),
        "gammaB_opt": round(gammaB_opt_list[k], 4),
    }
    save_optimal_info.append(entry)
    

    # also print for visibility
    print(f"\nCluster {k}:")
    print(f"  gammaB (from combinations): {entry['gammaB_combinations']}")
    print(f"  Loss: {entry['Loss']}")
    print(f"  r: {entry['r']}")
    print(f"  M_prime: {entry['M_prime']}")
    print(f"  Partition: {entry['Partition']}")
    print(f"  gamma_m (from optimization): {entry['gamma_m_optimization']}")
    print(f"  B_m (from optimization): {entry['B_m_optimization']}")
    print(f"  delta_F (from optimization): {entry['delta_F_optimization']}")
    print(f"  gammaB_opt (gamma_m * B_m): {entry['gammaB_opt']}")

# Save plotted data to CSV
df = pd.DataFrame(save_optimal_info)
df.to_csv(f"{path}/optimal_info_EUBA.csv", index=False)


T_vals = np.arange(12000, 60000, 500) #(4000, 12000, 500)
save_optimal_info = []  # make it a list, not dict
for T_val in T_vals:
    n_list = [sum(len(cluster_device_labels[k][d]) for d in range(devices_per_cluster)) for k in range(num_clusters)]
    x0 = []
    for _ in range(num_clusters):
        x0.extend([0.1, 0.5, T_val / num_clusters])

    bounds_lower = [1e-6, 0.0, 1e-6] * num_clusters
    bounds_upper = [np.inf, 1.0, np.inf] * num_clusters

    res = least_squares(
        full_equations,
        x0,
        bounds=(bounds_lower, bounds_upper),
        args=(T_val, n_list, params),
        max_nfev=5000
    )

    vars_opt = res.x
    delta_F_list = [vars_opt[3*k] for k in range(num_clusters)]
    gamma_m_list = [vars_opt[3*k + 1] for k in range(num_clusters)]
    B_m_list = [vars_opt[3*k + 2] for k in range(num_clusters)]

    gammaB_opt_list = [gamma_m_list[k] * B_m_list[k] for k in range(num_clusters)]
    gammaB_opt_list = np.array(gammaB_opt_list, dtype=float)
    T_B = np.sum(gammaB_opt_list)
    if T_B > T_val:
        gammaB_opt_list = (T_val / T_B) * gammaB_opt_list

    optimal_info = []
    for k in range(num_clusters):
        gammaB_opt = gammaB_opt_list[k]
        combinations=all_best_combinations_list[k]
        # combinations = all_combinations_list[k]
        best_comb = min(combinations, key=lambda x: abs(x["gammaB"] - gammaB_opt))
        optimal_info.append({
            "gammaB": best_comb["gammaB"],
            "loss": best_comb["acc_loss"],
            "r": best_comb["r"],
            "partition": best_comb["partition"],
            "M_prime":best_comb["M_prime"]
        })

    gammaB_arr = np.array([info["gammaB"] for info in optimal_info], dtype=float)
    M_prime_arr = np.array([info["M_prime"] for info in optimal_info], dtype=float)
    T_B = np.sum(gammaB_arr)

    if T_B > T_val:
        scale = T_val / T_B
        gammaB_arr = scale * gammaB_arr
        M_prime_arr = scale * M_prime_arr  # <--- apply same scaling

    for k in range(num_clusters):
        optimal_info[k]["gammaB"] = gammaB_arr[k]
        optimal_info[k]["M_prime"] = M_prime_arr[k]
    # print(f"Optimal T: {T_val}")
    # print(f"Optimal Utility: {utilities[best_idx]}")
    # print(f"Optimal loss: {avg_losses[best_idx]}")

    for k, info in enumerate(optimal_info):
        entry = {
            "T_value":T_val,
            "Cluster": k,
            "gammaB_combinations": round(info['gammaB'], 4),
            "Loss": round(info['loss'], 6),
            "r": round(info['r'], 4),
            "Partition": info['partition'],
            "M_prime":info["M_prime"],
            "gamma_m_optimization": round(gamma_m_list[k], 4),
            "B_m_optimization": round(B_m_list[k], 4),
            "delta_F_optimization": round(delta_F_list[k], 6),
            "gammaB_opt": round(gammaB_opt_list[k], 4),
        }
        save_optimal_info.append(entry)
        

        # also print for visibility
        # print(f"\nCluster {k}:")
        # print(f"  gammaB (from combinations): {entry['gammaB_combinations']}")
        # print(f"  Loss: {entry['Loss']}")
        # print(f"  r: {entry['r']}")
        # print(f"  M_prime: {entry['M_prime']}")
        # print(f"  Partition: {entry['Partition']}")
        # print(f"  gamma_m (from optimization): {entry['gamma_m_optimization']}")
        # print(f"  B_m (from optimization): {entry['B_m_optimization']}")
        # print(f"  delta_F (from optimization): {entry['delta_F_optimization']}")
        # print(f"  gammaB_opt (gamma_m * B_m): {entry['gammaB_opt']}")

    # Save plotted data to CSV
df = pd.DataFrame(save_optimal_info)
df.to_csv(f"{path}/tval_info_EUBA.csv", index=False)

