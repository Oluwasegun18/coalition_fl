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
path='results/coalition/result'
os.makedirs(path,exist_ok=True)
# Parameters
num_clusters = 3
devices_per_cluster = 10
total_devices = num_clusters * devices_per_cluster
communication_rounds = 5
privacy_sensitivity = 1
E_k = 1

# Shuffle and partition using Dirichlet (non-iid)
indices = np.arange(len(x_train))
np.random.shuffle(indices)

alpha = 0.1
# cluster_device_data,cluster_device_labels = balanced_dirichlet_split(x_train,y_train,num_clusters, devices_per_cluster,alpha)

# Example: 4 clusters, each with a different scheme
# strategies = ["dirichlet_low", "label_k", "majority", "zipf_major"]
strategies = ["dirichlet_low", "label_k", "majority"]
cluster_device_data,cluster_device_labels = split_non_iid_multi_cluster(
    x_train,y_train,
    num_clusters=num_clusters,
    devices_per_cluster=10,
    strategies=strategies,
    balanced_sizes=False,        # allow quantity skew (use True if you want equal sizes)
    dirichlet_alpha=0.1,        # make Dirichlet extra spiky
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

gammaB_values = np.arange(1000, 30000, 500)

cluster_results = []
gammaB_lists=[[] for _ in range(num_clusters)]
acc_losses_list=[[] for _ in range(num_clusters)]
all_combinations_list = [[] for _ in range(num_clusters)]

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
                        L_k = (len(coalition) - 1) * c_k * privacy_sensitivity
                        u_k = R_k - L_k - E_k
                        utilities[d] = u_k
                        rewards[d] = R_k

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
            total_payment1=sum(rewards.values())
            total_payment = sum(device_rewards.values()) #sum(rewards.values())

            if total_payment > gammaB / L:
                partition_updated = prev_partition
                r = prev_r
                break


            # Collect per-device rewards for analysis
            # reward_vector_current = np.array([rewards.get(d, 0.0) for d in range(N_k)])
            # reward_vector_sticky  = np.array([device_rewards.get(d, 0.0) for d in range(N_k)])  # only in Version A

            # deepcopy_partition = copy.deepcopy(partition_updated)

            # sum_current = float(reward_vector_current.sum())
            # sum_sticky = float(reward_vector_sticky.sum()) if 'reward_vector_sticky' in locals() else None

            # # logging.info("logging.......")
            # logging.info(
            #     f"cluster: {cluster_id}, gammaB: {gammaB}, r: {r}, "
            #     f"partition: {deepcopy_partition}, sum_current: {sum_current}, "
            #     f"sum_sticky: {sum_sticky}"
            # )


            # logging.info("logging.......")
            # logging.info(f" cluster: {cluster_id},  gammaB: {gammaB}, r: {r}, partition: {copy.deepcopy(partition_updated)}, sum_current: {float(reward_vector_current.sum())}, sum_sticky:  {float(reward_vector_sticky.sum()) if 'device_rewards' in locals() else None},")
           
            #     current: {reward_vector_current.tolist()},
            #     sticky:  {reward_vector_sticky.tolist() if 'device_rewards' in locals() else None}
            # ")

            prev_partition = copy.deepcopy(partition_updated)
            prev_r = r

            numerator = ((gammaB / L) - total_payment) * L
            denominator = total_payment if total_payment != 0 else 1e-6
            M_prime =  int(np.floor(numerator / denominator) + L)
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

            phi, epsilon, G, eta, beta, T_local = 0.1, 10, 1, 0.01, 0.1, 10
            acc_loss_bound = 1 / (M_prime * eta * phi * (T_local - 1) *
                                (1 - (beta * eta / 2) - (L * G * D_S) / (epsilon ** 2)))

            acc_losses.append(acc_loss_bound)
            gammaB_list.append(gammaB)
            r_list.append([r,acc_loss_bound,D_S])

            all_combinations.append({
                "gammaB": gammaB,
                "acc_loss": acc_loss_bound,
                "partition": copy.deepcopy(partition_updated),
                "r": r
            })


        best_r = min(r_list, key=lambda x: x[1])[0]
        best_combination = [entry for entry in all_combinations if (entry["r"] == best_r and entry["gammaB"]==gammaB)][0]
        best_combinations.append(best_combination)
    gamma_values = [entry["gammaB"] for entry in best_combinations]
    acc_losses_value = [entry["acc_loss"] for entry in best_combinations]
    gammaB_lists[cluster_id]=gamma_values
    acc_losses_list[cluster_id]=acc_losses_value
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
        eq1 = (1 / (n_list[k] * delta_F) / denom) * T - Bm

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
T_vals = np.arange(5000, 40000, 500)
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
        utility = avg_loss + 0.00001 * T

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
    combinations = all_combinations_list[k]
    best_comb = min(combinations, key=lambda x: abs(x["gammaB"] - gammaB_opt))
    optimal_info.append({
        "gammaB": best_comb["gammaB"],
        "loss": best_comb["acc_loss"],
        "r": best_comb["r"],
        "partition": best_comb["partition"]
    })
print(f"Optimal T: {T_opt}")
print(f"Optimal loss: {utilities[best_idx]}")

# Save plotted data to CSV
df = pd.DataFrame(optimal_info)
df.to_csv(f"{path}/optimal_info.csv", index=False)

for k, info in enumerate(optimal_info):
    print(f"\nCluster {k}:")
    print(f"  gammaB (from combinations): {info['gammaB']:.4f}")
    print(f"  Loss: {info['loss']:.6f}")
    print(f"  r: {info['r']:.4f}")
    print(f"  Partition: {info['partition']}")
    print(f"  gamma_m (from optimization): {gamma_m_list[k]:.4f}")
    print(f"  B_m (from optimization): {B_m_list[k]:.4f}")
    print(f"  delta_F (from optimization): {delta_F_list[k]:.6f}")
    print(f"  gammaB_opt (gamma_m * B_m): {gammaB_opt_list[k]:.4f}")












# from scipy.optimize import curve_fit, least_squares
# from scipy.special import lambertw
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import json

# cluster_results = []
# num_clusters = 3

# for cluster_id in range(num_clusters):
#     gammaBs = gammaB_lists[cluster_id]
#     accs = acc_losses_list[cluster_id]
#     combinations = all_combinations_list[cluster_id]

#     # 1. Fit exponential curve for this cluster
#     def exp_func(x, a, b, c):
#         bx = np.clip(-b * x, -700, 700)  # exp(-700) is still representable
#         return a * np.exp(bx) + c
#         return a * np.exp(-b * x) + c

#     try:
#         popt, _ = curve_fit(exp_func, gammaBs, accs, p0=[1, 0.001, 0.1], maxfev=10000)
#     except RuntimeError:
#         print(f"Curve fit failed for cluster {cluster_id + 1}")
#         continue

#     # 2. Get device label counts
#     devices_per_cluster = len(cluster_device_labels[cluster_id])
#     n_list = [len(cluster_device_labels[cluster_id][d]) for d in range(devices_per_cluster)]

#     # 3. Define system of equations
#     def equations(vars, T, n_list):
#         delta_F, gamma_m, Bm = vars
#         denom = sum(1 / (n_j * delta_F) for n_j in n_list)
#         eq1 = (1 / (n_list[0] * delta_F) / denom) * T - Bm
#         eq2 = exp_func(gamma_m * Bm, *popt) - delta_F

#         A, C, n_k = 1, 1, n_list[0]
#         term1 = -np.exp(-C / (A * n_k))
#         denom_l = A * n_k * (1 - gamma_m)
#         lambert_arg = term1 / denom_l
#         lambert_value = lambertw(lambert_arg).real
#         lambert_part = lambert_value + (C / (A * n_k))
#         eq3 = delta_F - (gamma_m * np.exp(lambert_part) - np.exp(lambert_part))
#         return [eq1, eq2, eq3]

#     # 4. Solve for range of T
#     T_vals = np.arange(10, 200, 1)
#     T_results = []
#     acc_loss_plot = []
#     utility_plot = []

#     for T in T_vals:
#         try:
#             bounds = ([1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf])
#             res = least_squares(equations, [10, 0.1, 1], bounds=bounds, args=(T, n_list))

#             if not res.success:
#                 raise RuntimeError(f"Solver failed at T={T}")

#             delta_F_val, gamma_val, B_val = res.x
#             avg_loss = (sum(n_list) * delta_F_val) / sum(n_list)
#             acc_loss_plot.append(avg_loss)
#             utility_plot.append(avg_loss + 0.1 * T)

#             T_results.append({
#                 "T": T,
#                 "gamma_m": gamma_val,
#                 "B_m": B_val,
#                 "acc_loss": avg_loss
#             })

#         except Exception as e:
#             acc_loss_plot.append(np.nan)
#             utility_plot.append(np.nan)
#             T_results.append({"T": T, "gamma_m": None, "B_m": None, "acc_loss": None})

#     # 5. Plotting
#     os.makedirs("results", exist_ok=True)
#     valid_results = [res for res in T_results if res["acc_loss"] is not None]
#     T_plot = [res["T"] for res in valid_results]
#     acc_plot = [res["acc_loss"] for res in valid_results]
#     util_plot = [res["acc_loss"] + 0.1 * res["T"] for res in valid_results]

#     plt.figure(figsize=(8, 5))
#     plt.plot(T_plot, acc_plot, marker='o', color='green')
#     plt.xlabel("T (Total Payment to Main Server)")
#     plt.ylabel("Accuracy Loss")
#     plt.title(f"Accuracy Loss vs T (Cluster {cluster_id + 1})")
#     plt.grid(True)
#     plt.savefig(f"results/accuracy_loss_cluster{cluster_id + 1}.png")
#     plt.close()

#     plt.figure(figsize=(8, 5))
#     plt.plot(T_plot, util_plot, marker='o', color='blue')
#     plt.xlabel("T (Total Payment to Main Server)")
#     plt.ylabel("Main Server Utility (Loss + Weighted T)")
#     plt.title(f"Server Utility vs T (Cluster {cluster_id + 1})")
#     plt.grid(True)
#     plt.savefig(f"results/server_utility_cluster{cluster_id + 1}.png")
#     plt.close()

#     # 6. Choose best T (min utility)
#     min_idx = np.nanargmin(util_plot)
#     best_T_info = valid_results[min_idx]
#     T_opt = best_T_info["T"]
#     gamma_final = best_T_info["gamma_m"]
#     B_final = best_T_info["B_m"]
#     acc_loss_final = best_T_info["acc_loss"]

#     # 7. Match final result to closest configuration
#     final_choice = None
#     min_diff = float('inf')
#     for comb in combinations:
#         diff = abs(comb["acc_loss"] - acc_loss_final)
#         if diff < min_diff:
#             min_diff = diff
#             final_choice = comb

#     cluster_results.append({
#         "cluster_id": cluster_id,
#         "T_opt": T_opt,
#         "gamma_m": gamma_final,
#         "B_m": B_final,
#         "acc_loss": acc_loss_final,
#         "closest_combination": final_choice
#     })

#     class NumpyEncoder(json.JSONEncoder):
#         def default(self, obj):
#             if isinstance(obj, (np.integer, np.int32, np.int64)):
#                 return int(obj)
#             elif isinstance(obj, (np.floating, np.float32, np.float64)):
#                 return float(obj)
#             elif isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             return super(NumpyEncoder, self).default(obj)
#     # Optional save to JSON
#     with open(f"results/best_T_cluster{cluster_id + 1}.json", "w") as f:
#         json.dump(cluster_results[-1], f, indent=4, cls=NumpyEncoder)











# from scipy.optimize import curve_fit, least_squares
# from scipy.special import lambertw
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import json

# # 1. Aggregate global gammaB_list and acc_losses
# gammaB_all = []
# acc_all = []
# n_list_global = []

# # for cluster_id in range(num_clusters):
# #     gammaBs = gammaB_lists[cluster_id]
# #     accs = acc_losses_list[cluster_id]
# #     combinations = all_combinations_list[cluster_id]

# for cluster_id in range(3):
#     gammaB_all.extend(gammaB_lists[cluster_id])
#     acc_all.extend(acc_losses_list[cluster_id])
#     n_list_global.extend([len(cluster_device_labels[cluster_id][d]) for d in range(len(cluster_device_labels[cluster_id]))])

# # 2. Fit global accuracy loss model
# def exp_func(x, a, b, c):
#     return a * np.exp(-b * x) + c

# popt, _ = curve_fit(exp_func, gammaB_all, acc_all, p0=[1, 0.001, 0.1], maxfev=10000)

# # 3. Define system of equations
# def equations(vars, T, n_list):
#     delta_F, gamma_m, Bm = vars
#     denom = sum(1 / (n_j * delta_F) for n_j in n_list)
#     eq1 = (1 / (n_list[0] * delta_F) / denom) * T - Bm
#     eq2 = exp_func(gamma_m * Bm, *popt) - delta_F

#     A, C, n_k = 1, 1, n_list[0]
#     term1 = -np.exp(-C / (A * n_k))
#     denom_l = A * n_k * (1 - gamma_m)
#     lambert_arg = term1 / denom_l
#     lambert_value = lambertw(lambert_arg).real
#     lambert_part = lambert_value + (C / (A * n_k))
#     eq3 = delta_F - (gamma_m * np.exp(lambert_part) - np.exp(lambert_part))
#     return [eq1, eq2, eq3]

# # 4. Loop over T to find optimal allocation
# T_vals = np.arange(10, 200, 1)
# T_results = []
# acc_loss_plot = []
# utility_plot = []

# for T in T_vals:
#     try:
#         bounds = ([1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf])
#         res = least_squares(equations, [10, 0.1, 1], bounds=bounds, args=(T, n_list_global))

#         if not res.success:
#             raise RuntimeError(f"Solver failed at T={T}")

#         delta_F_val, gamma_val, B_val = res.x
#         total_samples = sum(n_list_global)
#         avg_loss = total_samples * delta_F_val / total_samples  # = delta_F_val
#         acc_loss_plot.append(avg_loss)
#         utility_plot.append(avg_loss + 0.1 * T)

#         T_results.append({
#             "T": T,
#             "gamma_m": gamma_val,
#             "B_m": B_val,
#             "acc_loss": avg_loss
#         })

#     except Exception:
#         acc_loss_plot.append(np.nan)
#         utility_plot.append(np.nan)
#         T_results.append({"T": T, "gamma_m": None, "B_m": None, "acc_loss": None})

# # 5. Plot
# os.makedirs("results", exist_ok=True)
# valid_results = [res for res in T_results if res["acc_loss"] is not None]
# T_plot = [res["T"] for res in valid_results]
# acc_plot = [res["acc_loss"] for res in valid_results]
# util_plot = [res["acc_loss"] + 0.1 * res["T"] for res in valid_results]

# plt.figure(figsize=(8, 5))
# plt.plot(T_plot, acc_plot, marker='o', color='green')
# plt.xlabel("T (Total Payment to Main Server)")
# plt.ylabel("Global Accuracy Loss")
# plt.title("Global Accuracy Loss vs T")
# plt.grid(True)
# plt.savefig("results/global_accuracy_loss.png")
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.plot(T_plot, util_plot, marker='o', color='blue')
# plt.xlabel("T (Total Payment to Main Server)")
# plt.ylabel("Global Server Utility (Loss + weighted T)")
# plt.title("Global Server Utility vs T")
# plt.grid(True)
# plt.savefig("results/global_server_utility.png")
# plt.show()

# # 6. Select best T
# min_idx = np.nanargmin(util_plot)
# best_T_info = valid_results[min_idx]
# T_opt = best_T_info["T"]
# gamma_final = best_T_info["gamma_m"]
# B_final = best_T_info["B_m"]
# acc_loss_final = best_T_info["acc_loss"]

# # Save result
# final_output = {
#     "T_opt": T_opt,
#     "gamma_m": gamma_final,
#     "B_m": B_final,
#     "acc_loss": acc_loss_final
# }

# with open("results/global_optimal_resources.json", "w") as f:
#     json.dump(final_output, f, indent=4)

# print(f"\nOptimal global resources:")
# print(json.dumps(final_output, indent=2))























# def exp_func(x, a, b, c):
#     return a * np.exp(-b * x) + c

# popt, _ = curve_fit(exp_func, gamma_values, acc_losses_value, p0=[1, 0.001, 0.1], maxfev=10000)
# n_list = [len(cluster_device_labels[cluster_id][d]) for d in range(devices_per_cluster)]

# def equations(vars, T, n_list):
#     delta_F, gamma_m, Bm = vars
#     denom = sum(1 / (n_j * delta_F) for n_j in n_list)
#     eq1 = (1 / (n_list[0] * delta_F) / denom) * T - Bm
#     eq2 = exp_func(gamma_m * Bm, *popt) - delta_F

#     A, C, n_k = 1, 1, n_list[0]
#     term1 = -np.exp(-C / (A * n_k))
#     denom_l = A * n_k * (1 - gamma_m)
#     lambert_arg = term1 / denom_l
#     lambert_value = lambertw(lambert_arg).real
#     lambert_part = lambert_value + (C / (A * n_k))
#     eq3 = delta_F - (gamma_m * np.exp(lambert_part) - np.exp(lambert_part))
#     return [eq1, eq2, eq3]

# T_vals = np.arange(10, 200, 1)
# T_results = []
# acc_loss_plot = []
# utility_plot = []

# for T in T_vals:
#     try:
#         bounds = ([1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf])
#         res = least_squares(equations, [10, 0.1, 1], bounds=bounds, args=(T, n_list))
#         delta_F_val, gamma_val, B_val = res.x
#         avg_loss = (sum(n_list) * delta_F_val) / sum(n_list)
#         acc_loss_plot.append(avg_loss)
#         utility_plot.append(avg_loss + 0.1 * T)
#         T_results.append({
#             "T": T,
#             "gamma_m": gamma_val,
#             "B_m": B_val,
#             "acc_loss": avg_loss
#         })
#     except Exception:
#         acc_loss_plot.append(np.nan)
#         utility_plot.append(np.nan)
#         T_results.append({"T": T, "gamma_m": None, "B_m": None, "acc_loss": None})

# # Plot accuracy loss vs T
# os.makedirs("results", exist_ok=True)
# T_values_plot = [res["T"] for res in T_results if res["acc_loss"] is not None]
# acc_loss_clean_plot = [res["acc_loss"] for res in T_results if res["acc_loss"] is not None]

# plt.figure(figsize=(8, 5))
# plt.plot(T_values_plot, acc_loss_clean_plot, marker='o', color='green')
# plt.xlabel("T (Total Payment to Main Server)")
# plt.ylabel("Accuracy loss")
# plt.title(f"Accuracy loss vs T (Cluster {cluster_id + 1})")
# plt.grid(True)
# plt.savefig(f"results/accuracy_loss{cluster_id + 1}.png")
# plt.show()

# # Plot main server utility vs T
# valid_T_plot = [T for loss, T in zip(acc_loss_plot, T_vals) if not np.isnan(loss)]
# valid_utilities = [util for util in utility_plot if not np.isnan(util)]

# plt.figure(figsize=(8, 5))
# plt.plot(valid_T_plot, valid_utilities, marker='o', color='blue')
# plt.xlabel("T (Total Payment to Main Server)")
# plt.ylabel("Main Server Utility (Accuracy loss + weighted T)")
# plt.title(f"Main Server Utility vs T (Cluster {cluster_id + 1})")
# plt.grid(True)
# plt.savefig(f"results/server_utility{cluster_id + 1}.png")
# plt.show()

# # Choose best T
# min_idx = np.nanargmin(valid_utilities)
# best_T_info = T_results[min_idx]
# T_opt = best_T_info["T"]
# gamma_final = best_T_info["gamma_m"]
# B_final = best_T_info["B_m"]
# acc_loss_final = best_T_info["acc_loss"]

# final_choice = None
# min_diff = float('inf')
# for comb in all_combinations:
#     diff = abs(comb["acc_loss"] - acc_loss_final)
#     if diff < min_diff:
#         min_diff = diff
#         final_choice = comb

# cluster_results.append({
#     "Cluster": cluster_id + 1,
#     "Optimal_T": T_opt,
#     "Final_Bm": B_final,
#     "Final_gamma": gamma_final,
#     "Final_acc_loss": final_choice["acc_loss"],
#     "Final_partition": final_choice["partition"],
#     "Final_r": final_choice["r"]
# })

# for result in cluster_results:
#     print(f"\nCluster {result['Cluster']}:")
#     print(f"  Optimal T = {result['Optimal_T']:.2f}")
#     print(f"  Final B_m = {result['Final_Bm']:.2f}")
#     print(f"  Final gamma_m = {result['Final_gamma']:.2f}")
#     print(f"  Final Accuracy loss = {result['Final_acc_loss']:.4f}")
#     print(f"  Final Partition: {result['Final_partition']}")
#     print(f"  Final r = {result['Final_r']:.4f}")



