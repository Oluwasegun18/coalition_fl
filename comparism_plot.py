import pandas as pd
import matplotlib.pyplot as plt
import os
# import matplotlib

# matplotlib.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['patch.force_edgecolor'] = True
# plt.rcParams['patch.facecolor'] = 'none'
# plt.rcParams.update({'font.size': 15}) # Sets a global font size of 14




path = "results_new1/cifar10"
clusters =10
# Your CSVs
ours_loss = pd.read_csv(os.path.join(path,"proposed/result" ,"average_loss_vs_T.csv"))
ours_util = pd.read_csv(os.path.join(path, "proposed/result" ,"utility_vs_T.csv"))
ours_acc_loss = []  #pd.read_csv(os.path.join(path, "acc_losses_vs_gamma.csv"))  # gamma vs acc loss per cluster
for i in range(clusters):
    ours_acc_loss.append(pd.read_csv(os.path.join(path,"proposed/result" , f"cluster{i}_gammaB_Loss.csv")))

benchmarks = [
    ("equal_b/result","EUBA", "average_loss_vs_T.csv", "utility_vs_T.csv", "acc_losses_vs_gamma_benchmark1.csv"),
    ("coalition_based_r/result", "CSLRA","average_loss_vs_T.csv", "utility_vs_T.csv", "acc_losses_vs_gamma_benchmark2.csv"),
    # ("benchmark3/result", "average_loss_vs_T.csv", "utility_vs_T.csv", "acc_losses_vs_gamma_benchmark3.csv"),
    ("constant_r/result", "CLRA","average_loss_vs_T.csv", "utility_vs_T.csv", "acc_losses_vs_gamma_benchmark4.csv"),
    ("minimum_r/result", "MiLRA","average_loss_vs_T.csv", "utility_vs_T.csv", "acc_losses_vs_gamma_benchmark4.csv"),
    ("databased/result", "DBRA","average_loss_vs_T.csv", "utility_vs_T.csv", "acc_losses_vs_gamma_benchmark4.csv")
]

# ---------------------
# 1️⃣ Plot: T_vals vs Average Loss
plt.figure(figsize=(10, 6),dpi=600)
plt.plot(ours_loss["T_val"], ours_loss["Average Loss"], 'o-', linewidth=2, label="Proposed")

for name,labels, loss_file, _, _ in benchmarks:
    # if name == "benchmark2/result":
    #     bench_loss = pd.read_csv(os.path.join(path,name, loss_file))
    #     plt.plot(bench_loss["T_val"], 0.05*bench_loss["Average Loss"], '--', linewidth=2, label=name)
    # else:
    bench_loss = pd.read_csv(os.path.join(path,name, loss_file))
    plt.plot(bench_loss["T_val"], bench_loss["Average Loss"], '--', linewidth=2, label=labels)

plt.xlabel("T (Total Payment by Main Server)",fontsize=14)
plt.ylabel("Average Global Loss",fontsize=14)
# plt.title("Average Loss vs T — All Benchmarks",fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
# plt.tight_layout()
plt.savefig(os.path.join(path, "avg_loss_vs_T_all_benchmarks.pdf"), bbox_inches='tight')
plt.close()

# ---------------------
# 2️⃣ Plot: T_vals vs Utility
plt.figure(figsize=(10, 6),dpi=600)
plt.plot(ours_util["T_vals"], ours_util["Utility"], 'o-', linewidth=2, label="Proposed")

for name,labels, _, util_file, _ in benchmarks:
    # if name == "benchmark2/result":
    #     bench_util = pd.read_csv(os.path.join(path,name, util_file))
    #     plt.plot(bench_util["T_vals"], 0.05*bench_util["Utility"], '--', linewidth=2, label=name)
    # else:
    bench_util = pd.read_csv(os.path.join(path,name, util_file))
    plt.plot(bench_util["T_vals"], bench_util["Utility"], '--', linewidth=2, label=labels)

plt.xlabel("T (Total Payment by Main Server)",fontsize=14)
plt.ylabel("Global Cost",fontsize=14)
# plt.title("Utility vs T — All Benchmarks",fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
# plt.tight_layout()
plt.savefig(os.path.join(path, "utility_vs_T_all_benchmarks.pdf"), bbox_inches='tight')
plt.close()

# ---------------------
# 3️⃣ Plot: Gamma vs Accuracy Loss per Cluster
# clusters =3 # ours_acc_loss["cluster"].unique()

# bench_acc_loss =[] # pd.read_csv(os.path.join(path, acc_loss_file))
# for i in range(clusters):
#     bench_acc_loss.append(pd.read_csv(os.path.join(path,name , f"cluster{i}_gammaB_Loss.csv")))

for cluster in range(clusters):
    plt.figure(figsize=(10, 6),dpi=600)
    ours_c = ours_acc_loss[cluster]  # ours_acc_loss[ours_acc_loss["cluster"] == cluster]
    plt.plot(ours_c["gammaB"], ours_c["Accuracy Loss"], 'o-', linewidth=2, label="Proposed")

    for name,labels, _, _, acc_loss_file in benchmarks:
        # if name == "benchmark2/result":        
        #     bench_c = pd.read_csv(os.path.join(path,name , f"cluster{cluster}_gammaB_Loss.csv"))  #bench_acc_loss #bench_acc_loss[bench_acc_loss["cluster"] == cluster]
        #     plt.plot(bench_c["gammaB"], 0.05*bench_c["Accuracy Loss"], '--', linewidth=2, label=name)
        # else:
        bench_c = pd.read_csv(os.path.join(path,name , f"cluster{cluster}_gammaB_Loss.csv"))  #bench_acc_loss #bench_acc_loss[bench_acc_loss["cluster"] == cluster]
        plt.plot(bench_c["gammaB"], bench_c["Accuracy Loss"], '--', linewidth=2, label=labels)

    plt.xlabel(r"$\gamma_B$",fontsize=14)
    plt.ylabel("Accuracy Loss",fontsize=14)
    # plt.title(f"Accuracy Loss vs GammaB — Cluster {cluster}",fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    # plt.tight_layout()
    plt.savefig(os.path.join(path, f"acc_loss_vs_gamma_cluster{cluster}_all_benchmarks.pdf"), bbox_inches='tight')
    plt.close()

print("✅ All benchmarks compared on single plots")










# for bench_name, loss_file, util_file, acc_loss_file in benchmarks:
#     # Read benchmark data
#     bench_loss = pd.read_csv(os.path.join(path, loss_file))
#     bench_util = pd.read_csv(os.path.join(path, util_file))
#     bench_acc_loss =[] #pd.read_csv(os.path.join(path, acc_loss_file))
#     for i in range(0,cluster):
#         bench_acc_loss.append(pd.read_csv(os.path.join(path,bench_name , f"cluster{i}_gammaB_Loss.csv")))

#     # --- Plot 1: T_vals vs Average Loss ---
#     plt.figure(figsize=(8, 6))
#     plt.plot(ours_loss["T_val"], ours_loss["Average Loss"], 'o-', linewidth=2, label="Ours")
#     plt.plot(bench_loss["T_val"], bench_loss["Average Loss"], '--', linewidth=2, label=f"{bench_name}")
#     plt.xlabel("T (Total Payment to Main Server)")
#     plt.ylabel("Average Loss")
#     plt.title(f"Average Loss vs T — {bench_name}")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(path, f"avg_loss_vs_T_{bench_name}.png"))
#     plt.close()

#     # --- Plot 2: T_vals vs Utility ---
#     plt.figure(figsize=(8, 6))
#     plt.plot(ours_util["T_vals"], ours_util["Utility"], 'o-', linewidth=2, label="Ours")
#     plt.plot(bench_util["T_vals"], bench_util["Utility"], '--', linewidth=2, label=f"{bench_name}")
#     plt.xlabel("T (Total Payment to Main Server)")
#     plt.ylabel("Utility")
#     plt.title(f"Utility vs T — {bench_name}")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(path, f"utility_vs_T_{bench_name}.png"))
#     plt.close()

#     # --- Plot 3: gamma_values vs acc_losses_value per cluster ---
#     clusters = ours_acc_loss["cluster"].unique()
#     for cluster in clusters:
#         ours_c = ours_acc_loss[ours_acc_loss["cluster"] == cluster]
#         bench_c = bench_acc_loss[bench_acc_loss["cluster"] == cluster]

#         plt.figure(figsize=(8, 6))
#         plt.plot(ours_c["gamma_values"], ours_c["acc_losses_value"], 'o-', linewidth=2, label="Ours")
#         plt.plot(bench_c["gamma_values"], bench_c["acc_losses_value"], '--', linewidth=2, label=f"{bench_name}")
#         plt.xlabel("Gamma")
#         plt.ylabel("Accuracy Loss")
#         plt.title(f"Accuracy Loss vs Gamma — Cluster {cluster} — {bench_name}")
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(path, f"acc_loss_vs_gamma_cluster{cluster}_{bench_name}.png"))
#         plt.close()

# print("✅ All benchmark comparisons plotted")
