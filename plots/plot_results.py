import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

with open("results_and_fps.json", "r") as f:
    data = json.load(f)

# Define dataset groupings
mipnerf_scenes = {"garden", "stump", "kitchen"}
tt_scenes = {"truck", "train"}

def parse_mb(size_str):
    """Convert size like '814.8 MB' or '2.3 GB' to float MB (ignore GB)"""
    if "MB" in size_str:
        return float(size_str.replace("MB", "").strip())
    elif "GB" in size_str:
        return float(size_str.replace("GB", "").strip()) * 1024
    else:
        return 0.0

# Accumulators
results = defaultdict(lambda: {
    "MipNeRF": {"SSIM": [], "PSNR": [], "LPIPS": [], "Pointcloud_Size": [], "FPS":[]},
    "TnT": {"SSIM": [], "PSNR": [], "LPIPS": [], "Pointcloud_Size": [], "FPS":[]}
})

# Aggregate metrics
for method, scenes in data.items():
    for scene, metrics in scenes.items():
        if scene in mipnerf_scenes:
            group = "MipNeRF"
        elif scene in tt_scenes:
            group = "TnT"
        else:
            continue  # Skip unknown scenes

        results[method][group]["SSIM"].append(metrics["SSIM"])
        results[method][group]["PSNR"].append(metrics["PSNR"])
        results[method][group]["LPIPS"].append(metrics["LPIPS"])
        results[method][group]["FPS"].append(metrics["FPS"])
        results[method][group]["Pointcloud_Size"].append(parse_mb(metrics["Pointcloud_Size"]))

# Compute averages
averaged_results = {}
for method, groups in results.items():
    averaged_results[method] = {}
    for group_name, metrics in groups.items():
        averaged_results[method][group_name] = {
            "SSIM": round(sum(metrics["SSIM"]) / len(metrics["SSIM"]), 2),
            "PSNR": round(sum(metrics["PSNR"]) / len(metrics["PSNR"]), 2),
            "LPIPS": round(sum(metrics["LPIPS"]) / len(metrics["LPIPS"]),2),
            "FPS": round(sum(metrics["FPS"]) / len(metrics["FPS"]),2),
            "Avg_Pointcloud_Size_MB": round(sum(metrics["Pointcloud_Size"]) / len(metrics["Pointcloud_Size"]),1),
        }


import pprint
pprint.pprint(averaged_results)


def plot_memory_vs_x(dataset_name, metric="PSNR"):
    x_vals = []
    y_vals = []
    plt.figure(figsize=(8, 6))
    for method, metrics in averaged_results.items():
        if metric == "FPS" and (method == "compGS" or method == "Mini-Splatting-C"):
            continue
        if dataset_name in metrics:
            x = metrics[dataset_name]["Avg_Pointcloud_Size_MB"]
            y = metrics[dataset_name][metric]
            x_vals.append(x)
            y_vals.append(y)
            plt.scatter(x, y, label=method)
            plt.text(x, y, method, fontsize=8, ha='right', va='bottom')

    plt.title(f"Avg Pointcloud Size vs {metric} ({dataset_name})")
    plt.xlabel("Avg Pointcloud Size (MB)")
    #plt.xscale("log")
    min_x = min(x_vals) * 0.8
    max_x = max(x_vals) * 1.2
    ticks = [20,50,100,200,400,600,800]
    ticks = [t for t in ticks if min_x <= t <= max_x]

    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='x', which='major', rotation=45)
    plt.ylabel(f"Avg {metric}")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right' if (metric == "FPS" or metric == "LPIPS") else 'lower right', fontsize=7)
    plt.tight_layout()
    plt.savefig(f"pointcloud_vs_{metric}_{dataset_name}.png")
    plt.show()

# Generate plots
plot_memory_vs_x("MipNeRF", "FPS")
plot_memory_vs_x("TnT", "FPS")
plot_memory_vs_x("MipNeRF", "PSNR")
plot_memory_vs_x("TnT", "PSNR")
plot_memory_vs_x("MipNeRF", "SSIM")
plot_memory_vs_x("TnT", "SSIM")
plot_memory_vs_x("MipNeRF", "LPIPS")
plot_memory_vs_x("TnT", "LPIPS")


# average pointcloud size graph
plt.figure(figsize=(12, 6))
methods = list(averaged_results.keys())
mipnerf_sizes = [averaged_results[m]["MipNeRF"]["Avg_Pointcloud_Size_MB"] for m in methods]
tnt_sizes = [averaged_results[m]["TnT"]["Avg_Pointcloud_Size_MB"] for m in methods]

x = range(len(methods))
width = 0.35
plt.bar([i - width/2 for i in x], mipnerf_sizes, width, label='MipNeRF Scenes')
plt.bar([i + width/2 for i in x], tnt_sizes, width, label='TnT Scenes')

plt.xlabel('Methods')
plt.ylabel('Pointcloud Size (MB)')
plt.title('Average Pointcloud Size Comparison')
plt.xticks(x, methods, rotation=45, ha='right')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("average_pointcloud_size_comparison.png")
plt.show()

# average PSNR graph
plt.figure(figsize=(12, 6))
mipnerf_psnr = [averaged_results[m]["MipNeRF"]["PSNR"] for m in methods]
tnt_psnr = [averaged_results[m]["TnT"]["PSNR"] for m in methods]

plt.bar([i - width/2 for i in x], mipnerf_psnr, width, label='MipNeRF Scenes')
plt.bar([i + width/2 for i in x], tnt_psnr, width, label='TnT Scenes')

plt.xlabel('Methods')
plt.ylabel('PSNR')
plt.title('Average PSNR Comparison')
plt.xticks(x, methods, rotation=45, ha='right')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("average_psnr_comparison.png")
plt.show()

# average FPS graph
plt.figure(figsize=(12, 6))
mipnerf_fps = [averaged_results[m]["MipNeRF"]["FPS"] for m in methods]
tnt_fps = [averaged_results[m]["TnT"]["FPS"] for m in methods]

plt.bar([i - width/2 for i in x], mipnerf_fps, width, label='MipNeRF Scenes')
plt.bar([i + width/2 for i in x], tnt_fps, width, label='TnT Scenes')

plt.xlabel('Methods')
plt.ylabel('FPS')
plt.title('Average FPS Comparison')
plt.xticks(x, methods, rotation=45, ha='right')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("average_fps_comparison.png")
plt.show()



plt.figure(figsize=(10, 6))
colors = plt.cm.tab10.colors
method_colors = {method: colors[i % len(colors)] for i, method in enumerate(averaged_results.keys())}

all_fps = []
for metrics in averaged_results.values():
    all_fps.append(metrics["MipNeRF"]["FPS"])
    all_fps.append(metrics["TnT"]["FPS"])
fps_range = max(all_fps) - min(all_fps)
fps_offset = 0.02 * fps_range 

for method, metrics in averaged_results.items():
    if method in ["compGS", "Mini-Splatting-C"]:
        continue
    
    color = method_colors[method]
    
    # MipNeRF data (circles)
    x_mip = metrics["MipNeRF"]["Avg_Pointcloud_Size_MB"]
    y_mip = metrics["MipNeRF"]["FPS"]
    plt.scatter(x_mip, y_mip, marker='o', color=color, s=80, label=f'{method} (MipNeRF)' if method == list(averaged_results.keys())[0] else "")
    plt.text(x_mip, y_mip + fps_offset, method, fontsize=8, ha='center', va='bottom')
    
    # TnT data (squares)
    x_tnt = metrics["TnT"]["Avg_Pointcloud_Size_MB"]
    y_tnt = metrics["TnT"]["FPS"]
    plt.scatter(x_tnt, y_tnt, marker='s', color=color, s=80, label=f'{method} (TnT)' if method == list(averaged_results.keys())[0] else "")
    plt.text(x_tnt, y_tnt + fps_offset, method, fontsize=8, ha='center', va='bottom')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='MipNeRF Scenes',
          markerfacecolor='gray', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='TnT Scenes',
          markerfacecolor='gray', markersize=10)
]
plt.legend(handles=legend_elements, loc='upper right')

plt.title("Avg Pointcloud Size vs FPS (All Scenes)")
plt.xlabel("Avg Pointcloud Size (MB)")
plt.ylabel("FPS")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("combined_pointcloud_vs_fps.png", dpi=300)
plt.show()


plt.figure(figsize=(10, 6))
all_psnr = []
for metrics in averaged_results.values():
    all_psnr.append(metrics["MipNeRF"]["PSNR"])
    all_psnr.append(metrics["TnT"]["PSNR"])
psnr_range = max(all_psnr) - min(all_psnr)
psnr_offset = 0.02 * psnr_range

for method, metrics in averaged_results.items():
    color = method_colors[method]
    
    # MipNeRF data (circles)
    x_mip = metrics["MipNeRF"]["Avg_Pointcloud_Size_MB"]
    y_mip = metrics["MipNeRF"]["PSNR"]
    plt.scatter(x_mip, y_mip, marker='o', color=color, s=80, label=f'{method} (MipNeRF)' if method == list(averaged_results.keys())[0] else "")
    plt.text(x_mip, y_mip + psnr_offset, method, fontsize=8, ha='center', va='bottom')
    
    # TnT data (squares)
    x_tnt = metrics["TnT"]["Avg_Pointcloud_Size_MB"]
    y_tnt = metrics["TnT"]["PSNR"]
    plt.scatter(x_tnt, y_tnt, marker='s', color=color, s=80, label=f'{method} (TnT)' if method == list(averaged_results.keys())[0] else "")
    plt.text(x_tnt, y_tnt + psnr_offset, method, fontsize=8, ha='center', va='bottom')

plt.legend(handles=legend_elements, loc='lower right')
plt.title("Avg Pointcloud Size vs PSNR (All Scenes)")
plt.xlabel("Avg Pointcloud Size (MB)")
plt.ylabel("PSNR")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("combined_pointcloud_vs_psnr.png", dpi=300)
plt.show()