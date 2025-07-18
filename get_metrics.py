import os
import json
import csv
from pathlib import Path

def format_size(bytes_size):
    gb = 1024 ** 3
    mb = 1024 ** 2
    if bytes_size >= gb:
        return f"{bytes_size / gb:.1f} GB"
    else:
        return f"{bytes_size / mb:.1f} MB"

def read_base_results(folder):
    with open(os.path.join(folder, "results.json")) as f:
        data = json.load(f)
    key = list(data.keys())[0]
    return data[key]

def read_c3dgs_results(folder):
    with open(os.path.join(folder, "results.json")) as f:
        data = json.load(f)
    key = list(data.keys())[0]
    return data[key]

def read_light_metrics(folder):
    csv_path = os.path.join(folder, "metric.csv")
    best = None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['set'] == 'test':
                if best is None or float(row['iteration']) > float(best['iteration']):
                    best = row
    return {
        "SSIM": float(best["ssim"]),
        "PSNR": float(best["psnr"]),
        "LPIPS": float(best["lpips"])
    }

def find_checkpoint_size(folder):
    pth_files = list(Path(folder).glob("*.pth"))
    if not pth_files:
        return "not found"
    return format_size(pth_files[0].stat().st_size)

def find_pointcloud_size(folder, is_c3dgs=False):
    pointcloud_root = Path(folder) / "point_cloud"
    if not pointcloud_root.exists():
        return "not found"
    
    pattern = "**/point_cloud.npz" if is_c3dgs else "**/point_cloud.ply"
    files = list(pointcloud_root.glob(pattern))
    
    if not files:
        return "not found"
    
    return format_size(files[0].stat().st_size)

def build_results():
    scenes = ['garden', 'stump', 'kitchen']
    output = {
        "Gaussian-Splatting (Base)": {},
        "LightGaussian": {},
        "c3dgs": {}
    }

    for scene in scenes:
        base_folder = f"output/base_{scene}"
        light_folder = f"output/Light_{scene}"
        c3dgs_folder = f"output/c3dgs_{scene}"

        # BASE
        base_metrics = read_base_results(base_folder)
        base_ckpt = find_checkpoint_size(base_folder)
        base_pointcloud = find_pointcloud_size(base_folder)
        output["Gaussian-Splatting (Base)"][scene] = {
            "SSIM": base_metrics["SSIM"],
            "PSNR": base_metrics["PSNR"],
            "LPIPS": base_metrics["LPIPS"],
            "Pointcloud_Size": base_pointcloud,
            "Checkpoint_Size": base_ckpt
        }

        # LIGHT
        light_metrics = read_light_metrics(light_folder)
        light_ckpt = find_checkpoint_size(light_folder)
        light_pointcloud = find_pointcloud_size(light_folder)
        output["LightGaussian"][scene] = {
            "SSIM": light_metrics["SSIM"],
            "PSNR": light_metrics["PSNR"],
            "LPIPS": light_metrics["LPIPS"],
            "Pointcloud_Size": light_pointcloud,
            "Checkpoint_Size": light_ckpt
        }

        # C3DGS
        c3dgs_metrics = read_c3dgs_results(c3dgs_folder)
        c3dgs_pointcloud = find_pointcloud_size(c3dgs_folder, is_c3dgs=True)
        output["c3dgs"][scene] = {
            "SSIM": c3dgs_metrics["SSIM"],
            "PSNR": c3dgs_metrics["PSNR"],
            "LPIPS": c3dgs_metrics["LPIPS"],
            "Pointcloud_Size": c3dgs_pointcloud,
            "Checkpoint_Size": "no seperate checkpoint to base"
        }

    # Final output
    with open("combined_results.json", "w") as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    build_results()
