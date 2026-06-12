import os
import torch
import numpy as np
import cv2
import math
import pandas as pd
import time
from sklearn.metrics import confusion_matrix

from models.CAE import ConvolutionalAutoencoder
from data.preprocessing import VideoPreprocessor

os.environ["ALBUMENTATIONS_DISABLE_UPDATE_CHECK"] = "1"

IMG_SIZE       = 128
MAX_NUM_LAYERS = int(math.log2(IMG_SIZE // 4))

TEST_DIR = "data/my_data/Test"
VAL_DIR  = "data/my_data/Val"

LOG_DIR         = "outputs/log_latent"
LOG_EXPERIMENTS = os.path.join(LOG_DIR, "experiments_cae_search_latent.csv")
LOG_VIDEO_STATS = os.path.join(LOG_DIR, "video_stats_cae_search_laten.csv")
LOG_VAL_STATS   = os.path.join(LOG_DIR, "val_stats_cae_search_laten.csv")

LABELS_ANOMALY = {
    "output_video_pnp_0_1.avi": "labels_pnp_0_1.csv",
    "output_video_pnp_0_2.avi": "labels_pnp_0_2.csv",
    "output_video_pnp_0_3.avi": "labels_pnp_0_3.csv",
    "output_video_pnp_0_4.avi": "labels_pnp_0_4.csv",
}


LABELS_NORM = {
    "output_video_pnp_0.avi":  "labels_pnp_0.csv",
    "output_video_pnp_12.avi": "labels_pnp_12.csv",
    "output_video_pnp_13.avi": "labels_pnp_13.csv",
    "output_video_pnp_16.avi": "labels_pnp_16.csv",
    "output_video_pnp_17.avi": "labels_pnp_17.csv",
}

CONFIGS = [
    {"kernel_size": 7, "channels_out_1_layer": 32, "latent_dims": [1027, 316, 178],"num_layers":5},
    {"kernel_size": 4, "channels_out_1_layer": 64, "latent_dims": [1025, 328,  183],"num_layers":5},
    {"kernel_size": 3, "channels_out_1_layer": 32, "latent_dims": [403, 121,  59],"num_layers":5},
    {"kernel_size": 4, "channels_out_1_layer": 32, "latent_dims": [599, 175,  97],"num_layers":5},
    {"kernel_size": 3, "channels_out_1_layer": 32, "latent_dims": [604, 187,  92],"num_layers":4},
    {"kernel_size": 5, "channels_out_1_layer": 32, "latent_dims": [742, 224,  126],"num_layers":5},
    {"kernel_size": 4, "channels_out_1_layer": 32, "latent_dims": [687, 211,  102],"num_layers":4},
    ]


def log_to_csv(data: dict, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame([data])
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)


def load_model(params: dict, model_path: str, device) -> torch.nn.Module:
    model = ConvolutionalAutoencoder(
        input_channels=1,
        latent_dim=params['latent_dim'],
        kernel_size=params['kernel_size'],
        channels_out_1_layer=params['channels_out_1_layer'],
        num_layers=params['num_layers'],
        in_size=IMG_SIZE,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_frame_losses(model, video_path: str, preprocessor, device) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    losses = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            p_frame = preprocessor.process_frame(frame, validate_quality=False)
            tensor = torch.from_numpy(p_frame).view(1, 1, IMG_SIZE, IMG_SIZE).to(device).float()
            with torch.no_grad():
                recon = model(tensor)
                loss  = torch.mean((tensor - recon) ** 2).item()
                losses.append(loss)
        except Exception as e:
            print(f"  [WARN] {os.path.basename(video_path)}: {e}")
            break
    cap.release()
    return np.array(losses)


def evaluate_val(params: dict, model, device) -> dict | None:
    preprocessor = VideoPreprocessor(
        target_size=(IMG_SIZE, IMG_SIZE),
        convert_to_grayscale=True,
        quality_threshold=0.0,
    )

    video_files = sorted([
        f for f in os.listdir(VAL_DIR)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ])
    if not video_files:
        print(f"  [VAL] Нет видео в {VAL_DIR}")
        return None

    all_losses = []
    for vf in video_files:
        losses = get_frame_losses(model, os.path.join(VAL_DIR, vf), preprocessor, device)
        all_losses.extend(losses)

    arr = np.array(all_losses)
    print(f"  [VAL] {len(arr)} кадров  mean={np.mean(arr):.6f}  p90={np.percentile(arr,90):.6f}")

    return {
        "exp_id":               params['exp_id'],
        "kernel_size":          params['kernel_size'],
        "channels_out_1_layer": params['channels_out_1_layer'],
        "latent_dim":           params['latent_dim'],
        "mean_loss_val":        round(float(np.mean(arr)),             8),
        "max_loss_val":         round(float(np.max(arr)),              8),
        "std_val":              round(float(np.std(arr)),              8),
        "p90_val":              round(float(np.percentile(arr, 90)),   8),
    }


def evaluate_model(params: dict, model_path: str, device):
    print(f"\n{'='*65}")
    print(f"  exp_id={params['exp_id']}  kernel={params['kernel_size']}  "
          f"filters={params['channels_out_1_layer']}  latent={params['latent_dim']}")
    print(f"{'='*65}")

    model = load_model(params, model_path, device)

    preprocessor = VideoPreprocessor(
        target_size=(IMG_SIZE, IMG_SIZE),
        convert_to_grayscale=True,
        quality_threshold=0.0,
    )

    video_files = sorted([
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ])
    if not video_files:
        print(f"  Нет видео в {TEST_DIR}")
        return []

    norm_losses_all  = []
    norm_labels_all  = []
    norm_video_data  = {}   # vf → (losses, labels)
    anom_video_data  = {}   # vf → (losses, labels)

    video_rows = []

    for vf in video_files:
        t0     = time.time()
        losses = get_frame_losses(model, os.path.join(TEST_DIR, vf), preprocessor, device)
        t_proc = round(time.time() - t0, 2)

        row = {
            "exp_id":               params['exp_id'],
            "kernel_size":          params['kernel_size'],
            "channels_out_1_layer": params['channels_out_1_layer'],
            "latent_dim":           params['latent_dim'],
            "video":                vf,
            "video_proc_time_sec":  t_proc,
  
            "mean_loss":            round(float(np.mean(losses)),  8) if len(losses) else None,
  
            "normal_mean_loss":     None,
            "anomaly_mean_loss":    None,
            "separability_ratio":   None,

            "specificity":          None,
            "TN":                   None,
            "FP":                   None,
        }

        if vf in LABELS_ANOMALY:
            labels = np.genfromtxt(LABELS_ANOMALY[vf], delimiter=',', usecols=1, skip_header=1)
            norm_l = losses[labels == 0]
            anom_l = losses[labels == 1]
            sep = (float(np.mean(anom_l)) / float(np.mean(norm_l))
                   if len(norm_l) and np.mean(norm_l) > 0 else 0.0)

            print(f"  {vf}")
            print(f"    норма    ({len(norm_l):4d} кадров): mean={np.mean(norm_l):.6f}")
            print(f"    аномалия ({len(anom_l):4d} кадров): mean={np.mean(anom_l):.6f}")
            print(f"    разделимость: {sep:.3f}x")

            row.update({
                "normal_mean_loss":   round(float(np.mean(norm_l)),  8),
                "anomaly_mean_loss":  round(float(np.mean(anom_l)),  8),
                "separability_ratio": round(sep, 3),
            })
            anom_video_data[vf] = (losses, labels)

        elif vf in LABELS_NORM:
            labels = np.genfromtxt(LABELS_NORM[vf], delimiter=',', usecols=1, skip_header=1)
            norm_video_data[vf] = (losses, labels)
            norm_losses_all.extend(losses)
            norm_labels_all.extend(labels)

        video_rows.append(row)

    if not norm_losses_all:
        print("  [WARN] Нет нормальных видео для расчёта порога — специфичность не считается")
        return video_rows

    threshold = float(np.percentile(np.array(norm_losses_all), 90))
    print(f"\n  Порог (p90 нормы): {threshold:.6f}")

    print(f"\n  Специфичность (нормальные видео):")
    for vf, (v_losses, v_labels) in norm_video_data.items():
        preds = (v_losses >= threshold).astype(int)
        cm    = confusion_matrix(v_labels, preds, labels=[0, 1])
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        print(f"    {vf:<35} spec={spec:.4f}  TN={tn}  FP={fp}")

        for row in video_rows:
            if row["video"] == vf:
                row.update({"specificity": round(spec, 4), "TN": tn, "FP": fp})
                break

    return video_rows

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")


    experiments = []
    exp_id = 1
    for cfg in CONFIGS:
        for ld in cfg["latent_dims"]:
            experiments.append({
                "exp_id":               exp_id,
                "kernel_size":          cfg["kernel_size"],
                "channels_out_1_layer": cfg["channels_out_1_layer"],
                "latent_dim":           ld,
                "num_layers":           cfg["num_layers"],
            })
            exp_id += 1

    print(f"Всего экспериментов: {len(experiments)}\n")

    for params in experiments:
        try:
            model_path = f"outputs/models/latent_2/cae_exp_{params['exp_id']}.pth"


            model = load_model(params, model_path, device)
            val_row = evaluate_val(params, model, device)
            if val_row:
                log_to_csv(val_row, LOG_VAL_STATS)

  
            video_rows = evaluate_model(params, model_path, device)

            sep_vals  = [r["separability_ratio"] for r in video_rows if r["separability_ratio"] is not None]
            spec_vals = [r["specificity"]        for r in video_rows if r["specificity"]        is not None]

            exp_row = {
                "exp_id":               params["exp_id"],
                "kernel_size":          params["kernel_size"],
                "channels_out_1_layer": params["channels_out_1_layer"],
                "latent_dim":           params["latent_dim"],
                "mean_sep":             round(float(np.mean(sep_vals)),  4) if sep_vals  else None,
                "min_sep":              round(float(np.min(sep_vals)),   4) if sep_vals  else None,
                "mean_specificity":     round(float(np.mean(spec_vals)), 4) if spec_vals else None,
                "min_specificity":      round(float(np.min(spec_vals)),  4) if spec_vals else None,
            }
            log_to_csv(exp_row, LOG_EXPERIMENTS)

            for row in video_rows:
                log_to_csv(row, LOG_VIDEO_STATS)

            print(f"\n  Логи сохранены (exp_id={params['exp_id']})\n")

        except Exception as e:
            print(f"  [ERROR] exp_id={params['exp_id']}: {e}")
