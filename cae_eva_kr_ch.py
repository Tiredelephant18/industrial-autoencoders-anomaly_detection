import os
import torch
import numpy as np
import cv2
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import time
from sklearn.metrics import fbeta_score, confusion_matrix

from models.cae_without_latent import ConvolutionalAutoencoder
from data.dataset import VideoDataset
from data.preprocessing import VideoPreprocessor
from metrics import Evaluation

os.environ["ALBUMENTATIONS_DISABLE_UPDATE_CHECK"] = "1"

img_size = 128
max_num_layers = int(math.log2(img_size // 4))

labels_map = {
    "output_video_pnp_0_1.avi": "labels_pnp_0_1.csv",
    "output_video_pnp_0_2.avi": "labels_pnp_0_2.csv",
    "output_video_pnp_0_3.avi": "labels_pnp_0_3.csv",
    "output_video_pnp_0_4.avi": "labels_pnp_0_4.csv",
}

labels_norm_map = {
    "output_video_pnp_0.avi":  "labels_pnp_0.csv",
    "output_video_pnp_12.avi": "labels_pnp_12.csv",
    "output_video_pnp_13.avi": "labels_pnp_13.csv",
    "output_video_pnp_16.avi": "labels_pnp_16.csv",
    "output_video_pnp_17.avi": "labels_pnp_17.csv",
}

TEST_DIR = "data/my_data/Test"
VAL_DIR  = "data/my_data/Val"

LOG_EXPERIMENTS = "outputs/log_norm/experiments.csv"
LOG_VIDEO_STATS = "outputs/log_norm/video_stats.csv"
LOG_VAL_STATS   = "outputs/log_norm/val_stats.csv"




def log_to_csv(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame([data])
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)


def load_model(params, model_path, device):
    model = ConvolutionalAutoencoder(
        input_channels=1,
        kernel_size=params['kernel_size'],
        channels_out_1_layer=params['channels_out_1_layer'],
        num_layers=max_num_layers,
        in_size=img_size,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_frame_losses(model, video_path, preprocessor, device):
    cap = cv2.VideoCapture(video_path)
    losses = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            p_frame = preprocessor.process_frame(frame, validate_quality=False)
            tensor = torch.from_numpy(p_frame).view(1, 1, img_size, img_size).to(device).float()
            with torch.no_grad():
                recon = model(tensor)
                loss  = torch.mean((tensor - recon) ** 2).item()
                losses.append(loss)
        except Exception as e:
            print(f"  [DEBUG] {os.path.basename(video_path)}: {e}")
            break
    cap.release()
    return np.array(losses)




def evaluate_val(params, model, device):
    preprocessor = VideoPreprocessor(
        target_size=(img_size, img_size),
        convert_to_grayscale=True,
        quality_threshold=0.0,
    )

    video_files = sorted([
        f for f in os.listdir(VAL_DIR)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ])

    if not video_files:
        print(f"  Нет видео в {VAL_DIR}")
        return None

    all_losses = []
    for vf in video_files:
        losses = get_frame_losses(model, os.path.join(VAL_DIR, vf), preprocessor, device)
        all_losses.extend(losses)

    arr = np.array(all_losses)
    print(f"  Val норма: {len(arr)} кадров  mean={np.mean(arr):.6f}  max={np.max(arr):.6f}")

    return {
        "exp_id":               params['exp_id'],
        "kernel_size":          params['kernel_size'],
        "channels_out_1_layer": params['channels_out_1_layer'],
        "num_layers":           params['num_layers'],
        "mean_loss_val":        round(np.mean(arr),            8),
        "max_loss_val":         round(np.max(arr),             8),
        "min_loss_val":         round(np.min(arr),             8),
        "std_val":              round(np.std(arr),             8),
        "p90_val":              round(np.percentile(arr, 90),  8),
        "p95_val":              round(np.percentile(arr, 95),  8),
        "p99_val":              round(np.percentile(arr, 99),  8),
    }


def evaluate_model(device, params, model_path):

    print(f"\nСтарт оценки эксперимента {params['exp_id']}")
    print("-" * 65)

    model = load_model(params, model_path, device)

    preprocessor = VideoPreprocessor(
        target_size=(img_size, img_size),
        convert_to_grayscale=True,
        quality_threshold=0.0,
    )

    video_files = sorted([
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ])

    if not video_files:
        print(f"  Файлы не найдены в {TEST_DIR}")
        return [], {}

    per_video_data = {}   # аномальные видео  → (losses, labels)
    norm_video_data = {}  # нормальные видео   → (losses, labels)

    all_anomaly_losses = []
    all_anomaly_labels = []
    all_norm_losses    = []
    all_norm_labels    = []

    video_rows = []

    for video_file in video_files:
        video_path  = os.path.join(TEST_DIR, video_file)
        video_start = time.time()

        losses_arr = get_frame_losses(model, video_path, preprocessor, device)

        video_proc_time = time.time() - video_start
        print(f"  {video_file}: {video_proc_time:.2f} сек  "
              f"mean={np.mean(losses_arr):.6f}" if len(losses_arr) else f"  {video_file}: нет кадров")

        row = {
            "exp_id":               params['exp_id'],
            "kernel_size":          params['kernel_size'],
            "channels_out_1_layer": params['channels_out_1_layer'],
            "num_layers":           params['num_layers'],
            "video":                video_file,
            "video_proc_time_sec":  round(video_proc_time, 2),
            "mean_loss":            round(float(np.mean(losses_arr)),  8) if len(losses_arr) else None,
            "max_loss":             round(float(np.max(losses_arr)),   8) if len(losses_arr) else None,
            "min_loss":             round(float(np.min(losses_arr)),   8) if len(losses_arr) else None,
            "normal_mean_loss":     None,
            "normal_max_loss":      None,
            "anomaly_mean_loss":    None,
            "anomaly_min_loss":     None,
            "separability_ratio":   None,

        }


        if video_file in labels_map:
            true_label = np.genfromtxt(
                labels_map[video_file], delimiter=',', usecols=(1), skip_header=1
            )
            normal_losses  = losses_arr[true_label == 0]
            anomaly_losses = losses_arr[true_label == 1]
            sep = (np.mean(anomaly_losses) / np.mean(normal_losses)
                   if len(normal_losses) and np.mean(normal_losses) > 0 else 0)

            print(f"  [{video_file}]")
            print(f"    Норма    ({len(normal_losses):4d}): "
                  f"mean={np.mean(normal_losses):.6f}  max={np.max(normal_losses):.6f}")
            print(f"    Аномалия ({len(anomaly_losses):4d}): "
                  f"mean={np.mean(anomaly_losses):.6f}  min={np.min(anomaly_losses):.6f}")
            print(f"    Разделимость: {sep:.2f}x")

            row.update({
                "normal_mean_loss":   round(float(np.mean(normal_losses)),  8),
                "normal_max_loss":    round(float(np.max(normal_losses)),   8),
                "anomaly_mean_loss":  round(float(np.mean(anomaly_losses)), 8),
                "anomaly_min_loss":   round(float(np.min(anomaly_losses)),  8),
                "separability_ratio": round(sep, 3),
            })

            per_video_data[video_file] = (losses_arr, true_label)
            all_anomaly_losses.extend(losses_arr)
            all_anomaly_labels.extend(true_label)

  
        elif video_file in labels_norm_map:
            raw_labels = np.genfromtxt(
                labels_norm_map[video_file], delimiter=',', usecols=(1), skip_header=1
            )
            norm_video_data[video_file] = (losses_arr, raw_labels)
            all_norm_losses.extend(losses_arr)
            all_norm_labels.extend(raw_labels)

        video_rows.append(row)


    evaluator = Evaluation()
    metrics = evaluator.specificity_evaluation(
        np.array(all_norm_labels),
        np.array(all_norm_losses),
        exp_id=f"exp_{params['exp_id']}",
    )

    global_threshold = metrics["threshold"]
    specificity_global = metrics["specificity"]

    print(f"\n--- Метрики (exp {params['exp_id']}) ---")
    print(f"  Порог (p90 нормы): {global_threshold:.6f}  "
          f"Специфичность: {specificity_global:.4f}")


    print(f"\n  Специфичность по нормальным видео:")
    for vf, (v_losses, v_labels) in norm_video_data.items():
        preds = (v_losses >= global_threshold).astype(int)
        cm    = confusion_matrix(v_labels, preds, labels=[0, 1])
        tn, fp = cm[0, 0], cm[0, 1]
        spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        print(f"  {vf:<30} spec={spec:.4f}  TN={tn}  FP={fp}")

        for row in video_rows:
            if row["video"] == vf:
                row["specificity"] = round(spec, 4)
                break

    return video_rows, metrics




if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    """
    kernel_sizes = [4, 3, 5, 7, 9]
    channels_list = [16, 32, 64] 
    exp_id = 1
    """
    kernel_sizes = [3]
    channels_list = [64] 
    experiments = []
    exp_id = 6

    experiments = []

    for kernel_size in kernel_sizes:
        for channels_out in channels_list:
            experiments.append({
                'num_layers':           max_num_layers,
                'kernel_size':          kernel_size,
                'channels_out_1_layer': channels_out,
                'exp_id':               exp_id,
            })
            exp_id += 1

    print(f"Всего экспериментов: {len(experiments)}")

    for params in experiments:
        try:
            model_path = f"outputs/models/cae_kr_ch/CAE_exp_{params['exp_id']}.pth"
            #model_path = f"outputs/second_stat/models/cae_exp_{params['exp_id']}.pth"
            model = load_model(params, model_path, device)
            val_row = evaluate_val(params, model, device)
            if val_row:
                log_to_csv(val_row, LOG_VAL_STATS)


            video_rows, metrics = evaluate_model(device, params, model_path)

            exp_row = {
                "exp_id":               params['exp_id'],
                "kernel_size":          params['kernel_size'],
                "channels_out_1_layer": params['channels_out_1_layer'],
                "num_layers":           params['num_layers'],
                "threshold":            metrics["threshold"],
                "specificity":          metrics["specificity"],
            }
            log_to_csv(exp_row, LOG_EXPERIMENTS)

            for row in video_rows:
                log_to_csv(row, LOG_VIDEO_STATS)

            print(f"  Логи обновлены (exp_id={params['exp_id']})\n")

        except Exception as e:
            print(f"  Ошибка в эксп {params['exp_id']}: {e}")