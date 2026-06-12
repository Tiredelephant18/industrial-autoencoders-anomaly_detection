import os
import torch
import numpy as np
import cv2
import math
import pandas as pd
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix

from models.cae_without_latent import ConvolutionalAutoencoder
from data.dataset import VideoDataset
from data.preprocessing import VideoPreprocessor

os.environ["ALBUMENTATIONS_DISABLE_UPDATE_CHECK"] = "1"

img_size = 128
max_num_layers = int(math.log2(img_size // 4))

VAL_DIR  = "data/my_data/Val"
TEST_DIR = "data/my_data/Test"
LOG_PATH_VAL  = "outputs/log_num/stat_val.csv"
LOG_PATH_SEP  = "outputs/log_num/stat_sep.csv"
LOG_PATH_SPEC = "outputs/log_num/stat_spec.csv"   

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

COMBINATIONS = [
    {'kernel_size': 3, 'channels_out_1_layer': 32},

]

TRAIN_DIR      = "data/my_data/Train"
MODEL_DIR      = "outputs/models/num"
BATCH_SIZE     = 16
LEARNING_RATE  = 0.0001
EPOCHS         = 20
LOG_PATH_TRAIN = "outputs/log_num/train.csv"


def log_to_csv(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame([data])
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)


def train(params, device):
    """Обучение одного эксперимента. Возвращает (model, model_path, train_log)."""
    start_time = time.time()

    os.makedirs(MODEL_DIR, exist_ok=True)

    preprocessor = VideoPreprocessor(
        target_size=(img_size, img_size),
        quality_threshold=0.0,
    )

    train_files = []
    for root, _, files in os.walk(TRAIN_DIR):
        for f in sorted(files):
            if f.lower().endswith(('.tif', '.png', '.jpg')):
                train_files.append(os.path.join(root, f))

    dataset = VideoDataset(
        train_files,
        frame_size=(img_size, img_size),
        preprocessor=preprocessor,
        mode="frame",
    )
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = ConvolutionalAutoencoder(
        input_channels=1,
        kernel_size=params['kernel_size'],
        channels_out_1_layer=params['channels_out_1_layer'],
        num_layers=params['num_layers'],
        in_size=img_size,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"\n--- Обучение exp_{params['exp_id']}  "
          f"kernel={params['kernel_size']}  "
          f"ch={params['channels_out_1_layer']}  "
          f"layers={params['num_layers']} ---")

    epoch_losses = []
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            if torch.isnan(loss):
                print(f"  NaN на эпохе {epoch}, батч {batch_idx} — пропуск")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"  Эпоха [{epoch + 1:>2}/{EPOCHS}]  loss={avg_loss:.6f}")

    training_time = time.time() - start_time
    model_path = os.path.join(MODEL_DIR, f"cae_exp_{params['exp_id']}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  Сохранено: {model_path}  ({training_time:.1f} сек)")

    train_log = {
        "exp_id":               params['exp_id'],
        "kernel_size":          params['kernel_size'],
        "channels_out_1_layer": params['channels_out_1_layer'],
        "num_layers":           params['num_layers'],
        "epochs":               EPOCHS,
        "start_loss":           round(epoch_losses[0],  6),
        "final_loss":           round(epoch_losses[-1], 6),
        "train_time_sec":       round(training_time, 1),
    }
    return model, model_path, train_log


def load_model(params, model_path, device):
    model = ConvolutionalAutoencoder(
        input_channels=1,
        kernel_size=params['kernel_size'],
        channels_out_1_layer=params['channels_out_1_layer'],
        num_layers=params['num_layers'],
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
    """Статистика лосса на нормальных видео — одна строка на эксперимент."""
    preprocessor = VideoPreprocessor(
        target_size=(img_size, img_size),
        convert_to_grayscale=True,
        quality_threshold=0.0
    )

    video_files = sorted([
        f for f in os.listdir(VAL_DIR)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
        and f not in labels_map
    ])

    if not video_files:
        print(f"  Нет нормальных видео в {VAL_DIR}")
        return None

    all_losses = []
    for video_file in video_files:
        losses = get_frame_losses(model, os.path.join(VAL_DIR, video_file), preprocessor, device)
        all_losses.extend(losses)

    arr = np.array(all_losses)
    print(f"  Val норма: {len(arr)} кадров, mean={np.mean(arr):.6f}, max={np.max(arr):.6f}")

    return {
        "exp_id":               params['exp_id'],
        "kernel_size":          params['kernel_size'],
        "channels_out_1_layer": params['channels_out_1_layer'],
        "num_layers":           params['num_layers'],
        "mean_loss_val":        round(np.mean(arr),           8),
        "max_loss_val":         round(np.max(arr),            8),
        "min_loss_val":         round(np.min(arr),            8),
        "std_val":              round(np.std(arr),            8),
        "p90_val":              round(np.percentile(arr, 90), 8),
        "p95_val":              round(np.percentile(arr, 95), 8),
        "p99_val":              round(np.percentile(arr, 99), 8),
    }


def evaluate_separability(params, model, device):
    """
    Разделимость — отдельная строка на каждое аномальное видео.
    Возвращает список строк.
    """
    preprocessor = VideoPreprocessor(
        target_size=(img_size, img_size),
        convert_to_grayscale=True,
        quality_threshold=0.0
    )

    rows = []

    for video_file, label_file in labels_map.items():
        video_path = os.path.join(TEST_DIR, video_file)
        if not os.path.exists(video_path):
            print(f"  Не найдено: {video_path}")
            continue

        losses_arr = get_frame_losses(model, video_path, preprocessor, device)
        true_label = np.genfromtxt(label_file, delimiter=',', usecols=(1), skip_header=1)

        normal_losses  = losses_arr[true_label == 0]
        anomaly_losses = losses_arr[true_label == 1]

        if len(normal_losses) == 0 or len(anomaly_losses) == 0:
            print(f"  [{video_file}] пропуск — нет нормы или аномалий")
            continue

        sep = np.mean(anomaly_losses) / np.mean(normal_losses) if np.mean(normal_losses) > 0 else 0

        print(f"  [{video_file}]")
        print(f"    Норма    ({len(normal_losses):4d} кадров): "
              f"mean={np.mean(normal_losses):.6f}  max={np.max(normal_losses):.6f}  "
              f"min={np.min(normal_losses):.6f}")
        print(f"    Аномалия ({len(anomaly_losses):4d} кадров): "
              f"mean={np.mean(anomaly_losses):.6f}  min={np.min(anomaly_losses):.6f}")
        print(f"    Разделимость: {sep:.2f}x")

        rows.append({
            "exp_id":               params['exp_id'],
            "kernel_size":          params['kernel_size'],
            "channels_out_1_layer": params['channels_out_1_layer'],
            "num_layers":           params['num_layers'],
            "video":                video_file,
            "normal_mean_loss":     round(np.mean(normal_losses), 8),
            "normal_max_loss":      round(np.max(normal_losses),  8),
            "normal_min_loss":      round(np.min(normal_losses),  8),
            "anomaly_mean_loss":    round(np.mean(anomaly_losses), 8),
            "anomaly_min_loss":     round(np.min(anomaly_losses),  8),
            "anomaly_max_loss":     round(np.max(anomaly_losses),  8),
            "separability_ratio":   round(sep, 3),
        })

    return rows


def evaluate_specificity(params, model, device):
    """
    Специфичность на нормальных видео из TEST_DIR.
    Порог — p90 по всем нормальным кадрам (аналогично specificity_evaluation из metrics.py).
    Возвращает: (общая строка эксперимента, список строк по каждому видео).
    """
    preprocessor = VideoPreprocessor(
        target_size=(img_size, img_size),
        convert_to_grayscale=True,
        quality_threshold=0.0,
    )

    all_losses = []
    per_video  = {}   # video_file -> losses_arr

    for video_file, label_file in labels_norm_map.items():
        video_path = os.path.join(TEST_DIR, video_file)
        if not os.path.exists(video_path):
            print(f"  [spec] Не найдено: {video_path}")
            continue
        losses_arr = get_frame_losses(model, video_path, preprocessor, device)
        per_video[video_file] = losses_arr
        all_losses.extend(losses_arr)

    if not all_losses:
        print("  [spec] Нет нормальных видео для расчёта специфичности")
        return None, []

    all_arr   = np.array(all_losses)
    threshold = float(np.percentile(all_arr, 90))   # p90 нормы — порог

    # --- глобальная специфичность (все нормальные кадры вместе) ---
    preds_all    = (all_arr >= threshold).astype(int)
    tn_all       = int(np.sum(preds_all == 0))
    fp_all       = int(np.sum(preds_all == 1))
    spec_global  = tn_all / (tn_all + fp_all) if (tn_all + fp_all) > 0 else 0.0

    print(f"  [spec] Порог (p90): {threshold:.6f}  "
          f"Специфичность общая: {spec_global:.4f}  TN={tn_all}  FP={fp_all}")

    exp_row = {
        "exp_id":               params['exp_id'],
        "kernel_size":          params['kernel_size'],
        "channels_out_1_layer": params['channels_out_1_layer'],
        "num_layers":           params['num_layers'],
        "threshold_p90":        round(threshold, 8),
        "specificity_global":   round(spec_global, 4),
        "tn_global":            tn_all,
        "fp_global":            fp_all,
    }

 
    video_rows = []
    for video_file, losses_arr in per_video.items():
        preds = (losses_arr >= threshold).astype(int)
        tn    = int(np.sum(preds == 0))
        fp    = int(np.sum(preds == 1))
        spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        print(f"    {video_file:<35} spec={spec:.4f}  TN={tn}  FP={fp}")

        video_rows.append({
            "exp_id":               params['exp_id'],
            "kernel_size":          params['kernel_size'],
            "channels_out_1_layer": params['channels_out_1_layer'],
            "num_layers":           params['num_layers'],
            "video":                video_file,
            "threshold_p90":        round(threshold, 8),
            "specificity":          round(spec, 4),
            "tn":                   tn,
            "fp":                   fp,
        })

    return exp_row, video_rows


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    experiments = []
    exp_id = 1
    for combo in COMBINATIONS:
        for num_layers in range(1, max_num_layers + 1):
            experiments.append({
                **combo,
                'num_layers': num_layers,
                'exp_id':     exp_id,
            })
            exp_id += 1

    print(f"Всего экспериментов: {len(experiments)}")

    for params in experiments:
        try:

            model, model_path, train_log = train(params, device)
            log_to_csv(train_log, LOG_PATH_TRAIN)

            val_row = evaluate_val(params, model, device)
            if val_row:
                log_to_csv(val_row, LOG_PATH_VAL)

            sep_rows = evaluate_separability(params, model, device)
            for row in sep_rows:
                log_to_csv(row, LOG_PATH_SEP)

            spec_exp_row, spec_video_rows = evaluate_specificity(params, model, device)
            if spec_exp_row:
                log_to_csv(spec_exp_row, LOG_PATH_SPEC)
            for row in spec_video_rows:
                log_to_csv(row, LOG_PATH_SPEC.replace("stat_spec", "stat_spec_per_video"))

            print(f"  Логи обновлены (exp_id={params['exp_id']})\n")

        except Exception as e:
            print(f"  Ошибка в эксп {params['exp_id']}: {e}")
