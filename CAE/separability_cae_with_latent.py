import os
import torch
import numpy as np
import cv2
import math
import pandas as pd

from models.CAE import ConvolutionalAutoencoder
from data.preprocessing import VideoPreprocessor
from data.dataset import VideoDataset
from data.preprocessing import VideoPreprocessor
from metrics import Evaluation
import matplotlib.pyplot as plt


os.environ["ALBUMENTATIONS_DISABLE_UPDATE_CHECK"] = "1"

img_size = 128
max_num_layers = int(math.log2(img_size // 4))

VAL_DIR  = "data/my_data/Val"
TEST_DIR = "data/my_data/Test"
LOG_PATH_VAL = "outputs/log/stat_val.csv"
LOG_PATH_SEP = "outputs/log/stat_sep.csv"

PATH_MODEL = "outputs/models/latent_2/"
PATH_HIST = "outputs/hist/"
PATH_CURVE = "outputs/curve/"
PATH_IMAGE = "outputs/image/"

labels_map = {
    "output_video_pnp_0_1.avi": "labels_pnp_0_1.csv",
    "output_video_pnp_0_2.avi": "labels_pnp_0_2.csv",
    "output_video_pnp_0_3.avi": "labels_pnp_0_3.csv",
    "output_video_pnp_0_4.avi": "labels_pnp_0_4.csv",
}

COMBINATIONS = [
    {"exp_id": 2,"kernel_size": 7, "channels_out_1_layer": 32, "latent_dim":  316,"num_layers":5},
    {"exp_id":5 ,"kernel_size": 4, "channels_out_1_layer": 64, "latent_dim":  328,"num_layers":5},
    #{"exp_id": 8,"kernel_size": 3, "channels_out_1_layer": 32, "latent_dim": 121,"num_layers":5},
    {"exp_id": 10,"kernel_size": 4, "channels_out_1_layer": 32, "latent_dim": 599,"num_layers":5},
    #{"exp_id":15 ,"kernel_size": 3, "channels_out_1_layer": 32, "latent_dim":  92,"num_layers":4},
    {"exp_id": 18,"kernel_size": 5, "channels_out_1_layer": 32, "latent_dim":   126,"num_layers":5},
    {"exp_id": 19,"kernel_size": 4, "channels_out_1_layer": 32, "latent_dim": 687,"num_layers":4},
   
]


def log_to_csv(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame([data])
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

def show_results(input_tensor, reconstructed_tensor, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    """
    img_in = input_tensor.cpu().detach()[0]
    img_out = reconstructed_tensor.cpu().detach()[0]

    if img_in.dim() == 4 and img_in.shape[1] == 1: # T, C, H, W
        img_in = img_in[-1, 0]
        img_out = img_out[-1, 0]

    elif img_in.dim() == 4:
        img_in  = img_in[0,-1]
        img_out  = img_out[0,-1]

    elif img_in.dim() == 3:
        img_in  = img_in[0]
        img_out  = img_out[0]

    img_in = img_in.numpy()
    img_out = img_out.numpy()
    """
    img_in  = input_tensor.cpu().detach()[0, 0].numpy()   # (H, W)
    img_out = reconstructed_tensor.cpu().detach()[0, 0].numpy()  # (H, W)

    img_in  = np.clip(img_in,  0, 1)
    img_out = np.clip(img_out, 0, 1)
    comparison = np.hstack((img_in, img_out))
    comparison = np.clip(comparison, 0, 1)
    
    comparison = (comparison * 255).astype(np.uint8)
        
    cv2.imwrite(save_path, comparison)

def load_model(params, model_path, device):
    model = ConvolutionalAutoencoder(
        input_channels=1,
        latent_dim=params['latent_dim'],
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


def get_frame_losses(model, video_path, preprocessor, device, exp_id =0,video_file = None):
    cap = cv2.VideoCapture(video_path)
    losses = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            p_frame = preprocessor.process_frame(frame, validate_quality=False)
            tensor = torch.from_numpy(p_frame).view(1, 1, img_size, img_size).to(device).float()
            
            with torch.no_grad():
                recon = model(tensor)
                if frame_idx % 20 == 0 :
                    save_path = os.path.join(
                        PATH_IMAGE,
                        f"exp{exp_id}_{video_file}_frame{frame_idx:04d}.jpg"
                    )
                    show_results(tensor,recon,save_path )
                loss  = torch.mean((tensor - recon) ** 2).item()
                losses.append(loss)
            frame_idx += 1
        except Exception as e:
            print(f"  [DEBUG] {os.path.basename(video_path)}: {e}")
            break
    cap.release()
    return np.array(losses)


def evaluate_val(params, model, device):
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
        losses = get_frame_losses(model, os.path.join(VAL_DIR, video_file), preprocessor, device, params['exp_id'],video_file)
        all_losses.extend(losses)

    arr = np.array(all_losses)
    print(f"  Val норма: {len(arr)} кадров, mean={np.mean(arr):.6f}, max={np.max(arr):.6f}")

    return {
        "exp_id":               params['exp_id'],
        "kernel_size":          params['kernel_size'],
        "channels_out_1_layer": params['channels_out_1_layer'],
        "num_layers":           params['num_layers'],
        "latent_dim":           params['latent_dim'],
        "mean_loss_val":        round(np.mean(arr),           8),
        "max_loss_val":         round(np.max(arr),            8),
        "min_loss_val":         round(np.min(arr),            8),
        "std_val":              round(np.std(arr),            8),
        "p90_val":              round(np.percentile(arr, 90), 8),

    }


def evaluate_separability(params, model, device):
 
    
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

        losses_arr = get_frame_losses(model, video_path, preprocessor, device,exp_id=params['exp_id'], video_file=video_file)
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

        evaluator = Evaluation()
    
        metrics = evaluator.separability_evaluation(
                    np.array(normal_losses), 
                    np.array(anomaly_losses),
                    save_path_hist=PATH_HIST,
                    save_path_curve=PATH_CURVE, 
                    exp_id=f"exp_{params['exp_id']}_{video_file}"
                )
        
        rows.append({
            "exp_id":               params['exp_id'],
            "kernel_size":          params['kernel_size'],
            "channels_out_1_layer": params['channels_out_1_layer'],
            "num_layers":           params['num_layers'],
            "latent_dim":           params['latent_dim'],
            "video":                video_file,
            "normal_mean_loss":     round(np.mean(normal_losses), 8),
            "normal_max_loss":      round(np.max(normal_losses),  8),

            "anomaly_mean_loss":    round(np.mean(anomaly_losses), 8),
            "anomaly_min_loss":     round(np.min(anomaly_losses),  8),

            "separability_ratio":   round(sep, 3),
            "roc_auc":              metrics['roc_auc'],
            "pr_auc" :              metrics['pr_auc'],
            "overlap" :             metrics['overlap'],


        })

    return rows
    

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    
    experiments = []


    for combo in COMBINATIONS:
  
            experiments.append({
                **combo,


            })



    print(f"Всего экспериментов: {len(experiments)}")

    for params in experiments:
        try:
            model_path = f"{PATH_MODEL}cae_exp_{params['exp_id']}.pth"
            model = load_model(params, model_path, device)

            
            val_row = evaluate_val(params, model, device)
            sep_rows = evaluate_separability(params, model, device)
            """
            if val_row:
                #log_to_csv(val_row, LOG_PATH_VAL)


            sep_rows = evaluate_separability(params, model, device)
            for row in sep_rows:
                log_to_csv(row, LOG_PATH_SEP)

            print(f"  Лог обновлён (exp_id={params['exp_id']})\n")
            """
        except Exception as e:
            print(f"  Ошибка в эксп {params['exp_id']}: {e}")