
import os
import torch
import numpy as np
import cv2
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time
from sklearn.metrics import  fbeta_score,confusion_matrix


from models.CAE import ConvolutionalAutoencoder
from data.dataset import VideoDataset
from data.preprocessing import VideoPreprocessor
from metrics import Evaluation
import matplotlib.pyplot as plt

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
    "output_video_pnp_0.avi": "labels_pnp_0.csv",
    "output_video_pnp_12.avi": "labels_pnp_12.csv",
    "output_video_pnp_13.avi": "labels_pnp_13.csv",
    "output_video_pnp_16.avi": "labels_pnp_16.csv",
    "output_video_pnp_17.avi": "labels_pnp_17.csv",
}


input_dir='data/my_data/Test'

def log_to_csv(data, filename):
    df = pd.DataFrame([data])
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)


    
def train_stable(params, device):
    start_time = time.time()
    kernel_size = params['kernel_size']
    c_out_1l = params['channels_out_1_layer']

    latent_dim = params['latent_dim']


    DATA_PATH = "/home/liza/Desktop/diplom/code/code_python/data/my_data/Train"
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001  
    EPOCHS = 20
   
    
    print(f"--- Запуск обучения на {device} эксперимента {params['exp_id']}---")


    preprocessor = VideoPreprocessor(target_size=(img_size , img_size ), quality_threshold=0.0)
    
    train_files = []
    for root, _, files in os.walk(DATA_PATH):
        for file in sorted(files):
            if file.endswith(('.tif', '.png', '.jpg')):
                train_files.append(os.path.join(root, file))
    
    dataset = VideoDataset(train_files, frame_size=(img_size , img_size ), preprocessor=preprocessor, mode="frame")
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

   
    model = ConvolutionalAutoencoder(input_channels=1,
                                         latent_dim = latent_dim,
                                         kernel_size = kernel_size,
                                         channels_out_1_layer = c_out_1l,
                                         num_layers = params['num_layers'],
                                         in_size =img_size,
                                         ).to(device)
    print(model.encoder)
    print(model.decoder)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

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
                print(f" Обнаружен NaN на эпохе {epoch}, батч {batch_idx}. Пропускаем...")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Эпоха [{epoch+1}/{EPOCHS}] | Средний Loss: {avg_loss:.6f}")
    training_time = time.time() - start_time
        
    model_path = f"outputs/models/latent_2/cae_with_latent_exp_{params['exp_id']}.pth"
    torch.save(model.state_dict(), model_path)
    print(f" Обучение эксперимента {params['exp_id']} завершено ")
    experiment_log = {
        "exp_id": params["exp_id"],
        "kernel_size": params["kernel_size"],
        "channels_out_1_layer": params["channels_out_1_layer"],
        "final_loss": epoch_losses[-1],
        "start_loss": epoch_losses[0],
        "train_time_sec": training_time,
    }
    
    return model_path, experiment_log


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
   

   

    kernel_sizes = [3,3,4]
    num_layers = [5,4,4]
    channels_layers =[32,32,32]
    latent_dims = [403,121,59,604,187,92,687,211,102]

    experiments = []
    exp_id = 1
    step = 3
    
    for i, (k_size, ch_out,num_layer) in enumerate(zip(kernel_sizes, channels_layers, num_layers)):
        

        start = i * step
        end = start + step
        current_latents = latent_dims[start:end]

        for l_dim in current_latents:
            experiments.append({
                'exp_id': exp_id,
                'kernel_size': k_size,
                'channels_out_1_layer': ch_out,
                'latent_dim': l_dim,
                'num_layers': num_layer
            })
            exp_id += 1


    print(f"Всего создано экспериментов: {len(experiments)}")
    for params in experiments:
        try:
  
            model_path, experiment_log = train_stable(params, device)

            log_to_csv(experiment_log, "outputs/log_latent/train_experiments.csv")

            print(f"Логи обновлены (exp_id={params['exp_id']})")

        except Exception as e:
            print(f"Ошибка в эксп {params['exp_id']}: {e}")