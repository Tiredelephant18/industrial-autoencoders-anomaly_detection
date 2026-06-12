
import math
import torch
import torch.nn as nn
from collections import OrderedDict


def conv_out(size: int, kernel: int, stride: int, padding: int):
    return (size + 2 * padding - kernel) // stride + 1


def convT_out(size: int, kernel: int, stride: int, padding: int, out_pad: int):
    return (size - 1) * stride - 2 * padding + kernel + out_pad


def compute_temporal_params(T_in: int, T_out: int, num_layers: int, t_kernel: int):
    
    total_reduction = T_in / T_out

    if total_reduction < 1:
        raise ValueError(f"T_in={T_in} должен быть >= T_out={T_out}")
    
    if total_reduction != int(total_reduction):
        raise ValueError(
            f"T_in/T_out={total_reduction} должно быть целым числом."
        )

    n_stride2 = int(round(math.log2(total_reduction)))

    if 2 ** n_stride2 != int(total_reduction):
        raise ValueError(
            f"T_in/T_out={int(total_reduction)} не является степенью 2. "
        )
    
    if n_stride2 > num_layers:
        raise ValueError(
            f"Нужно {n_stride2} слоёв со stride=2 по T, "
            f"но num_layers={num_layers}."
        )


    if n_stride2 == 0:
        stride2_positions = set()
    elif n_stride2 == 1:
        stride2_positions = {0}
    else:
        step = num_layers / n_stride2
        stride2_positions = {int(i * step) for i in range(n_stride2)}


    T_cur = T_in
    enc_params = []  # (t_stride, t_pad, T_before, T_after)

    for i in range(num_layers):

        T_before = T_cur

        if i in stride2_positions:
            t_stride = 2
            T_target = T_cur // 2

            chosen_pad = None

            for t_pad in range(t_kernel + 1):
                if conv_out(T_cur, t_kernel, t_stride, t_pad) == T_target:
                    chosen_pad = t_pad
                    break

            if chosen_pad is None:
                best_pad, best_diff = 0, float("inf")
                for t_pad in range(t_kernel + 1):
                    T_next = conv_out(T_cur, t_kernel, t_stride, t_pad)
                    if T_next >= 1 and abs(T_next - T_target) < best_diff:
                        best_diff = abs(T_next - T_target)
                        best_pad  = t_pad
                chosen_pad = best_pad
        else:
            t_stride   = 1

            chosen_pad = (t_kernel - 1) // 2

            if conv_out(T_cur, t_kernel, 1, chosen_pad) != T_cur:
                if conv_out(T_cur, t_kernel, 1, chosen_pad + 1) == T_cur:
                    chosen_pad += 1


        T_next = conv_out(T_cur, t_kernel, t_stride, chosen_pad)

        if T_next < 1:
            raise ValueError(
                f"Слой {i}: T={T_cur} → T_next={T_next} < 1 "
                f"(kernel={t_kernel} stride={t_stride} pad={chosen_pad}). "
                f"Уменьши t_kernel или num_layers."
            )

        enc_params.append((t_stride, chosen_pad, T_before, T_next))

        T_cur = T_next

    if T_cur != T_out:
        trajectory = f"{T_in} → " + " → ".join(str(p[3]) for p in enc_params)
        raise ValueError(
            f"После энкодера T={T_cur}, ожидалось T_out={T_out}.\n"
            f"  Траектория T: {trajectory}\n"
            f"  stride=2 на слоях: {sorted(stride2_positions)}\n"
            f"  Попробуй другую комбинацию in_time/T_out/num_layers/t_kernel."
        )

    dec_params = []
    for t_stride, t_pad, T_enc_in, T_enc_out in reversed(enc_params):
        chosen_out_pad = None
        for out_pad in range(max(t_stride, 4)):
            if convT_out(T_enc_out, t_kernel, t_stride, t_pad, out_pad) == T_enc_in:
                chosen_out_pad = out_pad
                break
        if chosen_out_pad is None:
            candidates = ", ".join(
                f"op={op}→{convT_out(T_enc_out, t_kernel, t_stride, t_pad, op)}"
                for op in range(8)
            )
            raise ValueError(
                f"Декодер: не могу восстановить T={T_enc_in} из T={T_enc_out} "
                f"(kernel={t_kernel} stride={t_stride} pad={t_pad}).\n"
                f"  Варианты: {candidates}"
            )
        dec_params.append((t_stride, t_pad, chosen_out_pad))

    enc_result = [(s, p) for s, p, _, _ in enc_params]
    return enc_result, dec_params



class C3DAutoencoder(nn.Module):

    def __init__(
        self,
        input_channels: int       = 1,

        spatial_kernel: int       = 7,
        in_size: int              = 128,

        channels_out_1_layer: int = 32,
        num_layers: int           = 5,

        temporal_kernel: int      = 3,
        in_time: int              = 8,
        T_out: int                = 4,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.in_time        = in_time
        self.in_size        = in_size

        sp_pad = (spatial_kernel - 1) // 2

 
        enc_t_params, dec_t_params = compute_temporal_params(in_time, T_out, num_layers, temporal_kernel)

      
        enc    = OrderedDict()
        in_ch  = input_channels
        out_ch = channels_out_1_layer

        for i, (t_stride, t_pad) in enumerate(enc_t_params):
            enc[f"conv_{i}"] = nn.Conv3d(
                in_ch, out_ch,
                kernel_size=(temporal_kernel, spatial_kernel, spatial_kernel),
                stride     =(t_stride, 2, 2),
                padding    =(t_pad, sp_pad, sp_pad),
            )
            enc[f"bn_{i}"]   = nn.BatchNorm3d(out_ch)
            enc[f"relu_{i}"] = nn.LeakyReLU(0.2, inplace=True)
            in_ch   = out_ch
            out_ch *= 2

        self.encoder  = nn.Sequential(enc)
        self.enc_out_channels = in_ch

        dec    = OrderedDict()
        cur_in = self.enc_out_channels

        for i, (t_stride, t_pad, t_out_pad) in enumerate(dec_t_params):
            is_last = (i == num_layers - 1)
            cur_out = cur_in // 2 if not is_last else input_channels

            dec[f"deconv_{i}"] = nn.ConvTranspose3d(
                cur_in, cur_out,
                kernel_size   =(temporal_kernel, spatial_kernel, spatial_kernel),
                stride        =(t_stride, 2, 2),
                padding       =(t_pad, sp_pad, sp_pad),
                output_padding=(t_out_pad, 1, 1),
            )
            if not is_last:
                dec[f"bn_{i}"]   = nn.BatchNorm3d(cur_out)
                dec[f"relu_{i}"] = nn.ReLU(inplace=True)
            else:
                dec["sigmoid"] = nn.Sigmoid()

            cur_in = cur_out

        self.decoder = nn.Sequential(dec)

        self._verify_shape()


    def _compute_enc_shape(self) -> tuple:
        with torch.no_grad():
            dummy = torch.zeros(
                1, self.input_channels, self.in_time, self.in_size, self.in_size
            )
            out = self.encoder(dummy)
        return out.shape, int(out[0].numel())

    def _verify_shape(self):
        with torch.no_grad():
            dummy = torch.zeros(
                1, self.input_channels, self.in_time, self.in_size, self.in_size
            )
            out = self.forward(dummy)
        assert out.shape == dummy.shape, (
            f"Форма не совпадает: вход {tuple(dummy.shape)} → "
            f"выход {tuple(out.shape)}"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
  
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def describe(self):
        print(f"\n{'─'*60}")
        print(f"  C3DAutoencoder")
        print(f"  in_time={self.in_time}  in_size={self.in_size}")

        print(f"{'─'*60}")

        x = torch.zeros(
            1, self.input_channels, self.in_time, self.in_size, self.in_size
        )
        print(f"  input     : {list(x.shape)}")
        with torch.no_grad():
            for name, layer in self.encoder.named_children():
                x = layer(x)
                if "conv" in name:
                    B, C, T, H, W = x.shape
                    print(f"  enc {name}: [B={B} C={C} T={T} H={H} W={W}]")

      
        with torch.no_grad():
            dummy = torch.zeros(
                1, self.input_channels, self.in_time, self.in_size, self.in_size
            )
            x2 = self.encoder(dummy)
            for name, layer in self.decoder.named_children():
                x2 = layer(x2)
                if "deconv" in name:
                    B, C, T, H, W = x2.shape
                    print(f"  dec {name}: [B={B} C={C} T={T} H={H} W={W}]")

        print(f"  output    : {list(x2.shape)}")
        print(f"{'─'*60}\n")



if __name__ == "__main__":
    test_cases = [
        # (in_time, T_out, temporal_kernel, num_layers)
        (8,  4, 3, 5),
        (8,  4, 4, 5),
        (8,  4, 5, 5),
        (16, 4, 3, 5),
        (16, 4, 4, 5),
        (16, 4, 5, 5),
        (16, 8, 3, 5),
        (16, 8, 4, 5),
        (16, 4, 3, 4),
        (32, 4, 3, 5),
        (32, 4, 4, 5),
        (32, 4, 5, 5),
        (32, 8, 3, 5),
        (32, 8, 4, 5),
        (32, 4, 3, 4),
    ]

    for in_time, T_out, t_kernel, num_layers in test_cases:
        label = (f"in_time={in_time} T_out={T_out} "
                 f"t_kernel={t_kernel} layers={num_layers}")
        try:
            model = C3DAutoencoder(
                input_channels=1,
                spatial_kernel=4,
                in_size=128,
                channels_out_1_layer=32,
                num_layers=num_layers,
                temporal_kernel=t_kernel,
                in_time=in_time,
                T_out=T_out,
            )
            model.describe()
            print(f"  да {label}\n")
        except (ValueError, AssertionError) as e:
            print(f"  нет {label}")
            print(f"    {e}\n")