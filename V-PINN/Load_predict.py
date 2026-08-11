import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os


class VectorPotentialPINN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=512, num_layers=5):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)


def compute_B_scaled(model, coords_norm, inv_scale):
    """
    计算原始尺度下的磁场 B = -∇φ
    coords_norm: (batch, 3) 归一化坐标，requires_grad=True
    inv_scale: (3,) 缩放因子
    returns: (batch, 3) 磁场 Bx, By, Bz
    """
    phi = model(coords_norm)  # (batch, 1)

    inv_sx, inv_sy, inv_sz = inv_scale[0], inv_scale[1], inv_scale[2]

    # 计算 phi 对归一化坐标的梯度
    grad_phi = torch.autograd.grad(
        outputs=phi,
        inputs=coords_norm,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]  # (batch, 3)

    # 处理可能的 None 梯度
    if grad_phi is None:
        grad_phi = torch.zeros_like(coords_norm, requires_grad=False)

    # 转换到原始尺度：∂φ/∂x_raw = ∂φ/∂x_norm * inv_sx，等
    # B = -∇φ，所以 Bx = -∂φ/∂x_raw = - (∂φ/∂x_norm * inv_sx)
    Bx = -grad_phi[:, 0:1] * inv_sx
    By = -grad_phi[:, 1:2] * inv_sy
    Bz = -grad_phi[:, 2:3] * inv_sz

    return torch.cat([Bx, By, Bz], dim=-1)


def compute_dT(B, alpha, beta, gamma):
    """计算总场异常 dT"""
    return alpha * B[:, 0] + beta * B[:, 1] + gamma * B[:, 2]

def load_model_and_params(model_path='vector_potential_pinn.pth',
                          norm_params_path='normalization_params.npz',
                          device='cpu'):
    """加载训练好的模型和归一化参数"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    input_dim = 3
    hidden_dim = 512
    num_layers = 5

    model = VectorPotentialPINN(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    model.load_state_dict(checkpoint)
    npz_file = np.load(norm_params_path)
    norm_params = {
        'coord_min': npz_file['coord_min'],
        'coord_max': npz_file['coord_max']
    }

    print(f"模型已从 {model_path} 加载")
    print(f"归一化参数：coord_min shape={norm_params['coord_min'].shape}, coord_max shape={norm_params['coord_max'].shape}")

    return model, norm_params


def normalize_coords(coords_raw, coord_min, coord_max):
    """将坐标归一化到 [-1, 1]"""
    coords_norm = 2 * (coords_raw - coord_min) / (coord_max - coord_min) - 1
    return coords_norm


def predict_dT(model, coords_norm, inv_scale, alpha, beta, gamma, batch_size=4096):
    """使用模型预测 dT"""
    device = next(model.parameters()).device
    model.eval()

    dT_pred_list = []

    for i in range(0, len(coords_norm), batch_size):
        coords_batch = torch.tensor(coords_norm[i:i + batch_size], dtype=torch.float32, device=device, requires_grad=True)
        B = compute_B_scaled(model, coords_batch, inv_scale)
        dT_batch = compute_dT(B, alpha, beta, gamma).detach().cpu().numpy()
        dT_pred_list.append(dT_batch)

    dT_pred = np.concatenate(dT_pred_list, axis=0)
    return dT_pred


def plot_residual_contour_eval(coords_raw, dT_pred, dT_true, save_dir='evaluation_results', plot_type='hexbin'):
    """绘制残差分布图（单高度）"""
    os.makedirs(save_dir, exist_ok=True)

    residuals = dT_pred.flatten() - dT_true.flatten()

    df = pd.DataFrame({
        'X': coords_raw[:, 0],
        'Y': coords_raw[:, 1],
        'Z': coords_raw[:, 2],
        'residual': residuals
    })

    z_value = coords_raw[0, 2]

    if plot_type == 'hexbin':
        fig, ax = plt.subplots(figsize=(10, 8))
        vmin_val = -150
        vmax_val = 150

        hb = ax.hexbin(df['Y'], df['X'],
                       C=df['residual'],
                       gridsize=50, cmap='RdBu_r',
                       mincnt=1, reduce_C_function=np.mean,
                       vmin=vmin_val, vmax=vmax_val)

        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label('Mean Residual (nT)', fontsize=12)

        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

        ax.set_xlabel('Y (m)', fontsize=12)
        ax.set_ylabel('X (m)', fontsize=12)
        ax.set_title(f'Residual Hexbin at Z={z_value:.0f} m', fontsize=14)
        ax.set_aspect('equal')

        plt.tight_layout()
        save_file = os.path.join(save_dir, f'residual_hexbin_z{int(z_value)}.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"六边形分箱图已保存至 {save_file}")

    elif plot_type == 'scatter':
        fig, ax = plt.subplots(figsize=(10, 8))
        vmin_val = -150
        vmax_val = 150

        scatter = ax.scatter(df['Y'], df['X'],
                             c=df['residual'],
                             cmap='RdBu_r', s=10, alpha=0.6,
                             edgecolors='none', linewidths=0,
                             vmin=vmin_val, vmax=vmax_val)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Residual (nT)', fontsize=12)

        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

        ax.set_xlabel('Y (m)', fontsize=12)
        ax.set_ylabel('X (m)', fontsize=12)
        ax.set_title(f'Residual Distribution at Z={z_value:.0f} m', fontsize=14)
        ax.set_aspect('equal')

        plt.tight_layout()
        save_file = os.path.join(save_dir, f'residual_scatter_z{int(z_value)}.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"散点图已保存至 {save_file}")

    elif plot_type == 'contour':
        x = df['X'].values
        y = df['Y'].values
        residual = df['residual'].values
        vmin_val = -150
        vmax_val = 150

        if len(x) < 3:
            print(f"警告：z={z_value} 高度数据点不足，跳过")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        contour = ax.tricontourf(y, x, residual, levels=20, cmap='RdBu_r', alpha=0.8, vmin=vmin_val, vmax=vmax_val)
        contour_lines = ax.tricontour(y, x, residual, levels=10, colors='black',
                                      linewidths=0.5, alpha=0.6)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')

        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Residual (nT)', fontsize=12)

        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

        ax.set_xlabel('Y (m)', fontsize=12)
        ax.set_ylabel('X (m)', fontsize=12)
        ax.set_title(f'Residual Contour at Z={z_value:.0f} m', fontsize=14)
        ax.set_aspect('equal')

        plt.tight_layout()
        save_file = os.path.join(save_dir, f'residual_contour_z{int(z_value)}.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"等值线图已保存至 {save_file}")

    plt.close('all')


def calculate_error_metrics(dT_pred, dT_true):
    """计算各种误差指标"""
    residuals = dT_pred.flatten() - dT_true.flatten()

    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    max_abs_error = np.max(np.abs(residuals))
    mean_error = np.mean(residuals)
    std_error = np.std(residuals)

    r_squared = 1 - np.sum(residuals ** 2) / np.sum((dT_true - np.mean(dT_true)) ** 2)

    mape = np.mean(np.abs(residuals / (dT_true.flatten() + 1e-10))) * 100

    percentiles = {
        'p60': np.percentile(np.abs(residuals), 60),
        'p90': np.percentile(np.abs(residuals), 90)
    }

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'Max_Absolute_Error': max_abs_error,
        'Mean_Error': mean_error,
        'Std_Error': std_error,
        'R_Squared': r_squared,
        'MAPE(%)': mape,
        'Percentiles': percentiles
    }

    return metrics


def print_error_report(metrics):
    """打印误差统计报告"""
    print("\n" + "=" * 60)
    print("误差统计报告")
    print("=" * 60)
    print(f"平均绝对误差 (MAE):          {metrics['MAE']:.6e} nT")
    print(f"均方根误差 (RMSE):           {metrics['RMSE']:.6e} nT")
    print(f"最大绝对误差：               {metrics['Max_Absolute_Error']:.6e} nT")
    print(f"平均误差：                   {metrics['Mean_Error']:.6e} nT")
    print(f"误差标准差：                 {metrics['Std_Error']:.6e} nT")
    print(f"决定系数 (R²):              {metrics['R_Squared']:.6f}")
    print(f"平均绝对百分比误差 (MAPE):   {metrics['MAPE(%)']:.4f}%")
    print("-" * 60)
    print("残差绝对值分位数:")
    print(f"  60% 分位数：{metrics['Percentiles']['p60']:.6e} nT")
    print(f"  90% 分位数：{metrics['Percentiles']['p90']:.6e} nT")
    print("=" * 60)


def main():
    csv_path = "../data/interpolated_result.csv"
    model_path = 'scalar_potential_pinn_final.pth'
    norm_params_path = 'normalization_params_scalar.npz'
    I_deg = 43.1
    D_deg = -2.0
    manual_height = -85  # 手动输入的高度值（单位：米），当 CSV 中没有 z 列时使用

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备：{device}")

    print(f"\n正在加载数据：{csv_path}")
    df = pd.read_csv(csv_path)

    # 检查是否有 z 列
    if 'z' in df.columns:
        coords_raw = df[['X', 'Y', 'z']].values.astype(np.float32)
        print(f"检测到 z 列，直接从 CSV 读取坐标")
    else:
        print(f"未检测到 z 列，将使用手动输入的高度：{manual_height} 米")
        coords_XY = df[['X', 'Y']].values.astype(np.float32)
        z_values = np.full((len(coords_XY), 1), manual_height, dtype=np.float32)
        coords_raw = np.hstack([coords_XY, z_values])

    dT_true = df['dT'].values.astype(np.float32)
    print(f"数据加载完成，共 {len(coords_raw)} 个样本")

    print(f"\n正在加载模型和归一化参数...")
    model, norm_params = load_model_and_params(
        model_path=model_path,
        norm_params_path=norm_params_path,
        device=device
    )

    coord_min = norm_params['coord_min']
    coord_max = norm_params['coord_max']

    print(f"\n正在归一化坐标...")
    coords_norm = normalize_coords(coords_raw, coord_min, coord_max)

    I_rad = np.radians(I_deg)
    D_rad = np.radians(D_deg)
    alpha = np.cos(I_rad) * np.cos(D_rad)
    beta = np.cos(I_rad) * np.sin(D_rad)
    gamma = np.sin(I_rad)

    print(f"地磁场方向余弦：alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}")

    inv_scale = 2.0 / (coord_max - coord_min).squeeze()

    print(f"\n正在进行预测...")
    dT_pred = predict_dT(model, coords_norm, inv_scale, alpha, beta, gamma, batch_size=4096)
    print(f"预测完成！")

    print(f"\n正在计算误差指标...")
    metrics = calculate_error_metrics(dT_pred, dT_true)
    print_error_report(metrics)

    print(f"\n正在绘制残差分布图...")
    plot_residual_contour_eval(
        coords_raw=coords_raw,
        dT_pred=dT_pred,
        dT_true=dT_true.reshape(-1, 1),
        save_dir='evaluation_results',
        plot_type='contour'
    )

    predictions_df = pd.DataFrame({
        'X': coords_raw[:, 0],
        'Y': coords_raw[:, 1],
        'Z': coords_raw[:, 2],
        'dT_true': dT_true,
        'dT_pred': dT_pred.flatten(),
        'residual': dT_pred.flatten() - dT_true
    })

    predictions_df.to_csv('evaluation_results/predictions_comparison.csv', index=False)
    print(f"\n预测对比结果已保存至 evaluation_results/predictions_comparison.csv")


if __name__ == '__main__':
    main()
