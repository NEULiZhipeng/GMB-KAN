from typing import Dict
import numpy as np
import torch as th
import torchinfo
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from EasyTSAD.DataFactory import TSData
from EasyTSAD.Exptools import EarlyStoppingTorch
from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory.TorchDataSet import PredictWindow
from EasyTSAD.DataFactory.TorchDataSet.PredictWindow import UTSOneByOneDataset


class KANADModel(nn.Module):
    """
    KANADModel: A Physics-Informed Kolmogorov-Arnold Network for Anomaly Detection.

    This model utilizes a Mixture-of-Experts (MoE) approach to disentangle time series
    signals into three fundamental components:
    1. Local Transients (via Discrete Wavelet Transform)
    2. Global Trends (via Chebyshev Polynomials)
    3. Periodicity (via Fourier Sine/Cosine Bases)
    """

    def __init__(self, window: int, order: int, *args, **kwargs) -> None:
        super().__init__()
        self.window = window
        self.order = order  # Order defines the decomposition depth for wavelets and polynomials

        # Feature Channel Calculation:
        # - Wavelet (DWT): 'order' levels produce 'order' detail coeffs + 1 approx coeff = order + 1
        # - Chebyshev: 'order' degrees = order
        # - Fourier: 'order' frequencies for both sin and cos = 2 * order
        # - Raw Input: 1
        # Total = (order + 1) + (order) + (2 * order) + 1 = 4 * order + 2
        self.channels = 4 * self.order + 2

        # Pre-compute Chebyshev basis functions for numerical stability during training
        self.register_buffer(
            "chebyshev_basis",
            self._create_chebyshev_basis().unsqueeze(0),  # Shape: (1, order, window)
        )

        # Physics-Informed Gating Mechanism (MoE Selector)
        # This network analyzes the input signal's morphology to dynamically assign
        # weights to the three physical experts (Wavelet, Chebyshev, Fourier).
        self.gating_network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 3),
            nn.Sigmoid()  # Sigmoid allows multiple experts to be highly active simultaneously
        )

        # Reconstruction Network Architecture
        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)

        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        # Final projection to map features back to the reconstruction space
        self.final_conv = nn.Conv1d(1, 1, window, padding=0, stride=1, dilation=1)

    def _dwt_features(self, x: th.Tensor, level: int) -> th.Tensor:
        """
        Performs multi-level Discrete Wavelet Transform (DWT) using Haar wavelets.
        Used to extract local transient features and high-frequency anomalies.
        """
        coeffs = []
        current_cA = x  # Input shape: (B, 1, L)

        for _ in range(level):
            L_in = current_cA.shape[-1]
            if L_in % 2 != 0:
                current_cA = F.pad(current_cA, (0, 1), 'reflect')

            # Haar Filters: Approximation (cA) and Detail (cD) coefficients
            cA = (current_cA[..., ::2] + current_cA[..., 1::2]) * 0.70710678118
            cD = (current_cA[..., ::2] - current_cA[..., 1::2]) * 0.70710678118

            # Upsample coefficients to maintain the original window resolution for concatenation
            cD_upsampled = F.interpolate(cD, size=self.window, mode='linear', align_corners=False)
            coeffs.append(cD_upsampled)
            current_cA = cA

        cA_upsampled = F.interpolate(current_cA, size=self.window, mode='linear', align_corners=False)
        coeffs.append(cA_upsampled)

        return th.cat(coeffs, dim=1)  # (B, level + 1, L)

    def forward(self, x: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        Main Forward Pass:
        1. Extract domain-specific features (Wavelet, Chebyshev, Fourier).
        2. Apply dynamic gating to select relevant experts.
        3. Reconstruct the signal through a residual convolutional block.
        """
        x_unsqueezed = x.unsqueeze(1)  # (B, 1, L)
        B = x.size(0)

        # Calculate expert weights via the gating network
        gates = self.gating_network(x_unsqueezed)
        w_wavelet = gates[:, 0].view(B, 1, 1)
        w_cheby = gates[:, 1].view(B, 1, 1)
        w_fourier = gates[:, 2].view(B, 1, 1)

        # 1. Local Transient Expert (Wavelets)
        wavelet_features = self._dwt_features(x_unsqueezed, self.order)

        # 2. Global Trend Expert (Polynomials)
        chebyshev_features = self.chebyshev_basis.repeat(B, 1, 1)

        # 3. Periodic Seasonality Expert (Fourier)
        fourier_features = th.cat([
                                      th.cos(k * x_unsqueezed) for k in range(1, self.order + 1)
                                  ] + [
                                      th.sin(k * x_unsqueezed) for k in range(1, self.order + 1)
                                  ], dim=1)

        raw_features = x_unsqueezed

        # Apply gating to enforce physics-informed inductive bias
        wavelet_gated = wavelet_features * w_wavelet
        chebyshev_gated = chebyshev_features * w_cheby
        fourier_gated = fourier_features * w_fourier

        # Concatenate gated features for the reconstruction network
        combined_features = th.concat(
            [wavelet_gated, chebyshev_gated, fourier_gated, raw_features],
            dim=1,
        )

        # Residual reconstruction blocks
        res = [x_unsqueezed, combined_features]
        ff = self.init_conv(combined_features)
        ff = self.bn1(ff)
        ff = self.act(ff)
        ff = self.inner_conv(ff) + res.pop()
        ff = self.bn2(ff)
        ff = self.act(ff)
        ff = self.out_conv(ff) + res.pop()
        ff = self.bn3(ff)
        ff = self.act(ff)
        ff = self.final_conv(ff)
        return ff.squeeze(1)

    def _create_chebyshev_basis(self) -> th.Tensor:
        """
        Generates Chebyshev Polynomials of the first kind using a recursive formula.
        Used as a basis for modeling smooth, global trends in time series data.
        """
        range_value = th.arange(self.window, dtype=th.float32)
        # Normalize time range to [-1, 1] for Chebyshev stability
        normalized_range = (2.0 * range_value / (self.window - 1)) - 1.0

        T_n_minus_2 = th.ones(self.window, dtype=th.float32)  # T_0
        T_n_minus_1 = normalized_range.clone()  # T_1

        result_list = []
        if self.order == 0:
            return th.empty(0, self.window, dtype=th.float32)

        for i in range(1, self.order + 1):
            if i == 1:
                result_list.append(T_n_minus_1)
            else:
                # Recursive formula: T_n = 2x * T_{n-1} - T_{n-2}
                T_n = 2.0 * normalized_range * T_n_minus_1 - T_n_minus_2
                result_list.append(T_n)
                T_n_minus_2 = T_n_minus_1
                T_n_minus_1 = T_n

        return th.stack(result_list, dim=0)


class GMB_KAN(BaseMethod):
    """
    Experimental Wrapper for GMB_KAN.
    Handles the training loop, validation phases, and anomaly score generation.
    """

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.__anomaly_score = None

        # Device configuration
        self.device = th.device("cuda" if th.cuda.is_available() and params.get("cuda", True) else "cpu")
        print(f"Device initialized: {self.device}")

        # Hyperparameters
        self.batch_size = params["batch_size"]
        self.window = params["window"]
        self.debug = params.get("debug", False)
        self.epochs = params["epochs"]

        # Model and Optimization setup
        self.model = KANADModel(**params).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["lr"])

        # Learning rate scheduler to handle plateauing loss
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        self.loss = nn.MSELoss()
        self.early_stopping = EarlyStoppingTorch(save_path=None, patience=5)

    def train_valid_phase(self, tsTrain: TSData):
        """Executes the training and validation loops for a specific dataset."""
        train_loader = DataLoader(
            dataset=UTSOneByOneDataset(tsTrain, "train", window_size=self.window),
            batch_size=self.batch_size, shuffle=True,
        )
        valid_loader = DataLoader(
            dataset=UTSOneByOneDataset(tsTrain, "valid", window_size=self.window),
            batch_size=self.batch_size, shuffle=False,
        )

        for epoch in range(1, self.epochs + 1):
            # Training Phase
            self.model.train()
            avg_train_loss = 0
            for x, target in tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch}"):
                x, target = x.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                avg_train_loss += loss.item()

            # Validation Phase
            self.model.eval()
            avg_valid_loss = 0
            with th.no_grad():
                for x, target in valid_loader:
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    avg_valid_loss += self.loss(output, target).item()

            valid_loss = avg_valid_loss / len(valid_loader)
            self.scheduler.step(valid_loss)
            self.early_stopping(valid_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    def test_phase(self, tsData: TSData):
        """Calculates reconstruction-based anomaly scores for the test set."""
        test_loader = DataLoader(
            dataset=UTSOneByOneDataset(tsData, "test", window_size=self.window),
            batch_size=self.batch_size, shuffle=False,
        )

        self.model.eval()
        scores = []
        with th.no_grad():
            for x, target in tqdm.tqdm(test_loader, desc="Testing"):
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                # Anomaly score is defined as the absolute reconstruction error
                mse = th.abs(output - target)
                scores.append(mse.cpu())

        # Collect scores and finalize the anomaly score array
        final_scores = th.cat(scores, dim=0)[..., -1].numpy().flatten()
        final_scores[np.isnan(final_scores)] = 1e3  # Handle edge-case NaNs
        self.__anomaly_score = final_scores

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score  # type: ignore

    def param_statistic(self, save_file: str):
        """Generates a summary of the model parameters and architecture."""
        try:
            dummy_input = th.randn(self.batch_size, 1, self.window).to(self.device)
            stats = torchinfo.summary(self.model, input_data=dummy_input, verbose=0)
            with open(save_file, "w") as f:
                f.write(str(stats))
        except Exception as e:
            print(f"Stats generation failed: {e}")