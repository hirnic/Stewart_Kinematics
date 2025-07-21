# Thanks to Cha Guh Puh Tuh for teaching me how to train neural networks in Python and for cleaning up this code.

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import RandomizerRedacted as rr


# ==================== CONFIGURATION ====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
PARAMS_DIR = Path("Neural_Net_Parameters")

# ==================== DATA GENERATION ====================

def generate_Euler_poses(size=2**16, angle=0.5, trans=0.5):
    """Generates (input, target) pairs for Euler-angle-based training."""
    Table = np.array([rr.TableID[i].ToPureVec() for i in range(6)])
    Base = np.array([rr.BaseID[i].ToPureVec() for i in range(6)])
    x, y = [], []

    for _ in range(size):
        pose = rr.MakeEulerPose(angle, trans)
        y.append(pose)
        x.append(rr.euler_lengths(pose, Table, Base))

    x = torch.tensor(np.array(x), dtype=torch.float32).to(device)
    y = torch.tensor(np.array(y), dtype=torch.float32).to(device)
    return x, y


def generate_DQ_poses(size=2**16, angle=0.5, trans=0.5):
    """Generates (input, target) pairs for dual-quaternion-based training."""
    x, y = [], []
    for _ in range(size):
        pose = rr.MakeRandomPose(angle, trans)
        x.append(rr.LegLengthsRedacted(pose))
        y.append(pose.ToFullVec())

    x = torch.tensor(np.array(x), dtype=torch.float32).to(device)
    y = torch.tensor(np.array(y), dtype=torch.float32).to(device)
    return x, y


# ==================== MODEL DEFINITION ====================

class StewartNet(nn.Module):
    def __init__(self, hidden_dim, output_dim=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.2, b=0.2)
                nn.init.uniform_(m.bias, a=-0.2, b=0.2)

    def forward(self, x):
        return self.model(x)


# ==================== TRAINING ====================

def train(num_epochs=5000, hidden_dim=19):
    model = StewartNet(hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")

    def adjust_lr(event):
        if event.key == "up":
            optimizer.param_groups[0]['lr'] *= 1.1
        elif event.key == "down":
            optimizer.param_groups[0]['lr'] *= 0.9

    fig.canvas.mpl_connect("key_press_event", adjust_lr)

    for i in range(10):
        x_train, y_train = generate_Euler_poses(angle=0.1*(i+1), trans=0.1*(i+1))
        losses = []
        line, = ax.plot(losses)

        for epoch in range(num_epochs):
            model.train()
            preds = model(x_train)
            loss = criterion(preds, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 25 == 0:
                print(f"[{numerals[i]}] Epoch {epoch+1} | Loss: {loss:.4e} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            if epoch > 0 and epoch % 100 == 0:
                losses.append(loss.item())
                line.set_ydata(losses)
                line.set_xdata(range(len(losses)))
                ax.relim()
                ax.autoscale_view()

                fig.canvas.draw()
                fig.canvas.flush_events()

        save_path = PARAMS_DIR / "Euler_Angles" / f"s_{str(hidden_dim)}_{numerals[i]}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"[{numerals[i]}] Training complete. Model saved to {save_path}")

    plt.ioff()
    plt.show()


# ==================== TESTING ====================

def test_euler_nets(hidden_dim=20):
    model = StewartNet(hidden_dim).to(device)
    model.eval()

    for i in range(10):
        weights_path = PARAMS_DIR / "Euler_Angles" / f"s_20_{numerals[i]}.pth"
        model.load_state_dict(torch.load(weights_path))
        x, y = generate_Euler_poses(size=2**10, angle=0.1*(i+1), trans=0.1*(i+1))
        preds = model(x).detach()
        mse = F.mse_loss(preds, y).item()
        print(f"[Euler {numerals[i]}] MSE: {mse:.6f}")


def test_DQ_nets(hidden_dim=17):
    x_train, y_train = [], []
    model = StewartNet(17, 8).to(device)
    model.eval()
    for i in range(10):
        model.load_state_dict(torch.load("Neural_Net_Parameters\Dual_Quaternions\s_17_" + str(numerals[i]) +".pth"))
        x_train, y_train = generate_DQ_poses(2**10, 0.1*(1+i), 0.1*(1+i))
        predicted_pose = model(x_train)
        error_y_train = y_train - predicted_pose.detach()
        average_mse = torch.mean(error_y_train ** 2).item()
        print(f"Model {numerals[i]} has average MSE: {average_mse:.6f} before normalization")

        #normalize the predicted pose with DQClass normalization
        predicted_np = predicted_pose.detach().cpu().numpy() # I didn't want to use torch in my DQClass script.
        predicted_pose = [rr.DQClass.ToDualQuaternion(pose).normalization() for pose in predicted_np]
        predicted_pose = torch.tensor([pose.ToFullVec() for pose in predicted_pose]).to(device)
        error_y_train = y_train - predicted_pose.detach()
        average_mse = torch.mean(error_y_train ** 2).item()
        print(f"Model {numerals[i]} has average MSE: {average_mse:.6f} after normalization")


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    train()
    # test_euler_nets()
    # test_DQ_nets()
