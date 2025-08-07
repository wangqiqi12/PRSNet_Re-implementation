import torch
import torch.nn as nn

# NOTE： 目前修改为只预测一个对称平面的版本

class PRSNet(nn.Module):
    """
    PRSNet
    Input : [B, 1, 32, 32, 32]
    Output: 24 floats  -> 3 planes(3x4) + 3 quaternions(3x4)
    """
    def __init__(self):
        super().__init__()

        # 5-layer 3D CNN encoder
        self.encoder = nn.Sequential(
            # Layer 1: 1 -> 4 channels
            nn.Conv3d(1, 4, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True), # 32³ *4
            nn.MaxPool3d(2),                                           # 16³ *4

            # Layer 2: 4 -> 8 channels
            nn.Conv3d(4, 8, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True), # 16³ -> 8³
            nn.MaxPool3d(2),                                           # 8³ *8

            # Layer 3: 8 -> 16 channels
            nn.Conv3d(8, 16, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),# 8³ -> 4³
            nn.MaxPool3d(2),                                           # 4³ *16

            # Layer 4: 16 -> 32 channels
            nn.Conv3d(16, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),# 4³ -> 2³
            nn.MaxPool3d(2),                                            # 2³ *32

            # Layer 5: 32 -> 64 channels
            nn.Conv3d(32, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),# 2³ -> 1³
            nn.MaxPool3d(2),                                            # 1³ *64
        )

        # Fully connected layers 1 (64 -> 32 -> 16 -> 24)
        self.fc1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 4)
        )
        # Fully connected layers 2 (64 -> 32 -> 16 -> 24)
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 4)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 4)
        )
        
    def forward(self, x):
        """
        x: [B, 1, 32, 32, 32]
        returns:
          planes : [B, 1, 4]
        """
        B = x.size(0)
        feat = self.encoder(x).view(B, -1)      # [B, 64]
        out1  = self.fc1(feat)                    # [B, 4]
        out2 = self.fc2(feat)
        out3 = self.fc3(feat)                      # [B, 4]
        planes1 = out1.view(B, 1, 4)                # 1 plane
        planes2 = out2.view(B, 1, 4)                # 1 plane
        planes3 = out3.view(B, 1, 4)                # 1 plane

        # L2-normalize
        planes_norm1 = planes1.clone()
        planes_norm1[:, :, :3] = nn.functional.normalize(planes1[:, :, :3], dim=-1)

        planes_norm2 = planes2.clone()
        planes_norm2[:, :, :3] = nn.functional.normalize(planes2[:, :, :3], dim=-1)

        planes_norm3 = planes3.clone()
        planes_norm3[:, :, :3] = nn.functional.normalize(planes3[:, :, :3], dim=-1)

        return {"p1": planes_norm1, "p2": planes_norm2, "p3": planes_norm3}


# ---------------- simple test ----------------
if __name__ == "__main__":
    net = PRSNet().eval()
    dummy = torch.randn(2, 1, 32, 32, 32)   # 2 samples
    with torch.no_grad():
        out = net(dummy)
    print("planes shape:", out["p1"].shape)   # [2,1,4]
