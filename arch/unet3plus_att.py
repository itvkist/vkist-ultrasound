import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================== Self-Attention Block ======================
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, max(1, in_dim // 8), kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, max(1, in_dim // 8), kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B, N, C'
        key = self.key_conv(x).view(B, -1, H * W)                       # B, C', N
        energy = torch.bmm(query, key)                                  # B, N, N
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(B, -1, H * W)                   # B, C, N
        out = torch.bmm(value, attention.permute(0, 2, 1))              # B, C, N
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out


# ====================== Attention Gate ======================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (deeper) ; x: skip connection (shallower)
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # ensure same spatial size
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # ensure psi matches x spatially
        if psi.size()[2:] != x.size()[2:]:
            psi = F.interpolate(psi, size=x.size()[2:], mode='bilinear', align_corners=True)

        return x * psi


# ====================== Basic Conv Block ======================
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# ====================== UNet3+ with Attention (fixed channels) ======================
class UNet3Plus_Attention(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, filters=[64, 128, 256, 512, 1024]):
        super(UNet3Plus_Attention, self).__init__()
        self.num_classes = num_classes
        F1, F2, F3, F4, F5 = filters  # e.g. 64,128,256,512,1024

        # ---------------- Encoder ----------------
        self.encoder1 = conv_block(in_channels, F1)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = conv_block(F1, F2)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = conv_block(F2, F3)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = conv_block(F3, F4)
        self.pool4 = nn.MaxPool2d(2)
        self.encoder5 = conv_block(F4, F5)

        # ---------------- Self-Attention at bottleneck ----------------
        self.self_att = SelfAttention(F5)

        # ---------------- 1x1 conv for multi-scale fusion ----------------
        # These compress encoder features to F1 channels for easier fusion where used.
        self.conv_1x1_1 = nn.Conv2d(F1, F1, kernel_size=1)
        self.conv_1x1_2 = nn.Conv2d(F2, F1, kernel_size=1)
        self.conv_1x1_3 = nn.Conv2d(F3, F1, kernel_size=1)
        self.conv_1x1_4 = nn.Conv2d(F4, F1, kernel_size=1)
        self.conv_1x1_5 = nn.Conv2d(F5, F1, kernel_size=1)

        # ---------------- Decoder Blocks ----------------
        # IMPORTANT: set in_channels = actual concatenation channels at each level
        # Decoder4 concatenates upsampled conv_1x1 outputs (all compressed to F1) => 5*F1
        self.decoder4_cat_conv = conv_block(F1 * 5, F4)   # output channels F4

        # Decoder3 concatenation channels:
        # d1_up (conv_1x1_1) : F1
        # d2_up (conv_1x1_2) : F1
        # d3_up (conv_1x1_3) : F1
        # d4_up (decoder4 output) : F4
        # d5_up (conv_1x1_5) : F1
        dec3_in = F1 + F1 + F1 + F4 + F1  # = 4*F1 + F4
        self.decoder3_cat_conv = conv_block(dec3_in, F3)

        # Decoder2 concatenation channels:
        # d1_up: F1
        # d2_up: F1
        # d3_up: decoder3 output F3
        # d4_up: decoder4 output F4
        # d5_up: F1
        dec2_in = F1 + F1 + F3 + F4 + F1
        self.decoder2_cat_conv = conv_block(dec2_in, F2)

        # Decoder1 concatenation channels:
        # d1_up: F1
        # d2_up: decoder2 output F2
        # d3_up: decoder3 output F3
        # d4_up: decoder4 output F4
        # d5_up: F1
        dec1_in = F1 + F2 + F3 + F4 + F1
        self.decoder1_cat_conv = conv_block(dec1_in, F1)

        # ---------------- Attention Gates ----------------
        self.att4 = AttentionGate(F_g=F5, F_l=F4, F_int=max(1, F4 // 2))
        self.att3 = AttentionGate(F_g=F4, F_l=F3, F_int=max(1, F3 // 2))
        self.att2 = AttentionGate(F_g=F3, F_l=F2, F_int=max(1, F2 // 2))
        self.att1 = AttentionGate(F_g=F2, F_l=F1, F_int=max(1, F1 // 2))

        # ---------------- Final Fusion ----------------
        self.final_reduce = nn.Sequential(
            nn.Conv2d(F1 + F2 + F3 + F4 + F1, F1, kernel_size=1),
            nn.BatchNorm2d(F1),
            nn.ReLU(inplace=True)
        )
        # Note: final_reduce input channels use same sum as dec1 input (we fuse d1 + up(d2) + up(d3) + up(d4) + up(x5)
        # but to keep consistent we will recompute in forward and pass through this module (in_ch matches dec1_in)
        # For simplicity, we use a 1x1 to F1.

        self.final_conv = nn.Conv2d(F1, self.num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ---------------- Forward ----------------
    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)                         # [B, F1, H, W]
        x2 = self.encoder2(self.pool1(x1))            # [B, F2, H/2, W/2]
        x3 = self.encoder3(self.pool2(x2))            # [B, F3, H/4, W/4]
        x4 = self.encoder4(self.pool3(x3))            # [B, F4, H/8, W/8]
        x5 = self.encoder5(self.pool4(x4))            # [B, F5, H/16, W/16]

        # Bottleneck self-attention
        x5 = self.self_att(x5)

        # Attention gates on skip connections (ensure spatial alignment inside gate)
        x4 = self.att4(x5, x4)
        x3 = self.att3(x4, x3)
        x2 = self.att2(x3, x2)
        x1 = self.att1(x2, x1)

        # ---- Prepare compressed feature maps (1x1) ----
        # These are used for multi-scale fusion where appropriate
        e1c = self.conv_1x1_1(x1)                     # [B, F1, H, W]
        e2c = self.conv_1x1_2(x2)                     # [B, F1, H/2, W/2]
        e3c = self.conv_1x1_3(x3)                     # [B, F1, H/4, W/4]
        e4c = self.conv_1x1_4(x4)                     # [B, F1, H/8, W/8]
        e5c = self.conv_1x1_5(x5)                     # [B, F1, H/16, W/16]

        # ---------------- Decoder 4 (produce d4 with out channels F4) ----------------
        target4 = x4.size()[2:]  # spatial of level 4
        d1_u4 = F.interpolate(e1c, size=target4, mode='bilinear', align_corners=True)  # F1
        d2_u4 = F.interpolate(e2c, size=target4, mode='bilinear', align_corners=True)  # F1
        d3_u4 = F.interpolate(e3c, size=target4, mode='bilinear', align_corners=True)  # F1
        d4_c = e4c  # F1
        d5_u4 = F.interpolate(e5c, size=target4, mode='bilinear', align_corners=True)  # F1

        d4_in = torch.cat([d1_u4, d2_u4, d3_u4, d4_c, d5_u4], dim=1)  # 5*F1
        d4 = self.decoder4_cat_conv(d4_in)  # out channels = F4

        # ---------------- Decoder 3 (out channels F3) ----------------
        target3 = x3.size()[2:]
        d1_u3 = F.interpolate(e1c, size=target3, mode='bilinear', align_corners=True)  # F1
        d2_u3 = F.interpolate(e2c, size=target3, mode='bilinear', align_corners=True)  # F1
        d3_c  = e3c  # F1
        d4_u3 = F.interpolate(d4, size=target3, mode='bilinear', align_corners=True)   # F4
        d5_u3 = F.interpolate(e5c, size=target3, mode='bilinear', align_corners=True)  # F1

        d3_in = torch.cat([d1_u3, d2_u3, d3_c, d4_u3, d5_u3], dim=1)  # channels = 4*F1 + F4
        d3 = self.decoder3_cat_conv(d3_in)  # out channels = F3

        # ---------------- Decoder 2 (out channels F2) ----------------
        target2 = x2.size()[2:]
        d1_u2 = F.interpolate(e1c, size=target2, mode='bilinear', align_corners=True)  # F1
        d2_c  = e2c  # F1
        d3_u2 = F.interpolate(d3,   size=target2, mode='bilinear', align_corners=True)  # F3
        d4_u2 = F.interpolate(d4,   size=target2, mode='bilinear', align_corners=True)  # F4
        d5_u2 = F.interpolate(e5c,  size=target2, mode='bilinear', align_corners=True)  # F1

        d2_in = torch.cat([d1_u2, d2_c, d3_u2, d4_u2, d5_u2], dim=1)  # channels = F1 + F1 + F3 + F4 + F1
        d2 = self.decoder2_cat_conv(d2_in)  # out channels = F2

        # ---------------- Decoder 1 (out channels F1) ----------------
        target1 = x1.size()[2:]
        d1_c  = e1c  # F1
        d2_u1 = F.interpolate(d2, size=target1, mode='bilinear', align_corners=True)  # F2
        d3_u1 = F.interpolate(d3, size=target1, mode='bilinear', align_corners=True)  # F3
        d4_u1 = F.interpolate(d4, size=target1, mode='bilinear', align_corners=True)  # F4
        d5_u1 = F.interpolate(e5c, size=target1, mode='bilinear', align_corners=True)  # F1

        d1_in = torch.cat([d1_c, d2_u1, d3_u1, d4_u1, d5_u1], dim=1)  # channels = F1 + F2 + F3 + F4 + F1
        d1 = self.decoder1_cat_conv(d1_in)  # out channels = F1

        # ---------------- Final feature fusion ----------------
        # Fuse d1 and upsampled coarser decoders + upsampled bottleneck
        # We will form a concat consistent with final_reduce 1x1 (which maps dec1_in -> F1)
        f_d2 = F.interpolate(d2, size=d1.size()[2:], mode='bilinear', align_corners=True)
        f_d3 = F.interpolate(d3, size=d1.size()[2:], mode='bilinear', align_corners=True)
        f_d4 = F.interpolate(d4, size=d1.size()[2:], mode='bilinear', align_corners=True)
        f_x5 = F.interpolate(e5c, size=d1.size()[2:], mode='bilinear', align_corners=True)

        final_concat = torch.cat([d1, f_d2, f_d3, f_d4, f_x5], dim=1)  # channels = F1 + F2 + F3 + F4 + F1
        # Reduce to F1
        final_feat = self.final_reduce(final_concat)
        out = self.final_conv(final_feat)  # [B, num_classes, H, W]

        return out


# quick sanity test (only run if file executed directly)
if __name__ == "__main__":
    model = UNet3Plus_Attention(in_channels=3, num_classes=7).cuda()
    x = torch.randn(2, 3, 512, 512).cuda()
    y = model(x)
    print("Output:", y.shape)  # expect [2, 7, 512, 512]
