from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


def _conv(in_channels: int, out_channels: int) -> nn.Sequential:
    groups = min(32, out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(),
    )


class FiLM(nn.Module):
    def __init__(self, latent_dim: int, channels: int) -> None:
        super().__init__()
        self.projection = nn.Linear(latent_dim, channels * 2)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, feature: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        scale, shift = self.projection(latent).chunk(2, dim=-1)
        return feature * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class InverseDynamicsModel(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim * 2),
        )

    def forward(self, current: torch.Tensor, future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current_feature = self.encoder(current).flatten(1)
        future_feature = self.encoder(future).flatten(1)
        stats = self.head(
            torch.cat((current_feature, future_feature, future_feature - current_feature), dim=-1)
        )
        return stats.chunk(2, dim=-1)


class ForwardDynamicsModel(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.input = _conv(3, 64)
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), _conv(128, 128))
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), _conv(256, 256))
        self.down3 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), _conv(512, 512))
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), _conv(512, 512))
        self.up3 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        self.dec3 = _conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.dec2 = _conv(512, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec1 = _conv(256, 128)
        self.up0 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec0 = _conv(128, 64)
        self.film3 = FiLM(latent_dim, 512)
        self.film2 = FiLM(latent_dim, 256)
        self.film1 = FiLM(latent_dim, 128)
        self.film0 = FiLM(latent_dim, 64)
        self.output = nn.Sequential(nn.Conv2d(64, 3, 1), nn.Sigmoid())

    def forward(self, current: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        x0 = self.input(current)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        bottleneck = self.bottleneck(x3)
        d3 = self.film3(self.dec3(torch.cat((self.up3(bottleneck), x3), dim=1)), latent)
        d2 = self.film2(self.dec2(torch.cat((self.up2(d3), x2), dim=1)), latent)
        d1 = self.film1(self.dec1(torch.cat((self.up1(d2), x1), dim=1)), latent)
        d0 = self.film0(self.dec0(torch.cat((self.up0(d1), x0), dim=1)), latent)
        return self.output(d0)


class LAPOTeacher(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.idm = InverseDynamicsModel(latent_dim)
        self.fdm = ForwardDynamicsModel(latent_dim)
        self.latent_dim = latent_dim

    def encode(self, current: torch.Tensor, future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.idm(current, future)

    def forward(
        self, current: torch.Tensor, future: torch.Tensor, *, deterministic: bool = False
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(current, future)
        logvar = logvar.clamp(-10, 10)
        if deterministic:
            latent = mu
        else:
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        reconstruction = self.fdm(current, latent)
        return {"reconstruction": reconstruction, "mu": mu, "logvar": logvar, "latent": latent}


def teacher_loss(
    output: dict[str, torch.Tensor], target: torch.Tensor, beta: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstruction = torch.nn.functional.mse_loss(output["reconstruction"], target)
    kl = -0.5 * (1 + output["logvar"] - output["mu"].square() - output["logvar"].exp()).mean()
    loss = reconstruction + beta * kl
    return loss, {"loss": loss.detach(), "reconstruction": reconstruction.detach(), "kl": kl.detach()}

