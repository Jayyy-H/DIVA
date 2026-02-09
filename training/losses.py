import torch
import torch.nn.functional as F

def info_nce(z1, z2, T=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / T
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def orth_loss(zs, zu):
    return (zs.T @ zu).pow(2).mean()


class Stage1Loss:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, logits_gen, logits_und,
                 labels_gen, labels_und,
                 z_sh_und, z_sh_gen):

        loss_gen = F.cross_entropy(
            logits_gen.view(-1, logits_gen.size(-1)),
            labels_gen.view(-1),
            ignore_index=-100
        )

        loss_und = F.cross_entropy(
            logits_und.view(-1, logits_und.size(-1)),
            labels_und.view(-1),
            ignore_index=-100
        )

        return loss_gen + loss_und


class Stage2Loss:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(
        self,
        logits_gen, logits_und,
        labels_gen, labels_und,
        z_sh_und, z_sh_gen,
        z_un_und, z_un_gen,
        club_und, club_gen
    ):

        loss_gen = F.cross_entropy(
            logits_gen.view(-1, logits_gen.size(-1)),
            labels_gen.view(-1),
            ignore_index=-100
        )

        loss_und = F.cross_entropy(
            logits_und.view(-1, logits_und.size(-1)),
            labels_und.view(-1),
            ignore_index=-100
        )

        loss_align = (
            info_nce(z_sh_und, z_sh_gen.detach()) +
            info_nce(z_sh_gen, z_sh_und.detach())
        ) * 0.5

        loss_uni = (
            club_und(z_un_und, z_un_gen.detach()) +
            club_gen(z_un_gen, z_un_und.detach())
        )

        loss_orth = (
            orth_loss(z_sh_und, z_un_und) +
            orth_loss(z_sh_gen, z_un_gen)
        )

        loss_total = (
            loss_gen + loss_und
            + self.cfg.lambda_sha * loss_align
            + self.cfg.lambda_uni * loss_uni
            + self.cfg.lambda_orth * loss_orth
        )

        return loss_total

