from random import randint

import torch
import torch.nn.functional as F


def apply_augmentations(
        y: torch.Tensor, cond: torch.Tensor,
        types: list, prob: float = .25):
    """
    :param y: Generated or original image
    :param cond: Conditional image (e.g., landcover)
    :param types: List of augmentations to apply ('color', 'translation', 'cutout', 'noise', 'blit')
    :param prob: Probability that a single augmentation is happening.
    :return:
    """
    batch_size = y.shape[0]
    for p in types:
        # print(p, AUGMENT_FNS[p])
        # print(len(AUGMENT_FNS[p]))
        for f, f3 in AUGMENT_FNS[p]:
        # for f, f2, f3 in AUGMENT_FNS[p]:
            apply_aug = torch.rand(batch_size) < prob
            if apply_aug.any():
                no_aug_cond = ~ apply_aug.view(-1, 1, 1, 1).to(y.device)
                tmp_y, params = f(y[apply_aug])
                yy = torch.empty_like(y)
                yy[apply_aug] = tmp_y
                y = y.where(no_aug_cond, yy)

                if cond != None:
                    if params is None:
                        tmp_cond, _ = f3(cond[apply_aug])
                    else:
                        tmp_cond, _ = f3(cond[apply_aug], params)

                    cc = torch.empty_like(cond)
                    cc[apply_aug] = tmp_cond
                    cond = cond.where(no_aug_cond, cc)
    return y, cond


def hflip(x):
    return torch.flip(x, dims=(3,)), None


def vflip(x):
    return torch.flip(x, dims=(2,)), None


def rand_rot90(x):
    k = randint(1, 3)
    return rot90_k(x, k)


def rot90_k(x, k):
    return x.rot90(k, (2, 3)), k


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x, None


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x, None


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x, None


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    x, _ = rand_translation_params(x, [x.shape, translation_x, translation_y])

    return x, [x.shape, translation_x, translation_y]


def rand_translation_params(x, params):
    shape, translation_x, translation_y = params[:3]
    if len(translation_x.shape) != 3:
        translation_x.squeeze_(1)
        translation_y.squeeze_(1)
    ratio = x.shape[3] / shape[3]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + (translation_x * ratio).long() + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + (translation_y * ratio).long() + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1], "reflect")
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)

    return x, params


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    x, _ = rand_cutout_params(x, [x.shape, offset_x, offset_y, cutout_size])

    return x, [x.shape, offset_x, offset_y, cutout_size]


def rand_cutout_params(x, params):
    shape, offset_x, offset_y, cutout_size = params
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    ratio = x.shape[3] / shape[3]
    grid_x = torch.clamp(
        grid_x + (offset_x * ratio).long() - int(cutout_size[0] * ratio) // 2, min=0, max=x.size(2) - 1
    )
    grid_y = torch.clamp(
        grid_y + (offset_y * ratio).long() - int(cutout_size[1] * ratio) // 2, min=0, max=x.size(3) - 1
    )
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x, params


def rand_nnoise(x, sigma=.03):
    # augment the segmentation mask with normal noise
    noise = torch.randn_like(x).abs() * sigma * -(x - .5).sign()
    return x + noise, None


def rand_unoise(x, sigma=.1):
    # augment the segmentation mask with uniform noise
    noise = torch.rand_like(x) * sigma * -(x - .5).sign()
    return x + noise, None


def noop(x): return x, None


def noop_params(x, params): return x, None


AUGMENT_FNS = {
    'color': [  # no color change on segmentation map
        (rand_brightness, noop),
        (rand_saturation, noop),
        (rand_contrast, noop)
    ],
    'translation': [
        (rand_translation, rand_translation_params)
    ],
    'cutout': [
        (rand_cutout, rand_cutout_params)
    ],
    'noise': [
        (rand_nnoise, noop),
        (rand_unoise, noop),
        # (rand_nnoise, noop),
        # (rand_unoise, noop)
    ],
    'blit': [
        (hflip, hflip),
        (vflip, vflip),
        (rand_rot90, rot90_k)
    ]
}