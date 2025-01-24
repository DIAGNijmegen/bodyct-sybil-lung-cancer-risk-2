import numpy as np
import torch
import torch.nn.functional as F
from sybil.serie import Serie
from typing import Dict, List, Union
import os
import imageio


def visualize_attentions(
    series: Union[Serie, List[Serie]],
    attentions: List[Dict[str, np.ndarray]],
    save_directory: str = None,
    gain: int = 3,
    series_uids: Union[str, List[str]] = None,
) -> List[List[np.ndarray]]:
    """
    Args:
        series (Serie): series object
        attentions (Dict[str, np.ndarray]): attention dictionary output from model
        save_directory (str, optional): where to save the images. Defaults to None.
        gain (int, optional): how much to scale attention values by for visualization. Defaults to 3.

    Returns:
        List[List[np.ndarray]]: list of list of overlayed images
    """

    if isinstance(series, Serie):
        series = [series]

    if isinstance(series_uids, str):
        series_uids = [series_uids]

    series_overlays = []
    for serie_idx, serie in enumerate(series):
        a1 = attentions[serie_idx]["image_attention_1"]
        v1 = attentions[serie_idx]["volume_attention_1"]

        a1 = torch.Tensor(a1)
        v1 = torch.Tensor(v1)

        # take mean attention over ensemble
        a1 = torch.exp(a1).mean(0)
        v1 = torch.exp(v1).mean(0)

        attention = a1 * v1.unsqueeze(-1)
        attention = attention.view(1, 25, 16, 16)

        # get original image
        images = serie.get_raw_images()

        N = len(images)
        attention_up = F.interpolate(
            attention.unsqueeze(0), (N, 512, 512), mode="trilinear"
        )

        overlayed_images = []
        for i in range(N):
            overlayed = np.zeros((512, 512, 3))
            # overlayed[..., 2] = images[i]
            # overlayed[..., 1] = images[i]
            overlayed[..., 0] = np.clip(
                (attention_up[0, 0, i] * gain * 256),
                a_min=0,
                a_max=256,
            )

            overlayed_images.append(np.uint8(overlayed))

        if save_directory is not None:
            if series_uids is not None:
                # save_path = os.path.join(
                #     save_directory, f"serie_{series_uids[serie_idx]}"
                # )
                save_images(
                    overlayed_images, save_directory, f"serie_{series_uids[serie_idx]}"
                )
            else:
                # save_path = os.path.join(save_directory, f"serie_{serie_idx}")
                save_images(overlayed_images, save_directory, f"serie_{serie_idx}")

        series_overlays.append(overlayed_images)
    return series_overlays


def save_images(img_list: List[np.ndarray], directory: str, name: str):
    """
    Saves a list of images as a GIF in the specified directory with the given name.

    Args:
        ``img_list`` (List[np.ndarray]): A list of numpy arrays representing the images to be saved.
        ``directory`` (str): The directory where the GIF should be saved.
        ``name`` (str): The name of the GIF file.

    Returns:
        None
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.gif")
    imageio.mimsave(path, img_list)
