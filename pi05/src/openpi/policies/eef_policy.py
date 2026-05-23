import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class EEFInputs(transforms.DataTransformFn):
    """Inputs for dual-arm end-effector-pose control data."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = _parse_image(in_images["cam_high"])
        left_wrist = _parse_image(in_images["cam_left_wrist"]) if "cam_left_wrist" in in_images else np.zeros_like(base_image)
        right_wrist = (
            _parse_image(in_images["cam_right_wrist"]) if "cam_right_wrist" in in_images else np.zeros_like(base_image)
        )

        inputs = {
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if "cam_left_wrist" in in_images else np.False_,
                "right_wrist_0_rgb": np.True_ if "cam_right_wrist" in in_images else np.False_,
            },
            "state": np.asarray(data["state"], dtype=np.float32),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class EEFOutputs(transforms.DataTransformFn):
    action_dim: int = 16

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
