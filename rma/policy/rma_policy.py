"""RoboMemArena (RMA) policy transforms.

Same shape as ``robomme_policy`` but for the RMA dataset:

  * state  : 8-d = ee_states(6) | gripper_states(2)  (matches the RMA eval
             harness's ``build_eval26_policy_input`` / ``_extract_state``:
             eef_pos | quat2axisangle(eef_quat) | gripper_qpos)
  * actions: native 7-d delta-OSC in [-1,1] -> outputs slice ``[:, :7]``
             (NO DeltaActions/AbsoluteActions round trip)
  * images : the RMA pkls store JPEG-encoded bytes, so ``_parse_image`` grows a
             decode shim.  It is a no-op for raw uint8 arrays, i.e. the existing
             RoboMME pkls flow through unchanged.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_rma_example() -> dict:
    """Random input example (agentview + wrist + 8-d state)."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pour wine into the mug twice.",
    }


def decode_image_bytes(buf) -> np.ndarray:
    """JPEG/PNG bytes -> RGB uint8 (H, W, 3)."""
    import cv2

    arr = np.frombuffer(bytes(buf), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"failed to decode {len(arr)} encoded image bytes")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _parse_image(image) -> np.ndarray:
    """Accept raw arrays (existing RoboMME pkls) or encoded bytes (RMA pkls)."""
    # ---- JPEG decode shim (benign for array-valued images) ----
    if isinstance(image, (bytes, bytearray, memoryview)):
        image = decode_image_bytes(image)
    elif isinstance(image, np.ndarray) and image.dtype == object and image.ndim == 0:
        # torch/np collation can wrap a bytes payload in a 0-d object array
        inner = image.item()
        if isinstance(inner, (bytes, bytearray, memoryview)):
            image = decode_image_bytes(inner)

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        assert np.abs(image).max() <= 1.0, "Image is not normalized"
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RMAInputs(transforms.DataTransformFn):
    """Dataset -> model input format for RoboMemArena (training + inference)."""

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
            },
            # perceptual memory
            "static_image_emb": data.get("static_image_emb", None),   # (budget, d1)
            "static_pos_emb": data.get("static_pos_emb", None),       # (budget, d2)
            "static_state_emb": data.get("static_state_emb", None),   # (budget, d3)
            "static_mask": data.get("static_mask", None),             # (budget)
            # recurrent memory
            "recur_image_emb": data.get("recur_image_emb", None),
            "recur_pos_emb": data.get("recur_pos_emb", None),
            "recur_state_emb": data.get("recur_state_emb", None),
            "recur_mask": data.get("recur_mask", None),
            # symbolic memory
            "simple_subgoal": data.get("simple_subgoal", None),
            "grounded_subgoal": data.get("grounded_subgoal", None),
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RMAOutputs(transforms.DataTransformFn):
    """Model -> RMA action format: 7-d delta OSC (dpos 3 | daxisangle 3 | gripper)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
