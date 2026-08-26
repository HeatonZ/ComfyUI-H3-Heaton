"""HT_H3_BUNDLE: the canonical internal format every Heaton node speaks.

Business nodes never touch third-party bundle types; bridging to other packs
lives in compat/ adapters.
"""

from dataclasses import dataclass, field

BUNDLE_TYPE = "HT_H3_BUNDLE"


@dataclass
class HTBundle:
    """Unified H3 load result.

    unet_model is the first-pass MODEL; second_model defaults to None (the
    same model serves both passes) and may hold a distinct MODEL when a
    dedicated second-sampling transformer was loaded. base_info carries
    user-facing facts (fps, frame length policy) that downstream nodes read.
    """

    unet_model: object
    clip: object
    video_vae: object
    audio_vae: object
    fps: float = 24.0
    loaded_report: str = ""
    lora_stack: tuple = ()
    second_model: object = None
    base_info: dict = field(default_factory=dict)

    def report(self) -> str:
        return self.loaded_report
