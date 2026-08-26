"""First-frame anchored i2v conditioning (MiniMax H3 keyframes) and an input-image loader."""
import logging
try:
    from comfy_extras import nodes_minimax_h3 as h3
except ImportError:
    h3 = None
try:
    import folder_paths
    import node_helpers
    import nodes
except ImportError:
    folder_paths = node_helpers = nodes = None

log = logging.getLogger("HT.i2v")


def _need_comfy():
    if h3 is None or node_helpers is None or nodes is None or folder_paths is None:
        raise RuntimeError("HT I2V nodes require a running ComfyUI installation")


class HTI2VConditioning:
    """Anchor generation to a first (and optionally last) frame via H3 keyframes."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "bundle": ("HT_H3_BUNDLE",),
                    "prompt": ("STRING", {"multiline": True}),
                    "width": ("INT", {"default": 960, "step": 32}),
                    "height": ("INT", {"default": 544, "step": 32}),
                    "seconds": ("FLOAT", {"default": 7.0}),
                    "first_frame": ("IMAGE",),
                },
                "optional": {"last_frame": ("IMAGE",)}}
    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT"); RETURN_NAMES = ("conditioning", "latent", "frame_count")
    FUNCTION = "condition"; CATEGORY = "H3-Heaton/i2v"

    def condition(self, bundle, prompt, width, height, seconds, first_frame, last_frame=None):
        _need_comfy()
        target = max(5, seconds * 24)
        length = int(round((target - 5) / 17)) * 17 + 5
        latent, frame_count = h3._empty_av_latent(width, height, length)
        images = []; keyframes = []
        if first_frame is not None:
            image = h3._resize(first_frame[:1], width, height, "disabled")
            images.append(image)
            keyframes.append({"resolved_frame_index": 0, "image": image})
        if last_frame is not None:
            image = h3._resize(last_frame[:1], width, height, "center")
            images.append(image)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": image})
        tokens = bundle.clip.tokenize(prompt, images=images)
        conditioning = bundle.clip.encode_from_tokens_scheduled(tokens)
        if keyframes:
            for keyframe in keyframes:
                keyframe["latent"] = bundle.video_vae.encode(keyframe.pop("image"))
            conditioning = node_helpers.conditioning_set_values(conditioning, {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": frame_count,
            })
        return (conditioning, latent, frame_count)


class HTLoadImage:
    """Thin wrapper around ComfyUI's LoadImage returning only the IMAGE output."""
    @classmethod
    def INPUT_TYPES(cls):
        files = sorted(folder_paths.get_filename_list("input")) if folder_paths else []
        try:
            files = folder_paths.filter_files_content_types(files, ["image"])
        except AttributeError:
            pass
        return {"required": {"image": (files, {"image_upload": True})}}
    RETURN_TYPES = ("IMAGE",); RETURN_NAMES = ("image",); FUNCTION = "load"; CATEGORY = "H3-Heaton/i2v"

    def load(self, image):
        _need_comfy()
        return (nodes.LoadImage().load_image(image)[0],)


NODE_CLASS_MAPPINGS = {"HT_I2V_Conditioning": HTI2VConditioning, "HT_LoadImage": HTLoadImage}
NODE_DISPLAY_NAME_MAPPINGS = {"HT_I2V_Conditioning": "HT · I2V Conditioning", "HT_LoadImage": "HT · Load Image"}
