import logging
import os
from pathlib import Path

import torch
import yaml
from packaging.version import InvalidVersion, Version

# vanilla and LTS compatibility snippet
try:
    from comfy_compatibility.vanilla import prepare_vanilla_environment

    prepare_vanilla_environment()

    from comfy.model_downloader import add_known_models
    from comfy.model_downloader_types import HuggingFile

    capability = torch.cuda.get_device_capability(0 if torch.cuda.is_available() else None)
    sm = f"{capability[0]}{capability[1]}"
    precision = "fp4" if sm == "120" else "int4"

    # add known models

    models_yaml_path = Path(__file__).parent / "test_data" / "models.yaml"
    with open(models_yaml_path, "r") as f:
        nunchaku_models_yaml = yaml.safe_load(f)

    NUNCHAKU_SVDQ_MODELS = []
    for model in nunchaku_models_yaml["models"]:
        filename = model["filename"]
        if not filename.startswith("svdq-"):
            continue
        if "{precision}" in filename:
            filename = filename.format(precision=precision)
        NUNCHAKU_SVDQ_MODELS.append(HuggingFile(repo_id=model["repo_id"], filename=filename))

    NUNCHAKU_SVDQ_TEXT_ENCODER_MODELS = [
        HuggingFile(repo_id="nunchaku-tech/nunchaku-t5", filename="awq-int4-flux.1-t5xxl.safetensors"),
    ]

    add_known_models("diffusion_models", *NUNCHAKU_SVDQ_MODELS)
    add_known_models("text_encoders", *NUNCHAKU_SVDQ_TEXT_ENCODER_MODELS)
except (ImportError, ModuleNotFoundError):
    pass

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("=" * 40 + " ComfyUI-nunchaku Initialization " + "=" * 40)

from .utils import get_package_version, get_plugin_version

nunchaku_full_version = get_package_version("nunchaku").split("+")[0].strip()

logger.info(f"Nunchaku version: {nunchaku_full_version}")
logger.info(f"ComfyUI-nunchaku version: {get_plugin_version()}")


min_nunchaku_version = "1.0.0"
nunchaku_version = nunchaku_full_version.split("+")[0].strip()
nunchaku_major_minor_patch_version = ".".join(nunchaku_version.split(".")[:3])

try:
    if Version(nunchaku_major_minor_patch_version) < Version(min_nunchaku_version):
        logger.warning(
            f"ComfyUI-nunchaku {get_plugin_version()} requires nunchaku >= v{min_nunchaku_version}, "
            f"but found nunchaku {nunchaku_full_version}. Please update nunchaku."
        )
except InvalidVersion:
    logger.warning(
        f"Could not parse nunchaku version: {nunchaku_full_version}. "
        f"Please ensure you have at least v{min_nunchaku_version}."
    )

NODE_CLASS_MAPPINGS = {}


def _log_import_failure(message: str, error: Exception):
    err_text = str(error)
    if isinstance(error, ModuleNotFoundError) and getattr(error, "name", "") == "nunchaku":
        logger.warning(f"{message} (cpu compatibility mode): {error}")
    elif "No module named 'nunchaku'" in err_text:
        logger.warning(f"{message} (cpu compatibility mode): {error}")
    else:
        logger.exception(message)


def _safe_file_options(folder_name: str, empty_option: str | None = None):
    try:
        from .nodes.utils import get_filename_list

        values = list(get_filename_list(folder_name))
        if empty_option is not None:
            return [empty_option] + values
        return values if values else [""]
    except Exception:
        return [empty_option] if empty_option is not None else [""]


def _register_generic_fallback_node(class_name: str, title: str, return_types=("STRING",), function_name="run"):
    class _FallbackNode:
        RETURN_TYPES = return_types
        FUNCTION = function_name
        CATEGORY = "Nunchaku"
        TITLE = title

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"{class_name} is running in CPU frontend compatibility mode. "
            "Execute on a GPU ComfyUI instance with nunchaku installed."
        )

    setattr(_FallbackNode, function_name, _raise)
    _FallbackNode.__name__ = class_name
    NODE_CLASS_MAPPINGS[class_name] = _FallbackNode

try:
    from .nodes.models.flux import NunchakuFluxDiTLoader

    NODE_CLASS_MAPPINGS["NunchakuFluxDiTLoader"] = NunchakuFluxDiTLoader
except ImportError as e:
    _log_import_failure("Node `NunchakuFluxDiTLoader` import failed", e)
    # CPU/UI-only fallback:
    # Keep this node class discoverable in /object_info and /validate_prompt
    # when CUDA nunchaku backend is unavailable on frontend-only CPU pods.
    class NunchakuFluxDiTLoader:
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load_model"
        CATEGORY = "Nunchaku"
        TITLE = "Nunchaku FLUX DiT Loader"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model_path": (
                        _safe_file_options("diffusion_models"),
                        {
                            "tooltip": "Select a model from diffusion_models.",
                            "tooltip": "CPU fallback placeholder. Real loading happens on GPU worker.",
                        },
                    ),
                    "cache_threshold": (
                        "FLOAT",
                        {"default": 0, "min": 0, "max": 1, "step": 0.001},
                    ),
                    "attention": (
                        ["nunchaku-fp16", "flash-attention2"],
                        {"default": "nunchaku-fp16"},
                    ),
                    "cpu_offload": (
                        ["auto", "enable", "disable"],
                        {"default": "auto"},
                    ),
                    "device_id": (
                        "INT",
                        {"default": 0, "min": 0, "max": 32, "step": 1},
                    ),
                    "data_type": (
                        ["bfloat16", "float16"],
                        {"default": "bfloat16"},
                    ),
                },
                "optional": {"i2f_mode": (["enabled", "always"], {"default": "enabled"})},
            }

        def load_model(self, *args, **kwargs):
            raise RuntimeError(
                "NunchakuFluxDiTLoader is running in CPU frontend compatibility mode. "
                "Run this workflow on a GPU ComfyUI instance with nunchaku installed."
            )

    NODE_CLASS_MAPPINGS["NunchakuFluxDiTLoader"] = NunchakuFluxDiTLoader

try:
    from .nodes.models.qwenimage import NunchakuQwenImageDiTLoader

    NODE_CLASS_MAPPINGS["NunchakuQwenImageDiTLoader"] = NunchakuQwenImageDiTLoader
except ImportError as e:
    _log_import_failure("Node `NunchakuQwenImageDiTLoader` import failed", e)
    class NunchakuQwenImageDiTLoader:
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load_model"
        CATEGORY = "Nunchaku"
        TITLE = "Nunchaku Qwen-Image DiT Loader"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model_name": (_safe_file_options("diffusion_models"), {}),
                    "cpu_offload": (["auto", "enable", "disable"], {"default": "auto"}),
                    "use_pin_memory": (["enable", "disable"], {"default": "disable"}),
                    "num_blocks_on_gpu": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                }
            }

        def load_model(self, *args, **kwargs):
            raise RuntimeError("NunchakuQwenImageDiTLoader is in CPU frontend compatibility mode. Execute on GPU.")

    NODE_CLASS_MAPPINGS["NunchakuQwenImageDiTLoader"] = NunchakuQwenImageDiTLoader

try:
    from .nodes.lora.flux import NunchakuFluxLoraLoader, NunchakuFluxLoraStack

    NODE_CLASS_MAPPINGS["NunchakuFluxLoraLoader"] = NunchakuFluxLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuFluxLoraStack"] = NunchakuFluxLoraStack
except ImportError as e:
    _log_import_failure("Nodes `NunchakuFluxLoraLoader` and `NunchakuFluxLoraStack` import failed", e)
    class NunchakuFluxLoraLoader:
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load_lora"
        CATEGORY = "Nunchaku"
        TITLE = "Nunchaku FLUX LoRA Loader"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model": ("MODEL", {}),
                    "lora_name": (_safe_file_options("loras"), {}),
                    "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                }
            }

        def load_lora(self, *args, **kwargs):
            raise RuntimeError(
                "NunchakuFluxLoraLoader is in CPU frontend compatibility mode. "
                "Execute on a GPU ComfyUI instance with nunchaku installed."
            )

    class NunchakuFluxLoraStack:
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load_lora_stack"
        CATEGORY = "Nunchaku"
        TITLE = "Nunchaku FLUX LoRA Stack"

        @classmethod
        def INPUT_TYPES(cls):
            inputs = {
                "required": {
                    "model": ("MODEL", {}),
                },
                "optional": {},
            }
            for i in range(1, 16):
                inputs["optional"][f"lora_name_{i}"] = (_safe_file_options("loras", empty_option="None"), {})
                inputs["optional"][f"lora_strength_{i}"] = (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                )
            return inputs

        def load_lora_stack(self, *args, **kwargs):
            raise RuntimeError(
                "NunchakuFluxLoraStack is in CPU frontend compatibility mode. "
                "Execute on a GPU ComfyUI instance with nunchaku installed."
            )

    NODE_CLASS_MAPPINGS["NunchakuFluxLoraLoader"] = NunchakuFluxLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuFluxLoraStack"] = NunchakuFluxLoraStack

try:
    from .nodes.models.text_encoder import NunchakuTextEncoderLoader, NunchakuTextEncoderLoaderV2

    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoader"] = NunchakuTextEncoderLoader
    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoaderV2"] = NunchakuTextEncoderLoaderV2
except ImportError as e:
    _log_import_failure("Nodes `NunchakuTextEncoderLoader` and `NunchakuTextEncoderLoaderV2` import failed", e)
    class NunchakuTextEncoderLoader:
        RETURN_TYPES = ("CLIP",)
        FUNCTION = "load_text_encoder"
        CATEGORY = "Nunchaku"
        TITLE = "Nunchaku Text Encoder Loader"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model_type": (["flux", "flux.1"], {"default": "flux"}),
                    "text_encoder1": (_safe_file_options("text_encoders"), {}),
                    "text_encoder2": (_safe_file_options("text_encoders"), {}),
                    "use_4bit_t5": (["disable", "enable"], {"default": "disable"}),
                    "int4_model": (["none"] + _safe_file_options("text_encoders"), {"default": "none"}),
                    "t5_min_length": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                }
            }

        def load_text_encoder(self, *args, **kwargs):
            raise RuntimeError("NunchakuTextEncoderLoader is in CPU frontend compatibility mode. Execute on GPU.")

    class NunchakuTextEncoderLoaderV2:
        RETURN_TYPES = ("CLIP",)
        FUNCTION = "load_text_encoder"
        CATEGORY = "Nunchaku"
        TITLE = "Nunchaku Text Encoder Loader V2"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model_type": (["flux.1", "flux"], {"default": "flux.1"}),
                    "text_encoder1": (_safe_file_options("text_encoders"), {}),
                    "text_encoder2": (_safe_file_options("text_encoders"), {}),
                    "t5_min_length": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                }
            }

        def load_text_encoder(self, *args, **kwargs):
            raise RuntimeError("NunchakuTextEncoderLoaderV2 is in CPU frontend compatibility mode. Execute on GPU.")

    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoader"] = NunchakuTextEncoderLoader
    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoaderV2"] = NunchakuTextEncoderLoaderV2

try:
    from .nodes.preprocessors.depth import FluxDepthPreprocessor

    NODE_CLASS_MAPPINGS["NunchakuDepthPreprocessor"] = FluxDepthPreprocessor
except ImportError as e:
    _log_import_failure("Node `NunchakuDepthPreprocessor` import failed", e)
    class NunchakuDepthPreprocessor:
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "depth_preprocess"
        CATEGORY = "Nunchaku"
        TITLE = "FLUX Depth Preprocessor (Deprecated)"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE", {}), "model_path": (_safe_file_options("checkpoints"), {})}}

        def depth_preprocess(self, *args, **kwargs):
            raise RuntimeError("NunchakuDepthPreprocessor is in CPU frontend compatibility mode. Execute on GPU.")

    NODE_CLASS_MAPPINGS["NunchakuDepthPreprocessor"] = NunchakuDepthPreprocessor

try:
    from .nodes.models.pulid import (
        NunchakuFluxPuLIDApplyV2,
        NunchakuPulidApply,
        NunchakuPulidLoader,
        NunchakuPuLIDLoaderV2,
    )

    NODE_CLASS_MAPPINGS["NunchakuPulidApply"] = NunchakuPulidApply
    NODE_CLASS_MAPPINGS["NunchakuPulidLoader"] = NunchakuPulidLoader
    NODE_CLASS_MAPPINGS["NunchakuPuLIDLoaderV2"] = NunchakuPuLIDLoaderV2
    NODE_CLASS_MAPPINGS["NunchakuFluxPuLIDApplyV2"] = NunchakuFluxPuLIDApplyV2
except ImportError as e:
    _log_import_failure(
        "Nodes `NunchakuPulidApply`,`NunchakuPulidLoader`, "
        "`NunchakuPuLIDLoaderV2` and `NunchakuFluxPuLIDApplyV2` import failed",
        e,
    )
    class NunchakuPuLIDLoaderV2:
        RETURN_TYPES = ("PULID_PIPELINE",)
        FUNCTION = "load"
        CATEGORY = "Nunchaku"
        TITLE = "Nunchaku PuLID Loader V2"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model": ("MODEL", {}),
                    "pulid_file": (_safe_file_options("pulid"), {}),
                    "eva_clip_file": (_safe_file_options("clip"), {}),
                    "insight_face_provider": (["gpu", "cpu"], {"default": "gpu"}),
                }
            }

        def load(self, *args, **kwargs):
            raise RuntimeError("NunchakuPuLIDLoaderV2 is in CPU frontend compatibility mode. Execute on GPU.")

    class NunchakuFluxPuLIDApplyV2:
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "apply"
        CATEGORY = "Nunchaku"
        TITLE = "Nunchaku FLUX PuLID Apply V2"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model": ("MODEL", {}),
                    "pulid_pipline": ("PULID_PIPELINE", {}),
                    "image": ("IMAGE", {}),
                    "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                }
            }

        def apply(self, *args, **kwargs):
            raise RuntimeError("NunchakuFluxPuLIDApplyV2 is in CPU frontend compatibility mode. Execute on GPU.")

    NODE_CLASS_MAPPINGS["NunchakuPuLIDLoaderV2"] = NunchakuPuLIDLoaderV2
    NODE_CLASS_MAPPINGS["NunchakuFluxPuLIDApplyV2"] = NunchakuFluxPuLIDApplyV2
    _register_generic_fallback_node("NunchakuPulidApply", "Nunchaku Pulid Apply", ("MODEL",), "apply")
    _register_generic_fallback_node("NunchakuPulidLoader", "Nunchaku Pulid Loader", ("PULID",), "load")
try:
    from .nodes.models.ipadapter import NunchakuFluxIPAdapterApply, NunchakuIPAdapterLoader

    NODE_CLASS_MAPPINGS["NunchakuFluxIPAdapterApply"] = NunchakuFluxIPAdapterApply
    NODE_CLASS_MAPPINGS["NunchakuIPAdapterLoader"] = NunchakuIPAdapterLoader
except ImportError as e:
    _log_import_failure("Nodes `NunchakuFluxIPAdapterApply` and `NunchakuIPAdapterLoader` import failed", e)
    _register_generic_fallback_node("NunchakuFluxIPAdapterApply", "Nunchaku FLUX IPAdapter Apply", ("MODEL",), "apply")
    _register_generic_fallback_node("NunchakuIPAdapterLoader", "Nunchaku IPAdapter Loader", ("IPADAPTER",), "load")

try:
    from .nodes.models.zimage import NunchakuZImageDiTLoader

    NODE_CLASS_MAPPINGS["NunchakuZImageDiTLoader"] = NunchakuZImageDiTLoader
except ImportError as e:
    _log_import_failure("Nodes `NunchakuZImageDiTLoader` import failed", e)
    class NunchakuZImageDiTLoader:
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load_model"
        CATEGORY = "Nunchaku"
        TITLE = "Nunchaku Z-Image DiT Loader"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"model_name": (_safe_file_options("diffusion_models"), {})}}

        def load_model(self, *args, **kwargs):
            raise RuntimeError("NunchakuZImageDiTLoader is in CPU frontend compatibility mode. Execute on GPU.")

    NODE_CLASS_MAPPINGS["NunchakuZImageDiTLoader"] = NunchakuZImageDiTLoader

try:
    from .nodes.tools.merge_safetensors import NunchakuModelMerger

    NODE_CLASS_MAPPINGS["NunchakuModelMerger"] = NunchakuModelMerger
except ImportError as e:
    _log_import_failure("Node `NunchakuModelMerger` import failed", e)
    class NunchakuModelMerger:
        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("status",)
        FUNCTION = "run"
        CATEGORY = "Nunchaku"
        TITLE = "Nunchaku Model Merger"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model_folder": (_safe_file_options("diffusion_models"), {}),
                    "save_name": ("STRING", {}),
                }
            }

        def run(self, *args, **kwargs):
            raise RuntimeError("NunchakuModelMerger is in CPU frontend compatibility mode. Execute on GPU.")

    NODE_CLASS_MAPPINGS["NunchakuModelMerger"] = NunchakuModelMerger

try:
    from .nodes.tools.installers import NunchakuWheelInstaller

    NODE_CLASS_MAPPINGS["NunchakuWheelInstaller"] = NunchakuWheelInstaller
except ImportError as e:
    _log_import_failure("Node `NunchakuWheelInstaller` import failed", e)
    class NunchakuWheelInstaller:
        OUTPUT_NODE = True
        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("status",)
        FUNCTION = "run"
        CATEGORY = "Nunchaku"
        TITLE = "Nunchaku Installer"

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "version": (["none"], {}),
                    "dev_version": (["none"], {"default": "none"}),
                    "mode": (["install", "uninstall", "update node"], {"default": "install"}),
                }
            }

        def run(self, *args, **kwargs):
            raise RuntimeError("NunchakuWheelInstaller is in CPU frontend compatibility mode. Execute on GPU.")

    NODE_CLASS_MAPPINGS["NunchakuWheelInstaller"] = NunchakuWheelInstaller

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
logger.info("=" * (80 + len(" ComfyUI-nunchaku Initialization ")))
