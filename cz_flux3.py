"""crispz-flux3 - the FLUX.3 Image Klein engine (diffusers ModularPipeline).

FLUX.3 does not ship a classic pipeline class: diffusers serves it through
`Flux3DistilledModularPipeline` / `Flux3ModularPipeline`, a chain of blocks
(text encoder -> vae encoder -> denoise loop -> decode) called with keyword inputs.
The rest of crispz (cz_pipeline, about thirty call sites, the detailer, the tiles,
the protocol) was written against the Flux2Klein* calling convention:

    pipe(prompt=..., image=..., mask_image=..., strength=..., width=..., height=...,
         num_inference_steps=..., generator=..., callback_on_step_end=...).images

`Flux3Pipe` below is the adapter between the two. It owns the modular pipeline, the
Qwen3-VL text encoder (loaded apart: crispz encodes the prompt itself so it can cache
the embeddings), and adds what the modular blocks do not offer:

  - img2img (`strength`): SDEdit on the flow-matching path. The init image is encoded,
    noised to the first kept sigma, and only the tail of the schedule is run. FLUX.3 has
    no Flux3...Img2ImgPipeline: the refine / upscale / detailer passes of crispz depend
    on this.
  - inpaint (`mask_image`): at every step, the tokens outside the mask are put back to
    the init image noised to the current sigma (RePaint-style latent blending). The
    caller composites the result over the original afterwards (cz_pipeline already
    does, see _composite_back).
  - `callback_on_step_end(pipe, i, t, {"latents": ...})` and `pipe._interrupt`: the
    live preview, the VRAM guard and the Stop button ride on them. A step hook is
    inserted into the denoise loop of each workflow.
  - model / sequential CPU offload, through accelerate, with the same method names as a
    DiffusionPipeline (enable_model_cpu_offload, maybe_free_model_hooks, _all_hooks).

The prompt upsampler (`flux-3-image-klein-pup`, a 9B Qwen3.5) is NOT loaded: it would
need ~28 GB on top of the generator. Its block is removed from the chain, the prompt
is encoded as given. crispz's own "Improve prompt" (Ollama) plays that role.

Depends on torch / diffusers / transformers only (never on cz_pipeline or gradio), so
it can be tested in isolation with tiny random components (tests/test_flux3_adapter.py).
"""

import json
import os
import types

import numpy as np
import torch
from PIL import Image

FLUX3_DISTILLED = "black-forest-labs/flux-3-image-klein"
FLUX3_BASE = "black-forest-labs/flux-3-image-klein-base"
FLUX3_REPOS = (FLUX3_DISTILLED, FLUX3_BASE)

# The distilled checkpoint is a flow map: 16 steps, no CFG. The base one: 50 steps,
# CFG 3.5 (BFL's model card; the diffusers blog shows 5.0 as an example value).
DEFAULTS = {"distilled": {"steps": 16, "guidance": 1.0},
            "base": {"steps": 50, "guidance": 3.5}}

# The FLUX.3 transformer ships in three precisions inside the official repos:
# bf16 (transformer/), and two BFL pre-quantized single files under variants/.
# fp8r needs compute capability 8.9+ for its speed-up (it still RUNS below, dequantized
# on every forward); nvfp4 needs Blackwell + flashinfer.
PRECISIONS = ("bf16", "fp8r", "nvfp4")
_VARIANT_FILE = {"fp8r": "variants/fp8r/model.safetensors",
                 "nvfp4": "variants/nvfp4/model.safetensors"}

# Hidden layers of Qwen3-VL whose states are stacked into the embedding (diffusers'
# default for FLUX.3: 8 x 2560 = the transformer's joint_attention_dim of 20480).
TEXT_OUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
TEXT_PAD_MULTIPLE = 80
MAX_SEQUENCE_LENGTH = 8192


# --- Family detection ---------------------------------------------------------------
# A FLUX.2-klein-4B and FLUX.3 Image Klein share the SAME hidden size (24 heads x 128 =
# 3072): the 4B/9B guard inherited from crispz-klein, which reads that dimension, would
# wave a FLUX.2 checkpoint or LoRA through and let it die gigabytes later inside
# load_state_dict. The family is therefore read from tensor NAMES, which do differ.
# Both the original (BFL/ComfyUI) and the diffusers layouts are covered.
_FLUX3_MARKERS = ("early_stream_modulations.", "single_stream_modulations.",
                  "content_mode_blocks.", "mode_blocks.", "txt_mode_blocks.",
                  "ref_time_in.", "target_timestep_embedder.", "x_embedders.")
_FLUX2_MARKERS = ("double_stream_modulation_img.", "double_stream_modulation_txt.",
                  "single_stream_modulation.", "double_blocks.")


def _segments(key):
    """'diffusion_model.double_blocks.0.img_attn' -> the dotted prefixes to test against
    the markers, so 'single_stream_modulation.' never matches inside
    'single_stream_modulations.'."""
    parts = key.split(".")
    return [".".join(parts[i:]) for i in range(len(parts))]


def family_from_keys(keys):
    """'flux3', 'flux2' or None from an iterable of tensor names (a safetensors header,
    a LoRA). Votes rather than first hit: a file can carry a stray key."""
    n3 = n2 = 0
    for k in keys:
        if not isinstance(k, str) or k == "__metadata__":
            continue
        segs = _segments(k)
        if any(s.startswith(m) for s in segs for m in _FLUX3_MARKERS):
            n3 += 1
        elif any(s.startswith(m) for s in segs for m in _FLUX2_MARKERS):
            n2 += 1
        elif any(s.startswith("transformer_blocks.") for s in segs) and \
                not any(s.startswith("single_transformer_blocks.") for s in segs):
            # diffusers' FLUX.2 double-stream blocks; FLUX.3 calls them mode_blocks
            n2 += 1
    if n3 == 0 and n2 == 0:
        return None
    return "flux3" if n3 >= n2 else "flux2"


def _modular_index(repo):
    """The repo's modular_model_index.json as a dict, or None. Local folder first, then
    the HF cache, then the hub (a few KB; gated repos answer 401 without a token)."""
    try:
        path = os.path.join(repo, "modular_model_index.json")
        if not os.path.isfile(path):
            from huggingface_hub import hf_hub_download
            try:
                path = hf_hub_download(repo, "modular_model_index.json", local_files_only=True)
            except Exception:
                path = hf_hub_download(repo, "modular_model_index.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_flux3_repo(repo):
    """True for the official repos, a mirror carrying 'flux-3' / 'flux3' in its name, or
    a local folder whose modular index declares a Flux3 pipeline."""
    r = str(repo or "").strip()
    if not r:
        return False
    low = r.lower().replace("_", "-")
    if r in FLUX3_REPOS or "flux-3" in low or "flux3" in low:
        return True
    if os.path.isdir(r):
        idx = _modular_index(r) or {}
        return str(idx.get("_class_name", "")).startswith("Flux3")
    return False


def is_base_repo(repo):
    """The undistilled checkpoint (50 steps + CFG) rather than the 16-step flow map.
    The modular index says it ('Flux3ModularPipeline' vs 'Flux3DistilledModularPipeline');
    without it, the '-base' suffix of the official name decides."""
    r = str(repo or "").strip()
    if r.lower().rstrip("/").endswith("-base"):
        return True
    if os.path.isdir(r):
        idx = _modular_index(r) or {}
        return idx.get("_class_name") == "Flux3ModularPipeline"
    return False


def variant_of(repo):
    """'base' or 'distilled'."""
    return "base" if is_base_repo(repo) else "distilled"


# --- The step hook ------------------------------------------------------------------
class _Interrupted(Exception):
    """Raised by the step hook when `_interrupt` is set: unwinds the modular loop, which
    has no break of its own. Carries nothing: the adapter keeps the last latents."""


def _make_hook_block():
    """A loop sub-block that hands every step to the adapter. Built lazily so importing
    this module never requires the modular API (smoke tests, the CLI's --help)."""
    from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks

    class CrispzStepHook(ModularPipelineBlocks):
        model_name = "flux3"

        @property
        def description(self):
            return "crispz: per-step hook (inpaint blending, callback, interrupt)."

        @property
        def inputs(self):
            return []

        @property
        def intermediate_outputs(self):
            return []

        @torch.no_grad()
        def __call__(self, components, block_state, i, t):
            owner = getattr(self, "owner", None)
            if owner is not None:
                owner._on_step(block_state, i, t)
            return components, block_state

    return CrispzStepHook()


def _insert_hooks(blocks):
    """Puts a CrispzStepHook at the end of every denoise loop of `blocks` (the
    text-to-image AND the image-conditioned workflow). Returns how many were placed."""
    from diffusers.modular_pipelines.modular_pipeline import LoopSequentialPipelineBlocks
    n = 0

    def walk(b):
        nonlocal n
        subs = getattr(b, "sub_blocks", None)
        if not subs:
            return
        if isinstance(b, LoopSequentialPipelineBlocks):
            subs.insert("crispz_hook", _make_hook_block(), len(subs))
            n += 1
            return
        for child in list(subs.values()):
            walk(child)
    walk(blocks)
    return n


def _find_hooks(blocks):
    out = []

    def walk(b):
        subs = getattr(b, "sub_blocks", None) or {}
        for k, child in subs.items():
            if k == "crispz_hook":
                out.append(child)
            else:
                walk(child)
    walk(blocks)
    return out


def build_blocks(base=False):
    """The block chain crispz runs: the official auto blocks minus the prompt upsampler
    (not loaded, see the module docstring) and minus the text encoder (crispz encodes
    and caches the prompt itself), plus the step hooks."""
    from diffusers.modular_pipelines import Flux3AutoBlocks, Flux3DistilledAutoBlocks
    blocks = Flux3AutoBlocks() if base else Flux3DistilledAutoBlocks()
    for name in ("prompt_upsampler", "text_encoder"):
        if name in blocks.sub_blocks:
            blocks.sub_blocks.pop(name)
    _insert_hooks(blocks)
    return blocks


# --- Latent helpers -------------------------------------------------------------------
def _patchify(latents):
    """(B, C, H, W) -> (B, C*4, H/2, W/2), the FLUX.2/3 packed layout."""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def _unpatchify(latents):
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def _bn_stats(vae, device, dtype):
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(device, dtype)
    return mean, std


def shifted_sigmas(n, shift):
    """The sigmas the FLUX.3 scheduler actually walks for n steps: diffusers feeds it
    linspace(1, 1/n, n) and FlowMatchEulerDiscreteScheduler applies its constant shift."""
    s = np.linspace(1.0, 1.0 / n, n)
    return shift * s / (1 + (shift - 1) * s)


def img2img_schedule(n, strength, shift):
    """(sigmas to hand the pipeline, sigma of the first kept step, steps run).

    SDEdit on a flow-matching model: x_t = (1 - sigma) * x0 + sigma * noise. With n
    steps and a strength s in (0, 1], the steps whose shifted sigma is above s are
    skipped; at least one step always runs (a 0.05 denoise must still touch the image).
    The UNSHIFTED tail is returned because the scheduler applies the shift itself."""
    n = max(1, int(n))
    s = float(min(max(strength, 0.0), 1.0))
    raw = np.linspace(1.0, 1.0 / n, n)
    sh = shifted_sigmas(n, shift)
    start = int(np.searchsorted(-sh, -s, side="left"))      # first index with sh <= s
    start = min(start, n - 1)
    return list(raw[start:]), float(sh[start]), n - start


# --- The adapter ---------------------------------------------------------------------
class Flux3Pipe:
    """A FLUX.3 modular pipeline behind the Flux2Klein calling convention (see the
    module docstring). Unknown attributes are forwarded to the modular pipeline, so
    pipe.transformer, pipe.vae, pipe.scheduler, pipe.load_lora_weights(...) keep
    working."""

    _OWN = {"_mp", "_text_encoder", "_tokenizer", "repo", "variant", "_interrupt",
            "_all_hooks", "_seq_offload", "_offload_device", "_cfg", "_step_ctx",
            "_last_latents", "guidance_scale"}

    def __init__(self, mp, text_encoder=None, tokenizer=None, repo="", variant="distilled"):
        object.__setattr__(self, "_mp", mp)
        object.__setattr__(self, "_text_encoder", text_encoder)
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "_interrupt", False)
        object.__setattr__(self, "_all_hooks", [])
        object.__setattr__(self, "_seq_offload", False)
        object.__setattr__(self, "_offload_device", None)
        object.__setattr__(self, "_cfg", {"is_distilled": variant == "distilled"})
        object.__setattr__(self, "_step_ctx", None)
        object.__setattr__(self, "_last_latents", None)
        object.__setattr__(self, "guidance_scale", DEFAULTS[variant]["guidance"])
        for h in _find_hooks(getattr(mp, "_blocks", None) or mp.blocks):
            h.owner = self

    # -- attribute plumbing --
    def __getattr__(self, name):
        if name in ("text_encoder",):
            return self._text_encoder
        if name in ("tokenizer",):
            return self._tokenizer
        return getattr(object.__getattribute__(self, "_mp"), name)

    def __setattr__(self, name, value):
        if name in self._OWN:
            object.__setattr__(self, name, value)
        elif name == "text_encoder":
            object.__setattr__(self, "_text_encoder", value)
        elif name == "tokenizer":
            object.__setattr__(self, "_tokenizer", value)
        elif name in ("transformer", "vae", "scheduler", "image_processor", "guider"):
            self._mp.update_components(**{name: value})
        else:
            setattr(self._mp, name, value)

    @property
    def config(self):
        return types.SimpleNamespace(**self._cfg)

    def register_to_config(self, **kw):
        self._cfg.update(kw)

    @property
    def components(self):
        comps = dict(self._mp.components)
        if self._text_encoder is not None:
            comps["text_encoder"] = self._text_encoder
        return comps

    @property
    def _execution_device(self):
        if self._offload_device is not None:
            return torch.device(self._offload_device)
        return self._mp._execution_device

    @property
    def dtype(self):
        return self._mp.transformer.dtype

    def _modules(self):
        """The torch modules in the order a generation uses them."""
        out = []
        for m in (self._text_encoder, getattr(self._mp, "transformer", None),
                  getattr(self._mp, "vae", None)):
            if isinstance(m, torch.nn.Module):
                out.append(m)
        return out

    # -- placement / offload (DiffusionPipeline-compatible names) --
    def to(self, *args, **kwargs):
        self.remove_all_hooks()
        for m in self._modules():
            m.to(*args, **kwargs)
        return self

    def remove_all_hooks(self):
        try:
            from accelerate.hooks import remove_hook_from_module
            for m in self._modules():
                remove_hook_from_module(m, recurse=True)
        except Exception:
            pass
        self._all_hooks = []
        self._seq_offload = False
        self._offload_device = None

    def enable_model_cpu_offload(self, gpu_id=None, device="cuda"):
        """One model on the GPU at a time (text encoder -> transformer -> VAE), the others
        parked in RAM. Same chain as diffusers' model_cpu_offload_seq."""
        from accelerate import cpu_offload_with_hook
        dev = torch.device(device if gpu_id is None else f"{device}:{gpu_id}")
        self.remove_all_hooks()
        hooks, prev = [], None
        for m in self._modules():
            m.to("cpu")
            _, prev = cpu_offload_with_hook(m, dev, prev_module_hook=prev)
            hooks.append(prev)
        self._all_hooks = hooks
        self._offload_device = dev
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    def enable_sequential_cpu_offload(self, gpu_id=None, device="cuda"):
        """Layer by layer from RAM: the slowest mode, the one that fits anywhere."""
        from accelerate import cpu_offload
        dev = torch.device(device if gpu_id is None else f"{device}:{gpu_id}")
        self.remove_all_hooks()
        for m in self._modules():
            m.to("cpu")
            cpu_offload(m, execution_device=dev)
        self._seq_offload = True
        self._offload_device = dev
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    def maybe_free_model_hooks(self):
        for h in self._all_hooks or []:
            try:
                h.offload()
            except Exception:
                pass

    def set_progress_bar_config(self, **kwargs):
        try:
            self._mp.set_progress_bar_config(**kwargs)
        except Exception:
            pass

    # -- text --
    @torch.no_grad()
    def encode_prompt(self, prompt, device=None, max_sequence_length=None, **_):
        """(prompt_embeds,) for one prompt, as the Qwen3-VL text-encoder step of diffusers
        computes it (chat template, right padding to a multiple of 80 tokens, 8 hidden
        states stacked). Kept as a method so cz_pipeline's embedding cache works as it
        did on FLUX.2."""
        from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
        if self._text_encoder is None or self._tokenizer is None:
            raise RuntimeError("FLUX.3: no text encoder loaded")
        dev = device or self._execution_device
        if self._all_hooks:
            # model offload: the hook moves the encoder on its forward; tokens go there
            dev = self._offload_device
        emb = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
            text_encoder=self._text_encoder, tokenizer=self._tokenizer,
            prompt=prompt or "", device=dev,
            max_sequence_length=int(max_sequence_length or MAX_SEQUENCE_LENGTH),
            pad_multiple=TEXT_PAD_MULTIPLE, hidden_states_layers=TEXT_OUT_LAYERS)
        return (emb,)

    # -- image helpers --
    @torch.no_grad()
    def _encode_image(self, image, width, height):
        """Normalized packed latents (B, 128, H/16, W/16) of `image` resized to the
        output size: the x0 of img2img and inpaint."""
        vae = self._mp.vae
        dev = self._execution_device
        img = image.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
        x = x.permute(2, 0, 1)[None].to(dev, vae.dtype)
        lat = vae.encode(x).latent_dist.mode()
        lat = _patchify(lat)
        mean, std = _bn_stats(vae, lat.device, lat.dtype)
        return (lat - mean) / std

    @staticmethod
    def _mask_tokens(mask_image, width, height, device):
        """The mask on the token grid (1, seq, 1): 1 = repaint, 0 = keep. A token is
        repainted as soon as ANY of its pixels is (max-pool): a thin stroke must not
        vanish below the 16-pixel grid."""
        m = mask_image.convert("L")
        if m.size != (width, height):
            m = m.resize((width, height), Image.NEAREST)
        t = torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)[None, None]
        t = torch.nn.functional.max_pool2d(t, kernel_size=16, stride=16)
        return t.flatten(2).transpose(1, 2).to(device)

    @staticmethod
    def _is_full_mask(mask_image):
        try:
            lo, _hi = mask_image.convert("L").getextrema()
            return lo >= 250
        except Exception:
            return False

    @torch.no_grad()
    def _decode_tokens(self, tokens, width, height, output_type="pil"):
        """Decodes latent TOKENS (B, seq, 128): used when a Stop interrupts the loop, so
        the caller still gets the image as it stood."""
        from diffusers.modular_pipelines.flux3.decoders import Flux3DecodeStep  # noqa: F401
        vae = self._mp.vae
        b, _seq, c = tokens.shape
        lh, lw = height // 16, width // 16
        lat = tokens.view(b, lh, lw, c).permute(0, 3, 1, 2)
        mean, std = _bn_stats(vae, lat.device, lat.dtype)
        lat = _unpatchify(lat * std + mean)
        img = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
        return self._mp.image_processor.postprocess(img, output_type=output_type)

    # -- the step hook --
    def _on_step(self, block_state, i, t):
        ctx = self._step_ctx or {}
        lat = block_state.latents
        # Inpaint: outside the mask, the tokens are put back to the init image noised to
        # the sigma the step just reached (the next timestep; 0 after the last one).
        if ctx.get("mask") is not None and ctx.get("x0") is not None:
            ts = block_state.timesteps
            nxt = float(ts[i + 1]) / 1000.0 if i + 1 < len(ts) else 0.0
            x0, noise, m = ctx["x0"], ctx["noise"], ctx["mask"]
            known = (1.0 - nxt) * x0 + nxt * noise
            lat = m * lat + (1.0 - m) * known.to(lat.dtype)
            block_state.latents = lat
        self._last_latents = block_state.latents
        cb = ctx.get("callback")
        if cb is not None:
            ret = cb(self, i, t, {"latents": block_state.latents})
            if isinstance(ret, dict) and ret.get("latents") is not None:
                block_state.latents = ret["latents"]
        if self._interrupt:
            raise _Interrupted()

    # -- the call --
    def __call__(self, prompt=None, image=None, mask_image=None, strength=None,
                 width=None, height=None, num_inference_steps=None, generator=None,
                 guidance_scale=None, num_images_per_prompt=1, prompt_embeds=None,
                 negative_prompt=None, negative_prompt_embeds=None,
                 callback_on_step_end=None, callback_on_step_end_tensor_inputs=None,
                 output_type="pil", max_sequence_length=None, return_dict=True, **_ignored):
        self._interrupt = False
        steps = int(num_inference_steps or DEFAULTS[self.variant]["steps"])
        if isinstance(prompt, (list, tuple)):
            # one prompt per call in FLUX.3: a list is run prompt by prompt
            imgs = []
            for j, p in enumerate(prompt):
                g = generator[j] if isinstance(generator, (list, tuple)) else generator
                imgs += self(prompt=p, image=image, mask_image=mask_image, strength=strength,
                             width=width, height=height, num_inference_steps=steps,
                             generator=g, guidance_scale=guidance_scale,
                             callback_on_step_end=callback_on_step_end,
                             output_type=output_type).images
            return types.SimpleNamespace(images=imgs)

        if prompt_embeds is None:
            prompt_embeds = self.encode_prompt(prompt or "", max_sequence_length=max_sequence_length)[0]
        dev = self._execution_device
        prompt_embeds = prompt_embeds.to(dev)
        kw = {"prompt_embeds": prompt_embeds, "num_images_per_prompt": int(num_images_per_prompt or 1),
              "generator": generator, "output_type": output_type}

        # Real CFG on the base checkpoint only (the distilled one has no unconditional
        # branch at all). The guider is swapped per call: a shared pipe must not keep the
        # previous caller's scale.
        if self.variant == "base":
            g = float(self.guidance_scale if guidance_scale is None else guidance_scale)
            try:
                self._mp.guider = self._mp.guider.new(guidance_scale=g)
            except Exception:
                pass
            if g > 1.0:
                if negative_prompt_embeds is None:
                    negative_prompt_embeds = self.encode_prompt(negative_prompt or "")[0]
                kw["negative_prompt_embeds"] = negative_prompt_embeds.to(dev)

        refs = image if isinstance(image, (list, tuple)) else ([image] if image is not None else [])
        refs = [r for r in refs if r is not None]
        img2img = strength is not None or mask_image is not None
        ctx = {"callback": callback_on_step_end}

        if img2img and refs:
            # img2img / inpaint: the init image is NOT a reference here, it is the start
            # point. Output size = the asked one, else the image's own, on the 16 grid.
            init = refs[0]
            w = int(width or init.size[0]) // 16 * 16
            h = int(height or init.size[1]) // 16 * 16
            shift = float(getattr(self._mp.scheduler.config, "shift", 6.93) or 6.93)
            s = 1.0 if strength is None else float(strength)
            sigmas, sigma0, _n = img2img_schedule(steps, s, shift)
            x0 = self._encode_image(init, w, h)
            noise_gen = generator[0] if isinstance(generator, (list, tuple)) else generator
            noise = torch.randn(x0.shape, generator=noise_gen, dtype=torch.float32,
                                device=noise_gen.device if noise_gen is not None else "cpu")
            noise = noise.to(x0.device, x0.dtype)
            start = (1.0 - sigma0) * x0 + sigma0 * noise
            kw.update(latents=start, sigmas=sigmas, width=w, height=h)
            if mask_image is not None and not self._is_full_mask(mask_image):
                tok = lambda z: z.flatten(2).transpose(1, 2)  # noqa: E731
                ctx.update(mask=self._mask_tokens(mask_image, w, h, x0.device).to(x0.dtype),
                           x0=tok(x0), noise=tok(noise))
            out_w, out_h = w, h
        else:
            if refs:
                kw["image"] = refs if len(refs) > 1 else refs[0]
            if width and height:
                kw["width"] = int(width) // 16 * 16
                kw["height"] = int(height) // 16 * 16
            kw["num_inference_steps"] = steps
            out_w, out_h = kw.get("width"), kw.get("height")

        self._step_ctx = ctx
        self._last_latents = None
        try:
            images = self._mp(output="images", **kw)
        except _Interrupted:
            lat = self._last_latents
            if lat is None or not out_w or not out_h:
                images = []
            else:
                images = self._decode_tokens(lat, out_w, out_h, output_type)
        finally:
            self._step_ctx = None
            self.maybe_free_model_hooks()
        if not isinstance(images, list):
            images = list(images) if output_type == "pil" else images
        return types.SimpleNamespace(images=images) if return_dict else (images,)


# --- Loading ---------------------------------------------------------------------------
def load_text_encoder(repo, dtype, token=None):
    """The Qwen3-VL-4B encoder and its processor, from the repo's text_encoder/."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    kw = {"subfolder": "text_encoder"}
    if token:
        kw["token"] = token
    enc = Qwen3VLForConditionalGeneration.from_pretrained(repo, dtype=dtype, **kw)
    enc.eval()
    tok = AutoProcessor.from_pretrained(repo, **kw)
    return enc, tok


def load_transformer(repo, dtype, precision="bf16", single_file=None, token=None):
    """The FLUX.3 transformer: a single-file override first (a community .safetensors,
    BFL's or ComfyUI's layout, converted by diffusers), else the repo's precision
    variant (fp8r / nvfp4 from variants/), else transformer/ in bf16."""
    from diffusers import Flux3Transformer2DModel
    if single_file:
        kw = {"config": repo, "subfolder": "transformer", "torch_dtype": dtype}
        if token:
            kw["token"] = token
        quant = _bfl_quant_of(single_file)
        if quant:
            from diffusers import BFLQuantizationConfig
            kw["quantization_config"] = BFLQuantizationConfig(compute_dtype=dtype)
        return Flux3Transformer2DModel.from_single_file(single_file, **kw)
    precision = (precision or "bf16").lower()
    if precision in _VARIANT_FILE:
        from diffusers import BFLQuantizationConfig
        from huggingface_hub import hf_hub_download
        if os.path.isdir(repo):
            path = os.path.join(repo, _VARIANT_FILE[precision])
        else:
            path = hf_hub_download(repo, _VARIANT_FILE[precision], token=token)
        return Flux3Transformer2DModel.from_single_file(
            path, config=repo, subfolder="transformer", torch_dtype=dtype, token=token,
            quantization_config=BFLQuantizationConfig(compute_dtype=dtype))
    kw = {"subfolder": "transformer", "torch_dtype": dtype}
    if token:
        kw["token"] = token
    return Flux3Transformer2DModel.from_pretrained(repo, **kw)


def _bfl_quant_of(path):
    """'fp8r' / 'nvfp4' when a single file carries BFL's scale tensors, else None."""
    try:
        import struct
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(min(n, 10_000_000)).decode("utf-8", "ignore"))
    except Exception:
        return None
    keys = [k for k in hdr if k != "__metadata__"]
    if any(k.endswith("weight_scale_2") for k in keys):
        return "nvfp4"
    if any(k.endswith("weight_scale") for k in keys):
        return "fp8r"
    return None


def load(repo, dtype=torch.bfloat16, precision="bf16", transformer=None, text_encoder=None,
         token=None):
    """Builds a ready Flux3Pipe for `repo` (on the CPU: the caller places it).

    transformer: an already-loaded Flux3Transformer2DModel (single-file override) or
    None for the repo's own at `precision`. text_encoder: an (encoder, processor) pair
    to reuse, or None to load the repo's."""
    base = is_base_repo(repo)
    blocks = build_blocks(base=base)
    mp = blocks.init_pipeline(repo)
    if transformer is None:
        transformer = load_transformer(repo, dtype, precision=precision, token=token)
    vae_kw = {"subfolder": "vae", "torch_dtype": dtype}
    if token:
        vae_kw["token"] = token
    from diffusers import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained(repo, **vae_kw)
    mp.update_components(transformer=transformer, vae=vae)
    enc, tok = text_encoder if text_encoder else load_text_encoder(repo, dtype, token=token)
    return Flux3Pipe(mp, enc, tok, repo=repo, variant="base" if base else "distilled")
