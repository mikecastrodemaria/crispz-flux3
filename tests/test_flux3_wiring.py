"""cz_pipeline on a FLUX.3 engine: the call sites written for Flux2Klein* must run through
the ONE Flux3Pipe (txt2img, edit, refine, inpaint, outpaint), with the FLUX.3 policies
(no derived pipeline, Euler only, guidance by variant, no encoder trim). CPU, a tiny
random Flux3Pipe injected as the cached base: nothing is downloaded, nothing is loaded.
"""
import os
import sys

import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cz_flux3  # noqa: E402
import cz_pipeline as P  # noqa: E402
from flux3_tiny import tiny_pipe  # noqa: E402

TMP = os.path.join(os.environ.get("TEMP") or "/tmp", "cz_flux3_wiring")
os.makedirs(TMP, exist_ok=True)


@pytest.fixture
def engine(monkeypatch):
    """A tiny Flux3Pipe installed as cz_pipeline's loaded base, exactly as _ensure_base
    leaves it (key, derived map, scheduler config), on the CPU in float32."""
    pipe = tiny_pipe()
    monkeypatch.setattr(P, "DEVICE", "cpu")
    monkeypatch.setattr(P, "DTYPE", torch.float32)
    monkeypatch.setattr(P, "LORAS", [])
    monkeypatch.setattr(P, "EDIT_LORAS", [], raising=False)
    monkeypatch.setattr(P, "_APPLIED_LORAS", [])
    monkeypatch.setattr(P, "_APPLIED_LOKRS", [])
    monkeypatch.setattr(P, "_BASE_PIPE", pipe)
    monkeypatch.setattr(P, "_DERIVED", {"txt2img": pipe})
    monkeypatch.setattr(P, "_LOADED_KEY", (P.BASE_REPO, P.ZIMAGE_TRANSFORMER, P.OFFLOAD_MODE,
                                           P.PRECISION))
    monkeypatch.setattr(P, "_BASE_SCHED_CONFIG", dict(pipe.scheduler.config))
    monkeypatch.setattr(P, "GUIDANCE", 1.0)
    monkeypatch.setattr(P, "SAMPLER", "euler")
    monkeypatch.setattr(P, "SCHEDULE", "sgm_uniform")
    monkeypatch.setattr(P, "FORCE_RATIO", "")
    monkeypatch.setattr(P, "EDIT_SPEED", None, raising=False)
    P._embed_cache_clear()
    return pipe


def _logs(monkeypatch):
    lines = []
    monkeypatch.setattr(P, "_log", lambda m: lines.append(str(m)))
    return lines


def test_every_kind_is_the_one_flux3_pipe(engine):
    assert P.get_pipe("txt2img") is engine
    assert P.get_pipe("img2img") is engine
    assert P.get_pipe("inpaint") is engine
    assert P.get_pipe("omni") is engine
    assert P._DERIVED["img2img"] is engine and P._DERIVED["inpaint"] is engine
    assert P._BASE_PIPE is engine                     # no reload was triggered


def test_generate_gives_the_asked_size(engine):
    img = P.generate("a cube", 64, 48, 2, 7)
    assert isinstance(img, Image.Image) and img.size == (64, 48)
    img = P.generate("a cube", 70, 50, 1, 7, negative_prompt="ugly")   # rounded, negative ignored
    assert img.size == (64, 48)


def test_generate_omni_edits_one_and_two_references(engine):
    r1 = Image.new("RGB", (64, 48), (200, 30, 30))
    r2 = Image.new("RGB", (48, 64), (30, 200, 30))
    out = P.generate_omni([r1], "make it blue", "", 0, 0, 2, 7)
    assert out.size == (64, 48)
    out = P.generate_omni([r1, r2], "put <ref_image_1> on <ref_image_0>", "", 0, 0, 1, 7)
    assert out.size[0] % 16 == 0 and out.size[1] % 16 == 0
    out = P.generate_omni([r1], "x", "", 96, 64, 1, 7, honor_size=True)
    assert out.size == (96, 64)
    with pytest.raises(ValueError):
        P.generate_omni([], "x", "", 0, 0, 1, 7)


def test_process_one_refines_through_the_adapter(engine):
    src = Image.new("RGB", (64, 48), (90, 140, 200))
    out, timings = P.process_one(src, None, 1.0, 0.5, 3, "sharpen", 7, 512, 32,
                                 refine_tile=0, do_esrgan=False)
    assert out.size == (64, 48) and timings["refine"] > 0 and timings["esrgan"] == 0
    # denoise 0: nothing runs, the image comes back as is
    out, _ = P.process_one(src, None, 1.0, 0.0, 3, "sharpen", 7, 512, 32,
                           refine_tile=0, do_esrgan=False)
    assert out.size == (64, 48)


def test_inpaint_run_keeps_the_original_size(engine):
    bg = Image.new("RGB", (80, 64), (90, 140, 200))
    mask = Image.new("L", (80, 64), 0)
    mask.paste(255, (40, 0, 80, 64))
    out = P.inpaint_run(bg, mask, "a window", 2, 0.9, 7)
    assert out.size == (80, 64)
    # the unmasked left half is the original, pixel for pixel (composite back)
    assert out.getpixel((5, 5)) == (90, 140, 200)


def test_outpaint_fills_a_canvas_at_the_ratio(engine, monkeypatch):
    monkeypatch.setattr(P, "_ratio_canvas", lambda rw, rh, target_px=0: (128, 96))
    src = Image.new("RGB", (64, 64), (90, 140, 200))
    out = P.outpaint(src, 4, 3, "more sky", 2, 7)
    assert out.size == (128, 96)
    out = P.outpaint_directions(src, None, ["right"], "more", 1, 7, expand=0.5)
    assert out.size[0] > 64 and out.size[1] == 64


def test_the_sampler_choice_is_ignored_once_and_said(engine, monkeypatch):
    lines = _logs(monkeypatch)
    sched = engine.scheduler
    monkeypatch.setattr(P, "SAMPLER", "dpmpp2m")
    monkeypatch.setattr(P, "SCHEDULE", "karras")
    P._SAMPLER_IGNORED_SAID.discard(("dpmpp2m", "karras"))
    P._apply_sampler(engine)
    P._apply_sampler(engine)
    assert engine.scheduler is sched                   # untouched
    assert sum("not applied" in m for m in lines) == 1  # said once
    P._SAMPLER_IGNORED_SAID.discard(("dpmpp2m", "karras"))


def test_guidance_follows_the_variant(engine, monkeypatch):
    lines = _logs(monkeypatch)
    seen = {}
    real = type(engine).__call__

    def spy(self, **kw):
        seen.update(kw)
        return real(self, **kw)
    monkeypatch.setattr(type(engine), "__call__", spy)
    monkeypatch.setattr(P, "GUIDANCE", 4.0)
    P._CFG_IGNORED_SAID.discard(4.0)
    P.generate("x", 32, 32, 1, 7, negative_prompt="bad")
    assert seen["guidance_scale"] == 1.0               # distilled: a flow map, no CFG
    assert "negative_prompt" not in seen
    assert any("guidance 4 ignored" in m for m in lines)
    # the base variant takes the slider and the negative
    base = tiny_pipe(base=True)
    monkeypatch.setattr(P, "_BASE_PIPE", base)
    monkeypatch.setattr(P, "_DERIVED", {"txt2img": base})
    monkeypatch.setattr(P, "BASE_REPO", cz_flux3.FLUX3_BASE)
    monkeypatch.setattr(P, "_LOADED_KEY", (P.BASE_REPO, P.ZIMAGE_TRANSFORMER, P.OFFLOAD_MODE,
                                           P.PRECISION))
    monkeypatch.setattr(type(base), "__call__", spy)
    seen.clear()
    P.generate("x", 32, 32, 1, 7, negative_prompt="bad")
    assert seen["guidance_scale"] == 4.0 and seen["negative_prompt"] == "bad"


def test_the_encoder_trim_is_announced_not_silent(engine, monkeypatch):
    lines = _logs(monkeypatch)
    monkeypatch.setattr(P, "TRIM_TEXT_ENCODER", True)
    P._trim_text_encoder(engine)
    assert P._ENCODER_TRIMMED is False
    assert any("NOT trimmed" in m and "Qwen3-VL" in m for m in lines)


def test_swap_transformer_goes_through_register_modules(engine, monkeypatch):
    new = tiny_pipe().transformer
    monkeypatch.setattr(P, "_load_transformer", lambda *a, **k: new)
    monkeypatch.setattr(P, "OFFLOAD_MODE", "none")
    monkeypatch.setattr(P, "_LOADED_KEY", (P.BASE_REPO, None, "none", P.PRECISION))
    monkeypatch.setattr(P, "ZIMAGE_TRANSFORMER", os.path.join(TMP, "x.safetensors"))
    assert P._swap_transformer(engine) is True
    assert engine.transformer is new and engine._mp.transformer is new
    assert P._DERIVED == {}                             # re-derived (to the base) on demand
    assert P.get_pipe("img2img") is engine


def test_load_transformer_routes_single_files_through_cz_flux3(engine, monkeypatch):
    from safetensors.torch import save_file
    calls = []
    monkeypatch.setattr(cz_flux3, "load_transformer",
                        lambda *a, **k: calls.append((a, k)) or "tr")
    monkeypatch.setattr(P, "_load_monitor", lambda label, fn: fn())
    monkeypatch.setattr(P, "PRECISION", "fp8r")
    # the base repo at the chosen precision
    assert P._load_transformer(None) == "tr"
    assert calls[-1][1]["precision"] == "fp8r" and calls[-1][1]["single_file"] is None
    # a FLUX.3 single file
    ok = os.path.join(TMP, "flux3.safetensors")
    save_file({"mode_blocks.0.img_attn.qkv.weight": torch.zeros(2, 2),
               "early_stream_modulations.0.lin.weight": torch.zeros(2, 2)}, ok)
    assert P._load_transformer(ok) == "tr" and calls[-1][1]["single_file"] == ok
    # a FLUX.2 single file is refused BEFORE any load, with the studio that runs it
    bad = os.path.join(TMP, "flux2.safetensors")
    save_file({"double_blocks.0.img_attn.qkv.weight": torch.zeros(2, 2),
               "single_stream_modulation.lin.weight": torch.zeros(2, 2)}, bad)
    n = len(calls)
    with pytest.raises(RuntimeError, match="crispz-klein"):
        P._load_transformer(bad)
    assert len(calls) == n
    assert P.checkpoint_refusal(bad) and "FLUX.2" in P.checkpoint_refusal(bad)
    assert P.checkpoint_refusal(ok) is None
    # the menu names the family, not a 4B/9B
    assert P.model_info(ok)["variant"] == "FLUX.3"
    assert P.model_info(bad)["variant"] == "FLUX.2"


def test_a_gguf_single_file_is_read_by_family_and_kept_quantized(engine, monkeypatch):
    try:
        import numpy as np
        from gguf import GGUFWriter
    except Exception as e:
        pytest.skip(f"gguf writer unavailable: {e}")
    path = os.path.join(TMP, "flux3-Q8_0.gguf")
    w = GGUFWriter(path, "flux3")
    w.add_tensor("early_stream_modulations.0.lin.weight", np.zeros((4, 4), dtype=np.float16))
    w.add_tensor("mode_blocks.0.img_attn.qkv.weight", np.zeros((4, 4), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    assert P._file_family(path) == "flux3"
    assert P.checkpoint_refusal(path) is None
    assert P.model_info(path)["variant"] == "FLUX.3"
    calls = []
    monkeypatch.setattr(cz_flux3, "load_transformer",
                        lambda *a, **k: calls.append((a, k)) or "tr")
    monkeypatch.setattr(P, "_load_monitor", lambda label, fn: fn())
    assert P._load_transformer(path) == "tr"
    assert calls[-1][1]["single_file"] == path
