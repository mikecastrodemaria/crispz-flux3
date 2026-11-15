"""Selecting a model or a Performance preset must never push a value the Generate sliders
refuse.

Seen on 2026-11-15 (flux3-image-klein-base single-file on the 5090): the steps slider kept
crispz-klein's maximum of 40, the FLUX.3 BASE profile and the "Base (50 steps, CFG 3.5)"
preset ask for 50, and Gradio refused every click after the selection ("Value 50 is
greater than maximum value 40"). The bound now covers every built-in profile and preset,
and a config.txt one asking for more raises it (said in the log).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _inside(steps, guidance, U):
    lo, hi = U.GEN_STEPS_BOUNDS
    glo, ghi = U.GUIDANCE_BOUNDS
    return lo <= int(steps) <= hi and glo <= float(guidance) <= ghi


def test_every_model_profile_fits_the_sliders():
    import cz_ui as U
    for name, prof in U.MODEL_PROFILES.items():
        assert _inside(prof["steps"], prof["guidance"], U), (name, prof, U.GEN_STEPS_BOUNDS)
    d = U.DEFAULT_MODEL_PROFILE
    assert _inside(d["steps"], d["guidance"], U)


def test_every_performance_preset_fits_the_sliders():
    import cz_ui as U
    for name, (st, g) in U.PERFORMANCE.items():
        assert _inside(st, g, U), (name, st, g, U.GEN_STEPS_BOUNDS)


def test_the_flux3_base_profile_is_reachable():
    import cz_ui as U
    st, g = U.profile_for_model("black-forest-labs/flux-3-image-klein-base")
    assert (st, g) == (50, 3.5)
    assert _inside(st, g, U)


def test_a_config_profile_above_the_cap_raises_it_and_says_which():
    import cz_ui as U
    cap, raised = U._steps_cap({"my-model": {"steps": 150, "guidance": 4.0}},
                               {"Slow": (120, 3.0)})
    assert cap == 150
    assert raised == ["model_profiles.my-model"]
    cap, raised = U._steps_cap({"a": {"steps": 30}}, {"Fast": (8, 1.0)})
    assert (cap, raised) == (U._GEN_STEPS_MAX, [])
    # unreadable values are skipped, not fatal
    assert U._steps_cap({"bad": {"steps": "lots"}}, {"Odd": ()})[0] == U._GEN_STEPS_MAX
