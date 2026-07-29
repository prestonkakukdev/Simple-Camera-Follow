"""Config loading. A typo here should be loud, not silent."""

from pathlib import Path

import pytest

from camerafollow.config import AppConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_build():
    cfg = AppConfig()
    assert cfg.backend.kind == "virtual"
    assert cfg.pan.deadzone > 0


@pytest.mark.parametrize("name", ["virtual_4k.yaml", "visca_ptz.yaml", "diy_serial.yaml"])
def test_shipped_configs_load(name):
    cfg = AppConfig.load(CONFIG_DIR / name)
    assert cfg.detector.kind in ("yolo", "hog")
    assert cfg.runtime.control_rate > cfg.detector.confidence


def test_nested_values_are_applied():
    cfg = AppConfig.from_dict({"pan": {"kp": 3.5}, "backend": {"kind": "null"}})
    assert cfg.pan.kp == 3.5
    assert cfg.backend.kind == "null"
    assert cfg.tilt.kp != 3.5  # untouched sections keep their defaults


def test_backend_options_pass_through_untouched():
    cfg = AppConfig.from_dict({"backend": {"kind": "visca", "options": {"host": "10.0.0.5"}}})
    assert cfg.backend.options == {"host": "10.0.0.5"}


def test_unknown_key_is_rejected_with_a_useful_message():
    with pytest.raises(KeyError) as exc:
        AppConfig.from_dict({"pan": {"dead_zone": 0.1}})
    assert "dead_zone" in str(exc.value)
    assert "deadzone" in str(exc.value)  # tells you what you meant


def test_dotted_overrides():
    cfg = AppConfig().apply_overrides({"pan.kp": 2.5, "runtime.lead_time": 0.2})
    assert cfg.pan.kp == 2.5
    assert cfg.runtime.lead_time == 0.2


def test_dotted_override_coerces_strings_from_the_cli():
    cfg = AppConfig().apply_overrides({"pan.kp": "2.5", "runtime.show_preview": "false"})
    assert cfg.pan.kp == 2.5
    assert cfg.runtime.show_preview is False


def test_unknown_override_is_rejected():
    with pytest.raises(KeyError):
        AppConfig().apply_overrides({"pan.nope": 1})


def test_round_trip(tmp_path):
    original = AppConfig()
    original.pan.kp = 1.75
    path = tmp_path / "c.yaml"
    original.save(path)
    assert AppConfig.load(path).pan.kp == 1.75
