from maker3d import config


def test_all_models_have_hf_urls():
    for m in config.ALL_MODELS:
        assert m.url.startswith("https://huggingface.co/")
        assert m.filename.endswith(".safetensors")


def test_anima_turbo_v11_is_the_diffusion_model():
    names = [m.filename for m in config.ANIMA_MODELS]
    assert "anima-turbo-v1.1.safetensors" in names


def test_turbo_defaults_follow_model_card():
    d = config.AnimaDefaults()
    assert d.cfg == 1.0
    assert 8 <= d.steps <= 12


def test_default_toon_style_is_known():
    assert config.DEFAULT_TOON_STYLE in config.TOON_STYLES
