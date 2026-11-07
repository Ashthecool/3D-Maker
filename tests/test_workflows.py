import pytest

from maker3d.workflows import Trellis2Settings, anima_t2i, trellis2_i2m

# Number of outputs per node type (from ComfyUI v0.44.0 /object_info).
OUTPUTS = {
    "UNETLoader": 1, "CLIPLoader": 1, "VAELoader": 1, "CLIPTextEncode": 1,
    "EmptyLatentImage": 1, "KSampler": 1, "VAEDecode": 1, "SaveImage": 1, "LoadImage": 2,
    "CLIPVisionLoader": 1, "LoadBackgroundRemovalModel": 1, "RemoveBackground": 1,
    "ImageCropToMask": 1, "EmptyTrellis2LatentStructure": 1, "Trellis2Conditioning": 2,
    "CFGOverride": 1, "RescaleCFG": 1, "ModelSamplingSD3": 1, "VaeDecodeStructureTrellis2": 1,
    "Trellis2ShapeStage": 3, "Trellis2UpsampleStage": 3, "VaeDecodeShapeTrellis": 2,
    "Trellis2TextureStage": 3, "VaeDecodeTextureTrellis": 1, "RemeshMesh": 1,
    "DecimateMesh": 1, "MeshSmoothNormals": 1, "UnwrapMesh": 1, "BakeTextureFromVoxel": 3,
    "BakeNormalMapFromMesh": 1, "BakeAmbientOcclusion": 1, "ApplyTextureToMesh": 1,
    "SaveGLB": 0,
}  # fmt: skip


def _links(wf):
    for node_id, node in wf.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                yield node_id, name, value


def _check_graph(wf):
    for node_id, name, (src, idx) in _links(wf):
        assert src in wf, f"node {node_id}.{name} links to missing node {src}"
        src_type = wf[src]["class_type"]
        assert idx < OUTPUTS[src_type], f"{node_id}.{name}: {src_type} has no output {idx}"
    for node in wf.values():
        assert node["class_type"] in OUTPUTS, f"unexpected node type {node['class_type']}"


def _by_title(wf, title):
    return next(n for n in wf.values() if n.get("_meta", {}).get("title") == title)


def test_anima_graph_is_valid_and_parametrised():
    wf = anima_t2i("1girl, silver hair", seed=123)
    _check_graph(wf)
    assert _by_title(wf, "prompt")["inputs"]["text"] == "1girl, silver hair"
    sampler = _by_title(wf, "sampler")["inputs"]
    assert sampler["seed"] == 123 and sampler["cfg"] == 1.0
    assert _by_title(wf, "latent")["inputs"]["batch_size"] == 4
    loaders = {n["inputs"].get("unet_name") for n in wf.values()}
    assert "anima-turbo-v1.1.safetensors" in loaders


@pytest.mark.parametrize("remove_bg", [True, False])
def test_trellis2_graph_is_valid(remove_bg):
    wf = trellis2_i2m("char.png", seed=7, settings=Trellis2Settings(remove_background=remove_bg))
    _check_graph(wf)
    types = [n["class_type"] for n in wf.values()]
    assert ("RemoveBackground" in types) is remove_bg
    assert types.count("KSampler") == 4
    assert types[-1] == "SaveGLB"
    assert _by_title(wf, "image")["inputs"]["image"] == "char.png"


def test_trellis2_has_no_eu_gated_models():
    wf = trellis2_i2m("char.png", seed=1)
    text = repr(wf).lower()
    assert "pixal" not in text and "moge" not in text


def test_trellis2_dynamic_combo_keys():
    wf = trellis2_i2m("char.png", seed=1)
    remesh = next(n for n in wf.values() if n["class_type"] == "RemeshMesh")["inputs"]
    assert remesh["sign_mode"] == "solid" and "sign_mode.fill" in remesh
