import json

from b2e.merge.run import GRID, copy_side_files, merge_config, tensor_dtypes, to_yaml


def test_the_grid_configs():
    experts = ["/m/r_fin", "/m/r_law", "/m/r_med"]
    avg = merge_config("avg", "/m/m0", experts)
    assert avg["merge_method"] == "task_arithmetic" and avg["base_model"] == "/m/m0"
    assert [m["parameters"] for m in avg["models"]] == [{"weight": 1 / 3}] * 3  # base + mean task vector
    ties = merge_config("ties_d0.2", "/m/m0", experts)
    assert ties["models"][0]["parameters"] == {"weight": 1.0, "density": 0.2} and "parameters" not in ties
    dare = merge_config("dare_d0.5", "/m/m0", experts)
    assert dare["merge_method"] == "dare_ties" and dare["parameters"] == {"normalize": True}  # mergekit sums by default
    assert all(merge_config(n, "/b", experts)["out_dtype"] == "bfloat16" for n in GRID)
    assert json.loads(to_yaml(dare)) == dare  # JSON is the YAML mergekit reads


def test_side_files_come_from_the_expert_and_weights_do_not(tmp_path):
    expert, out = tmp_path / "expert", tmp_path / "out"
    expert.mkdir(), out.mkdir()
    for name in ("config.json", "tokenizer.json", "chat_template.jinja", "model-00001-of-00002.safetensors",
                 "model.safetensors.index.json", ".complete"):
        (expert / name).write_text(name)
    (expert / "verl_saved").mkdir()
    (out / "model.safetensors.index.json").write_text("mergekit's")
    assert copy_side_files(expert, out) == ["chat_template.jinja", "config.json", "tokenizer.json"]
    assert (out / "model.safetensors.index.json").read_text() == "mergekit's"


def test_dtypes_are_read_from_the_safetensors_header(tmp_path):
    header = json.dumps({"a.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
                         "__metadata__": {"format": "pt"}}).encode()
    (tmp_path / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0")
    assert tensor_dtypes(tmp_path) == {"a.weight": "BF16"}
