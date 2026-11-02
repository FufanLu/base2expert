from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "rl"


def load(name):
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def test_p7_dapo_configs_are_dapo_fin_with_each_domains_cap_and_buffer():
    fin = load("dapo_fin")
    for name, cap in (("dapo_law", 4096), ("dapo_med", 8192)):
        c = load(name)
        assert c["defaults"] == fin["defaults"] == ["base", "_self_"], name
        assert c["actor_rollout_ref"] == fin["actor_rollout_ref"], name  # clip 0.2 / 0.28
        assert c["reward"]["reward_manager"] == fin["reward"]["reward_manager"], name
        kwargs, fin_kwargs = c["reward"]["reward_kwargs"], fin["reward"]["reward_kwargs"]
        assert kwargs["max_resp_len"] == fin_kwargs["max_resp_len"] == "${data.max_response_length}", name
        # the overlong zone is the cap's last quarter in every domain, as finance's 1,024 of 4,096
        assert kwargs["overlong_buffer_cfg"] == {**fin_kwargs["overlong_buffer_cfg"], "len": cap // 4}, name
        assert c["data"] == {"max_response_length": cap, "val_max_samples": -1}, name  # all 600 val rows
        assert c["trainer"] == {"experiment_name": name, "total_training_steps": 150}, name
    assert fin["reward"]["reward_kwargs"]["overlong_buffer_cfg"]["len"] == load("base")["data"]["max_response_length"] // 4


def test_every_run_names_itself_and_sets_its_length():
    # cluster/rl.sbatch reads trainer.total_training_steps from the yaml to size a leg (LEG_STEPS)
    for f in sorted(CONFIGS.glob("*.yaml")):
        if f.stem == "base":
            continue
        t = yaml.safe_load(f.read_text())["trainer"]
        assert t["experiment_name"] == f.stem and t["total_training_steps"] > 0, f.name
