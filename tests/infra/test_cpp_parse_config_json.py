import json
import pytest
import os
import sys

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# 喂真 config.json, 断言 10 个字段 == 实测值
def test_parse_real_config_all_fields():
    path = os.path.expanduser("~/models/Qwen2.5-0.5B/config.json")
    ref_cfg = json.loads(open(path).read())
    cfg = xstar_cpp.parse_config_json(open(path).read())

    assert cfg.hidden_size == ref_cfg["hidden_size"]
    assert cfg.num_attention_heads == ref_cfg["num_attention_heads"]
    assert cfg.num_key_value_heads == ref_cfg["num_key_value_heads"]
    assert cfg.num_hidden_layers == ref_cfg["num_hidden_layers"]
    assert cfg.intermediate_size == ref_cfg["intermediate_size"]
    assert cfg.max_position_embeddings == ref_cfg["max_position_embeddings"]
    assert cfg.vocab_size == ref_cfg["vocab_size"]
    # 兼任浮点边界测试(大浮点不被截断 + 科学计数法读对)
    assert abs(cfg.rms_norm_eps - ref_cfg["rms_norm_eps"]) < 1e-12
    assert abs(cfg.rope_theta - ref_cfg["rope_theta"]) < 1e-6
    assert cfg.tie_word_embeddings == ref_cfg["tie_word_embeddings"]


# 测抛 "top-level value is not a JSON object"
def test_parse_rejects_non_object_top_level():
    with pytest.raises(RuntimeError, match="top-level value is not a JSON object"):
        xstar_cpp.parse_config_json("[1,2,3]")


# 测抛 "empty config"
def test_parse_rejects_empty_config():
    with pytest.raises(RuntimeError, match="empty config"):
        xstar_cpp.parse_config_json("")


# 测抛 "missing required config field: vocab_size"
def test_parse_missing_field_names_it():
    with pytest.raises(RuntimeError, match="missing required config field: vocab_size"):
        xstar_cpp.parse_config_json(
            '{"hidden_size": 896,"intermediate_size": 4864,"max_position_embeddings": 32768,"num_attention_heads": 14,"num_hidden_layers": 24,"num_key_value_heads": 2,"rms_norm_eps": 1e-06,"rope_theta": 1000000.0,"tie_word_embeddings": true,}'
        )


# 测 read_number 错
def test_parse_rejects_wrong_value_type():
    with pytest.raises(RuntimeError, match="p not pointing at a digit or '-' or '+"):
        xstar_cpp.parse_config_json(json.dumps({"hidden_size": "x"}))


# 测 read_bool 错
def test_parse_rejects_bad_bool():
    with pytest.raises(
        RuntimeError, match="the bytes after 't' are not exactly \"rue\""
    ):
        xstar_cpp.parse_config_json('{"tie_word_embeddings": truX}')


#  含 5 种未知 value 类型(str/num/bool/arr/obj)+ 10 已知,不抛错、10 已知对
def test_parse_skips_unknown_fields():
    cfg = xstar_cpp.parse_config_json(
        '{"_str": "silu","hidden_size": 896,"intermediate_size": 4864,"_num": 0.02,"max_position_embeddings": 32768,"num_attention_heads": 14,"num_hidden_layers": 24,"num_key_value_heads": 2,"_bool": false,"rms_norm_eps": 1e-06,"rope_theta": 1000000.0,"tie_word_embeddings": true, "vocab_size": 151936, "_arr": [1, 2, 3],"_obj": {"a": 1} }'
    )

    assert cfg.hidden_size == 896
    assert cfg.num_attention_heads == 14
    assert cfg.num_key_value_heads == 2
    assert cfg.num_hidden_layers == 24
    assert cfg.intermediate_size == 4864
    assert cfg.max_position_embeddings == 32768
    assert cfg.vocab_size == 151936
    # 兼任浮点边界测试(大浮点不被截断 + 科学计数法读对)
    assert abs(cfg.rms_norm_eps - 1e-06) < 1e-12
    assert abs(cfg.rope_theta - 1000000.0) < 1e-6
    assert cfg.tie_word_embeddings == True
