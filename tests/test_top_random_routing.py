"""随机三态 × 实际通道，以及设置落盘与旧值兼容。"""
import pytest
import top_input_injection as top
from models import OpenAIMessage


@pytest.mark.parametrize('enabled', ['off', 'always'])
@pytest.mark.parametrize('random_mode', ['off', 'always', 'non_vertex_only'])
@pytest.mark.parametrize('channel', ['express', 'cookie', 'vertex'])
def test_mode_matrix(enabled, random_mode, channel, monkeypatch):
    plans = (top.TopInputPlan('fixed', '固定', 'system', '固定正文'),
             top.TopInputPlan('random', '随机', 'system', '随机正文'))
    config = top.TopInputInjectionConfig(enabled, plans, 'fixed', random_mode)
    calls = []
    monkeypatch.setattr(top.secrets, 'choice', lambda p: calls.append(True) or p[1])
    messages = [OpenAIMessage(role='user', content='原输入')]
    updated, notes = top.apply_top_input_injection(messages, config, channel=channel)
    if enabled == 'off':
        assert updated is messages
        assert not calls
        return
    random_expected = random_mode == 'always' or (random_mode == 'non_vertex_only' and channel != 'vertex')
    assert updated[0].content == ('随机正文' if random_expected else '固定正文')
    assert bool(calls) == random_expected
    assert ('随机' if random_expected else '指定') in notes[0]
    assert updated[1] is messages[0]


@pytest.mark.parametrize('value,expected', [(True, 'always'), (False, 'off'), ('false', 'off'), ('true', 'always'), ('non_vertex_only', 'non_vertex_only'), ('invalid', 'off')])
def test_legacy_random_value(value, expected):
    assert top.normalize_random_mode(value) == expected


@pytest.mark.parametrize('legacy', ['non_vertex_only', 'fake_stream_only'])
def test_old_gate_migrates_to_random_only(legacy):
    config, note = top.get_top_input_injection_config({
        top.SETTING_MODE: legacy, top.SETTING_RANDOM: False,
        top.SETTING_PLANS: [{'id': 'p', 'role': 'system', 'content': '原文'}],
    })
    assert note is None
    assert config.mode == 'always'
    assert config.random_mode == 'non_vertex_only'
    updated, _ = top.apply_top_input_injection([], config, channel='vertex')
    assert updated[0].content == '原文'


def test_random_mode_survives_restart(tmp_path, monkeypatch):
    import runtime_state
    monkeypatch.setattr(runtime_state, 'STATE_FILE', str(tmp_path / 'state.json'))
    state = runtime_state.AppState()
    state.update_settings({top.SETTING_MODE: 'always', top.SETTING_RANDOM: 'non_vertex_only',
                           top.SETTING_PLANS: [{'id': 'p', 'role': 'system', 'content': '原文'}]})
    restored = runtime_state.AppState()
    config, _ = top.get_top_input_injection_config(restored.get_settings())
    assert config.random_mode == 'non_vertex_only'
