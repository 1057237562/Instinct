import json
import pytest
import torch

from scripts.chat_tools import (
    ChatTextIteratorStreamer, calculate_expression, run_tool_calls,
    split_tool_calls, tool_request_message,
    ordered_response_parts,
    restore_thinking_prefix,
)


CALL = '<tool_call>{"name":"calculate_math","arguments":{"expression":"123 * 321"}}</tool_call>'


def test_tool_protocol_and_eos_are_separate_from_answer():
    body, calls = split_tool_calls('计算中。\n' + CALL + '\n<|im_end|>')
    assert body == '计算中。\n\n'
    assert len(calls) == 1
    assert calls[0]['arguments'] == {'expression': '123 * 321'}
    events = run_tool_calls(calls, {'calculate_math'},
                            lambda name, args: {'result': calculate_expression(args['expression'])})
    assert events[0]['result'] == {'result': 39483}
    message = tool_request_message(body, calls)
    assert '<tool_call>' not in message['content']
    assert message['tool_calls'][0]['function']['name'] == 'calculate_math'


def test_escaped_tags_and_multiple_calls_are_supported():
    escaped = CALL.replace('<', '\\<')
    body, calls = split_tool_calls(escaped + '\n' + CALL)
    assert not body.strip()
    assert len(calls) == 2
    assert all('error' not in call for call in calls)


def test_streaming_never_leaks_partial_protocol_or_json():
    for length in range(1, len(CALL) + 1):
        body, _ = split_tool_calls('Before\n' + CALL[:length], streaming=True)
        assert body == 'Before\n'
    body, _ = split_tool_calls('Before\n' + CALL + '\nAfter<|im_en', streaming=True)
    assert body == 'Before\n\nAfter'


@pytest.mark.parametrize('wrapped', [
    '`' + CALL + '`', '```xml\n' + CALL + '\n```',
    '<think>' + CALL + '</think>',
])
def test_examples_and_thinking_are_not_executed(wrapped):
    body, calls = split_tool_calls(wrapped)
    assert body == wrapped
    assert calls == []


def test_closing_tag_inside_json_string_is_not_a_delimiter():
    raw = json.dumps({'name': 'text_length', 'arguments': {'text': '</tool_call>'}})
    body, calls = split_tool_calls('<tool_call>' + raw + '</tool_call>')
    assert body == ''
    assert calls[0]['arguments'] == {'text': '</tool_call>'}


def test_tool_offsets_preserve_text_between_multiple_calls():
    body, calls = split_tool_calls('before' + CALL + 'between' + CALL + 'after')
    assert body == 'beforebetweenafter'
    assert [call['position'] for call in calls] == [6, 13]
    parts = ordered_response_parts(body, calls)
    assert [kind for kind, _, _ in parts] == ['text', 'tool', 'text', 'tool', 'text']
    assert [value for kind, value, _ in parts if kind == 'text'] == ['before', 'between', 'after']


def test_thinking_prefix_comes_from_prompt_state_and_is_not_duplicated():
    content = '分析</think>答案'
    assert restore_thinking_prefix(content, True) == '<think>\n' + content
    assert restore_thinking_prefix(content, False) == content
    assert restore_thinking_prefix('<think>' + content, True) == '<think>' + content


def test_unmatched_backtick_run_is_literal_and_hides_nothing():
    body, _ = split_tool_calls('a ``` b\n\n后续<|im_end|>')
    assert body == 'a ``` b\n\n后续'


def test_mid_sentence_backticks_in_thinking_do_not_leak_eos_marker():
    # A model quoting ```python mid-sentence used to wedge the inline-span
    # state, so the trailing <|im_end|> survived into the visible answer.
    reply = ('<think>Use ```python for the final code.</think>\n\n'
             '```python\nprint(1)\n```<|im_end|>')
    body, _ = split_tool_calls(reply)
    assert '<|im_end|>' not in body
    assert '```python\nprint(1)\n```' in body


def test_marker_inside_inline_code_stays_protected():
    body, _ = split_tool_calls('以 `<|im_end|>` 结束。')
    assert body == '以 `<|im_end|>` 结束。'


@pytest.mark.parametrize('raw', [
    '{"name":"calculate_math","arguments":{"expression":"123 \\* 321"}}',
    '[1, 2]', '{"name":12}', '{"name":"calculate_math","arguments":[]}',
])
def test_bad_json_or_schema_returns_visible_diagnostic(raw):
    _, calls = split_tool_calls('<tool_call>' + raw + '</tool_call>')
    assert 'error' in calls[0]
    executed = []
    events = run_tool_calls(calls, {'calculate_math'}, lambda *args: executed.append(args))
    assert not executed
    assert 'error' in events[0]['result']


def test_unclosed_call_is_hidden_with_an_error_on_completion():
    body, calls = split_tool_calls('Before<tool_call>{"name":')
    assert body == 'Before'
    assert 'missing' in calls[0]['error']


def test_disabled_tool_is_not_executed():
    _, calls = split_tool_calls(CALL)
    executed = []
    events = run_tool_calls(calls, set(), lambda *args: executed.append(args))
    assert not executed
    assert 'not enabled' in events[0]['error']


def test_string_arguments_are_decoded_and_executor_errors_are_reported():
    raw = json.dumps({'name': 'calculate_math', 'arguments': json.dumps({'expression': '1/0'})})
    _, calls = split_tool_calls('<tool_call>' + raw + '</tool_call>')
    def execute(name, args):
        return calculate_expression(args['expression'])
    events = run_tool_calls(calls, {'calculate_math'}, execute)
    assert 'division by zero' in events[0]['error']


def test_streamer_retains_real_tokenizer_control_tokens():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('model', trust_remote_code=True)
    streamer = ChatTextIteratorStreamer(tokenizer, skip_prompt=True)
    streamer.put(torch.tensor([[3]]))
    streamer.put(torch.tensor(tokenizer.encode(CALL + '<|im_end|>', add_special_tokens=False)))
    streamer.end()
    text = ''.join(streamer)
    assert '<tool_call>' in text and '</tool_call>' in text
    body, calls = split_tool_calls(text)
    assert body == ''
    assert calls[0]['name'] == 'calculate_math'


@pytest.mark.parametrize('expression, expected', [
    ('123 * 321', 39483), ('-(2 + 3) ** 2 + 7 // 2', -22), ('1.5 / 2', .75),
])
def test_arithmetic(expression, expected):
    assert calculate_expression(expression) == expected


@pytest.mark.parametrize('expression', [
    "__import__('os').getcwd()", 'open("x")', '[1, 2]', 'True + 1', '2 ** 1000000',
])
def test_arithmetic_rejects_code_and_unbounded_powers(expression):
    with pytest.raises(ValueError):
        calculate_expression(expression)


@pytest.mark.parametrize('first_reply, failed', [
    (CALL, False),
    ('<tool_call>{"name":"calculate_math","arguments":{"expression":"123 \\* 321"}}</tool_call>', True),
])
def test_webui_tool_details_are_collapsed_and_final_reply_survives_rerun(first_reply, failed):
    from types import SimpleNamespace
    from streamlit.testing.v1 import AppTest
    from transformers import AutoTokenizer
    from model.model_instinct import InstinctConfig

    tokenizer = AutoTokenizer.from_pretrained('model', trust_remote_code=True)
    replies = iter([first_reply, '123 × 321 = 39483。'])
    prompts = []

    def generate(input_ids, streamer, **kwargs):
        prompts.append(tokenizer.decode(input_ids[0].cpu(), skip_special_tokens=False))
        streamer.put(input_ids.cpu())
        reply = next(replies) + '<|im_end|>'
        streamer.put(torch.tensor(tokenizer.encode(reply, add_special_tokens=False)))
        streamer.end()

    app = AppTest.from_file('scripts/web_demo.py', default_timeout=60).run()
    app.session_state.model = SimpleNamespace(config=InstinctConfig(), generate=generate)
    app.session_state.tokenizer = tokenizer
    app.session_state.model_loaded = True
    app.session_state.loaded_weight_path = 'test.pth'
    app.session_state.loaded_config_path = 'test.json'
    app.run()
    app.checkbox(key='tool_calculate_math').check().run()
    app.chat_input[0].set_value('123 * 321 是多少？').run()
    assert not app.exception, [item.message for item in app.exception]
    message = app.session_state.messages[-1]
    assert message['content'] == '123 × 321 = 39483。'
    assert '<tool_call>' not in message['content'] and 'ToolCalled' not in message['content']
    assert len(message['tool_events']) == 1
    assert ('error' in message['tool_events'][0]) is failed
    if not failed:
        assert message['tool_events'][0]['result'] == {'result': 39483}
        assert '39483' in prompts[1]
    else:
        assert 'error' in prompts[1]
    assert 'ToolCalled' not in prompts[1]
    status = 'Failed' if failed else 'Completed'
    cards = [item for item in app.expander if status in item.label]
    assert len(cards) == 1 and not cards[0].proto.expanded
    assert 'generation_task' not in app.session_state.filtered_state
    app.run()
    assert not app.exception
    assert app.session_state.messages[-1] == message
    assert len([item for item in app.expander if status in item.label]) == 1


@pytest.mark.parametrize('multiple_rounds', [False, True])
def test_webui_renders_tool_cards_at_call_positions_live_and_after_refresh(multiple_rounds):
    from types import SimpleNamespace
    from streamlit.testing.v1 import AppTest
    from transformers import AutoTokenizer
    from model.model_instinct import InstinctConfig

    tokenizer = AutoTokenizer.from_pretrained('model', trust_remote_code=True)
    responses = ['调用前的说明。\n' + CALL + '\n调用后的说明。']
    if multiple_rounds:
        responses.append('第二次调用前。\n' + CALL + '\n第二次调用后。')
    responses.append('最终答案是 39483。')
    replies = iter(responses)

    def generate(input_ids, streamer, **kwargs):
        streamer.put(input_ids.cpu())
        reply = next(replies) + '<|im_end|>'
        streamer.put(torch.tensor(tokenizer.encode(reply, add_special_tokens=False)))
        streamer.end()

    app = AppTest.from_file('scripts/web_demo.py', default_timeout=60).run()
    app.session_state.model = SimpleNamespace(config=InstinctConfig(), generate=generate)
    app.session_state.tokenizer = tokenizer
    app.session_state.model_loaded = True
    app.session_state.loaded_weight_path = 'test.pth'
    app.session_state.loaded_config_path = 'test.json'
    app.run()
    app.checkbox(key='tool_calculate_math').check().run()
    app.chat_input[0].set_value('请计算。').run()
    assert not app.exception, [item.message for item in app.exception]

    def assert_order():
        order = []
        for node in app.main:
            if node.type == 'expander' and 'Completed' in node.label:
                order.append('tool')
                assert not node.proto.expanded
            elif node.type == 'html':
                for phrase in ('调用前的说明。', '调用后的说明。', '第二次调用前。',
                               '第二次调用后。', '最终答案是 39483。'):
                    if phrase in node.proto.body:
                        order.append(phrase)
        expected = ['调用前的说明。', 'tool', '调用后的说明。']
        if multiple_rounds:
            expected += ['第二次调用前。', 'tool', '第二次调用后。']
        expected.append('最终答案是 39483。')
        assert order == expected

    assert_order()
    message = app.session_state.messages[-1]
    assert all(event['position'] > 0 for event in message['tool_events'])
    assert '<tool_call>' not in message['content']
    app.run()
    assert not app.exception
    assert_order()
    assert app.session_state.messages[-1] == message


def test_webui_recognizes_prefilled_thinking_and_keeps_labels_after_refresh():
    from types import SimpleNamespace
    from streamlit.testing.v1 import AppTest
    from transformers import AutoTokenizer
    from model.model_instinct import InstinctConfig

    tokenizer = AutoTokenizer.from_pretrained('model', trust_remote_code=True)
    prompts = []

    def generate(input_ids, streamer, **kwargs):
        prompts.append(tokenizer.decode(input_ids[0].cpu(), skip_special_tokens=False))
        streamer.put(input_ids.cpu())
        # <think> was supplied by the prompt, so generate only emits its body.
        reply = '先分析条件。\n</think>\n这是最终回答。<|im_end|>'
        streamer.put(torch.tensor(tokenizer.encode(reply, add_special_tokens=False)))
        streamer.end()

    app = AppTest.from_file('scripts/web_demo.py', default_timeout=60).run()
    app.session_state.model = SimpleNamespace(config=InstinctConfig(), generate=generate)
    app.session_state.tokenizer = tokenizer
    app.session_state.model_loaded = True
    app.session_state.loaded_weight_path = 'test.pth'
    app.session_state.loaded_config_path = 'test.json'
    app.run()
    app.checkbox[0].check().run()  # thinking toggle precedes logit lens/tool toggles
    app.chat_input[0].set_value('你好').run()
    assert not app.exception, [item.message for item in app.exception]
    assert prompts[0].rstrip().endswith('<think>')
    assert app.session_state.messages[-1]['content'].startswith('<think>')

    def assert_labels():
        thoughts = [item for item in app.expander if '思考过程' in item.label]
        assert len(thoughts) == 1
        assert thoughts[0].label.endswith('思考过程 · 已结束')
        assert 'data:image/svg+xml;base64,' in thoughts[0].label
        assert not thoughts[0].proto.expanded
        assert any('先分析条件。' in item.proto.body for item in thoughts[0].get('html'))
        assert any(item.value == '正文' for item in app.caption)
        assert not any('这是最终回答。' in item.proto.body for item in thoughts[0].get('html'))

    assert_labels()
    app.run()
    assert not app.exception
    assert_labels()
