import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from scripts import serve_openai_api as api


@pytest.fixture
def client(monkeypatch):
    class Inputs(dict):
        def to(self, device):
            return self

        def __getattr__(self, name):
            return self[name]

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, text, **kwargs):
            return Inputs(input_ids=torch.tensor([[1, 3]]), attention_mask=torch.ones(1, 2))

        def decode(self, tokens, **kwargs):
            return "Hello!"

    def generate(input_ids, **kwargs):
        if "streamer" in kwargs:
            kwargs["streamer"].on_finalized_text("Hello!", stream_end=True)
        return torch.tensor([[1, 3, 4, 2]])

    monkeypatch.setattr(api, "tokenizer", Tokenizer(), raising=False)
    monkeypatch.setattr(api, "model", SimpleNamespace(generate=generate), raising=False)
    monkeypatch.setattr(api, "device", "cpu", raising=False)
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.mark.parametrize("stream", [None, False])
def test_nonstream_request_returns_json(client, stream):
    # Cherry Studio's connection test omits stream and expects ordinary JSON.
    body = {"model": "instinct", "messages": [{"role": "system", "content": "test"}, {"role": "user", "content": "hi"}]}
    if stream is not None:
        body["stream"] = stream
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "instinct"
    assert payload["choices"][0]["message"] == {"role": "assistant", "content": "Hello!"}
    assert payload["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}


def test_stream_returns_complete_openai_events(client):
    response = client.post("/v1/chat/completions", json={
        "model": "instinct", "messages": [{"role": "user", "content": "hi"}], "stream": True,
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = response.text.strip().split("\n\n")
    assert frames[-1] == "data: [DONE]"
    chunks = [json.loads(frame.removeprefix("data: ")) for frame in frames[:-1]]
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert len({chunk["created"] for chunk in chunks}) == 1
    for chunk in chunks:
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "instinct"
        assert chunk["choices"][0]["index"] == 0
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks) == "Hello!"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_stream_tool_calls_have_indices(client, monkeypatch):
    monkeypatch.setattr(api, "generate_stream_response", lambda **kwargs: iter([
        json.dumps({"choices": [{"delta": {"tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
        ]}}]}),
        json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
    ]))
    response = client.post("/v1/chat/completions", json={
        "model": "instinct", "messages": [{"role": "user", "content": "hi"}], "stream": True,
    })
    chunks = [json.loads(frame[6:]) for frame in response.text.strip().split("\n\n")[:-1]]
    assert chunks[1]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("path", ["/v1/responses", "/responses", "/v1/response", "/response"])
def test_responses_json_and_aliases(client, path):
    response = client.post(path, json={"model": "instinct", "input": "hi", "store": False})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    payload = response.json()
    assert payload["id"].startswith("resp_")
    assert payload["object"] == "response"
    assert payload["status"] == "completed"
    assert payload["output"][0]["content"][0]["text"] == "Hello!"
    assert payload["usage"]["input_tokens"] == 2
    assert payload["usage"]["output_tokens"] == 2


def response_events(response):
    assert response.headers["content-type"].startswith("text/event-stream")
    events = []
    for frame in response.text.strip().split("\n\n"):
        name, data = frame.split("\n", 1)
        payload = json.loads(data.removeprefix("data: "))
        assert name == f"event: {payload['type']}"
        events.append(payload)
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    return events


def test_responses_stream_lifecycle_and_usage(client):
    events = response_events(client.post("/v1/responses", json={"model": "instinct", "input": "hi", "stream": True}))
    assert events[0]["type"] == "response.created"
    assert events[0]["response"]["output"] == []
    assert events[1]["type"] == "response.in_progress"
    assert events[-1]["type"] == "response.completed"
    assert "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta") == "Hello!"
    final = events[-1]["response"]
    assert final["id"] == events[0]["response"]["id"]
    assert final["output"][0]["status"] == "completed"
    assert final["usage"]["output_tokens"] == 2
    assert client.get(f"/v1/responses/{final['id']}").json() == final


def test_responses_input_instructions_and_previous_id(client, monkeypatch):
    prompts = []
    monkeypatch.setattr(api.tokenizer, "apply_chat_template", lambda messages, **kwargs: prompts.append(messages) or "prompt")
    first = client.post("/v1/responses", json={"model": "instinct", "instructions": "old instructions", "input": "hi"}).json()
    response = client.post("/v1/responses", json={
        "model": "instinct", "instructions": "new instructions", "previous_response_id": first["id"],
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "again"}]}],
    })
    assert response.status_code == 200
    assert prompts[-1] == [
        {"role": "system", "content": "new instructions"}, {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "again"},
    ]


def test_responses_store_false_and_missing_history(client):
    result = client.post("/v1/responses", json={"model": "instinct", "input": "hi", "store": False}).json()
    assert client.get(f"/v1/responses/{result['id']}").status_code == 404
    response = client.post("/v1/responses", json={"model": "instinct", "input": "again", "previous_response_id": result["id"]})
    assert response.status_code == 404
    assert "error" in response.json()


@pytest.mark.parametrize("stream", [False, True])
def test_responses_function_call_and_tool_result(client, monkeypatch, stream):
    tool_text = '<tool_call>{"name":"lookup","arguments":{"query":"hi"}}</tool_call>'
    monkeypatch.setattr(api.tokenizer, "decode", lambda *args, **kwargs: tool_text)

    def generate(input_ids, **kwargs):
        if "streamer" in kwargs:
            for character in tool_text:
                kwargs["streamer"].on_finalized_text(character)
            kwargs["streamer"].on_finalized_text("", stream_end=True)
        return torch.tensor([[1, 3, 4, 2]])

    monkeypatch.setattr(api.model, "generate", generate)
    response = client.post("/v1/responses", json={
        "model": "instinct", "input": "hi", "stream": stream,
        "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
    })
    if stream:
        events = response_events(response)
        assert not any(event["type"] == "response.output_text.delta" for event in events)
        assert any(event["type"] == "response.function_call_arguments.done" for event in events)
        result = events[-1]["response"]
    else:
        result = response.json()
    assert result["output"][0]["type"] == "function_call"
    call = result["output"][0]
    assert call["name"] == "lookup"
    assert json.loads(call["arguments"]) == {"query": "hi"}
    prompts = []
    monkeypatch.setattr(api.tokenizer, "apply_chat_template", lambda messages, **kwargs: prompts.append(messages) or "prompt")
    followup = client.post("/v1/responses", json={
        "model": "instinct", "previous_response_id": result["id"],
        "input": [{"type": "function_call_output", "call_id": call["call_id"], "output": "found"}],
    })
    assert followup.status_code == 200
    assert prompts[-1][-1] == {"role": "tool", "tool_call_id": call["call_id"], "content": "found"}


def test_responses_reasoning_stream_and_fragmented_markers(client, monkeypatch):
    def generate(input_ids, **kwargs):
        if "streamer" in kwargs:
            for character in 'Let me think.</think>Hello!':
                kwargs["streamer"].on_finalized_text(character)
            kwargs["streamer"].on_finalized_text("", stream_end=True)
        return torch.tensor([[1, 3, 4, 2]])
    monkeypatch.setattr(api.model, "generate", generate)
    events = response_events(client.post("/v1/responses", json={
        "model": "instinct", "input": "hi", "stream": True, "reasoning": {"effort": "medium"},
    }))
    assert "".join(event["delta"] for event in events if event["type"] == "response.reasoning_summary_text.delta") == "Let me think."
    assert "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta") == "Hello!"
    assert [item["type"] for item in events[-1]["response"]["output"]] == ["reasoning", "message"]


def test_responses_stream_failure_has_no_completed_event(client, monkeypatch):
    monkeypatch.setattr(api, "generate_stream_response", lambda **kwargs: iter([json.dumps({"error": "test failure"})]))
    events = response_events(client.post("/v1/responses", json={"model": "instinct", "input": "hi", "stream": True}))
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"]["message"] == "test failure"
    assert not any(event["type"] == "response.completed" for event in events)


@pytest.mark.parametrize("extra", [
    {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "example"}]}]},
    {"tools": [{"type": "web_search"}]}, {"background": True},
    {"text": {"format": {"type": "json_schema"}}}, {"tool_choice": "required"},
])
def test_responses_rejects_unsupported_features(client, extra):
    response = client.post("/v1/responses", json={"model": "instinct", "input": "hi", **extra})
    assert response.status_code == 400
    assert "error" in response.json()


def test_responses_openai_sdk_json_and_stream(client):
    openai = pytest.importorskip("openai")
    if not hasattr(openai.OpenAI, "responses"):
        pytest.skip("Installed OpenAI SDK predates Responses")
    sdk = openai.OpenAI(base_url="http://testserver/v1", api_key="local", http_client=client)
    response = sdk.responses.create(model="instinct", input="hi", store=False)
    assert response.output_text == "Hello!"
    with sdk.responses.stream(model="instinct", input="hi", store=False) as stream:
        final = stream.get_final_response()
    assert final.output_text == "Hello!"
    assert final.usage.output_tokens == 2


@pytest.mark.parametrize("stream", [False, True])
def test_responses_output_limit_and_greedy_sampling(client, monkeypatch, stream):
    calls = []
    def generate(input_ids, **kwargs):
        calls.append(kwargs)
        if "streamer" in kwargs:
            kwargs["streamer"].on_finalized_text("Hello!", stream_end=True)
        return torch.tensor([[1, 3, 4]])  # Budget exhausted without EOS.
    monkeypatch.setattr(api.model, "generate", generate)
    response = client.post("/v1/responses", json={
        "model": "instinct", "input": "hi", "max_output_tokens": 1, "temperature": 0, "stream": stream,
    })
    result = response_events(response)[-1]["response"] if stream else response.json()
    assert result["status"] == "incomplete"
    assert result["incomplete_details"] == {"reason": "max_output_tokens"}
    assert calls[0]["max_new_tokens"] == 1
    assert calls[0]["do_sample"] is False


@pytest.mark.parametrize("kind", ["reasoning", "function_call"])
def test_responses_sdk_stream_accumulates_reasoning_and_tools(client, monkeypatch, kind):
    openai = pytest.importorskip("openai")
    if not hasattr(openai.OpenAI, "responses"):
        pytest.skip("Installed OpenAI SDK predates Responses")
    text = 'Thinking.</think>Hello!' if kind == "reasoning" else '<tool_call>{"name":"lookup","arguments":{"query":"hi"}}</tool_call>'
    def generate(input_ids, **kwargs):
        for character in text:
            kwargs["streamer"].on_finalized_text(character)
        kwargs["streamer"].on_finalized_text("", stream_end=True)
        return torch.tensor([[1, 3, 4, 2]])
    monkeypatch.setattr(api.model, "generate", generate)
    sdk = openai.OpenAI(base_url="http://testserver/v1", api_key="local", http_client=client, max_retries=0)
    options = {"reasoning": {"effort": "medium"}} if kind == "reasoning" else {"tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]}
    with sdk.responses.stream(model="instinct", input="hi", store=False, **options) as stream:
        final = stream.get_final_response()
    assert final.output[0].type == kind
    if kind == "reasoning":
        assert final.output[0].summary[0].text == "Thinking."
        assert final.output_text == "Hello!"
    else:
        assert final.output[0].name == "lookup"
        assert json.loads(final.output[0].arguments) == {"query": "hi"}


@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_performance_transmitted_with_usage_and_sdk_compatibility(client, monkeypatch, protocol, stream):
    calls = []
    def generate(input_ids, **kwargs):
        # Token callbacks happen in both JSON and SSE modes. Text length is unrelated to token count.
        assert api.generation_lock.locked()
        assert torch.is_inference_mode_enabled()
        tracker = kwargs['streamer']
        calls.append(tracker)
        tracker.put(input_ids)
        tracker.put(torch.tensor([[4]]))
        tracker.put(torch.tensor([[2]]))
        tracker.end()
        return torch.tensor([[1, 3, 4, 2]])

    # Bypass text decoding in the delegate so this test does not need a real tokenizer.
    monkeypatch.setattr(api.CustomStreamer, "put", lambda self, tokens: None)
    monkeypatch.setattr(api.CustomStreamer, "end", lambda self: self.on_finalized_text("Hello!", stream_end=True))
    monkeypatch.setattr(api.model, "generate", generate)
    if protocol == 'chat':
        response = client.post('/v1/chat/completions', json={'model': 'instinct', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': stream})
        result = json.loads(response.text.strip().split('\n\n')[-2][6:]) if stream else response.json()
        assert result['usage']['completion_tokens'] == 2
    else:
        response = client.post('/v1/responses', json={'model': 'instinct', 'input': 'hi', 'stream': stream})
        result = response_events(response)[-1]['response'] if stream else response.json()
        assert result['usage']['output_tokens'] == 2
        assert client.get(f"/v1/responses/{result['id']}").json()['performance'] == result['performance']
    assert response.status_code == 200
    stats = result['performance']
    assert stats['output_tokens'] == 2
    assert stats['timing_source'] == 'token_delivery'
    assert stats['ttft_ms'] >= stats['generation_ttft_ms'] >= 0
    assert stats['total_ms'] >= stats['generation_ms'] > 0
    assert stats['tokens_per_second'] > 0
    assert stats['tpot_ms'] * stats['tokens_per_second'] == pytest.approx(1000)
    assert calls[0].flush_first_token is True


def test_openai_sdk_keeps_performance_extension(client, monkeypatch):
    openai = pytest.importorskip('openai')
    sdk = openai.OpenAI(base_url='http://testserver/v1', api_key='local', http_client=client, max_retries=0)
    chat = sdk.chat.completions.create(model='instinct', messages=[{'role': 'user', 'content': 'hi'}])
    assert chat.model_extra['performance']['output_tokens'] == 2
    chunks = list(sdk.chat.completions.create(model='instinct', messages=[{'role': 'user', 'content': 'hi'}], stream=True))
    assert chunks[-1].model_extra['performance']['output_tokens'] == 2
    assert chunks[-1].usage.completion_tokens == 2
    if hasattr(openai.OpenAI, 'responses'):
        with sdk.responses.stream(model='instinct', input='hi') as stream:
            final = stream.get_final_response()
        assert final.model_extra['performance']['output_tokens'] == 2


@pytest.mark.parametrize('device,dtype,mode,expected_dtype,warmed', [
    ('cuda', 'auto', None, torch.bfloat16, True),
    ('cuda', 'fp16', 'off', torch.float16, False),
    ('cpu', 'auto', None, torch.float16, False),
])
def test_api_loader_uses_webui_inference_runtime(tmp_path, monkeypatch, device, dtype, mode, expected_dtype, warmed):
    from unittest.mock import Mock
    config_path = tmp_path / 'best.json'
    config_path.write_text(json.dumps({'hidden_size': 512, 'use_moe': True}), encoding='utf-8')
    loaded = Mock()
    loaded.config = SimpleNamespace(hidden_size=512, use_moe=True)
    loaded.parameters.return_value = [torch.zeros(1)]
    loaded.to.return_value = loaded
    loaded.eval.return_value = loaded
    loaded._static_cache_ok = device.startswith('cuda') and mode != 'off'
    weights = Mock(return_value={})
    optimize = Mock(return_value=loaded)
    warmup = Mock()
    selected = Mock(return_value=torch.bfloat16 if device.startswith('cuda') else torch.float16)
    monkeypatch.setattr(api, 'InstinctForCausalLM', Mock(return_value=loaded))
    monkeypatch.setattr(api.AutoTokenizer, 'from_pretrained', Mock(return_value='tokenizer'))
    monkeypatch.setattr(api.torch, 'load', weights)
    monkeypatch.setattr(api, 'optimize_inference', optimize)
    monkeypatch.setattr(api, 'warmup_decode', warmup)
    monkeypatch.setattr(api, 'select_inference_dtype', selected)
    monkeypatch.delenv('INSTINCT_INFERENCE_COMPILE', raising=False)
    args = SimpleNamespace(load_from='../model', config_path=str(config_path), weight_path='../out/best.pth',
                           device=device, dtype=dtype, inference_compile=mode, lora_weight='None')
    result, tokenizer = api.init_model(args)
    assert result is loaded
    assert tokenizer == 'tokenizer'
    weights.assert_called_once_with('../out/best.pth', map_location='cpu', weights_only=True)
    loaded.to.assert_any_call(dtype=expected_dtype)
    loaded.to.assert_any_call(device)
    optimize.assert_called_once_with(loaded, mode or 'auto')
    assert warmup.called is warmed
