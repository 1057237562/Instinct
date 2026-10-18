"""Text and function-call Responses API adapter for the local inference server."""

from collections import OrderedDict
from copy import deepcopy
import json
from threading import Lock
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


class ResponseRequest(BaseModel):
    model: str
    input: str | list[dict]
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=0.7, ge=0, le=2)
    top_p: float | None = Field(default=0.92, gt=0, le=1)
    stream: bool = False
    tools: list[dict] = Field(default_factory=list)
    tool_choice: str | dict = "auto"
    previous_response_id: str | None = None
    store: bool = True
    metadata: dict = Field(default_factory=dict)
    reasoning: dict | None = None
    text: dict | None = None
    background: bool = False
    open_thinking: bool = False
    chat_template_kwargs: dict | None = None


def input_messages(items):
    """Translate Responses input items into the model's chat template format."""
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    messages = []
    for item in items:
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            if role not in {"user", "assistant", "system", "developer"}:
                raise HTTPException(400, f"Unsupported message role: {role}")
            content = item.get("content", "")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict) or part.get("type") not in {"input_text", "output_text"} or not isinstance(part.get("text"), str):
                        raise HTTPException(400, "Only text input/output content is supported")
                    parts.append(part["text"])
                content = "".join(parts)
            if not isinstance(content, str):
                raise HTTPException(400, "Message content must be text or text parts")
            messages.append({"role": "system" if role == "developer" else role, "content": content})
        elif kind == "function_call":
            if not all(isinstance(item.get(key), str) for key in ("call_id", "name", "arguments")):
                raise HTTPException(400, "Function calls require call_id, name and arguments strings")
            messages.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": item["call_id"], "type": "function",
                "function": {"name": item["name"], "arguments": item["arguments"]},
            }]})
        elif kind == "function_call_output":
            if not isinstance(item.get("call_id"), str):
                raise HTTPException(400, "Function results require call_id")
            output = item.get("output", "")
            if isinstance(output, list):
                output = input_messages([{"role": "user", "content": output}])[0]["content"]
            if not isinstance(output, str):
                raise HTTPException(400, "Function output must be text")
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": output})
        elif kind == "reasoning":
            # Reasoning items returned by this server can be replayed with output.
            summary = item.get("summary", [])
            if not isinstance(summary, list) or any(not isinstance(part, dict) or not isinstance(part.get("text"), str) for part in summary):
                raise HTTPException(400, "Invalid reasoning summary")
            if summary:
                messages.append({"role": "assistant", "content": "", "reasoning_content": "".join(part["text"] for part in summary)})
        else:
            raise HTTPException(400, f"Unsupported Responses input item: {kind}")
    return messages


def function_tools(request):
    if request.tool_choice == "none":
        return []
    if request.tool_choice != "auto":
        raise HTTPException(400, "Only auto/none tool_choice is supported")
    tools = []
    for tool in request.tools:
        if tool.get("type") != "function" or not isinstance(tool.get("name"), str):
            raise HTTPException(400, "Only named function tools are supported")
        tools.append({"type": "function", "function": {
            key: tool[key] for key in ("name", "description", "parameters", "strict") if key in tool
        }})
    return tools


def response_object(request):
    return {
        "id": f"resp_{uuid.uuid4().hex}", "object": "response", "created_at": int(time.time()),
        "status": "in_progress", "error": None, "incomplete_details": None,
        "model": request.model, "output": [], "usage": None,
        "instructions": request.instructions, "max_output_tokens": request.max_output_tokens,
        "previous_response_id": request.previous_response_id, "store": request.store,
        "metadata": request.metadata, "temperature": request.temperature, "top_p": request.top_p,
        "tools": request.tools, "tool_choice": request.tool_choice, "parallel_tool_calls": False,
        "reasoning": request.reasoning, "text": {"format": {"type": "text"}},
    }


def response_usage(usage):
    if usage is None:
        return None
    return {
        "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"], "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


def output_items(message):
    items = []
    if message.get("reasoning_content"):
        items.append({"id": f"rs_{uuid.uuid4().hex}", "type": "reasoning", "status": "completed",
                      "summary": [{"type": "summary_text", "text": message["reasoning_content"]}]})
    if message.get("content") or not message.get("tool_calls"):
        items.append({"id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant", "status": "completed",
                      "content": [{"type": "output_text", "text": message.get("content", ""), "annotations": [], "logprobs": []}]})
    for call in message.get("tool_calls", []):
        items.append({"id": f"fc_{uuid.uuid4().hex}", "type": "function_call", "status": "completed",
                      "call_id": call["id"], "name": call["function"]["name"], "arguments": call["function"]["arguments"]})
    return items


class ResponseStore:
    """Bounded, process-local history; instructions are scoped to each request."""
    def __init__(self, limit=128):
        self.limit = limit
        self.records = OrderedDict()
        self.lock = Lock()

    def get(self, response_id):
        with self.lock:
            if response_id not in self.records:
                raise HTTPException(404, "Response not found (expired, not stored, or server restarted)")
            return deepcopy(self.records[response_id])

    def save(self, request, response, conversation):
        if request.store:
            with self.lock:
                self.records[response["id"]] = deepcopy((response, conversation + input_messages(response["output"])))
                while len(self.records) > self.limit:
                    self.records.popitem(last=False)


def stream_response(chat_request, response, generate, on_complete):
    """Emit Responses lifecycle events while the model produces chat deltas."""
    sequence = 0

    def event(kind, **data):
        nonlocal sequence
        payload = {"type": kind, "sequence_number": sequence, **data}
        sequence += 1
        return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def finish_item(index):
        item = response["output"][index]
        if item["status"] == "completed":
            return
        item["status"] = "completed"
        base = {"item_id": item["id"], "output_index": index}
        if item["type"] == "message":
            part = item["content"][0]
            yield event("response.output_text.done", **base, content_index=0, text=part["text"], logprobs=[])
            yield event("response.content_part.done", **base, content_index=0, part=part)
        elif item["type"] == "reasoning":
            part = item["summary"][0]
            yield event("response.reasoning_summary_text.done", **base, summary_index=0, text=part["text"])
            yield event("response.reasoning_summary_part.done", **base, summary_index=0, part=part)
        else:
            yield event("response.function_call_arguments.done", **base, name=item["name"], arguments=item["arguments"])
        yield event("response.output_item.done", output_index=index, item=item)

    yield event("response.created", response=response)
    yield event("response.in_progress", response=response)
    text_index = reasoning_index = None
    try:
        for chunk in generate(
            messages=chat_request.messages, temperature=chat_request.temperature, top_p=chat_request.top_p,
            max_tokens=chat_request.max_tokens, tools=chat_request.tools, open_thinking=chat_request.get_open_thinking(),
            request_started=chat_request._request_started,
        ):
            payload = json.loads(chunk)
            if "error" in payload:
                raise RuntimeError(payload["error"])
            if "usage" in payload:
                response["usage"] = response_usage(payload["usage"])
            if "performance" in payload:
                response["performance"] = payload["performance"]
            for choice in payload.get("choices", []):
                delta = choice.get("delta", {})
                for key in ("reasoning_content", "content"):
                    value = delta.get(key)
                    if not value:
                        continue
                    is_reasoning = key == "reasoning_content"
                    index = reasoning_index if is_reasoning else text_index
                    if not is_reasoning and reasoning_index is not None:
                        yield from finish_item(reasoning_index)
                    if index is None:
                        index = len(response["output"])
                        if is_reasoning:
                            reasoning_index = index
                            item = {"id": f"rs_{uuid.uuid4().hex}", "type": "reasoning", "status": "in_progress", "summary": []}
                        else:
                            text_index = index
                            item = {"id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant", "status": "in_progress", "content": []}
                        response["output"].append(item)
                        yield event("response.output_item.added", output_index=index, item=item)
                        part = {"type": "summary_text", "text": ""} if is_reasoning else {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
                        item["summary" if is_reasoning else "content"].append(part)
                        if is_reasoning:
                            yield event("response.reasoning_summary_part.added", item_id=item["id"], output_index=index, summary_index=0, part=part)
                        else:
                            yield event("response.content_part.added", item_id=item["id"], output_index=index, content_index=0, part=part)
                    item = response["output"][index]
                    item["summary" if is_reasoning else "content"][0]["text"] += value
                    if is_reasoning:
                        yield event("response.reasoning_summary_text.delta", item_id=item["id"], output_index=index, summary_index=0, delta=value)
                    else:
                        yield event("response.output_text.delta", item_id=item["id"], output_index=index, content_index=0, delta=value, logprobs=[])
                for call in delta.get("tool_calls", []):
                    for index in range(len(response["output"])):
                        yield from finish_item(index)
                    index = len(response["output"])
                    item = {"id": f"fc_{uuid.uuid4().hex}", "type": "function_call", "status": "in_progress",
                            "call_id": call["id"], "name": call["function"]["name"], "arguments": ""}
                    response["output"].append(item)
                    yield event("response.output_item.added", output_index=index, item=item)
                    item["arguments"] = call["function"]["arguments"]
                    yield event("response.function_call_arguments.delta", item_id=item["id"], output_index=index, delta=item["arguments"])
                if choice.get("finish_reason") == "length":
                    response["incomplete_details"] = {"reason": "max_output_tokens"}
        for index in range(len(response["output"])):
            yield from finish_item(index)
        response["status"] = "incomplete" if response["incomplete_details"] else "completed"
        on_complete(response)
        yield event(f"response.{response['status']}", response=response)
    except Exception as exc:
        response["status"] = "failed"
        response["error"] = {"code": "server_error", "message": str(exc)}
        yield event("response.failed", response=response)


def create_responses_router(chat_request_type, complete, generate):
    router = APIRouter()
    store = ResponseStore()

    @router.post("/v1/responses")
    @router.post("/responses", include_in_schema=False)
    @router.post("/v1/response", include_in_schema=False)
    @router.post("/response", include_in_schema=False)
    async def create_response(request: ResponseRequest):
        try:
            started = time.perf_counter()
            if request.background:
                raise HTTPException(400, "Background generation is not supported")
            if request.text:
                text_format = request.text.get("format") or {}
                if not isinstance(text_format, dict) or text_format.get("type", "text") != "text":
                    raise HTTPException(400, "Only plain text output is supported")
            conversation = store.get(request.previous_response_id)[1] if request.previous_response_id else []
            conversation += input_messages(request.input)
            if not conversation:
                raise HTTPException(400, "input must contain at least one message")
            messages = ([{"role": "system", "content": request.instructions}] if request.instructions else []) + conversation
            reasoning_effort = (request.reasoning or {}).get("effort")
            chat_request = chat_request_type(
                model=request.model, messages=messages, tools=function_tools(request),
                max_tokens=request.max_output_tokens or 8192, temperature=request.temperature if request.temperature is not None else 0.7,
                top_p=request.top_p if request.top_p is not None else 0.92,
                open_thinking=request.open_thinking or reasoning_effort not in (None, "none"),
                chat_template_kwargs=request.chat_template_kwargs, stream=False,
            )
            chat_request._request_started = started
            response = response_object(request)
            if request.stream:
                return StreamingResponse(
                    stream_response(chat_request, response, generate, lambda final: store.save(request, final, conversation)),
                    media_type="text/event-stream", headers={"Cache-Control": "no-cache"},
                )
            completion = await complete(chat_request)
            response["output"] = output_items(completion["choices"][0]["message"])
            response["usage"] = response_usage(completion.get("usage"))
            if "performance" in completion:
                response["performance"] = completion["performance"]
            response["status"] = "completed"
            if completion["choices"][0]["finish_reason"] == "length":
                response["status"] = "incomplete"
                response["incomplete_details"] = {"reason": "max_output_tokens"}
            store.save(request, response, conversation)
            return response
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"error": {
                "message": str(exc.detail), "type": "invalid_request_error" if exc.status_code < 500 else "server_error", "code": None,
            }})

    @router.get("/v1/responses/{response_id}")
    def retrieve_response(response_id: str):
        return store.get(response_id)[0]

    return router
