"""Parse chat tool protocol separately from the visible assistant answer."""

import json
import re
import ast
import operator

from transformers import TextIteratorStreamer


class ChatTextIteratorStreamer(TextIteratorStreamer):
    """Keep thinking/tool delimiters for the UI's protocol parser."""

    def __init__(self, tokenizer, **kwargs):
        kwargs['skip_special_tokens'] = False
        super().__init__(tokenizer, **kwargs)


_TOKENS = re.compile(
    r'`+|~{3,}|</?think>|\\?</?tool_call>|<\|im_(?:start|end)\|>|<\|endoftext\|>'
)
_MARKERS = ('<|im_start|>', '<|im_end|>', '<|endoftext|>')


def has_matching_backtick_run(content, start, length):
    """An inline span opens only if an equal-length run follows on the same line.

    A backtick string with no closer on its own line (``` quoted mid-sentence)
    is literal text; letting it open a span would wedge the state until a much
    later fence swallowed the </think>/<|im_end|> handling after it.
    """
    line_end = content.find('\n', start)
    if line_end < 0:
        line_end = len(content)
    pos = start
    while pos < line_end:
        if content[pos] == '`':
            end = pos + 1
            while end < line_end and content[end] == '`':
                end += 1
            if end - pos == length:
                return True
            pos = end
        else:
            pos += 1
    return False


def restore_thinking_prefix(content, initial_thinking=False):
    """Persist the opening tag supplied by the prompt rather than generated."""
    if initial_thinking and not content.lstrip().startswith('<think>'):
        return '<think>\n' + content
    return content


def _inside_json_string(text):
    quoted = escaped = False
    for char in text:
        if escaped:
            escaped = False
        elif quoted and char == '\\':
            escaped = True
        elif char == '"':
            quoted = not quoted
    return quoted


def _parse_call(raw):
    event = {'raw': raw, 'name': '', 'arguments': {}}
    try:
        call = json.loads(raw.strip())
        if not isinstance(call, dict):
            raise ValueError('Tool call must be a JSON object')
        if not isinstance(call.get('name'), str) or not call['name'].strip():
            raise ValueError('Tool call requires a non-empty name')
        event['name'] = call['name']
        args = call.get('arguments', {})
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict):
            raise ValueError('Tool arguments must be a JSON object')
        event['arguments'] = args
    except (ValueError, TypeError) as exc:
        event['error'] = str(exc)
    return event


def split_tool_calls(content, *, streaming=False):
    """Hide protocol blocks, preserving fenced/inline examples and thinking.

    Markdown-escaped tool delimiters are accepted, while JSON itself remains
    strict. Malformed calls become explicit errors rather than silent no-ops.
    """
    visible, calls = [], []
    cursor = 0
    opened = None
    opened_position = 0
    fence = None
    inline = 0
    thinking = False
    for match in _TOKENS.finditer(content):
        token = match.group().lstrip('\\')
        if opened is not None:
            if token == '</tool_call>' and not _inside_json_string(content[opened:match.start()]):
                call = _parse_call(content[opened:match.start()])
                call['position'] = opened_position
                calls.append(call)
                opened = None
                cursor = match.end()
            continue
        prefix = content[content.rfind('\n', 0, match.start()) + 1:match.start()]
        prefix = prefix.replace('<think>', '').replace('</think>', '')
        is_fence = token[0] in '`~' and len(token) >= 3 and len(prefix) <= 3 and not prefix.strip()
        if fence:
            if is_fence and token[0] == fence[0] and len(token) >= len(fence):
                fence = None
                inline = 0
            continue
        if is_fence:
            fence = token
            inline = 0
            continue
        if token[0] == '`':
            if inline == len(token):
                inline = 0
            elif not inline and has_matching_backtick_run(content, match.end(), len(token)):
                inline = len(token)
            continue
        if inline:
            continue
        if token in ('<think>', '</think>'):
            thinking = token == '<think>'
        elif token in _MARKERS:
            visible.append(content[cursor:match.start()])
            cursor = match.end()
        elif token == '<tool_call>' and not thinking:
            visible.append(content[cursor:match.start()])
            opened_position = sum(map(len, visible))
            opened = match.end()
            cursor = match.end()
    if opened is not None:
        if not streaming:
            calls.append({'raw': content[opened:], 'name': '', 'arguments': {},
                          'position': opened_position,
                          'error': 'Tool call is missing </tool_call>'})
    else:
        tail = content[cursor:]
        if streaming and not fence and not inline and not thinking:
            # A tag may arrive over multiple token blocks; hide its partial tail.
            tags = ('<tool_call>', '\\<tool_call>', *_MARKERS)
            for length in range(min(len(tail), max(map(len, tags))), 0, -1):
                if any(tag.startswith(tail[-length:]) for tag in tags):
                    tail = tail[:-length]
                    break
        visible.append(tail)
    return ''.join(visible), calls


def ordered_response_parts(content, events):
    """Interleave visible text and tool cards at saved character offsets."""
    cursor = 0
    parts = []
    for event in events:
        # Older messages did not record positions; retain their legacy layout.
        position = max(cursor, min(len(content), event.get('position', 0)))
        if position > cursor:
            parts.append(('text', content[cursor:position], cursor))
        parts.append(('tool', event, position))
        cursor = position
    if cursor < len(content):
        parts.append(('text', content[cursor:], cursor))
    return parts


def tool_request_message(body, calls):
    """Use canonical calls in model history, separate from UI tool results."""
    message = {'role': 'assistant', 'content': body}
    valid = []
    for call in calls:
        if 'error' in call:
            message['content'] += '\n<tool_call>' + call['raw'] + '</tool_call>'
        else:
            valid.append({'type': 'function', 'function': {
                'name': call['name'], 'arguments': call['arguments'],
            }})
    if valid:
        message['tool_calls'] = valid
    return message


def run_tool_calls(calls, enabled_names, execute):
    events = []
    for call in calls:
        event = dict(call)
        if 'error' in event:
            result = {'error': event['error']}
        elif event['name'] not in enabled_names:
            result = {'error': f"Tool is not enabled: {event['name']}"}
        else:
            try:
                result = execute(event['name'], event['arguments'])
            except Exception as exc:
                result = {'error': str(exc)}
        event['result'] = result
        if isinstance(result, dict) and 'error' in result:
            event['error'] = str(result['error'])
        events.append(event)
    return events


def calculate_expression(expression):
    """Evaluate arithmetic only; tool input must never execute Python code."""
    operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                 ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                 ast.Mod: operator.mod, ast.Pow: operator.pow}
    tree = ast.parse(expression, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError('Expression is too complex')

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp) and type(node.op) in operators:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError('Exponent is too large')
            value = operators[type(node.op)](left, right)
        else:
            raise ValueError('Only numbers and arithmetic operators are supported')
        if isinstance(value, complex) or (isinstance(value, int) and value.bit_length() > 16384):
            raise ValueError('Result is outside the supported numeric range')
        return value

    return evaluate(tree.body)
