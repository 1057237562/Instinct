"""Plain-text HumanEval I/O codecs and isolated official-test adapter.

JSON is used only for private build/judge manifests, never for problem stdin/stdout.
"""
import copy
import inspect
import json
import random
import subprocess
import sys
import traceback
from pathlib import Path

PROTOCOL_VERSION = 2


def infer_schema(values, annotation=None):
    present = [v for v in values if v is not None]
    if not present:
        return {'kind': 'none'}
    types = {type(v) for v in present}
    if len(present) != len(values) or not (
            len(types) == 1 or types <= {int, float}):
        return {'kind': 'variant'}
    if types <= {int, float}:
        return {'kind': 'number'}
    if types == {bool}:
        return {'kind': 'bool'}
    if types == {str}:
        return {'kind': 'text' if any('\n' in v for v in present) else 'str'}
    if types == {tuple}:
        sizes = {len(v) for v in present}
        if len(sizes) == 1:
            return {'kind': 'tuple', 'items': [infer_schema([v[i] for v in present])
                                              for i in range(len(present[0]))]}
        return {'kind': 'variant'}
    if types == {list}:
        elements = [e for v in present for e in v]
        item = infer_schema(elements) if elements else {'kind': 'number'}
        # Annotations matter when all official examples are empty lists.
        if not elements and annotation and 'str' in annotation:
            item = {'kind': 'str'}
        return {'kind': 'numbers'} if item['kind'] == 'number' else {'kind': 'list', 'item': item}
    if types == {dict}:
        return {'kind': 'dict',
                'key': infer_schema([k for v in present for k in v]),
                'value': infer_schema([e for v in present for e in v.values()])}
    raise ValueError(f'Unsupported values: {types}')


def encode(value, schema, output=False):
    """Return newline-terminated text; preserve empty strings and numeric types."""
    kind = schema['kind']
    if kind == 'number':
        return str(value) + '\n'
    if kind == 'bool':
        return ('True' if value else 'False') + '\n' if output else ('1' if value else '0') + '\n'
    if kind == 'none':
        return 'None\n'
    if kind == 'str':
        if '\n' in value:
            raise ValueError('Expected a single-line string')
        return value + '\n'
    if kind == 'text':
        lines = value.split('\n')
        return str(len(lines)) + '\n' + '\n'.join(lines) + '\n'
    if kind == 'numbers':
        row = ' '.join(str(v) for v in value) + '\n'
        return row if output else str(len(value)) + '\n' + row
    if kind == 'list':
        return str(len(value)) + '\n' + ''.join(encode(v, schema['item'], output) for v in value)
    if kind == 'tuple':
        return ''.join(encode(v, s, output) for v, s in zip(value, schema['items']))
    if kind == 'dict':
        return str(len(value)) + '\n' + ''.join(
            encode(k, schema['key'], output) + encode(v, schema['value'], output)
            for k, v in value.items())
    if kind == 'variant':
        tag = {int: 'int', float: 'float', str: 'str', bool: 'bool',
               type(None): 'none', list: 'list', tuple: 'tuple', dict: 'dict'}[type(value)]
        return tag + '\n' + encode(value, tagged_schema(tag, value), output)
    raise ValueError(f'Unknown kind: {kind}')


def tagged_schema(tag, value=None):
    if tag in ('int', 'float'):
        return {'kind': 'number'}
    if tag == 'str':
        return {'kind': 'text'}
    if tag in ('none', 'bool'):
        return {'kind': tag}
    if tag in ('list', 'tuple'):
        return {'kind': 'list', 'item': {'kind': 'variant'}}
    if tag == 'dict':
        return {'kind': 'dict', 'key': {'kind': 'variant'}, 'value': {'kind': 'variant'}}
    raise ValueError(f'Invalid type tag: {tag}')


def decode(lines, schema, output=False):
    kind = schema['kind']
    if kind == 'variant':
        tag = next(lines)
        result = decode(lines, tagged_schema(tag), output)
        return tuple(result) if tag == 'tuple' else result
    if kind == 'number':
        text = next(lines)
        try:
            return int(text)
        except ValueError:
            return float(text)
    if kind == 'bool':
        text = next(lines)
        valid = {'True': True, 'False': False} if output else {'1': True, '0': False}
        if text not in valid:
            raise ValueError('Invalid boolean')
        return valid[text]
    if kind == 'none':
        if next(lines) != 'None':
            raise ValueError('Expected None')
        return None
    if kind == 'str':
        return next(lines)
    if kind == 'text':
        return '\n'.join(next(lines) for _ in range(read_count(lines)))
    if kind == 'numbers':
        count = None if output else read_count(lines)
        tokens = next(lines).split()
        if count is not None and count != len(tokens):
            raise ValueError('Wrong array length')
        return [decode(iter([t]), {'kind': 'number'}) for t in tokens]
    if kind == 'list':
        return [decode(lines, schema['item'], output) for _ in range(read_count(lines))]
    if kind == 'tuple':
        return tuple(decode(lines, s, output) for s in schema['items'])
    if kind == 'dict':
        result = {}
        for _ in range(read_count(lines)):
            key = decode(lines, schema['key'], output)
            result[key] = decode(lines, schema['value'], output)
        return result
    raise ValueError(f'Unknown kind: {kind}')


def read_count(lines):
    count = int(next(lines))
    if not 0 <= count <= 1000000:
        raise ValueError('Invalid length')
    return count


def parse_output(text, schema):
    # Remove exactly the print terminator, not meaningful empty string lines.
    lines = iter((text[:-1] if text.endswith('\n') else text).split('\n'))
    result = decode(lines, schema, output=True)
    if any(line.strip() for line in lines):
        raise ValueError('Unexpected extra output')
    return result


def render_input(args, plan):
    if len(args) != len(plan['params']):
        raise ValueError('Wrong argument count')
    return ''.join(encode(v, p['schema']) for v, p in zip(args, plan['params']))


def describe(schema, output=False):
    kind = schema['kind']
    if kind == 'number':
        return 'one line containing a number (integer or real)'
    if kind == 'bool':
        return 'one line: True or False' if output else 'one line: 1 for true, 0 for false'
    if kind == 'none':
        return 'one line: None'
    if kind == 'str':
        return 'one full line of text, preserving spaces; an empty string is an empty line'
    if kind == 'text':
        return 'a line L giving the number of text lines, followed by exactly L lines joined with newline characters'
    if kind == 'numbers':
        return ('one line of space-separated numbers (empty line for an empty array)' if output else
                'a line N giving the array length, then a line of N space-separated numbers (empty line when N=0)')
    if kind == 'list':
        return 'a line N giving the list length, followed by N elements, each encoded as: ' + describe(schema['item'], output)
    if kind == 'tuple':
        return 'the following fields in order: ' + '; '.join(describe(s, output) for s in schema['items'])
    if kind == 'dict':
        return ('a line N giving the number of dictionary entries, followed by N key/value pairs. Key: '
                + describe(schema['key'], output) + '. Value: ' + describe(schema['value'], output)
                + '. Entry order does not matter')
    if kind == 'variant':
        return ('a type tag on its own line (int, float, str, bool, none, list, tuple, dict), then its value. '
                'int/float: one numeric line; bool: ' + ('True/False' if output else '1/0')
                + '; none: a line None; str: a line L then L text lines; '
                'list/tuple: a line N then N tagged values; dict: a line N then N tagged key/value pairs')
    raise ValueError(kind)


def execute_job(job):
    """Trace or judge all original checks, including loops and relational assertions."""
    task = job['task']
    namespace = {}
    random.seed(job.get('seed', 42))
    exec(task['prompt'] + task['canonical_solution'] + '\n' + task['test'], namespace)
    canonical = namespace[task['entry_point']]
    signature = inspect.signature(canonical)
    records = []
    cache = {}

    def candidate(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = copy.deepcopy(list(bound.arguments.values()))
        if job['mode'] == 'trace':
            value = canonical(*args, **kwargs)
            records.append({'args_text': encode(arguments, {'kind': 'variant'}, output=True),
                            'result_text': encode(value, {'kind': 'variant'}, output=True)})
            return value
        stdin = render_input(arguments, job['plan'])
        if stdin not in cache:
            index = len(cache)
            try:
                proc = subprocess.run([sys.executable, '-I', job['program']], input=stdin,
                                      text=True, encoding='utf-8', capture_output=True,
                                      timeout=job['timeout'], cwd=job['cwd'])
            except subprocess.TimeoutExpired:
                raise TimeoutError(f'Case {index}: exceeded {job["timeout"]}s') from None
            if proc.returncode:
                raise RuntimeError(f'Case {index}: {proc.stderr[-1500:]}')
            try:
                cache[stdin] = parse_output(proc.stdout, job['plan']['output'])
            except (ValueError, StopIteration) as exc:
                raise ValueError(f'Case {index}: invalid stdout {proc.stdout[:300]!r}: {exc}') from exc
        return copy.deepcopy(cache[stdin])

    namespace['check'](candidate)
    return {'result': 'passed', 'records': records} if job['mode'] == 'trace' else {'result': 'passed'}


if __name__ == '__main__':
    job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    try:
        result = execute_job(job)
    except Exception as exc:
        result = {'result': 'timed out' if isinstance(exc, TimeoutError) else 'failed',
                  'error_type': 'Timeout' if isinstance(exc, TimeoutError) else
                                'WrongAnswer' if isinstance(exc, (AssertionError, ValueError, StopIteration)) else type(exc).__name__,
                  'error_message': str(exc)[:1500], 'traceback': traceback.format_exc()[-2000:]}
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
