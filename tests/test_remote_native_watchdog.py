import copy
from scripts.watch_remote_native_training import qualifies, stable_again


def remote():
    return dict(healthy=True, time=500, pid=123, process_created=100,
                status=dict(status='running', steps=12240, processed_valid_tokens=150000000))


def test_initial_1000_steps_and_unhealthy_states_cannot_trigger_shutdown():
    value = remote()
    assert qualifies(value)
    for key, changed in [('healthy', False), ('pid', None)]:
        item = copy.deepcopy(value); item[key] = changed
        assert not qualifies(item)
    for key, changed in [('steps', 11216), ('steps', 12215), ('status', 'starting'),
                         ('status', 'failed_held'), ('processed_valid_tokens', 134347008)]:
        item = copy.deepcopy(value); item['status'][key] = changed
        assert not qualifies(item)


def test_repeated_check_requires_same_run_fresh_progress_and_minimum_spacing():
    value = remote()
    first = dict(qualified=dict(time=300, pid=123, process_created=100, steps=12216))
    assert stable_again(value, first)
    assert not stable_again(value, {})
    for key, changed in [('pid', 124), ('process_created', 99), ('time', 301)]:
        item = dict(value, **{key: changed})
        assert not stable_again(item, first)
    item = copy.deepcopy(value); item['status']['steps'] = 12216
    assert not stable_again(item, first)
    item = dict(value, baseline_steps=12200)
    assert not qualifies(item)
