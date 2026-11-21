import pytest

from profrca.methods.engines.label_space import resolve, instruction, candidates
from profrca.methods.engines.plain_llm import to_predictions

SPACE = {'category': {'MANY_INSERTS': 'db_internal', 'MANY_UPDATES': 'db_internal'},
         'prompt': 'MANY_INSERTS: excessive inserts; MANY_UPDATES: excessive updates'}


def test_two_native_labels_in_same_category_remain_distinct():
    roots = to_predictions('{"root_causes":[{"mechanism_id":"MANY_INSERTS"},'
                           '{"mechanism_id":"MANY_UPDATES"}]}', SPACE)
    assert [r['mechanism_id'] for r in roots] == ['MANY_INSERTS', 'MANY_UPDATES']
    assert all(r['type'] == 'db_internal' for r in roots)
    assert 'excessive updates' in instruction(SPACE)


def test_unknown_id_preserved_for_independent_scoring():
    root = candidates([{'mechanism_id': 'fabricated'}], SPACE)[0]
    assert root['mechanism_id'] == 'fabricated' and root['type'] == 'unknown'
    with pytest.raises(ValueError):
        resolve({'category': {}})


def test_open_baseline_format_unchanged():
    roots = to_predictions('{"root_causes":[{"category":"cpu","mechanism":"high CPU"}]}')
    assert roots[0]['type'] == 'cpu' and 'mechanism_id' not in roots[0]
