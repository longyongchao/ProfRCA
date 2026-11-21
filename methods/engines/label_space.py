"""Optional closed answer protocol; the original open prompts remain the default."""
from ...errors import ConfigurationError


def resolve(space):
    if space == 'mechanisms':
        from ...pipeline import closed_label_space
        return closed_label_space()
    if space is not None and (not isinstance(space, dict) or not space.get('category')
                              or not isinstance(space['category'], dict)):
        raise ConfigurationError('label_space must be None, mechanisms, or a nonempty category mapping')
    return space


def instruction(space):
    return ("Allowed root-cause IDs and definitions (the complete candidate set, not case-specific clues):\n"
            + space.get('prompt', '\n'.join(space['category']))
            + '\nReturn up to 4 distinct candidates in descending likelihood. Do not force different categories. '
              'Use the exact ID in mechanism_id. Infer the number of causes from evidence; '
              'the gold number is not supplied. Evidence must refer only to the supplied observations.\n'
              'Return only JSON: {"root_causes":[{"mechanism_id":"ID","mechanism":"explanation",'
              '"location":"component","evidence":["observation"],"why":"short reason"}]}')


def candidates(items, space):
    out = []
    for rc in items[:4]:
        if not isinstance(rc, dict):
            continue
        mid = str(rc.get('mechanism_id', '')).strip()
        explanation = str(rc.get('mechanism', mid))
        out.append({'mechanism_id': mid, 'type': space['category'].get(mid, 'unknown'),
                    'headline': explanation, 'description': explanation + ' ' + str(rc.get('why', '')),
                    'location': str(rc.get('location', '')), 'evidence_templates': rc.get('evidence') or []})
    return out
