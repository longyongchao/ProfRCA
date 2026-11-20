"""Connect the shared input/result contract to each real diagnosis algorithm."""
from __future__ import annotations

from ..errors import ConfigurationError
from ..llm import current_config


def _only(options, allowed):
    unknown = set(options) - set(allowed)
    if unknown:
        raise ConfigurationError('Unsupported method option(s): ' + ', '.join(sorted(unknown)))


class ProfRCAMethod:
    name = 'profrca'

    def run(self, data, config, options):
        from ..pipeline import diagnose
        _only(options, ('tools', 'label_space', 'use_tools', 'line_refs', 'progress_cb', 'cancel_check',
                        'include_details', 'request_timeout', 'world_mode', 'ordering_mode',
                        'metrics_only_mode', 'candidate_diversity'))
        options.setdefault('include_details', config.include_details)
        # Existing progress/cancellation callbacks, tools and label-space settings are preserved.
        return diagnose(data.log, data.metrics, **options)


class PlainLLMMethod:
    name = 'llm'

    def run(self, data, config, options):
        from .engines import plain_llm
        _only(options, ('label_space',))
        return plain_llm.diagnose(data.log, data.metrics, **options)


class DbotMethod:
    name = 'dbot'

    def run(self, data, config, options):
        from .engines import dbot
        _only(options, ('seed',))
        # D-Bot parallelizes expert search: explicitly propagate per-call LLM configuration.
        return dbot.diagnose(data.log, data.metrics, llm_config=current_config(), **options)


class RCAAgentMethod:
    name = 'rca_agent'

    def run(self, data, config, options):
        from .engines import rca_agent
        _only(options, ('label_space', 'max_steps', 'kernel_factory', 'kernel_image', 'cell_timeout'))
        return rca_agent.diagnose(data.log, data.metrics, **options)
