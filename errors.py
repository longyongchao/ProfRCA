"""Public, predictable error types shared by every diagnosis method."""

class InputError(ValueError):
    pass

class ConfigurationError(ValueError):
    pass

class ExecutionUnavailable(RuntimeError):
    pass
