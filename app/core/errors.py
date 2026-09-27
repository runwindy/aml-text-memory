from __future__ import annotations


class ContractError(ValueError):
    """Raised when an internal contract assertion fails."""


class RequestConflictError(ValueError):
    """Raised when a request_id is reused with a different payload."""
