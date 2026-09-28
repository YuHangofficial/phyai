"""Exception types raised by phyai.parallel.

Kept narrow on purpose: each error class is identifiable by callers without
parsing message strings. Message text carries the diagnostic detail.
"""

from __future__ import annotations


class PhyaiDistError(Exception):
    """Base for all phyai.parallel errors."""


class NoBackendError(PhyaiDistError):
    """No registered backend can handle the requested op + mode + ctx."""
