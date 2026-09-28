"""Kernel selection APIs.

The names here are the working surface: what layers, the engine, and policy
files actually touch. Internal machinery (predicate node classes, trace
dataclasses, the op-declaration helpers) lives in the submodules and is
imported from there by the code that genuinely needs it.
"""

from phyai.kernel.call import (
    CallSite,
    select,
    explain,
    token_shape,
    torch_dtype,
    param_dtypes,
    backend_preference,
    freeze_kernel_choices,
    unfreeze_kernel_choices,
)
from phyai.kernel.facts import (
    Fact,
    Facts,
    FactKind,
    lib,
    attrs,
    dtype,
    model,
    quant,
    shape,
    device,
    facts_from_query,
)
from phyai.kernel.types import (
    KernelMode,
    KernelQuery,
    ModelContext,
    DeviceProfile,
    PhysicalSignature,
    dtype_name,
)
from phyai.kernel.config import KernelConfig
from phyai.kernel.device import probe_device
from phyai.kernel.policy import Policy, PolicyError, load_policy, policy_from_mapping
from phyai.kernel.registry import Catalog, build_catalog
from phyai.kernel.selector import Selector, NoKernelError
from phyai.kernel.bootstrap import (
    resolve_policy,
    get_kernel_selector,
    set_kernel_selector,
    kernel_selector_scope,
    reset_kernel_selector,
    initialize_kernel_system,
)
from phyai.kernel.predicate import same, all_of, any_of, implies, none_of

__all__ = [
    # Facts and predicates.
    "Fact",
    "FactKind",
    "Facts",
    "all_of",
    "any_of",
    "attrs",
    "device",
    "dtype",
    "facts_from_query",
    "implies",
    "lib",
    "model",
    "none_of",
    "quant",
    "same",
    "shape",
    # Catalog and policy.
    "Catalog",
    "build_catalog",
    "Policy",
    "PolicyError",
    "load_policy",
    "policy_from_mapping",
    "resolve_policy",
    # Selection and call sites.
    "NoKernelError",
    "Selector",
    "CallSite",
    "backend_preference",
    "explain",
    "freeze_kernel_choices",
    "param_dtypes",
    "select",
    "token_shape",
    "torch_dtype",
    "unfreeze_kernel_choices",
    # Lifecycle.
    "KernelConfig",
    "get_kernel_selector",
    "initialize_kernel_system",
    "kernel_selector_scope",
    "reset_kernel_selector",
    "set_kernel_selector",
    # Value types and probes.
    "DeviceProfile",
    "KernelMode",
    "KernelQuery",
    "ModelContext",
    "PhysicalSignature",
    "dtype_name",
    "probe_device",
]
