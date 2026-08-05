# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Jacobian lens: fit and apply the average input-output Jacobian as a readout
of decoder-transformer residuals."""

from jlens._logging import configure_logging
from jlens.fitting import fit, jacobian_for_prompt
from jlens.hf import HFLensModel, Layout, from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.multimodal import (
    ActivationSite,
    EstimatorConfig,
    MultimodalActivationRecorder,
    MultimodalJacobianLensBundle,
    MultimodalLensAdapter,
    MultimodalSample,
    PreparedMultimodalExample,
    evaluate_multimodal_sample,
    fit_multimodal,
)
from jlens.multimodal_hf import (
    Gemma4UnifiedLensAdapter,
    Qwen3VLLensAdapter,
    from_hf_multimodal,
    validate_adapter_parity,
)
from jlens.protocol import LensModel

__all__ = [
    "ActivationRecorder",
    "ActivationSite",
    "EstimatorConfig",
    "Gemma4UnifiedLensAdapter",
    "HFLensModel",
    "JacobianLens",
    "Layout",
    "LensModel",
    "MultimodalActivationRecorder",
    "MultimodalJacobianLensBundle",
    "MultimodalLensAdapter",
    "MultimodalSample",
    "PreparedMultimodalExample",
    "Qwen3VLLensAdapter",
    "configure_logging",
    "evaluate_multimodal_sample",
    "fit",
    "fit_multimodal",
    "from_hf",
    "from_hf_multimodal",
    "jacobian_for_prompt",
    "validate_adapter_parity",
]
