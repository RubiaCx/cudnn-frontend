# MXFP8 Attention recipe integration

Status: proposal for review in this fork. This change adds documentation only;
kernel integration and GPU acceptance are pending.

## Scope

Add explicit MXFP8 training recipes for D128 and D256 through the existing
Frontend graph and open-source engine infrastructure. Preserve existing API
defaults and existing engines. Start with D256, all-2D quantization, fixed P,
dynamic dS, and split backward on B300.

Inputs are BF16 Q/K/V and upstream dO at the training wrapper; prepared engine
inputs are E4M3 payloads and UE8M0 scale factors. Outputs and gradients are BF16;
row statistics and GEMM accumulations use FP32 where required by the recipe.
The initial scope is dense equal-length non-causal self-attention without
dropout. Packed and causal support require separate acceptance tests before
being advertised.

## Existing integration points

- `python/cudnn/sdpa/graph_analyzer.py` extracts graph facts for engine matching.
- `python/cudnn/engines/manifest.py` records engine identities.
- `python/cudnn/sdpa/fwd/kernels/sm100/` owns forward implementations.
- `python/cudnn/sdpa/bwd/api_dsl_mxfp8_sm100.py` adapts the existing D128/D256
  split backward, including SF repacking, dQ, and fused dK/dV.

The existing backward contract consumes independently quantized normal and
transposed 1x32 operands, fixed P scaling, and dynamic dS scaling. Our proposed
2D recipes share quantized values across transpose directions and allow
explicit P/dS grouping. They therefore require numerical attributes and layout
handling in addition to new engine entries.

## Numerical recipe and execution policy

The following are conceptual fields, not finalized public API names:

| Numerical attribute | Proposed choices |
| --- | --- |
| Q/K grouping | Matched 1D or 32x32 groups |
| V/dO grouping | Paired 1D or 32x32 groups in the consumed GEMM orientations |
| P mode | Fixed or dynamic, consistent between forward and backward |
| Dynamic P grouping | Explicit supported 1D/2D grouping |
| dS mode | Dynamic only |
| dS grouping | Explicit grouping for the dQ and dK consumers |
| Scale arithmetic | Explicit rounding, zero/subnormal handling, saturation, and fixed-P value |

For 1D groups, document the reduction axis: 1x32 and its transposed 32x1
representation are different operand contracts. A transpose of a quantized
payload does not reproduce independent quantization along the other axis.
Forward and backward must carry the same resolved P policy; matching the word
"fixed" alone does not establish matching scale arithmetic.

Numerics-changing choices belong in operation attributes and graph facts,
not autotuning knobs. Follow the repository's `python_only_attrs` mechanism
for Frontend-only semantics until an upstream API is agreed. Providers that
cannot implement a requested recipe must decline it, including backend
lowering. An unset recipe preserves current behavior.

Tile size, stages, warp roles and CTA geometry remain execution choices within
the requested recipe. Determinism is an execution requirement. Split, fused,
and dS-export paths may have different accumulation or residual-tile behavior;
admit them only after their numerical and determinism contracts are checked.

## Data flow and ownership

Use the common Frontend preparation/execution contract, keeping provider-specific
lowering internal. Provide a training sample that includes BF16-to-MXFP8
preparation and explicit backward state handoff; prepared FP8 engine timing
alone does not measure this full operation.

1. Quantize Q/K/V and prepare their scale representations. For 1D, generate the
   independently quantized transpose operands required by backward.
2. Run forward and retain output, LSE, resolved recipe, and required quantized
   operands in invocation-owned training state.
3. Quantize dO, prepare backward statistics/scales, and execute backward.
4. Release saved state according to autograd lifetime, including retained
   backward and activation-checkpoint recomputation.

For 32x32 groups, reuse payload storage where legal strides and TMA descriptors
allow it. Keep normal/transposed ports initially if that minimizes API changes;
validate the bound views. Expanding one square scale into repeated canonical
1x32 metadata is a possible correctness adapter, with its cost measured
explicitly. Compact SF consumption is the intended optimized path.

Workspace must be caller- or invocation-owned, sized during preparation, and
safe across independent streams and graph instances. Compilation caches must
not retain CUDA tensors. No hidden allocation, host readback, or layout copy
should be introduced into replay. Any required conversion belongs in the
documented execution chain and inclusive timing.

## Validation and performance

- Compare the integrated engine with the source implementation using identical
  inputs and an explicitly resolved recipe. Require byte equality for
  equivalent rearrangements; separately document any accumulation-order change.
- Check against an independent reference, including low-amplitude dO, outliers,
  residual tiles, and scale thresholds. Operator checks do not establish
  long-range training convergence.
- Exercise repeated and retained backward, activation checkpointing, changed
  inputs under Graph replay, fresh allocations, two streams/instances, and
  workspace release. Test determinism and batch invariance separately.
- Before enabling packed or causal cases, verify segment isolation, unused
  physical tails, fully masked tiles, and zero gradient stores.
- Run sanitizer checks and the repository's affected host/GPU suites. Update
  support declarations and determinism coverage with measured results.
- Benchmark D128/D256, H8/H24, S512 through S16384, and B64/S284 on the same GPU
  and software environment as the existing Frontend engines. Report prepared
  forward/backward, preparation, inclusive eager/Graph time, and peak allocated
  and reserved memory. Use independent processes and disclose recipe/layout
  differences. Do not derive total time by adding overlapping phase timings.

## Implementation sequence

- [ ] Agree on recipe operation attributes, scale layouts, and backward state.
- [ ] Integrate D256 all-2D fixed-P/dynamic-dS forward and split backward.
- [ ] Add the training sample, correctness tests, sanitizer checks, and paired
      performance results for that integration.
- [ ] Extend to dynamic P and supported mixed 1D/2D groups.
- [ ] Integrate D128 and validate its supported backward execution strategies.
- [ ] Evaluate D256 dS export as a separate optional optimization, including
      temporary workspace, residual-tile numerics, and memory-lifetime tests.

## Open decisions

Finalize public attribute names with maintainers. Confirm whether compact
square SF should receive a new layout declaration or initially use canonical
metadata. Preserve license headers and required third-party notices when
porting implementations. Performance and training claims will be added only
after the integrated version is measured; this proposal claims neither.
