# Verification Log — arxiv_2607_20415

## Revision 1 (initial comparison)

- **Timestamp**: 2026-07-26T12:40:00Z
- **SIR version used**: 1 (confidence 0.84)
- **Architecture plan version used**: 1
- Ran `evaluate.py --table all` at n=201 for Tables 7/8 (mesh choice not yet matched to
  paper's stated n=401 for Table 8)
- Flagged two deviations with an *unverified hypothesis* ("likely a mesh-size artifact")

## Revision 2 (follow-up investigation, user-requested)

- **Timestamp**: 2026-07-26T13:20:00Z
- **Trigger**: user explicitly asked to verify (not assume) the two flagged deviations
  before treating the repo as ready to share
- **Actions taken**:
  1. Changed Table 7 and Table 8 mesh from n=201 to n=401 (matching the paper's explicitly
     stated Table 8 mesh) and re-ran both.
  2. Ran a full mesh sweep (n=51, 101, 201, 401, 801) on the unlimited-scheme undershoot
     metric specifically, to test the mesh-size hypothesis directly rather than assume it.
  3. Found the mesh-size hypothesis was **false** for the undershoot metric: magnitude
     peaks at n=101 (-0.139) and *decreases* at both coarser and finer meshes, never
     reaching the paper's -0.255 anywhere in the sweep.
  4. While investigating the active-set pattern-update count, found and fixed a **real
     bug**: a redundant confirmatory solve inflating the count by +1 in every run. Fixed in
     `schemes/active_set.py`, added regression test
     `test_active_set_does_not_perform_redundant_confirmatory_solve`.
  5. Re-ran the full test suite (15/15 passed, including the new regression test).
  6. Re-ran a mesh sweep (n=101, 201, 401, 801) on the active-set pattern-update count after
     the fix, confirming the residual gap (2 vs. paper's 1) is mesh-independent — i.e. not
     explained by mesh size either.
  7. Regenerated all 5 table CSVs fresh with both fixes applied.
- **Net effect**: reproducibility score unchanged numerically (0.71 — both flagged items
  still exceed the formula's 50%-deviation cap even after improvement), but one item is now
  backed by a confirmed bug fix (not a guess) and the other is now an honestly-tested open
  question (not an assumed explanation). See `benchmark_comparison.md` for the full
  Root Cause Analysis.

## Paper metrics found vs. matched

- Paper metrics available in SIR `evaluation_protocol.reported_results`: 12
- Metrics matched (direct or closely-proxied comparison possible): 9
- Metrics unmatched (not reproduced this session): 3 — see `hallucination_report.md`
  "Omission Hallucinations" for the full list and suggested fixes

## Metrics compared (full list)

1. Coverage condition (a) holds from n=101 — `table2_coverage.csv`
2. Coverage condition (b) holds (never) — `table2_coverage.csv`
3. FCDF-B spatial L1 order, n=101/201/401 — `table3_ou_spatial_order.csv`
4. FCDF-DC spatial order spot-check, n=51/101/201 (ad hoc script, not in evaluate.py yet)
5. FCDF-DC temporal order, first dt-refinement (ad hoc script, not in evaluate.py yet)
6. Active-set pattern updates vs. gamma/gamma_pic, n=401 — `table8_active_set_cost.csv`
7. Unlimited-scheme min nodal value, all 4 scenario/dt combinations, n=401 — `table7_positivity_conservation.csv`
8. Max mass defect across all scheme/scenario/dt combinations — `table7_positivity_conservation.csv`
9. CC vs. FCDF-B error ratio across the Peclet sweep — `table5_peclet_sweep.csv`

## Raw output file checksums (SHA-256), final revision-2 versions

```
067339a74241f83eb2c2fc5770f007ba419b717e5e8a0e7c9bedb3cdb5b9cefe  table2_coverage.csv
7d9f3070968b9a87e3dccce95b80f49ce69bc4507bc902e5d37b7e61df41c142  table3_ou_spatial_order.csv
41d0107ebaaa1ce57b18f9cce02372384c9be4e1503b1537ccdd120e2fb2d9b1  table5_peclet_sweep.csv
68b47ae747ce1d42cdd57fd0e74a485ec3a98713878dc155f820dffeb6d21f50  table7_positivity_conservation.csv
e9115ed8ce8f3a7c819fabacbebb85d66587fe3bf1fbbd6e9e1cbd7bdf755fb1  table8_active_set_cost.csv
```

Note: table7 and table8 checksums differ from revision 1 (regenerated at n=401 with the
active-set bug fix applied); table2, table3, table5 checksums are unchanged from revision 1
(unaffected by either fix, confirmed by re-running rather than assumed).

## Mesh-sweep raw data (supporting evidence for Root Cause Analysis)

**Unlimited-scheme min value, front benchmark, small dt=2e-4, T=0.2:**

| n | h | Peh | min(p) |
|---|---|-----|--------|
| 51 | 2.00e-2 | 200.0 | -0.1303 |
| 101 | 1.00e-2 | 100.0 | -0.1390 |
| 201 | 5.00e-3 | 50.0 | -0.1229 |
| 401 | 2.50e-3 | 25.0 | -0.0772 |
| 801 | 1.25e-3 | 12.5 | -0.0178 |

**Active-set pattern updates at gamma=gamma_pic, front benchmark, post-fix:**

| n | pattern_updates | converged |
|---|-----------------|-----------|
| 101 | 2 | True |
| 201 | 2 | True |
| 401 | 2 | True |
| 801 | 2 | True |

## User-reported config modifications

`configs/config.yaml` was used as generated, with no manual edits. Two hardcoded mesh
values inside `evaluate.py`'s `table7_positivity_conservation()` and
`table8_active_set_cost()` functions were changed from `n=201` to `n=401` during the
revision-2 investigation, to match the paper's explicitly stated Table 8 mesh — this is a
code change to the evaluation script, disclosed here, not a config change.

## Test suite status at time of this revision

`pytest tests/ -v` → **15/15 passed** (14 from the original suite + 1 new regression test
added this revision), run immediately before finalizing this comparison.

## Manual review required

**Yes — for two specific, now well-characterized items.** See `benchmark_comparison.md`
Root Cause Analysis for the full detail on both. Everything else in this comparison
(spatial/temporal convergence orders, the positivity/conservation headline finding, the
active-set transition point, Chang-Cooper's degradation) is verified and not flagged for
further review.
