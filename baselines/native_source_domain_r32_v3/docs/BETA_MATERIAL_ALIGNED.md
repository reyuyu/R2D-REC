# Material-aligned BETA baseline

This variant preserves the original uploaded material rows and their three
`data_source` loss routes. It intentionally retains the current BETA user and
recommendation data, so it is a material-aligned comparison rather than a
bitwise reproduction of the packaged full multitask run.

Registered version: `BETA_material_aligned_v1`.

Unified path: `/data/lf_data_versions/alltrain/BETA_material_aligned_v1`.

Material contract:

- `material_sample`: 100,000 rows, SID/domain tokens weight 8, ordinary tokens
  weight 1, then apply the packaged material-domain factor.
- `sid_bucket_canonical_no_think`: 11,298 rows, every supervised response token
  weight 4.
- `sid_bucket_reverse`: 29,586 rows, SID/domain tokens weight 8 and ordinary
  tokens weight 1, without the `material_sample` domain factor.

The manifest records a projection hash proving that `system`, `prompt`,
`response`, `data_source`, and material row order are unchanged from the source
upload.
