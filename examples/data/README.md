# Portable theta benchmark example data

theta_demo.zip contains the original manifest and all 41 complete point bundles
from the local full-41points-tol5e-4 theta threshold reassessment campaign.
The notebook validates the archive SHA256 before extraction; the existing
ThetaContinuationSynthesis adapter then validates the manifest, every point,
gate action, leakage, and original QPY hashes.

- Archive SHA256: c7b346cd4c118c02a0e3d2008dca38ca2a64a165309bb1b3d114f81ae2fc6f1b
- Source manifest fingerprint: aeaae4fb938607ed00fe1cbb27b9e1b9de436061555f7c4a695e73d20cade090
- Original schema: theta_threshold_reassessment_v1
- CZ3 action/leakage tolerance: 5e-4; F3 tolerance: 1e-10.
- 575 members; 500891 bytes compressed.

Every archived file is a byte-for-byte copy of the original evidence. No
circuits, parameters, outcome flags, or hashes were modified. This is the complete
41-point gate grid, not a selection of winning encodings. Source metadata records
historical continuation and fallback methods and original provenance.

Only the replay inputs are included. Historical hardware/full-circuit results,
figures, recovery workspaces and the original baseline input are not required.
The absolute historical baseline path inside the original manifest is provenance;
replay does not read it. Fresh fitting requires an explicit existing baseline QPY.

The notebook performs a new backend compilation and routing run. Published example
outputs are finite-search compilation evidence, not QPU measurements, a global
optimum, or a guarantee of improvement on another backend/compiler version.
