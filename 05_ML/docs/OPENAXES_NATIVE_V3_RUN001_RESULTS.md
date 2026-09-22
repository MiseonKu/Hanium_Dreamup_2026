# OpenAxes Native V3 Run 001 — Primary-8 Results and Adaptive MoFe-only Follow-up

> Status: initial 48-fit Dense-versus-terminal analysis is sealed and complete; the adaptive MoFe-only follow-up and validated 72-fit analysis are complete.
>
> Initial and follow-up values below are copied from canonical receipts; scientific values were not manually recomputed for this document.

## Result scope and claim boundary

Run 001 asks whether a parameter-matched model can predict the UR5 base-frame TCP position in metres from the public OpenAXES recording when any non-empty subset of its four measured IMUs is retained under synthetic masking. This lineage is distinct from the workspace's collected four-WT901C/`vive_world` endpoint-regression corpus. The labels `humerus`, `radius`, `carpus`, and `digitus` refer to public OpenAXES sensor groups, not local WT901C devices.

The first sealed phase contains exactly **48 immutable fits**: Dense and terminal, eight primary LOSO folds, three fixed seeds, and contexts **0–23**.

After those Dense-versus-terminal outcomes were available, one already-preregistered candidate was selected for an adaptive follow-up: ordinal-4 two-stage DMoME+MoFe without CAGrad. Its model, parameter count, masks, folds, seeds, schedule, and metrics were fixed before the initial outcomes, but the decision to execute these additional 24 fits was outcome-informed. Expanded completion is exactly **72 attempts, 72 results, and zero `error.json` files**.

The initial terminal-versus-Dense contrast remains historical primary-8 evidence. Terminal-versus-MoFe is an adaptive follow-up on the same data and paired folds/seeds, not an independent confirmatory replication. Neither phase completes the preregistered 42-context grid, selects a final candidate across all six candidates, or authorizes a full ablation claim.

## Fixed methods

All three candidates have exactly **6,979 trainable parameters** and use the same 15 non-empty masks.

- **Dense:** `dense_6979__allmask_cardinality_balanced_erm`
- **Adaptive MoFe-only:** `dmome_6979__two_stage_allmask_cardinality_balanced_mofe`
- **Terminal:** `dmome_6979__two_stage_allmask_cardinality_balanced_mofe_cagrad`
- **DMoME:** availability-aware experts and output-space routing specialize predictions for observed sensor subsets.
- **MoFe:** a fixed 28-edge availability lattice penalizes degradation when a richer subset is available.
- **CAGrad:** four cardinality-group risks are combined while limiting destructive gradient conflict.
- **Two-stage optimization:** MoFe-only and terminal use 100 independent-expert plus 300 cooperative updates; Dense uses 400 cooperative updates.

MoFe-only and terminal share DMoME, two-stage optimization, all-mask training, and MoFe; CAGrad is the intended incremental difference. Because MoFe-only was executed adaptively, the contrast is reported with its frozen gate but without causal or independent-confirmation language.

These are established ideas assembled for this four-OpenAXES-IMU-to-TCP benchmark. The contribution claimed here is the parameter-matched missing-IMU evaluation, not invention of mixture-of-experts, monotonic regularization, or conflict-averse optimization.

## Sensor-availability augmentation verification

Training deterministically expands each recorded training row over the exact 15 non-empty subsets. An unavailable sensor's nine channels are set to exact zero and its availability bit is cleared. Scenario losses are balanced within each mask, masks within cardinality, and the four cardinalities equally. Normalization is fit on training rows only; outer features and labels are excluded from the trainer interface.

Focused tests verify the 15-mask roster, 28 adjacent edges, mask order, zeroing, risk arithmetic, deterministic seeding, train-only transform, and exact train-only controller projection. This is synthetic availability augmentation over fixed recordings, not additional physical data or independent fault evidence.

## Execution accounting

| Scope | Completed / planned | Status | Permitted interpretation |
|---|---:|---|---|
| Initial primary-8, Dense + terminal | **48 / 48** | Sealed complete | Historical Dense-versus-terminal contrast |
| Adaptive MoFe-only follow-up | **24 / 24** | Complete | Adaptive incremental comparison |
| Expanded primary-8, three candidates | **72 / 72** | Complete | Three-candidate summaries with adaptive disclosure |
| Six-candidate screen, contexts 0–11 | **36 / 72** | Incomplete | No six-candidate ranking |
| Three candidates, all 42 contexts | **72 / 126** | Incomplete | 54 report-only-fold fits absent |
| Full six-candidate × 42 grid | **72 / 252** | Deferred | No full ablation or ranking |

Dense and terminal were not refit: their 48 results are reused. The expanded set is initial 48 plus 24 new MoFe-only fits, not three new experiments or 72 independent replications. Overlapping counts must not be added.

### Fixed action accounting

Counts in the table are per fit; aggregate totals are stated immediately below it.

| Candidate | Fits | Updates | Training forwards | Checkpoint forwards | Gradient traversals | CAGrad solves | Outer feature/label/forward/evaluator calls |
|---|---:|---:|---:|---:|---:|---:|---|
| Dense | 24 | 400 | 400 | 400 | 400 | 0 | 1 / 1 / 1 / 1 |
| MoFe-only | 24 | 400 | 400 | 400 | 400 | 0 | 1 / 1 / 1 / 1 |
| Terminal | 24 | 400 | 400 | 400 | 1,300 | 300 | 1 / 1 / 1 / 1 |

The initial receipt binds 9,600 updates and checkpoint forwards per initial candidate, 9,600 Dense and 31,200 terminal gradient traversals, 7,200 terminal CAGrad solves, 360 view rows per candidate, null Dense stage-one hashes, and 24 terminal hash pairs. The expanded receipt additionally binds 24 MoFe-only stage-one model/optimizer hash pairs and, for MoFe-only, exactly 9,600 updates, 9,600 training forwards, 9,600 checkpoint forwards, 9,600 gradient traversals, zero CAGrad solves, 360 view rows, and 24 outer feature/label/forward/evaluator calls each. Dense retains zero stage-one hash pairs; terminal retains 24.

## Authority and provenance

| Field | Sealed value |
|---|---|
| Analysis-source HEAD | `dd9c25b778fa031be1378e48698ca0795e25fdf8` |
| Implementation / launch-source commit | `0580fda91fb929de314d5383e0efdbfd79f9c7d2` / `52c2d9510c689b2a57a845d0a88263c5081e09d6` |
| Preregistration/errata tip | `9aaf00e48c17b3ebeaacd776a2890c5ec8122253` |
| Config raw SHA-256 | `c9f828dfd1da82f0031c16a7802e3679081a3cd7ec37657435802e4d05a93381` |
| Parent exact-10 set SHA-256 | `72af3783c1c4a38c662d9afc8d97f243f3a9cc9f1e1807d41371109a445ae97d` |
| Copied parent receipt | `artifacts/openaxes_external_native_v3/run_openaxes-native-v3-001/parents/native_v2_compound_salvage_validations/receipt_set.json`; 4,250 B; raw `5d27d0e6f08843d691d6013a4909dcf9f06e64f69ff295b66672b62503277668`; payload `34ac59ffd0952a7fe5fbc23ad563fa9a3ace05eb7901d6644a373890214a3236` |
| Run ID/root | `openaxes-native-v3-allmask-dmome-001`; `artifacts/openaxes_external_native_v3/run_openaxes-native-v3-001` |
| Prepared Parquet | `artifacts/openaxes_external_v1/prepared_openaxes-v1-sealed-003/prepared.parquet`; 8,970,232 B; raw `98747e51fd95506b91facfeb273be23c454d07da563deea13a6f2192ded48703` |
| Task index | `task_index.json`; 31,045 B; raw `e339c7e7fd9c3be2c1b2bb5efae2685b3871c2c8da7c8ad6b7f2ca99e218df7f`; payload `c2e1c9dbc6cfa4ba4876bb30e0d4ca18434a4e9fd7b314d15cdd343319969f41` |
| Initial analysis | `receipts/openaxes_external_native_v3_run001_deadline_primary8_analysis.json`; 30,532 B; raw `cc87539ad2eb084d01a630f10fe8c62c98dfb6c77d0d72c59c869c7ef8ef20ee`; payload `36cf497c2fb8e8cb362b47a5b4fbac1556a14a1e9506f1164b2c8f9d3c06e4bc` |
| Initial document receipt | `receipts/openaxes_external_native_v3_run001_document_receipt.json`; 72,912 B; raw `9fc5b3f1f4c34b2edd9e42b0eea5e27858f8e5ff23a0e29ce9f09a296898c42a`; payload `59885110e65bd3de19fbe5fa90bfc04184a5b0116e4b5a9890328cec269543f2` |
| Initial evaluation | `reports/evaluation_partial_bb4270fcd0c3bdcb.json`; 3,384 B; raw `bd070ed99cc87ffcf4aafa4fecd8e018b34a1059b813cba3fac35630e6e791c5` |
| Initial receipt generator | `/home/shin-server/.codex-supervisors/capstone-code/openaxes_native_v3_run001_document_receipt.py`; 39,345 B; raw `541ddde28f320b91be1c3961e357401ff922188cb90e66bdee930fcb10f92a7c` |
| Expanded analysis | `receipts/openaxes_external_native_v3_run001_mofe_cagrad_primary8_analysis.json`; 45,359 B; raw `8cc2aec6021b77ab125b0c2923629707713910fa6cc99431b8151e7e04858c1b`; payload `562bf217a1ea3e27996fff1c75b62904145e8cb4d4fb79105f9a71b499868d49` |
| Expanded document receipt | `receipts/openaxes_external_native_v3_run001_mofe_document_receipt.json`; 118,301 B; raw `62d99a1701a6b5061f59b93aa3c1504cbb76f51dcde6e046a735f0c487b23b0e`; payload `e0f1e39d785133e2899217d9fe98e3321b452da5a85a26aab5a010481d0f9121` |
| Operational recovery supplement | `receipts/openaxes_external_native_v3_run001_operational_recovery_supplement.json`; 291,120 B; raw `c27d9dd9d41cd25842f1c8c2d672b18afb3d2ee179136f99022193408d6b2377`; payload `ac231af5f81179a4aaa1705165482041381ae3a417b3495404369740e03e9fc4` |
| Operational recovery supplement generator | `tools/openaxes_native_v3_run001_operational_recovery_supplement.py`; 55,811 B; raw `d4f06e656631bcabd6614e8480529a83d4c7d8561a5d2dd21faccd8ca9e8d982` |
| Operational recovery strict verifier | `tools/openaxes_native_v3_run001_operational_recovery_supplement_strict_verify.py`; 25,315 B; raw `2840b2370ac64385feb5113a18dc411018565acc3a0a838e9ffffb53047cdc49` |
| Expanded evaluation | `reports/evaluation_partial_2fdc6a253782c3ac.json`; 4,727 B; raw `4c86901309baa981220796f8bc3d25f42053f902a501f0a335f66cb5243d8d20`; payload `a3332c92cc4ecd08565ae414926369e302d6bcf8cd544c7f846be28c415a0608` |
| Expanded receipt generator | `/home/shin-server/.codex-supervisors/capstone-code/openaxes_native_v3_run001_mofe_document_receipt.py`; 34,756 B; raw `8f51f426d99a21cfae1e609fc9902a8001432ae851e865ad3c20cfbb80645ebc` |
| Adaptive follow-up launcher | `/home/shin-server/.codex-supervisors/capstone-code/openaxes_native_v3_run001_mofe_followup.sh`; 5,805 B; raw `f7511299fddf5e2ae727db989b8a000a2199ac4b3b4ce4756b5da09f74dcd34f` |
| Original expanded validation/analysis finalizer | `/home/shin-server/.codex-supervisors/capstone-code/openaxes_native_v3_run001_mofe_finalize.sh`; 4,451 B; raw `12f1f8da1164151e65526fbf5d8cfc561b24115bb2b24d226efee42e6119673a` |
| Original expanded document-receipt finalizer | `/home/shin-server/.codex-supervisors/capstone-code/openaxes_native_v3_run001_mofe_document_finalize.sh`; 1,909 B; raw `98abe3142cbfc6bc620cefc583a81f45e0c7df52bd6d95c203c96005da617537` |
| Successful recovery-v2 validation/analysis finalizer | `/home/shin-server/.codex-supervisors/capstone-code/openaxes_native_v3_run001_mofe_recovery_finalize.sh`; 8,051 B; raw `3f40038d49dbbc41c5c77c3d926195e72626d9dc2f18049348c01498cfd4ac9a` |
| Successful recovery-v2 document finalizer | `/home/shin-server/.codex-supervisors/capstone-code/openaxes_native_v3_run001_mofe_recovery_document_finalize.sh`; 2,034 B; raw `8b14516828bc272aac46df30eba59fce2527271167131eacfe5edafe6ce0c3ed` |
| Scope flags | `deadline_scoped_primary_8_not_full_42`; `full_42_context_grid_complete=false`; `final_claim_eligible=false`; `claim_authorized=false` |

Initial receipts are immutable historical evidence. Expanded receipts extend that lineage and must not replace or retroactively relabel the initial analysis.

### Operational recovery disclosure

The final poststate contains 72 unique attempts, 72 results, 72 checkpoints, 72 prediction arrays, and zero errors; all declare scientific fit attempt 1. The original two controller partitions returned 0 and 2 because the contexts 12–23 partition reached its global join check while the contexts 0–11 partition was still running; the follow-up service therefore exited 41, and its original analysis and document finalizers failed closed with statuses 52 and 60.

The first recovery attempt also failed closed before creating a lock, validation bundle, or analysis because its transient-service `PATH` did not contain the non-absolute `rg` command. Its validation/analysis and document-finalizer statuses were 127 and 80; the corresponding source hashes were `cf168c9100de659b6365f2c1972299b6280e81886d49eddac5de03d820a434bd` and `c94ff16cc8c5683e150f4e8e779591820b64df6e93e32ff45fc750909cf8a8de`.

Successful recovery-v2 made exactly **11 validator calls**: one preflight validation followed by the sealed bundle's exact ten validators. It also made one evaluator call and one analysis call, with zero `fit` and zero `internal-fit` calls. The preflight validator ran before creation of the recovery lock; that lock gated subsequent output creation, not the preflight, so total validation was **not** a one-shot operation. The no-retry assertion applies only to the sealed exact-10-plus-one bundle.

The user-journal records recovery-v2 starting at `2026-08-26T19:17:15+09:00`, completing validation and analysis at `19:18:03+09:00`, and completing the document finalizer at `19:18:04+09:00`; because both successful transient units were unloaded after exit, this journal is the success evidence rather than current live-unit state. The supplement binds the exact 72-member poststate, all scientific member mtimes preceding recovery, single-attempt lineage, and recovery source/journal evidence with no new fit invocation. However, no exact pre-recovery 72-member byte projection was sealed, so byte-for-byte pre/post equality, historical `O_EXCL` syscalls, and exact executed-source byte identity cannot be established cryptographically after the fact. The absence of retraining evidence is therefore a poststate/source/journal conclusion, not proof of every historical process action. The supplement does not extend the scientific claims or make this adaptive follow-up confirmatory.

The original supplement issuer's `validate` path is not sufficient as the sole acceptance authority because it does not completely reject mutations to `scope`, all claim fields, or raw metadata. Final acceptance additionally uses the strict verifier, which pins receipt raw `c27d9dd9d41cd25842f1c8c2d672b18afb3d2ee179136f99022193408d6b2377`, payload `ac231af5f81179a4aaa1705165482041381ae3a417b3495404369740e03e9fc4`, and issuer `d4f06e656631bcabd6614e8480529a83d4c7d8561a5d2dd21faccd8ca9e8d982`; enforces the full field set, live projections, mode `0600`, link count 1, and expected UID/GID; and completed both production verification and self-test with exit 0. Its negative suite rejected 17 semantic/temporary mutations as well as mode `0644`, link count 2, and a symlinked parent. Acceptance therefore depends on this stricter layer, not the issuer validator alone.

## Runtime accounting

| Phase | First attempt / last result UTC | Wall interval | Nominal allocated device-hours |
|---|---|---:|---:|
| Initial 48 | `2026-08-26T02:44:46Z` / `2026-08-26T08:12:50Z` | 19,684 s (5:28:04) | 21.871111111111112 |
| Adaptive MoFe-only 24 | `2026-08-26T08:13:42Z` / `2026-08-26T10:11:37Z` | 7,075 s (1:57:55) | 7.861111111111111 |
| First initial through final follow-up result | `2026-08-26T02:44:46Z` / `2026-08-26T10:11:37Z` | 26,811 s (7:26:51), a calendar span rather than training duration | Not reported as compute usage; the span may include adaptive-decision time |

Device-hour quantities are four physical devices times the stated execution interval, **not utilization-integrated GPU-hours**. The receipts bind the physical-device count of four. A contemporaneous supervisory host snapshot, not a canonical-receipt field, observed 4 × NVIDIA CMP 170HX (64 GiB), Python 3.12.3, PyTorch 2.12.1+cu132, CUDA 13.2, and cuDNN 92000.

## Fresh-process validation and evaluation

These checks establish validator reproducibility, not independent experiments.

### Initial 48-fit bundle — sealed complete

Root: `/home/shin-server/.codex-supervisors/capstone-code/openaxes-native-v3-run001-final-validation-001`

| Invocation | Start UTC | Finish UTC | Exit | stdout / stderr |
|---|---|---|---:|---|
| validate-00 | `2026-08-26T08:13:04Z` | `2026-08-26T08:13:04Z` | 0 | both 0 B, empty SHA |
| validate-01 | `2026-08-26T08:13:04Z` | `2026-08-26T08:13:05Z` | 0 | both 0 B, empty SHA |
| validate-02 | `2026-08-26T08:13:05Z` | `2026-08-26T08:13:05Z` | 0 | both 0 B, empty SHA |
| validate-03 | `2026-08-26T08:13:05Z` | `2026-08-26T08:13:05Z` | 0 | both 0 B, empty SHA |
| validate-04 | `2026-08-26T08:13:05Z` | `2026-08-26T08:13:05Z` | 0 | both 0 B, empty SHA |
| validate-05 | `2026-08-26T08:13:05Z` | `2026-08-26T08:13:05Z` | 0 | both 0 B, empty SHA |
| validate-06 | `2026-08-26T08:13:05Z` | `2026-08-26T08:13:05Z` | 0 | both 0 B, empty SHA |
| validate-07 | `2026-08-26T08:13:05Z` | `2026-08-26T08:13:06Z` | 0 | both 0 B, empty SHA |
| validate-08 | `2026-08-26T08:13:06Z` | `2026-08-26T08:13:06Z` | 0 | both 0 B, empty SHA |
| validate-09 | `2026-08-26T08:13:06Z` | `2026-08-26T08:13:06Z` | 0 | both 0 B, empty SHA |
| evaluate-00 | `2026-08-26T08:13:06Z` | `2026-08-26T08:13:06Z` | 0 | both 0 B, empty SHA |

Empty-file SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. `COMPLETE`: 39 B, raw `8935973fc39b791fd6a3419a0f5fb48a67f7850c7099830ad2c951dea616136b`, exact text `10 validators and 1 evaluator exited 0`. The 56-member bundle has ten sequential validators and one evaluator, all exit 0, no retry. Controller: `/home/shin-server/.codex-supervisors/capstone-code/openaxes_native_v3_run001_validate10.sh`; 5,110 B; raw `832bce63e33ea5dfdae3ddd76877c95ceac0aded55104016aaa19acc45c8b735`.

### Expanded 72-fit bundle — adaptive follow-up

Root: `/home/shin-server/.codex-supervisors/capstone-code/openaxes-native-v3-run001-mofe-final-validation-001`

| Invocation | Start UTC | Finish UTC | Exit | stdout / stderr |
|---|---|---|---:|---|
| validate-00 | `2026-08-26T10:17:15Z` | `2026-08-26T10:17:15Z` | 0 | `validate-00.stdout.log` / `validate-00.stderr.log`; both 0 B, empty SHA |
| validate-01 | `2026-08-26T10:17:15Z` | `2026-08-26T10:17:15Z` | 0 | `validate-01.stdout.log` / `validate-01.stderr.log`; both 0 B, empty SHA |
| validate-02 | `2026-08-26T10:17:15Z` | `2026-08-26T10:17:15Z` | 0 | `validate-02.stdout.log` / `validate-02.stderr.log`; both 0 B, empty SHA |
| validate-03 | `2026-08-26T10:17:15Z` | `2026-08-26T10:17:16Z` | 0 | `validate-03.stdout.log` / `validate-03.stderr.log`; both 0 B, empty SHA |
| validate-04 | `2026-08-26T10:17:16Z` | `2026-08-26T10:17:16Z` | 0 | `validate-04.stdout.log` / `validate-04.stderr.log`; both 0 B, empty SHA |
| validate-05 | `2026-08-26T10:17:16Z` | `2026-08-26T10:17:16Z` | 0 | `validate-05.stdout.log` / `validate-05.stderr.log`; both 0 B, empty SHA |
| validate-06 | `2026-08-26T10:17:16Z` | `2026-08-26T10:17:16Z` | 0 | `validate-06.stdout.log` / `validate-06.stderr.log`; both 0 B, empty SHA |
| validate-07 | `2026-08-26T10:17:16Z` | `2026-08-26T10:17:16Z` | 0 | `validate-07.stdout.log` / `validate-07.stderr.log`; both 0 B, empty SHA |
| validate-08 | `2026-08-26T10:17:16Z` | `2026-08-26T10:17:16Z` | 0 | `validate-08.stdout.log` / `validate-08.stderr.log`; both 0 B, empty SHA |
| validate-09 | `2026-08-26T10:17:16Z` | `2026-08-26T10:17:17Z` | 0 | `validate-09.stdout.log` / `validate-09.stderr.log`; both 0 B, empty SHA |
| evaluate-00 | `2026-08-26T10:17:17Z` | `2026-08-26T10:17:17Z` | 0 | `evaluate-00.stdout.log` / `evaluate-00.stderr.log`; both 0 B, empty SHA |

Empty-file SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. `COMPLETE`: 39 B, raw `8935973fc39b791fd6a3419a0f5fb48a67f7850c7099830ad2c951dea616136b`, exact text `10 validators and 1 evaluator exited 0`. The 56-member bundle has ten sequential fresh-process validators and one evaluator, all exit 0, with retry or replacement prohibited. Controller: `/home/shin-server/.codex-supervisors/capstone-code/openaxes_native_v3_run001_validate10.sh`; 5,110 B; raw `832bce63e33ea5dfdae3ddd76877c95ceac0aded55104016aaa19acc45c8b735`.

The two bundles are not “20 replications”; each revalidates one persisted scope ten times and evaluates it once.

## Initial sealed 48-fit primary-8 metrics

| Candidate | Fits | Amean (m) | WM (m) | WS (m) | J (m) | F (m) | CR | PE (m) | Cardinality 1 / 2 / 3 (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dense | 24 | 0.0869621040968043 | 0.24929323232818773 | 0.11434127828575213 | 0.15019887157024805 | 0.03430830346646642 | 0.3026862750193508 | 0.004193774320274617 | 0.1495707178010024 / 0.07605982849554355 / 0.04070690379449729 |
| Terminal | 24 | 0.08937796971194266 | 0.24796420309308906 | 0.12706496144543436 | 0.1548023780834887 | 0.05059202833156673 | 0.2955808091341122 | 0.003081123703617304 | 0.14874289431228033 / 0.07942349303013377 / 0.04494476013431831 |

### Initial fifteen-view summaries

Negative terminal − Dense values favor terminal; positive values favor Dense.

| View | Dense (m) | Terminal (m) | Terminal − Dense (m) |
|---|---:|---:|---:|
| full | 0.02809908039860877 | 0.03351162141139385 | 0.005412541012785078 |
| drop_humerus | 0.049857742005619134 | 0.05094599687024019 | 0.001088254864621059 |
| drop_radius | 0.0554299252455645 | 0.05864118415789407 | 0.0032112589123295746 |
| drop_carpus | 0.029729290943180955 | 0.03457192511844926 | 0.004842634175268305 |
| drop_digitus | 0.027810656983624574 | 0.03561993439068975 | 0.007809277407065177 |
| only_humerus_radius | 0.029838237927251536 | 0.03656161678838131 | 0.006723378861129774 |
| only_humerus_carpus | 0.05684599862908497 | 0.06010728305260644 | 0.003261284423521464 |
| only_humerus_digitus | 0.05614107471196515 | 0.05828325047635324 | 0.0021421757643880907 |
| only_radius_carpus | 0.05134984580951632 | 0.052629031295184515 | 0.0012791854856681953 |
| only_radius_digitus | 0.051888760473550656 | 0.05020238128368547 | -0.0016863791898651878 |
| only_carpus_digitus | 0.21029505342189264 | 0.2187573952845916 | 0.008462341862698952 |
| only_humerus | 0.057539652183901016 | 0.06056704757749361 | 0.0030273953935925912 |
| only_radius | 0.051292928139634604 | 0.05223634346022014 | 0.0009434153205855333 |
| only_carpus | 0.2401570585522863 | 0.23420398311831847 | -0.005953075433967836 |
| only_digitus | 0.24929323232818773 | 0.24796420309308906 | -0.001329029235098672 |

Per-seed `J` (1701, 2603, 3907):

- Dense: 0.15309151498422555, 0.1517411890074014, 0.15298057940069976
- Terminal: 0.15415953282067626, 0.15922944076158416, 0.15745100480580923

Fold scores (0–7):

- Dense: 0.09527745291849285, 0.0716748066234428, 0.07239287284932439, 0.08702522203087039, 0.11434127828575213, 0.08229259724609363, 0.08132113518159564, 0.09137146763886249
- Terminal: 0.10013163165084414, 0.0685474171442802, 0.06683605616085447, 0.08404880102404892, 0.12706496144543436, 0.08432843509481594, 0.08249708322000782, 0.10156937195525535

## Historical frozen terminal-versus-Dense contrast

| Condition | Observation | Pass? |
|---|---|---|
| `J` lower | terminal 0.1548023780834887 vs Dense 0.15019887157024805 | **False** |
| `WM` lower | terminal 0.24796420309308906 vs Dense 0.24929323232818773 | **True** |
| `CR` lower | terminal 0.2955808091341122 vs Dense 0.3026862750193508 | **True** |
| `PE` lower | terminal 0.003081123703617304 vs Dense 0.004193774320274617 | **True** |
| `F` non-harm within `1e-12 m` | terminal 0.05059202833156673 vs Dense 0.03430830346646642 | **False** |
| Fold wins | 3/8; threshold 6 | **False** |
| Seed-`J` wins | 0/3; threshold 2 | **False** |
| Frozen gate | not all conditions passed | **False** |
| Full-42 authorization | only contexts 0–23 | **False** |

Terminal did not improve the complete frozen gate. It reduced `WM`, `CR`, and `PE`, while `J` and `F` were higher. The largest absolute view change was `only_carpus_digitus`, +0.008462341862698952 m. The largest cardinality change was cardinality 3, +0.004237856339821017 m; cardinality 1 was −0.0008278234887220681 m. These are observations, not causal component evidence.

## Expanded three-candidate primary-8 metrics

Dense and terminal below are reused and must remain identical to the historical receipt.

| Candidate | Fits | Amean | WM | WS | J | F | CR | PE | Cardinality 1 / 2 / 3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dense | 24 | 0.0869621040968043 | 0.24929323232818773 | 0.11434127828575213 | 0.15019887157024805 | 0.03430830346646642 | 0.3026862750193508 | 0.004193774320274617 | 0.1495707178010024 / 0.07605982849554355 / 0.04070690379449729 |
| MoFe-only | 24 | 0.09153572782095491 | 0.249346780367841 | 0.1297633094856143 | 0.1568819392248034 | 0.05355406488712803 | 0.29726441228679407 | 0.00321223199006467 | 0.15100089564337488 / 0.08156290236677068 / 0.04702979817981136 |
| Terminal | 24 | 0.08937796971194266 | 0.24796420309308906 | 0.12706496144543436 | 0.1548023780834887 | 0.05059202833156673 | 0.2955808091341122 | 0.003081123703617304 | 0.14874289431228033 / 0.07942349303013377 / 0.04494476013431831 |

### Expanded fifteen-view projection

Negative signed differences favor the minuend candidate.

| View | Dense (m) | MoFe-only (m) | Terminal (m) | MoFe − Dense (m) | Terminal − MoFe (m) |
|---|---:|---:|---:|---:|---:|
| full | 0.02809908039860877 | 0.03562393008618739 | 0.03351162141139385 | 0.007524849687578619 | -0.0021123086747935407 |
| drop_humerus | 0.049857742005619134 | 0.05312810430279586 | 0.05094599687024019 | 0.0032703622971767266 | -0.0021821074325556675 |
| drop_radius | 0.0554299252455645 | 0.06045009472481845 | 0.05864118415789407 | 0.005020169479253954 | -0.0018089105669243793 |
| drop_carpus | 0.029729290943180955 | 0.03667160474192519 | 0.03457192511844926 | 0.006942313798744236 | -0.0020996796234759307 |
| drop_digitus | 0.027810656983624574 | 0.03786938894970592 | 0.03561993439068975 | 0.010058731966081348 | -0.0022494545590161716 |
| only_humerus_radius | 0.029838237927251536 | 0.03883041188859207 | 0.03656161678838131 | 0.008992173961340534 | -0.0022687951002107604 |
| only_humerus_carpus | 0.05684599862908497 | 0.062035680653920594 | 0.06010728305260644 | 0.005189682024835621 | -0.0019283976013141566 |
| only_humerus_digitus | 0.05614107471196515 | 0.0601135463196848 | 0.05828325047635324 | 0.003972471607719651 | -0.00183029584333156 |
| only_radius_carpus | 0.05134984580951632 | 0.05498096547181951 | 0.052629031295184515 | 0.0036311196623031866 | -0.0023519341766349913 |
| only_radius_digitus | 0.051888760473550656 | 0.05255788379040037 | 0.05020238128368547 | 0.0006691233168497138 | -0.0023555025067149016 |
| only_carpus_digitus | 0.21029505342189264 | 0.22085892607620672 | 0.2187573952845916 | 0.010563872654314077 | -0.002101530791615125 |
| only_humerus | 0.057539652183901016 | 0.06281353327087341 | 0.06056704757749361 | 0.005273881086972396 | -0.0022464856933798044 |
| only_radius | 0.051292928139634604 | 0.0548954800653577 | 0.05223634346022014 | 0.003602551925723098 | -0.0026591366051375645 |
| only_carpus | 0.2401570585522863 | 0.23694778886942747 | 0.23420398311831847 | -0.0032092696828588396 | -0.0027438057511089964 |
| only_digitus | 0.24929323232818773 | 0.249346780367841 | 0.24796420309308906 | 5.3548039653267265e-05 | -0.0013825772747519394 |

- MoFe per-seed `J` (1701/2603/3907): 0.15656269320580296, 0.16083734709979078, 0.15934892458954072
- MoFe fold scores (0–7): 0.10044187623304025, 0.06956767664321489, 0.06871740651832034, 0.08792449447469035, 0.1297633094856143, 0.08642750818137547, 0.08446753578875243, 0.1049760152426314

## Adaptive terminal-versus-MoFe contrast

This reports the preregistered `cagrad_vs_mofe` gate on adaptive evidence; it does not make the follow-up confirmatory.

| Condition | Observation | Pass? |
|---|---|---|
| `J` strictly lower | terminal 0.1548023780834887 vs MoFe-only 0.1568819392248034 | **True** |
| `WM` strictly lower | terminal 0.24796420309308906 vs MoFe-only 0.249346780367841 | **True** |
| `CR` non-harm within `1e-12` | terminal 0.2955808091341122 vs MoFe-only 0.29726441228679407 | **True** |
| `PE` non-harm within `1e-12 m` | terminal 0.003081123703617304 vs MoFe-only 0.00321223199006467 | **True** |
| `F` non-harm within `1e-12 m` | terminal 0.05059202833156673 vs MoFe-only 0.05355406488712803 | **True** |
| Fold wins | 8/8; threshold 6 | **True** |
| Seed-`J` wins | 3/3; threshold 2 | **True** |
| Frozen gate | all conditions passed | **True** |
| Final/full-42 authorization | both false | **False** |

## Descriptive MoFe-versus-Dense comparison

No frozen `mofe_vs_dense` gate is created.

- Largest view change: `only_carpus_digitus`, +0.010563872654314077 m (MoFe-only 0.22085892607620672 vs Dense 0.21029505342189264).
- Largest cardinality change: cardinality 3, +0.006322894385314068 m (MoFe-only 0.04702979817981136 vs Dense 0.04070690379449729).
- Other observations: MoFe-only had lower `CR` and `PE` than Dense but higher `Amean`, `WM`, `WS`, `J`, and `F`. Its only negative view delta was `only_carpus`, −0.0032092696828588396 m. This is descriptive; no new gate is defined.

## Interpretation

The initial evidence failed the terminal-versus-Dense gate. Terminal reduced worst-mask and adjacent-lattice violation measures but had higher `J` and `F`; three folds and zero seeds favored terminal on frozen diagnostics.

Adaptive follow-up: terminal passed every frozen `cagrad_vs_mofe` condition on this outcome-informed comparison, with 8/8 fold-score wins and 3/3 seed-`J` wins. Terminal was lower than MoFe-only on all 15 view summaries; the largest absolute view difference was `only_carpus`, −0.0027438057511089964 m, the largest cardinality difference was cardinality 1, −0.00225800133109455 m, and the full-view difference was −0.0021123086747935407 m. These paired observations do not establish a causal CAGrad effect or authorize a final/full-42 claim.

MoFe-only narrows an incremental terminal-versus-MoFe contrast but cannot establish that CAGrad caused a difference: execution selection was outcome-informed, data and comparator results are reused, and there is no independent physical replication or multiplicity-adjusted confirmation. Static experts, one-stage DMoME, and two-stage ERM remain unfit.

## Scope and limitations

- Missing sensors are simulated; this is not physical fault injection or deployment drift.
- The target is external OpenAXES UR5 base-frame TCP in metres. This does not validate local WT901C/`vive_world`, ShapeFK, an IMU-only controller, physical failure, online inference, or control readiness.
- One public dataset with eight scenarios is used; folds/seeds are paired contexts, not independent experiments.
- Parameter matching does not equalize optimization dynamics or inductive bias.
- MoFe-only was frozen in the roster but executed after initial outcomes; its contrasts are adaptive/exploratory.
- The six-candidate screen and report-only folds remain incomplete; no final selection or full-42 claim is allowed.
- Static expert, one-stage DMoME, and two-stage ERM remain absent; the ablation chain is incomplete.
- No broad state-of-the-art, causal-component, or architectural-novelty claim is supported.

## Related work and rationale

- [SimMLM (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.html) is the direct DMoME+MoFe precedent.
- [MoME (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Resilient_Sensor_Fusion_Under_Adverse_Sensor_Failures_via_Multi-Modal_Expert_CVPR_2025_paper.html) is separate expert-fusion prior work.
- [CAGrad (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html) supplies the conflict-averse gradient mechanism.
- [Benchmarking Multi-modal Semantic Segmentation under Sensor Failures (CVPRW 2025)](https://openaccess.thecvf.com/content/CVPR2025W/TMM-OpenWorld/html/Liao_Benchmarking_Multi-modal_Semantic_Segmentation_under_Sensor_Failures_Missing_and_Noisy_CVPRW_2025_paper.html) motivates separating entire/random missingness and equal-combination/Bernoulli failure evaluation. It is a different task; Run 001 evaluates the exact 15 deterministic masks and does **not** test noisy or Bernoulli sensor failures.
- [MDA-KD (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Dai_A_Study_of_Dropout-Induced_Modality_Bias_on_Robustness_to_Missing_CVPR_2024_paper.html), [GMC (ICML 2022)](https://proceedings.mlr.press/v162/poklukar22a.html), and [MODA (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_MODA_Motion-Drift_Augmentation_for_Inertial_Human_Motion_Analysis_CVPR_2025_paper.html) remain future alternatives, not evaluated components.

## Completion checklist

- [x] Bind initial analysis/document receipts and exact 10+1 bundle.
- [x] Report every historical terminal-versus-Dense gate condition.
- [x] Replace every follow-up placeholder only from canonical 72-fit receipts.
- [x] Verify 72 immutable results/attempts, zero errors, and unchanged Dense/terminal evidence.
- [x] Bind 24 MoFe-only stage-one hash pairs and fixed accounting.
- [x] Bind the second exact 10+1/56-file bundle and unique three-candidate evaluation.
- [x] Bind the append-only operational recovery supplement and its action-accounting limitations.
- [x] Confirm comparison order `cagrad_vs_mofe`, then `terminal_vs_dense_primary`.
- [x] Confirm all full-42/final-claim flags remain false.
- [x] Preserve adaptive disclosure and incomplete-grid limitations.
- [x] Confirm no follow-up placeholder remains before the final results commit.
