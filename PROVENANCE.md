# Files that differ from the code and records that produced the results

The result records name the SHA-256 of the code that wrote them and of every input they read. Every file copied into this repository from the authors' working repository is a byte-for-byte copy of the file at the same path there, except the 12 below. Their changes are path and machine settings only, and their diffs are given in full.

The other files were written for this release and have no counterpart that produced a result: `README.md`, `LICENSE`, `CITATION.cff`, this file, `requirements.txt`, `requirements-tables.txt`, `.gitignore`, `.gitattributes` and `config/example.env`. (The working repository has its own `requirements.txt`, `.gitignore` and `.gitattributes`; these are not copies of them.)

Two of the changes are not paths. Three modules pin the SHA-256 of `src/p2auroc_d424.py` and refuse to run when it differs; since that file's home path changed, those pins now hold its public SHA-256. The original SHA-256 in the second column is the value the result records and the original pins name.

In the diffs, the compute node's account name is written `<account>` on the removed lines. It is the only character run that is masked, and it occurs only inside the removed `/home/...` paths.

| File | SHA-256 of the original | SHA-256 here | Change | Lines -/+ |
|---|---|---|---|---|
| `src/config_table_d472.py` | `cb5f72142f9d1b805d02a8880b13787a8d9a0f4133dc9258cb1af87cd9e5f46e` | `26f13e909385d3bc2197a439263bc7a7b8f25a9991d78f2c78ac9fc2952a16c4` | output directory from GUARDRAIL_PAPER_DIR (default paper/, as before) | 1/3 |
| `src/overrefusal_p2_d440.py` | `ab9f3f894cddb37b1e0da1aaa820dae655fba2dd719ce619faac47904c5bdcd7` | `1395467046510231283f075566fa3764ceed8efd2f43727170f5ed4f1b9ac856` | the compute node's home path literal replaced by GUARDRAIL_NODE_HOME (default: the current user's home); its REUSED_SHA256 pin of src/p2auroc_d424.py set to that file's public bytes | 2/2 |
| `src/p2auroc_d424.py` | `dc6284489c17feb48781ce87e64a6915e6a1662c3b85ace09770f84ec98687c9` | `edc7346487c7d9369526602b0cd54176db04ffb9f319c4276ad4ae4f7d1c106f` | the compute node's home path literal replaced by GUARDRAIL_NODE_HOME (default: the current user's home) | 1/1 |
| `src/p2auroc_d424_grouped.py` | `647e45e5317098365df30ea4496965b66c372b2bc1b829e8b2a95d9afae6a8e6` | `68dc34c2c196cbcd64d48d5780735f2eb519470fe1255511a768d9941af0582b` | its REUSED_SHA256 pin of src/p2auroc_d424.py set to that file's public bytes (the pin is checked at run time; with the original value the module refuses to run) | 1/1 |
| `src/pairs_table_d468.py` | `4ce7aa30c7e8455468e65cfa0f9e62c338bd52b84c142b7edeb12fcb5c7b4ec7` | `0e2f6f06f158345edd18ce3eecbd4ee1762deb19ec9ccc2cfae50fa672cba3cc` | output directory from GUARDRAIL_PAPER_DIR (default paper/, as before) | 1/3 |
| `src/recalnoise_d433.py` | `ff92237abba65956c3eab3f7977ccd0a0dbffaabcfac9090121f5047352bfbf1` | `b55df292f37179ab97a9b73d538654403600284688b3d4a85fec88df431848f7` | the compute node's home path literal replaced by GUARDRAIL_NODE_HOME (default: the current user's home); its REUSED_SHA256 pin of src/p2auroc_d424.py set to that file's public bytes | 2/2 |
| `src/results_figures_d570.py` | `3ef5205594587ffdfb69248170b47c537dd6b964024713a0c209d7bdc5b265fe` | `d7af301df99b2cd8049f877e4bf5ad110bbc5bd5da55b520d29aa83487aae713` | output directory from GUARDRAIL_PAPER_DIR (default paper/, as before) | 2/4 |
| `src/results_tables_d504.py` | `8339267273ca3bf7d54fac7a5bf5ddf24f779ee7bf55f3943099dd3740b33710` | `8d5cef2259771d84e20181ac4c670e75300008fa23b3337e3380bf923b5450b4` | output directory from GUARDRAIL_PAPER_DIR (default paper/, as before) | 1/3 |
| `src/supp_tables_d585.py` | `f71a75e6034c12a6a46623b1ba998ab0d4abc888f8085b4058dd6557106701b1` | `754ec3d1c6e308244b527c9194f32ed69a26f6dcfad08ec0f04101064db400e9` | output directory, and the directory it reads the regenerated tables_results.tex from, from GUARDRAIL_PAPER_DIR (default paper/, as before) | 2/4 |
| `results/phase2/dataset_manifest_p2.json` | `b51e8419ccbf035581d7f4960d422a7a1a9d46eca35b3c47cfe46e6507d1621e` | `c0a12f87cb970ab48bb052710b3bb5887b68fc1b81a1bb1cd07a84ca74544227` | five absolute paths under the compute node's home directory: '/home/<account>' replaced by '<HOME>' | 5/5 |
| `results/phase2/dataset_manifest_p2sst.json` | `a339c9593e894f2be33ef96a5a8a932c0119bdf7db6f0cd8e5cb80932a409954` | `b056e5533224248029c9a14da52d17e5c1c0c0a45dfe60baefa725a7400d084a` | five absolute paths under the compute node's home directory: '/home/<account>' replaced by '<HOME>' | 5/5 |
| `results/phase3_capcost_wgtest_split.json` | `aac7f4e42ef84e98506bc6449167704e9970267329cc3c54a279aed142913a9c` | `99b0cd4ea674513fc8970e151beae609c0e0bcad8f03d7654619db1c425952d8` | five absolute paths under the compute node's home directory: '/home/<account>' replaced by '<HOME>' | 5/5 |

## Diffs

### `src/config_table_d472.py`

```diff
--- guardrail-ccf/src/config_table_d472.py
+++ src/config_table_d472.py
@@ -23,2 +23,3 @@
 import json
+import os
 import sys
@@ -28,3 +29,4 @@
 REC = "docs/paper/config_deltas_d471.json"
-OUT = ROOT / "paper" / "tables_config.tex"
+PAPER_DIR = ROOT / os.environ.get("GUARDRAIL_PAPER_DIR", "paper")  # where the .tex files are written (README, config/example.env)
+OUT = PAPER_DIR / "tables_config.tex"
 LABEL = {"rope_theta": "RoPE base", "rope_scaling": "RoPE scaling", "sliding_window": "sliding window",
```

### `src/overrefusal_p2_d440.py`

```diff
--- guardrail-ccf/src/overrefusal_p2_d440.py
+++ src/overrefusal_p2_d440.py
@@ -89,3 +89,3 @@
     "src/overrefusal_d156.py": "11bad9000dc6c70ee664d0784b167ad8bc08ee6a671be26b3a12e6be995d007f",
-    "src/p2auroc_d424.py": "dc6284489c17feb48781ce87e64a6915e6a1662c3b85ace09770f84ec98687c9",
+    "src/p2auroc_d424.py": "edc7346487c7d9369526602b0cd54176db04ffb9f319c4276ad4ae4f7d1c106f",
     **A.REUSED_SHA256,
@@ -695,3 +695,3 @@
     jobs: list[tuple[str, str, str]] = [("xstest_p2_sandbox", "-", "bash scripts/sandbox_xstest_p2.sh "
-                                         "/home/<account>/.cache/gccf_scratch/xstest_p2_sandbox_20261013")]
+                                         f"{os.environ.get('GUARDRAIL_NODE_HOME') or Path.home().as_posix()}/.cache/gccf_scratch/xstest_p2_sandbox_20261013")]
     for pooling in POOLINGS:
```

### `src/p2auroc_d424.py`

```diff
--- guardrail-ccf/src/p2auroc_d424.py
+++ src/p2auroc_d424.py
@@ -1079,3 +1079,3 @@
 LANE_OF = {"usermean": "C", "lastgen": "B"}               # D-424 ran on C; D-429's last-token list on B (idle, F-086)
-SANDBOX_ROOT = "/home/<account>/.cache/gccf_scratch"       # the node's NVMe scratch (outside the repo)
+SANDBOX_ROOT = f"{os.environ.get('GUARDRAIL_NODE_HOME') or Path.home().as_posix()}/.cache/gccf_scratch"  # the node's NVMe scratch (outside the repo)
 
```

### `src/p2auroc_d424_grouped.py`

```diff
--- guardrail-ccf/src/p2auroc_d424_grouped.py
+++ src/p2auroc_d424_grouped.py
@@ -48,3 +48,3 @@
 REUSED_SHA256 = {
-    "src/p2auroc_d424.py": "dc6284489c17feb48781ce87e64a6915e6a1662c3b85ace09770f84ec98687c9",
+    "src/p2auroc_d424.py": "edc7346487c7d9369526602b0cd54176db04ffb9f319c4276ad4ae4f7d1c106f",
     **Q.REUSED_SHA256,
```

### `src/pairs_table_d468.py`

```diff
--- guardrail-ccf/src/pairs_table_d468.py
+++ src/pairs_table_d468.py
@@ -29,2 +29,3 @@
 import json
+import os
 import sys
@@ -33,3 +34,4 @@
 ROOT = Path(__file__).resolve().parent.parent
-OUT = ROOT / "paper" / "tables_pairs.tex"
+PAPER_DIR = ROOT / os.environ.get("GUARDRAIL_PAPER_DIR", "paper")  # where the .tex files are written (README, config/example.env)
+OUT = PAPER_DIR / "tables_pairs.tex"
 
```

### `src/recalnoise_d433.py`

```diff
--- guardrail-ccf/src/recalnoise_d433.py
+++ src/recalnoise_d433.py
@@ -111,3 +111,3 @@
     "src/refsize_d382.py": "36d1a7709904b21cfdc522aa1feb61f151a0d0d10b14a57b99bbfb3ec507e81a",
-    "src/p2auroc_d424.py": "dc6284489c17feb48781ce87e64a6915e6a1662c3b85ace09770f84ec98687c9",
+    "src/p2auroc_d424.py": "edc7346487c7d9369526602b0cd54176db04ffb9f319c4276ad4ae4f7d1c106f",
     "src/twomoment_p2_d361.py": "843a6c13316545b550c3c1850840707a45a409bb8cdb63d5b8d4b863339ddea1",
@@ -150,3 +150,3 @@
 HEALTH = tuple(RS.HEALTH)                               # healthcheck_d156.health's five statistics (the anchor)
-NODE_HOME = "/home/<account>"                            # the node account's home (the lane lists' scratch paths)
+NODE_HOME = os.environ.get("GUARDRAIL_NODE_HOME") or Path.home().as_posix()  # the node account's home (the lane lists' scratch paths)
 SANDBOX_P1, SANDBOX_P2 = "repair_llama3_to_31", "mistral_v01_v02"   # two selected cells each; L* not selected in P2
```

### `src/results_figures_d570.py`

```diff
--- guardrail-ccf/src/results_figures_d570.py
+++ src/results_figures_d570.py
@@ -40,2 +40,3 @@
 import math
+import os
 import sys
@@ -49,4 +50,5 @@
 
-OUT_MAIN = ROOT / "paper" / "figures_results.tex"
-OUT_SUPP = ROOT / "paper" / "figures_supplement.tex"
+PAPER_DIR = ROOT / os.environ.get("GUARDRAIL_PAPER_DIR", "paper")  # where the .tex files are written (README, config/example.env)
+OUT_MAIN = PAPER_DIR / "figures_results.tex"
+OUT_SUPP = PAPER_DIR / "figures_supplement.tex"
 DATA = RT.DATA
```

### `src/results_tables_d504.py`

```diff
--- guardrail-ccf/src/results_tables_d504.py
+++ src/results_tables_d504.py
@@ -54,2 +54,3 @@
 import math
+import os
 import sys
@@ -64,3 +65,4 @@
 
-OUT = ROOT / "paper" / "tables_results.tex"
+PAPER_DIR = ROOT / os.environ.get("GUARDRAIL_PAPER_DIR", "paper")  # where the .tex files are written (README, config/example.env)
+OUT = PAPER_DIR / "tables_results.tex"
 EOL = " \\\\"
```

### `src/supp_tables_d585.py`

```diff
--- guardrail-ccf/src/supp_tables_d585.py
+++ src/supp_tables_d585.py
@@ -44,2 +44,3 @@
 import re
+import os
 import sys
@@ -53,4 +54,5 @@
 
-OUT = ROOT / "paper" / "tables_supplement.tex"
-HEADLINE_TEX = ROOT / "paper" / "tables_results.tex"
+PAPER_DIR = ROOT / os.environ.get("GUARDRAIL_PAPER_DIR", "paper")  # where the .tex files are written (README, config/example.env)
+OUT = PAPER_DIR / "tables_supplement.tex"
+HEADLINE_TEX = PAPER_DIR / "tables_results.tex"
 CALC = "docs/paper/review_20261016b_calc.json"
```

### `results/phase2/dataset_manifest_p2.json`

```diff
--- guardrail-ccf/results/phase2/dataset_manifest_p2.json
+++ results/phase2/dataset_manifest_p2.json
@@ -77,3 +77,3 @@
     "snapshot": "datasets--allenai--wildguardmix/snapshots/d29c47f41c8b51348b5c8e8c81c039b3132b66d1",
-    "local_snapshot_path": "/home/<account>/.cache/gccf_scratch/p2_datasets/hub/datasets--allenai--wildguardmix/snapshots/d29c47f41c8b51348b5c8e8c81c039b3132b66d1",
+    "local_snapshot_path": "<HOME>/.cache/gccf_scratch/p2_datasets/hub/datasets--allenai--wildguardmix/snapshots/d29c47f41c8b51348b5c8e8c81c039b3132b66d1",
     "snapshot_files_sha256": {
@@ -95,5 +95,5 @@
     "datasets_cache": {
-      "dir": "/home/<account>/.cache/hf/gccf_datasets_cache/phase2_freeze_splits.20261001T151112Z.1434541.1f45147e",
-      "root": "/home/<account>/.cache/hf/gccf_datasets_cache",
-      "mount_point": "/home/<account>/.cache",
+      "dir": "<HOME>/.cache/hf/gccf_datasets_cache/phase2_freeze_splits.20261001T151112Z.1434541.1f45147e",
+      "root": "<HOME>/.cache/hf/gccf_datasets_cache",
+      "mount_point": "<HOME>/.cache",
       "mount_source": "/dev/nvme1n1p1",
@@ -101,3 +101,3 @@
       "note": "HF_DATASETS_CACHE is a fresh directory made for this run on NVMe before datasets is imported. Any cleanup the datasets library does of its own Arrow cache (on a failed load its builder removes its '.incomplete' directory, freeze_splits.load_axis -> load_dataset) is therefore confined to this run's directory. That library-internal cleanup is the one remaining deletion in this pipeline, and it is the author's to accept. The directory is left after the run; removal is a human's scripts/safe_rm.sh call.",
-      "library_reported": "/home/<account>/.cache/hf/gccf_datasets_cache/phase2_freeze_splits.20261001T151112Z.1434541.1f45147e"
+      "library_reported": "<HOME>/.cache/hf/gccf_datasets_cache/phase2_freeze_splits.20261001T151112Z.1434541.1f45147e"
     }
```

### `results/phase2/dataset_manifest_p2sst.json`

```diff
--- guardrail-ccf/results/phase2/dataset_manifest_p2sst.json
+++ results/phase2/dataset_manifest_p2sst.json
@@ -84,3 +84,3 @@
     "snapshot": "datasets--nyu-mll--glue/snapshots/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
-    "local_snapshot_path": "/home/<account>/.cache/gccf_scratch/p2_datasets/hub/datasets--nyu-mll--glue/snapshots/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
+    "local_snapshot_path": "<HOME>/.cache/gccf_scratch/p2_datasets/hub/datasets--nyu-mll--glue/snapshots/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
     "snapshot_files_sha256": {
@@ -134,5 +134,5 @@
     "datasets_cache": {
-      "dir": "/home/<account>/.cache/hf/gccf_datasets_cache/phase2_freeze_splits.20261003T050212Z.2199293.273ab958",
-      "root": "/home/<account>/.cache/hf/gccf_datasets_cache",
-      "mount_point": "/home/<account>/.cache",
+      "dir": "<HOME>/.cache/hf/gccf_datasets_cache/phase2_freeze_splits.20261003T050212Z.2199293.273ab958",
+      "root": "<HOME>/.cache/hf/gccf_datasets_cache",
+      "mount_point": "<HOME>/.cache",
       "mount_source": "/dev/nvme1n1p1",
@@ -140,3 +140,3 @@
       "note": "HF_DATASETS_CACHE is a fresh directory made for this run on NVMe before datasets is imported. Any cleanup the datasets library does of its own Arrow cache (on a failed load its builder removes its '.incomplete' directory, freeze_splits.load_axis -> load_dataset) is therefore confined to this run's directory. That library-internal cleanup is the one remaining deletion in this pipeline, and it is the author's to accept. The directory is left after the run; removal is a human's scripts/safe_rm.sh call.",
-      "library_reported": "/home/<account>/.cache/hf/gccf_datasets_cache/phase2_freeze_splits.20261003T050212Z.2199293.273ab958"
+      "library_reported": "<HOME>/.cache/hf/gccf_datasets_cache/phase2_freeze_splits.20261003T050212Z.2199293.273ab958"
     }
```

### `results/phase3_capcost_wgtest_split.json`

```diff
--- guardrail-ccf/results/phase3_capcost_wgtest_split.json
+++ results/phase3_capcost_wgtest_split.json
@@ -101,3 +101,3 @@
       "snapshot": "datasets--allenai--wildguardmix/snapshots/d29c47f41c8b51348b5c8e8c81c039b3132b66d1",
-      "local_snapshot_path": "/home/<account>/.cache/gccf_scratch/p2_datasets/hub/datasets--allenai--wildguardmix/snapshots/d29c47f41c8b51348b5c8e8c81c039b3132b66d1",
+      "local_snapshot_path": "<HOME>/.cache/gccf_scratch/p2_datasets/hub/datasets--allenai--wildguardmix/snapshots/d29c47f41c8b51348b5c8e8c81c039b3132b66d1",
       "snapshot_files_sha256": {
@@ -115,5 +115,5 @@
       "datasets_cache": {
-        "dir": "/home/<account>/.cache/hf/gccf_datasets_cache/capcost_p3.20261008T130403Z.976791.f6b06bdd",
-        "root": "/home/<account>/.cache/hf/gccf_datasets_cache",
-        "mount_point": "/home/<account>/.cache",
+        "dir": "<HOME>/.cache/hf/gccf_datasets_cache/capcost_p3.20261008T130403Z.976791.f6b06bdd",
+        "root": "<HOME>/.cache/hf/gccf_datasets_cache",
+        "mount_point": "<HOME>/.cache",
         "mount_source": "/dev/nvme1n1p1",
@@ -121,3 +121,3 @@
         "note": "HF_DATASETS_CACHE is a fresh directory made for this run on NVMe before datasets is imported. Any cleanup the datasets library does of its own Arrow cache (on a failed load its builder removes its '.incomplete' directory, freeze_splits.load_axis -> load_dataset) is therefore confined to this run's directory. That library-internal cleanup is the one remaining deletion in this pipeline, and it is the author's to accept. The directory is left after the run; removal is a human's scripts/safe_rm.sh call.",
-        "library_reported": "/home/<account>/.cache/hf/gccf_datasets_cache/capcost_p3.20261008T130403Z.976791.f6b06bdd"
+        "library_reported": "<HOME>/.cache/hf/gccf_datasets_cache/capcost_p3.20261008T130403Z.976791.f6b06bdd"
       }
```
