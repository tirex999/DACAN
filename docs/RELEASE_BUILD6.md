# DACAN build 6

Release: **v0.1.42-build6**. Engine version **0.1.42**, `BUILD.json` field **`dacan: 6`**.

This update ships the current engine source snapshot and complete matching server/UI, including
four-slot controls, 1M context selection, KV/CPU settings, improved cancellation and slot reuse, and
conversation-cache retention controls. Windows and Linux portable engines cover CUDA75/80/86/89/120;
Linux targets glibc2.35+. The exact reference Linux archive is labelled separately.

The measured two-socket multi-user profile reaches **90.9–96.2 aggregate tokens/s** across four Russian
clients with 512 output tokens each, including admissions. See [workload and settings](MULTIUSER_BUILD6.md).
The profile uses a 1M context limit; the benchmark prompts were short. These figures are not per-client rates.

Update with `UPDATE.bat` / `./update.sh`. Existing model paths, chosen context and settings are preserved.
Hardware-specific tuning is documented as an optional measured profile, not imposed on every installation.
`RELEASE.json` records asset and server/UI hashes. Engine implementation source remains private.
