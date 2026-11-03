# DACAN build 5

Release: **v0.1.42-build5**. The engine's version string remains **0.1.42**; `BUILD.json` identifies this
snapshot with **`dacan: 5`**. Build 4 is the earlier public release, not this snapshot.

This release packages the current DACAN-2 unified engine code with the server and web interface from the
running reference installation. It includes the context/KV/thread controls and the RU/EN interface together.
The installer and MCP offer **1,048,576 tokens (1M, shown as 1024K)**. Setup uses YaRN with factor 4 for 1M.

The Windows and portable Linux engines are built from the same source snapshot, with CUDA architectures
75, 80, 86, 89 and 120. Linux targets glibc 2.35 and newer. A separately named reference archive, when present,
contains the exact Linux executable from the reference machine; its newer system requirements are recorded
in that archive's `BUILD.json`. The installer uses the portable archive.

The public release contains executables, the installer and the server/UI. Engine source stays private.
`RELEASE.json` records the binaries and the matching product-file hashes so the delivered version can be checked.

Update with `UPDATE.bat` / `./update.sh`. The build number causes build 4 to be replaced. Updating keeps an
existing model's selected context. For a new installation, choose 1M explicitly when the machine has sufficient
memory; the hardware recommendation is kept for smaller cards and machines.

The reference Flash profile uses 1M, int8 KV, a resident KV window of 32,768 and YaRN ×4. Prefill is 16,384;
the accepted single-sequence profile uses FIFO for multiple clients. GPU order, layer split, CPU threads and
model paths depend on the machine and are not copied blindly to other installations.

Historical 512K measurements remain labelled with their original context. They are not a current context limit.
