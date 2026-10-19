# DACAN's MCP server: let your AI assistant install and run DACAN

`tools/dacan_mcp.py` is an [MCP](https://modelcontextprotocol.io) server for the AI assistant you already use
(Claude Code, Claude Desktop, Cursor, VS Code, Codex ...). Add it once, then ask in plain words:

- "Install DACAN for this PC."
- "Start DACAN." / "Stop DACAN."
- "Is DACAN running? How much VRAM does it use?"
- "Which DACAN model fits my PC?"
- "DACAN didn't start - what does the log say?"
- "How fast is DACAN on this PC?" (a short speed test)
- "How do I connect Open WebUI / the OpenAI SDK / Claude Code to DACAN?"

The assistant calls the server's tools. The tools run DACAN's own setup and start the model the way the start
scripts do. It is one Python file that uses only the standard library, so it works before setup has created
`.venv`. It needs Python 3.10 or newer, 64-bit on Windows.

It is not the same thing as [Tools from MCP servers](DETAILS.md#tools-from-mcp-servers). That feature
(`serve/mcp.py`) lets the *DACAN model* call tools from your MCP servers in its chat page. This page is the other
direction: *your assistant* manages DACAN.

## Add it

You need a copy of DACAN: the zip from GitHub, or `git clone https://github.com/tirex999/DACAN`. Below,
`C:\Users\you\DACAN` (Windows) and `/home/you/DACAN` (Linux) stand for that folder. Use `python3` on Linux. On
Windows, use `py` if `python` is not on your PATH.

### Claude Code

```bash
claude mcp add dacan -- python C:\Users\you\DACAN\tools\dacan_mcp.py         # Windows
claude mcp add dacan -- python3 /home/you/DACAN/tools/dacan_mcp.py           # Linux
```

Add `--scope user` to have it in every project. `/mcp` in Claude Code shows that it is connected.

### Claude Desktop

Open Settings > Developer > Edit Config (`claude_desktop_config.json`) and add:

```json
{
  "mcpServers": {
    "dacan": {
      "command": "python",
      "args": ["C:\\Users\\you\\DACAN\\tools\\dacan_mcp.py"]
    }
  }
}
```

On Linux, use `"command": "python3"` and `"args": ["/home/you/DACAN/tools/dacan_mcp.py"]`. Restart Claude
Desktop after the change.

### Cursor

Use `~/.cursor/mcp.json` for every project, or `.cursor/mcp.json` in one project. The format is the same as
Claude Desktop's:

```json
{
  "mcpServers": {
    "dacan": {
      "command": "python",
      "args": ["C:\\Users\\you\\DACAN\\tools\\dacan_mcp.py"]
    }
  }
}
```

### VS Code (GitHub Copilot agent mode)

Use `.vscode/mcp.json` in a workspace, or run "MCP: Open User Configuration" from the command palette:

```json
{
  "servers": {
    "dacan": {
      "type": "stdio",
      "command": "python",
      "args": ["C:\\Users\\you\\DACAN\\tools\\dacan_mcp.py"]
    }
  }
}
```

On Linux, use `"command": "python3"` and `"args": ["/home/you/DACAN/tools/dacan_mcp.py"]`.

### Codex CLI

In `~/.codex/config.toml`:

```toml
[mcp_servers.dacan]
command = "python"
args = ['C:\Users\you\DACAN\tools\dacan_mcp.py']    # Linux: command = "python3", args = ["/home/you/DACAN/tools/dacan_mcp.py"]
tool_timeout_sec = 180                                # dacan_stop / dacan_benchmark can take longer than 60 s
```

### Any other MCP client

The command is `python <DACAN folder>/tools/dacan_mcp.py`. The transport is stdio. It speaks MCP revision
2025-06-18, and also 2025-03-26 and 2024-11-05.

`--root <folder>` manages a DACAN folder other than the one the file is in. For example:
`python D:\tools\dacan_mcp.py --root D:\DACAN`.

## The tools

| Tool | What it does | Arguments (all optional) |
| --- | --- | --- |
| `dacan_status` | Whether a model is running: the model, engine version, context, busy or idle, VRAM and RAM in use. What is installed: models, start scripts, engine version, data folder, downloads. An install in progress. This PC's hardware: OS, CPU, RAM, GPUs, page file, free disk. The model and context setup recommends for this PC. | `port`, `hardware` (true/false) |
| `dacan_models` | Every family and size setup offers, with download size, RAM needed, whether it fits this PC, and whether it is installed. Also the context lengths and the recommendation. | - |
| `dacan_install` | Runs setup without questions, in the background (see below). Without `confirm: true` it only returns the plan. A second call shows the progress. `cancel: true` stops the install. | `family` (qwen, swift, coder, unsloth), `model` (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, IQ1_M, UD-IQ4_XS, UD-Q4_K_XL), `context` (8192 ... 524288), `vision` (no, yes, gpu, cpu), `kv` (int8, q4_0, k8v4), `gpu` (a number) or `gpus` ("0,2", "all"), `backend` (auto, cuda, hip), `low_ram`, `port`, `data_dir`, `confirm`, `cancel` |
| `dacan_start` | Starts an installed model in the background and waits until it answers. If it is not ready after `wait_seconds` (default 50), it returns "loading". | `model` (an id from `dacan_status`, e.g. `q2_0`, `coder-iq1_m`; default: the most recently used), `port`, `gpu`, `wait_seconds` (0-600) |
| `dacan_stop` | Unloads the model through the server's own API, then ends the server it started. | `port`, `force` |
| `dacan_logs` | The last lines of the server log, the engine log or the setup log. | `source` (server, engine, setup), `model`, `lines` (1-500) |
| `dacan_benchmark` | Sends a short fixed request (greedy, thinking off) and reports output and prompt tokens/s. | `port`, `max_tokens` (16-512) |
| `dacan_connect_info` | The OpenAI and Anthropic base URLs, the model id, whether a key is needed, and settings for Claude Code, Codex, the OpenAI SDK, curl, Cursor, Continue and Open WebUI. | `port` |

Every result is JSON with a one-line `summary`. It is sent both as text and as MCP structured content.

## How it works

**Install.** `dacan_install` uses setup's own logic and tables (it imports `setup.py`'s `MODELS`, `FAMILIES`,
`CONTEXTS` and RAM checks) to fill in what you left out:

- the size setup itself would pick: IQ3_XXS from 60 GB of RAM, Q2_0 from about 44 GB, the Coder from about 28 GB,
  and the Coder in the low-RAM mode below that when the GPU makes up for the RAM;
- the context setup recommends for the GPU's VRAM;
- images off.

The first call returns the plan: the download size (58-111 GB), the free disk space, the RAM check, and the exact
setup command. The assistant should show you the plan and call again with `confirm: true` once you agree.

It then runs the same steps `START-HERE.bat` / `setup.sh` run:

1. It creates `.venv` with the Python it runs on, if `.venv` is missing.
2. It runs `setup.py --yes --no-start --family ... --model ... --context ... --vision ...` as a background process.
   The output goes to `.dacan-mcp/setup.log` in the DACAN folder.

The tool returns at once. Call it again to see the step (1-7), the download percentage, or setup's error message.
Everything setup has finished is skipped on the next run, and downloads resume. If the AI app closes, the install
keeps running. The install is refused while a model this server started is running, because setup may replace the
engine files that model uses.

**Start.** `dacan_start` runs the same command as the `run-<model>` script, `serve/server.py --engine dacan
--config dacan-<model>.json --port N`, with the venv's Python, without `--open` (no browser) and without a console
window. The server's output goes to `.dacan-mcp/server.log`. The tool then polls `GET /health` until the model
answers. Loading takes 1-3 minutes. If the server ends while loading, the tool returns the end of its log. The model
keeps running after the AI app closes, until `dacan_stop`. If the AI app runs its tools inside a Windows job that
does not allow processes to leave it, the model ends together with the app.

Two cases are refused:

- a model is already running: the tool returns it and starts nothing;
- another program uses the port: the tool says so.

**Stop.** `dacan_stop` first sends `POST /unload`. The server sends the engine `QUIT`, and the engine frees its
VRAM and pinned RAM itself, as when you close the window. Then the server ends:

- Linux: SIGTERM, which runs the server's own Ctrl+C path.
- Windows: the process ends. Its job object ends the vision encoder and the MCP servers it started.

While a request is running, the tool refuses unless you pass `force: true`.

**Status.** The tool reads `GET /health`, `/v1/status`, `/status` and, on older servers, `/metrics` on 127.0.0.1.
If a config of the folder sets an `api_key`, the tool sends that key with its own requests. It never shows the key.

## Safety

- **No shell, no free paths.** Every argument is checked against a fixed schema and against setup's own choices. An
  unknown argument, a model or family setup does not offer, or a `gpus` value other than digits and commas is
  refused before anything runs. Commands are passed as argument lists; nothing goes through a shell.
- **Paths.** The only path argument is `data_dir`. It must be one of:
  - the current DACAN data folder;
  - a folder inside the DACAN folder;
  - a new absolute folder whose name starts with `DACAN` (e.g. `D:\DACAN-data`), whose parent exists, and which is
    not in a system folder.

  `..`, relative paths and network (UNC) paths are refused. Logs are read only from `.dacan-mcp/` and from the
  engine log a config names, if that log is inside the DACAN folder or the data folder.
- **Only its own processes.** The server records the process id *and start time* of every process it starts, in
  `.dacan-mcp/*.json`. It only ends a process when both match, so a process id the OS has given to another program
  is never touched. A DACAN started some other way (its window, a run script, a service) is only asked over its
  API to unload the model. To end that server, close its window.
- **Big downloads need a yes.** `dacan_install` does nothing until it is called with `confirm: true`. The plan
  shows the download size and the free space first, and the install is refused if the disk is too small.
- **Local only.** The server checks `127.0.0.1` and never changes the network settings: it does not install with
  `--host 0.0.0.0` or an API key. To share DACAN on your network, run setup yourself (see
  [Using it](DETAILS.md#using-it)). Experimental options are left off too: the speed projection, rope scaling past
  262K and `--build`. Run setup by hand for those.

## Troubleshooting

- **The assistant does not see the tools.** Run the command by hand:
  `python C:\Users\you\DACAN\tools\dacan_mcp.py`. It should print `[dacan-mcp] ready (DACAN folder ...)` to
  stderr and wait for input (Ctrl+C ends it). If it reports `not a DACAN folder`, the path is wrong or `--root` is
  needed. Python older than 3.10 is refused.
- **The install failed.** Ask the assistant for the setup log (`dacan_logs` with `source: setup`), or open
  `.dacan-mcp/setup.log`. Setup's `[X]` line says what to fix. Then install again: every finished step is skipped.
- **The model does not start.** Get the server log with `dacan_logs`, and the engine log with `source: engine`.
  [Troubleshooting](DETAILS.md#troubleshooting) has the usual causes.
- **Tests.** `python -m unittest tools/test_dacan_mcp.py` runs the tests with a fake setup and server. They do not
  download anything or use the GPU.
