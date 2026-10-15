# -*- coding: utf-8 -*-
# One prompt at every temperature from 0.1 to 1.5 (step 0.1) on a running DACAN service, sent the way its web chat sends
# a message: one user message, no system prompt, reasoning_effort high, top_p 0.95, top_k 20, the temperature explicit,
# the answer capped at 65 536 tokens. The service has one batch slot, so before each run the script waits up to 5 min for
# a 60-s pause in other traffic (read from the service journal), then runs anyway.
#
#   python3 temp_progon.py <prompt.txt> <out_dir> [--url http://127.0.0.1:8000] [--temps 0.1,0.2,...]
#
# out_dir gets t<T>.html (the HTML file from the answer), t<T>.json (the whole answer), itogi.jsonl (one line per run:
# temperature, start, seconds, answer tokens, finish reason, reasoning length, HTML size) and progon.log.
# Used for docs/temperature/ of this site (15.10.2026, the pelican prompt first).
import argparse
import json
import os
import re
import subprocess
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--temps", default=",".join("%.1f" % (0.1 * i) for i in range(1, 16)))
a = ap.parse_args()
PROMPT = open(a.prompt, encoding="utf-8").read().strip()
TEMPS = [float(x) for x in a.temps.split(",")]
os.makedirs(a.out, exist_ok=True)
LOG = os.path.join(a.out, "progon.log")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def idle_s():
    # the service prints a progress line every second while it works: the age of its last journal line = idle time
    out = subprocess.run(["journalctl", "-u", "dacan.service", "-n", "1", "--no-pager", "-o", "short-unix"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    try:
        return time.time() - float(out[-1].split()[0])
    except (IndexError, ValueError):
        return 60.0


def html_of(text):
    m = re.search(r"```(?:html)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    m = re.search(r"(<!DOCTYPE html.*?</html>)", text, re.S | re.I)
    return m.group(1) if m else text


model = json.load(urllib.request.urlopen(a.url + "/v1/models", timeout=30))["data"][0]["id"]
log("модель %s, температуры %s" % (model, TEMPS))
for t in TEMPS:
    w0 = time.time()
    while idle_s() < 60 and time.time() - w0 < 300:
        time.sleep(10)
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "stream": False,
            "reasoning_effort": "high", "temperature": t, "top_p": 0.95, "top_k": 20, "max_tokens": 65536}
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    log("t=%.1f отправляю (ждал паузу %.0f с)" % (t, time.time() - w0))
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=7200))
    except Exception as e:  # one failed run must not stop the batch
        log("t=%.1f ОШИБКА %s" % (t, str(e)[:200]))
        continue
    msg = r["choices"][0]["message"]
    html = html_of(msg.get("content") or "")
    name = os.path.join(a.out, "t%.1f" % t)
    open(name + ".html", "w", encoding="utf-8").write(html)
    json.dump(r, open(name + ".json", "w", encoding="utf-8"), ensure_ascii=False)
    u = r.get("usage") or {}
    row = {"t": t, "start": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t0)), "sec": round(time.time() - t0),
           "tokens": u.get("completion_tokens"), "finish": r["choices"][0].get("finish_reason"),
           "reasoning_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or ""),
           "html_bytes": len(html.encode("utf-8")), "has_html": "<svg" in html.lower()}
    with open(os.path.join(a.out, "itogi.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log("t=%.1f готово: %s с, %s ток., %s, HTML %d байт" % (t, row["sec"], row["tokens"], row["finish"], row["html_bytes"]))
log("== ВСЕ ПРОГОНЫ ГОТОВЫ")
