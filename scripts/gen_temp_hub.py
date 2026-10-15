# -*- coding: utf-8 -*-
# Builds temperature/index.html of the DACAN site: one card per prompt folder (meta.json, prompt.txt, itogi.jsonl,
# t<T>.png), cover = the 0.6 shot. A new prompt is a new folder; re-run to list it.
#     python gen_temp_hub.py <temperature folder>
import html
import io
import json
import os
import sys

D = sys.argv[1]
cards = []
for slug in sorted(os.listdir(D)):
    p = os.path.join(D, slug)
    if not os.path.isfile(os.path.join(p, "meta.json")):
        continue
    meta = json.load(io.open(os.path.join(p, "meta.json"), encoding="utf-8"))
    rows = [json.loads(l) for l in io.open(os.path.join(p, "itogi.jsonl"), encoding="utf-8") if l.strip()]
    prompt = io.open(os.path.join(p, "prompt.txt"), encoding="utf-8").read().strip()
    cover = next((c for c in ("t0.6.png", "t0.5.png", "t0.1.png") if os.path.exists(os.path.join(p, c))), None)
    ts = sorted(r["t"] for r in rows)
    span = ("%.1f–%.1f" % (ts[0], ts[-1])) if ts else "—"
    cards.append(
        '<a class="pc" href="%s/index.html">%s<span class="pb"><span class="pt">%s</span>'
        '<span class="pp">%s</span><span class="pm">%d прогонов · температура %s · %s</span></span></a>'
        % (html.escape(slug), ('<img src="%s/%s" alt="" loading="lazy" width="1280" height="860">' % (html.escape(slug), cover))
           if cover else "", html.escape(meta["title"][:1].upper() + meta["title"][1:]),
           html.escape(prompt.splitlines()[0][:140]), len(rows), span, html.escape(rows[0]["start"][:10] if rows else "")))

page = u"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Температура выборки</title>
<meta name="description" content="Один промт — пятнадцать температур от 0.1 до 1.5 на DACAN, 2x RTX 2080 Ti: файлы, снимки, цифры.">
<style>
 :root{--bg:#faf8f5;--fg:#241f1b;--mut:#6d635a;--line:#e2dbd2;--acc:#8a5a2b;--card:#fffdf9;
   --cold:hsl(210 62%% 52%%);--hot:hsl(5 62%% 52%%)}
 @media (prefers-color-scheme:dark){:root{--bg:#17150f;--fg:#efe9e0;--mut:#a2988c;--line:#332e27;--acc:#d8a45f;--card:#1d1a14}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,"Segoe UI",Roboto,sans-serif;padding:28px 20px 72px}
 .wrap{max-width:1100px;margin:0 auto}
 a{color:var(--acc)}
 a.back{display:inline-block;margin:0 0 16px;font:600 12px/1 ui-monospace,Menlo,Consolas,monospace;letter-spacing:.09em;
   text-transform:uppercase;text-decoration:none}
 h1{font-size:clamp(28px,4vw,44px);line-height:1.05;margin:0 0 12px;letter-spacing:-.02em;text-wrap:balance}
 h1 em{font-style:normal;background:linear-gradient(90deg,var(--cold),var(--hot));-webkit-background-clip:text;background-clip:text;color:transparent}
 .rail{height:8px;border-radius:4px;background:linear-gradient(90deg,var(--cold),hsl(100 55%% 48%%),var(--hot));margin:0 0 6px;max-width:520px}
 .ticks{display:flex;justify-content:space-between;max-width:520px;font:11px/1 ui-monospace,Menlo,Consolas,monospace;color:var(--mut);margin:0 0 22px}
 p.sub{color:var(--mut);max-width:70ch;margin:0 0 12px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px;margin:26px 0 0}
 a.pc{display:block;text-decoration:none;color:inherit;border:1px solid var(--line);border-radius:14px;overflow:hidden;background:var(--card);
   border-top:4px solid transparent;border-image:linear-gradient(90deg,var(--cold),var(--hot)) 1 / 4px 0 0 0;transition:transform .15s,box-shadow .15s}
 a.pc:hover{transform:translateY(-3px);box-shadow:0 16px 34px -24px rgba(0,0,0,.6)}
 a.pc img{display:block;width:100%%;height:auto;aspect-ratio:1280/860;object-fit:cover;border-bottom:1px solid var(--line)}
 .pb{display:block;padding:13px 16px 15px}
 .pt{display:block;font-weight:700;font-size:18px;line-height:1.25;letter-spacing:-.01em}
 .pp{display:block;margin:6px 0 8px;font:13px/1.45 ui-monospace,Menlo,Consolas,monospace;color:var(--mut)}
 .pm{display:block;font:12px/1.3 ui-monospace,Menlo,Consolas,monospace;color:var(--acc)}
 .how{margin:34px 0 0;border:1px solid var(--line);border-radius:12px;padding:14px 18px;background:var(--card);max-width:80ch;font-size:14px;color:var(--mut)}
 .how b{color:var(--fg)}
</style></head><body><div class="wrap">
<a class="back" href="../index.html">← DACAN · главная</a>
<h1>Один промт — пятнадцать <em>температур</em></h1>
<div class="rail"></div><div class="ticks"><span>0.1 · точнее</span><span>0.8</span><span>1.5 · свободнее</span></div>
<p class="sub">Температура решает, насколько свободно модель выбирает следующее слово: ближе к нулю — почти всегда самое
вероятное, выше — разнообразнее. Здесь один и тот же промт прогнан на DACAN при каждой температуре от 0.1 до 1.5 с шагом
0.1. У каждого прогона — файл, который можно открыть и скачать, снимок и цифры.</p>
<div class="grid">%(cards)s</div>
<div class="how"><b>Как мерили.</b> Боевая служба DACAN (Swift 1.5 Flash-Next NVFP4 + Q8, две RTX 2080 Ti), запрос как из её
веб-чата: одно сообщение без системного промта, reasoning_effort high, top_p 0.95, top_k 20, температура задана явно,
лимит ответа 65 536 токенов, по одному прогону на значение. Снимок — браузер 1280×860 через 4 с анимации. Скрипты:
<a href="https://github.com/tirex999/DACAN/tree/gh-pages/scripts">scripts</a> (прогон, снимки, сборка страниц).</div>
</div></body></html>
"""
io.open(os.path.join(D, "index.html"), "w", encoding="utf-8", newline="\n").write(page % dict(cards="".join(cards)))
print("temperature/index.html: промтов", len(cards))
