# -*- coding: utf-8 -*-
# Builds temperature/index.html of the DACAN site: one card per prompt folder (meta.json, prompt.txt, itogi.jsonl,
# t<T>.png), cover = the 0.6 shot. A new prompt is a new folder; re-run to list it.
#     python gen_temp_hub.py <temperature folder>
# 21.10.2026 (tab 180b, Vadim «поставь на точно такие же тесты флеш 3.8… создай еще одну страницу такую же с тестами»):
# the same hub for another model — flash/ beside temperature/ — takes its texts from an optional <folder>/hub.json
# (title, desc, h1, sub, how); without it the page is byte for byte what it was (the Swift tests).
import html
import io
import json
import os
import sys

D = sys.argv[1]
HUB = {"title": u"Температура выборки",
       "desc": u"Один промт — пятнадцать температур от 0.1 до 1.5 на DACAN, 2x RTX 2080 Ti: файлы, снимки, цифры.",
       "h1": u"Один промт — пятнадцать <em>температур</em>",
       "sub": u"""Температура решает, насколько свободно модель выбирает следующее слово: ближе к нулю — почти всегда самое
вероятное, выше — разнообразнее. Здесь один и тот же промт прогнан на DACAN при каждой температуре от 0.1 до 1.5 с шагом
0.1. У каждого прогона — файл, который можно открыть и скачать, снимок и цифры.""",
       "how": u"""<b>Как мерили.</b> Боевая служба DACAN (Swift 1.5 Flash-Next NVFP4 + Q8, две RTX 2080 Ti), запрос как из её
веб-чата: одно сообщение без системного промта, reasoning_effort high, top_p 0.95, top_k 20, температура задана явно,
лимит ответа 65 536 токенов, по одному прогону на значение. Снимок — браузер 1280×860 через 4 с анимации. Скрипты:
<a href="https://github.com/tirex999/DACAN/tree/gh-pages/scripts">scripts</a> (прогон, снимки, сборка страниц)."""}
if os.path.isfile(os.path.join(D, "hub.json")):
    HUB.update(json.load(io.open(os.path.join(D, "hub.json"), encoding="utf-8")))
cards = []
slugs = sorted(os.listdir(D))
if HUB.get("poryadok"):   # hub.json: these folders first, in this order (flash/: the house first, Vadim 21.10 23:1x)
    slugs = [s for s in HUB["poryadok"] if s in slugs] + [s for s in slugs if s not in HUB["poryadok"]]
for slug in slugs:
    p = os.path.join(D, slug)
    if not os.path.isdir(p):
        continue
    if not os.path.isfile(os.path.join(p, "meta.json")):
        # 19.10.2026: a temperature × seed page (a copy of its claude.ai page with its own data.json) has no meta.json;
        # its card comes from karta.json (title, prompt line, meta line, cover), with the progress while it runs.
        # Without this the regenerated list dropped the pelican × seed card (16137c1).
        kp = os.path.join(p, "karta.json")
        if os.path.isfile(kp):
            k = json.load(io.open(kp, encoding="utf-8"))
            pm = k["pm"]
            if os.path.isfile(os.path.join(p, "data.json")):
                dj = json.load(io.open(os.path.join(p, "data.json"), encoding="utf-8"))
                if dj.get("current"):
                    pm += " · идёт: %d из %d прогонов" % (dj["runs_done"], dj["runs_total"])
            cover = k.get("cover") if k.get("cover") and os.path.exists(os.path.join(p, k["cover"])) else None
            cards.append(
                '<a class="pc" href="%s/index.html">%s<span class="pb"><span class="pt">%s</span>'
                '<span class="pp">%s</span><span class="pm">%s</span></span></a>'
                % (html.escape(slug), ('<img src="%s/%s" alt="" loading="lazy" width="1280" height="860">' % (html.escape(slug), cover))
                   if cover else "", html.escape(k["title"]), html.escape(k["prompt"][:140]), html.escape(pm)))
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
<title>%(title)s</title>
<meta name="description" content="%(desc)s">
<script>try{var t=localStorage.getItem("dacan.theme");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}catch(e){}</script>
<style>
 /* theme: «Авто» follows the device; «День»/«Ночь» stamp data-theme on <html> (the same choice on every page of the site) */
 :root{--bg:#faf8f5;--fg:#241f1b;--mut:#6d635a;--line:#e2dbd2;--acc:#8a5a2b;--card:#fffdf9;
   --cold:hsl(210 62%% 52%%);--hot:hsl(5 62%% 52%%);color-scheme:light}
 @media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#17150f;--fg:#efe9e0;--mut:#a2988c;--line:#332e27;
   --acc:#d8a45f;--card:#1d1a14;color-scheme:dark}}
 :root[data-theme=dark]{--bg:#17150f;--fg:#efe9e0;--mut:#a2988c;--line:#332e27;--acc:#d8a45f;--card:#1d1a14;color-scheme:dark}
 .topline{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin:0 0 16px}
 .topline a.back{margin:0}
 .theme{display:inline-flex;border:1px solid var(--line);border-radius:9px;overflow:hidden;background:var(--card)}
 .theme button{font:600 12.5px/1 -apple-system,"Segoe UI",Roboto,sans-serif;color:var(--mut);background:none;border:0;
   padding:7px 10px;cursor:pointer}
 .theme button+button{border-left:1px solid var(--line)}
 .theme button:hover{color:var(--fg)}
 .theme button[aria-pressed=true]{background:var(--fg);color:var(--bg)}
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
<div class="topline"><a class="back" href="../index.html">← DACAN · главная</a>
<div class="theme" role="group" aria-label="Тема"><button type="button" data-t="auto">Авто</button><button type="button" data-t="light" title="День">☀</button><button type="button" data-t="dark" title="Ночь">☾</button></div></div>
<h1>%(h1)s</h1>
<div class="rail"></div><div class="ticks"><span>0.1 · точнее</span><span>0.8</span><span>1.5 · свободнее</span></div>
<p class="sub">%(sub)s</p>
<div class="grid">%(cards)s</div>
<div class="how">%(how)s</div>
</div>
<script>
(function(){ // theme: auto (device) / day / night, remembered for the whole site
  var bs=document.querySelectorAll(".theme button");
  function cur(){return document.documentElement.getAttribute("data-theme")||"auto";}
  function show(){bs.forEach(function(b){b.setAttribute("aria-pressed",String(b.dataset.t===cur()));});}
  bs.forEach(function(b){b.addEventListener("click",function(){
    var t=b.dataset.t;
    if(t==="auto")document.documentElement.removeAttribute("data-theme");else document.documentElement.setAttribute("data-theme",t);
    try{if(t==="auto")localStorage.removeItem("dacan.theme");else localStorage.setItem("dacan.theme",t);}catch(e){}
    show();});});
  show();
})();
</script></body></html>
"""
io.open(os.path.join(D, "index.html"), "w", encoding="utf-8", newline="\n").write(page % dict(cards="".join(cards), **HUB))
print("%s/index.html: промтов %d" % (os.path.basename(os.path.normpath(D)), len(cards)))
