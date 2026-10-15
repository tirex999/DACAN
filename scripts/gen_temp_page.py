# -*- coding: utf-8 -*-
# Builds one prompt's page of the DACAN site's temperature tests: temperature/<slug>/index.html from the folder's
# t<T>.html (the model's file), t<T>.png (a 1280x860 browser shot), itogi.jsonl (one line per run, from
# temp_progon.py), prompt.txt, meta.json and an optional zametki.json with notes per temperature.
# The page: a temperature scale from cold to hot, the chosen run live in a sandboxed frame, a strip of shots, a chart
# of answer size, a table, download/open per run.        python gen_temp_page.py <temperature/slug folder>
import html
import io
import json
import os
import sys

D = sys.argv[1]
rows = [json.loads(l) for l in io.open(os.path.join(D, "itogi.jsonl"), encoding="utf-8") if l.strip()]
rows.sort(key=lambda r: r["t"])
zam = {}
if os.path.exists(os.path.join(D, "zametki.json")):
    zam = json.load(io.open(os.path.join(D, "zametki.json"), encoding="utf-8"))
OBSHEE = zam.pop("_obshee", "")


def hue(t):
    # cold blue (0.1) -> warm red (1.5), the colour of the temperature itself
    x = (t - 0.1) / 1.4
    return round(210 - 205 * x)


runs = []
for r in rows:
    t = "%.1f" % r["t"]
    runs.append({"t": t, "hue": hue(r["t"]), "tok": r.get("tokens") or 0, "sec": r.get("sec") or 0,
                 "kb": round((r.get("html_bytes") or 0) / 1024, 1), "fin": r.get("finish") or "",
                 "think": r.get("reasoning_chars") or 0, "svg": bool(r.get("has_html")),
                 "start": r.get("start", ""), "note": zam.get(t, ""),
                 "png": os.path.exists(os.path.join(D, "t%s.png" % t))})
maxtok = max([r["tok"] for r in runs] + [1])
# a round axis: step 1/2/5 x 10^k so that four steps cover the largest answer
_st = 10 ** (len(str(int(maxtok / 4))) - 1)
for _m in (1, 2, 5, 10):
    if _st * _m * 4 >= maxtok:
        _st *= _m
        break
STEP = _st
maxtok = STEP * 4
date = rows[0]["start"][:10] if rows else ""

cards = []
for r in runs:
    img = ('<img src="t%s.png" alt="Пеликан при температуре %s" loading="lazy" width="1280" height="860">' % (r["t"], r["t"])
           if r["png"] else '<div class="noimg">нет снимка</div>')
    cards.append(
        '<button class="shot" data-t="%(t)s" style="--h:%(hue)d" aria-label="Температура %(t)s">'
        '<span class="tlab">%(t)s</span>%(img)s'
        '<span class="smeta">%(kb)s КБ · %(tokk)s тыс. ток.</span></button>'
        % dict(r, img=img, tokk=("%.1f" % (r["tok"] / 1000)).replace(".", ",")))

bars = []
W, H, PAD = 720, 210, 34
bw = (W - 2 * PAD) / max(len(runs), 1)
for i, r in enumerate(runs):
    h = (H - 2 * PAD) * r["tok"] / maxtok
    x = PAD + i * bw + bw * 0.18
    y = H - PAD - h
    bars.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="3" fill="hsl(%d 62%% 52%%)" data-t="%s">'
                '<title>t %s: %s ток., %s</title></rect>' % (x, y, bw * 0.64, h, r["hue"], r["t"], r["t"], r["tok"], r["fin"]))
    if r["fin"] == "length":
        bars.append('<text x="%.1f" y="%.1f" class="cap">лимит</text>' % (x + bw * 0.32, y - 5))
    bars.append('<text x="%.1f" y="%d" class="ax">%s</text>' % (x + bw * 0.32, H - PAD + 15, r["t"]))
ticks = []
for k in range(0, 5):
    v = maxtok * k / 4
    y = H - PAD - (H - 2 * PAD) * k / 4
    ticks.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" class="grid"/><text x="%d" y="%.1f" class="ay">%s</text>'
                 % (PAD, W - PAD + 6, y, y, PAD - 6, y + 4, ("%g" % (v / 1000)) + "K" if v else "0"))

# the prompt and the page's name live in the prompt's own folder, so the next prompt is a new folder, not a code change
PROMPT = io.open(os.path.join(D, "prompt.txt"), encoding="utf-8").read().strip()
META = json.load(io.open(os.path.join(D, "meta.json"), encoding="utf-8"))

page = u"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Температура: %(title)s</title>
<meta name="description" content="Один промт, пятнадцать температур от 0.1 до 1.5: %(title)s — Swift 1.5 Flash-Next на DACAN, 2x RTX 2080 Ti.">
<style>
 :root{--bg:#faf8f5;--fg:#241f1b;--mut:#6d635a;--line:#e2dbd2;--acc:#8a5a2b;--pan:#f2ece4;--card:#fffdf9;
   --cold:hsl(210 62%% 52%%);--hot:hsl(5 62%% 52%%)}
 @media (prefers-color-scheme:dark){:root{--bg:#17150f;--fg:#efe9e0;--mut:#a2988c;--line:#332e27;--acc:#d8a45f;
   --pan:#201c15;--card:#1d1a14}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,"Segoe UI",Roboto,sans-serif;
   padding:28px 20px 72px}
 .wrap{max-width:1220px;margin:0 auto}
 .mono,.k{font-family:ui-monospace,Menlo,Consolas,monospace}
 a{color:var(--acc)} a:hover{text-decoration:none}
 a.back{display:inline-block;margin:0 0 16px;font:600 12px/1 ui-monospace,Menlo,Consolas,monospace;letter-spacing:.09em;
   text-transform:uppercase;text-decoration:none}
 h1{font-size:clamp(26px,3.6vw,40px);line-height:1.08;margin:0 0 10px;letter-spacing:-.02em;text-wrap:balance}
 h1 em{font-style:normal;background:linear-gradient(90deg,var(--cold),var(--hot));-webkit-background-clip:text;
   background-clip:text;color:transparent}
 h2{font-size:19px;margin:42px 0 12px;letter-spacing:-.01em}
 .sub{color:var(--mut);margin:0 0 22px;max-width:72ch}
 .top{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:18px;align-items:start}
 .box{border:1px solid var(--line);border-radius:12px;padding:15px 18px;background:var(--card)}
 .box h3{margin:0 0 8px;font:600 11px/1 ui-monospace,Menlo,Consolas,monospace;letter-spacing:.1em;text-transform:uppercase;
   color:var(--mut)}
 pre.prompt{margin:0;white-space:pre-wrap;font:13.5px/1.5 ui-monospace,Menlo,Consolas,monospace}
 dl.spec{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;margin:0;font-size:13.5px}
 dl.spec dt{color:var(--mut)} dl.spec dd{margin:0;font-family:ui-monospace,Menlo,Consolas,monospace}

 /* the scale: one notch per temperature, coloured from cold to hot */
 .scale{margin:30px 0 0;position:sticky;top:0;z-index:5;background:var(--bg);padding:12px 0 10px;
   border-bottom:1px solid var(--line)}
 .scale-row{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
 .tbig{font:800 clamp(34px,5vw,52px)/1 ui-monospace,Menlo,Consolas,monospace;letter-spacing:-.03em;min-width:3.2ch;
   color:hsl(var(--h,100) 58%% 46%%)}
 @media (prefers-color-scheme:dark){.tbig{color:hsl(var(--h,100) 70%% 64%%)}}
 .rng{flex:1 1 300px;min-width:0}
 input[type=range]{width:100%%;accent-color:var(--acc);height:28px;cursor:pointer}
 .rail{height:8px;border-radius:4px;background:linear-gradient(90deg,var(--cold),hsl(100 55%% 48%%),var(--hot));
   margin:2px 0 4px;opacity:.85}
 .ticks{display:flex;justify-content:space-between;font:11px/1 ui-monospace,Menlo,Consolas,monospace;color:var(--mut)}
 .stats{display:flex;flex-wrap:wrap;gap:6px 18px;font:13px/1.4 ui-monospace,Menlo,Consolas,monospace;color:var(--mut)}
 .stats b{color:var(--fg)}
 .acts{display:flex;gap:8px;flex-wrap:wrap}
 .btn{display:inline-flex;align-items:center;gap:6px;font-weight:600;font-size:13px;line-height:1;color:var(--fg);text-decoration:none;
   border:1px solid var(--line);border-radius:9px;padding:9px 13px;background:var(--card)}
 .btn:hover{border-color:var(--acc);color:var(--acc)}
 .btn.main{background:var(--acc);border-color:var(--acc);color:#fff}
 @media (prefers-color-scheme:dark){.btn.main{color:#1b160f}}

 .stage{margin:16px 0 0;border:1px solid var(--line);border-radius:14px;overflow:hidden;background:#0d0d0d;
   box-shadow:0 22px 50px -32px rgba(0,0,0,.6);position:relative;border-top:4px solid hsl(var(--h,100) 62%% 52%%)}
 .stage iframe{display:block;width:100%%;aspect-ratio:1280/860;border:0;background:#fff}
 .note{margin:10px 2px 0;color:var(--mut);font-size:14px;max-width:80ch;min-height:1.5em}

 .strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
 .shot{all:unset;cursor:pointer;display:block;border:1px solid var(--line);border-radius:11px;overflow:hidden;
   background:var(--card);border-top:4px solid hsl(var(--h) 62%% 52%%);transition:transform .15s,box-shadow .15s}
 .shot:hover{transform:translateY(-2px);box-shadow:0 12px 26px -20px rgba(0,0,0,.6)}
 .shot:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
 .shot[aria-current=true]{box-shadow:0 0 0 2px hsl(var(--h) 62%% 52%%)}
 .shot img{display:block;width:100%%;height:auto;aspect-ratio:1280/860;object-fit:cover}
 .noimg{aspect-ratio:1280/860;display:grid;place-items:center;color:var(--mut);font-size:12px}
 .tlab{display:block;padding:7px 10px 0;font:800 18px/1 ui-monospace,Menlo,Consolas,monospace;color:hsl(var(--h) 58%% 44%%)}
 @media (prefers-color-scheme:dark){.tlab{color:hsl(var(--h) 70%% 64%%)}}
 .smeta{display:block;padding:6px 10px 9px;font:12px/1.3 ui-monospace,Menlo,Consolas,monospace;color:var(--mut)}
 .shot .tlab+img,.shot .tlab+.noimg{margin-top:7px}

 .chart{border:1px solid var(--line);border-radius:12px;background:var(--card);padding:10px 10px 4px;overflow-x:auto}
 .chart svg{display:block;width:100%%;min-width:560px;height:auto}
 .chart .ax,.chart .ay,.chart .cap{font:11px ui-monospace,Menlo,Consolas,monospace;fill:var(--mut);text-anchor:middle}
 .chart .ay{text-anchor:end} .chart .cap{fill:var(--acc);font-weight:700}
 .chart .grid{stroke:var(--line);stroke-width:1}
 .chart rect{cursor:pointer}
 table{border-collapse:collapse;width:100%%;font-size:13.5px}
 .tw{overflow-x:auto;border:1px solid var(--line);border-radius:12px}
 th,td{padding:8px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
 th{font:600 11px/1.2 ui-monospace,Menlo,Consolas,monospace;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);
   background:var(--pan)}
 td:first-child,th:first-child{text-align:left}
 td .sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:7px;vertical-align:-1px}
 tr:last-child td{border-bottom:0}
 .foot{margin:34px 0 0;color:var(--mut);font-size:13px;max-width:80ch}
 @media (max-width:860px){.top{grid-template-columns:1fr}.scale{position:static}}
</style></head><body><div class="wrap">
<a class="back" href="../index.html">← Температура · все промты</a>
<h1>%(title_cap)s: пятнадцать <em>температур</em></h1>
<p class="sub">%(lede)s Модель Swift 1.5 Flash-Next (NVFP4 + Q8) на движке DACAN, две RTX 2080 Ti. Промт тот же, условия те
же, меняется только температура: от 0.1 до 1.5 с шагом 0.1, по одному прогону на значение. Каждый файл можно открыть
здесь же, скачать и посмотреть снимок.</p>
<div class="top">
 <div class="box"><h3>Промт, дословно</h3><pre class="prompt">%(prompt)s</pre></div>
 <div class="box"><h3>Условия</h3><dl class="spec">
  <dt>Модель</dt><dd>%(model)s</dd><dt>Движок</dt><dd>DACAN, 2× RTX 2080 Ti</dd>
  <dt>Размышление</dt><dd>reasoning_effort high</dd><dt>Выборка</dt><dd>top_p 0.95 · top_k 20</dd>
  <dt>Температура</dt><dd>0.1 … 1.5, шаг 0.1</dd><dt>Лимит ответа</dt><dd>65 536 токенов</dd>
  <dt>Дата</dt><dd>%(date)s</dd><dt>Прогонов</dt><dd>%(n)d, по одному на значение</dd></dl></div>
</div>
%(obshee)s
<div class="scale" id="scale">
 <div class="scale-row">
  <div class="tbig" id="tbig">0.6</div>
  <div class="rng"><input type="range" id="rng" min="0" max="%(maxi)d" step="1" value="0" aria-label="Температура">
   <div class="rail"></div><div class="ticks">%(ticklabels)s</div></div>
  <div class="acts"><a class="btn main" id="dl" href="#" download>Скачать файл</a><a class="btn" id="op" href="#" target="_blank" rel="noopener">Открыть отдельно</a></div>
 </div>
 <div class="stats" id="stats"></div>
</div>
<div class="stage" id="stage"><iframe id="fr" title="Файл модели" sandbox="allow-scripts" loading="eager"></iframe></div>
<p class="note" id="note"></p>

<h2>Все пятнадцать</h2>
<div class="strip" id="strip">%(cards)s</div>

<h2>Сколько модель написала</h2>
<div class="chart"><svg viewBox="0 0 %(W)d %(H)d" role="img" aria-label="Токены ответа по температуре">%(ticks)s%(bars)s</svg></div>

<h2>Таблица</h2>
<div class="tw"><table><thead><tr><th>Температура</th><th>Токенов ответа</th><th>Размышление, знаков</th><th>Время</th><th>Скорость</th><th>Файл</th><th>Конец</th></tr></thead>
<tbody>%(rows)s</tbody></table></div>
<p class="foot">Как мерили: запросы шли в боевую службу DACAN так же, как их шлёт её веб-чат (одно сообщение, без системного
промта, температура задана явно), по одному на значение и подряд; снимок — первый кадр после 4 с анимации в браузере 1280×860.
Одного прогона на значение мало, чтобы говорить о закономерности: одна и та же температура может дать и простую картинку,
и богатую. Скрипт прогона: <a href="../../scripts/temp_progon.py">temp_progon.py</a>.</p>
</div>
<script>
const RUNS=%(runs)s;
const $=s=>document.querySelector(s);
const rng=$("#rng"),fr=$("#fr");
function pick(i,scroll){
  i=Math.max(0,Math.min(RUNS.length-1,i)); const r=RUNS[i]; rng.value=i;
  document.documentElement.style.setProperty("--h",r.hue);
  $("#tbig").textContent=r.t; $("#stage").style.setProperty("--h",r.hue);
  const f="t"+r.t+".html"; if(fr.dataset.f!==f){fr.src=f;fr.dataset.f=f;}
  $("#dl").href=f; $("#dl").setAttribute("download","pelican_t"+r.t+".html"); $("#op").href=f;
  const sp=r.sec?(r.tok/r.sec).toFixed(1):"–";
  $("#stats").innerHTML="<span>ответ <b>"+r.tok.toLocaleString("ru")+"</b> ток.</span><span>размышление <b>"+r.think.toLocaleString("ru")+"</b> знаков</span><span>файл <b>"+r.kb.toLocaleString("ru")+"</b> КБ</span><span>время <b>"+Math.round(r.sec/60)+"</b> мин</span><span>скорость <b>"+sp+"</b> т/с</span>"+(r.fin==="length"?"<span><b>упёрся в лимит</b></span>":"");
  $("#note").textContent=r.note||"";
  document.querySelectorAll(".shot").forEach(b=>b.setAttribute("aria-current",String(b.dataset.t===r.t)));
  try{history.replaceState(null,"","#t"+r.t.replace(".","_"));}catch(e){}
  if(scroll) $("#scale").scrollIntoView({behavior:"smooth",block:"start"});
}
rng.addEventListener("input",()=>pick(+rng.value));
document.querySelectorAll(".shot,.chart rect").forEach(el=>el.addEventListener("click",()=>pick(RUNS.findIndex(r=>r.t===el.dataset.t),true)));
const h=(location.hash||"").match(/^#t(\\d)_(\\d)$/);
let start=RUNS.findIndex(r=>r.t===(h?h[1]+"."+h[2]:"0.6")); pick(start<0?0:start);
</script>
</body></html>
"""

trows = []
for r in runs:
    sp = ("%.1f" % (r["tok"] / r["sec"])).replace(".", ",") if r["sec"] else "–"
    trows.append('<tr><td><span class="sw" style="background:hsl(%d 62%% 52%%)"></span><a href="#t%s" data-t="%s">%s</a></td>'
                 '<td>%s</td><td>%s</td><td>%d мин %02d с</td><td>%s т/с</td><td><a href="t%s.html" download="pelican_t%s.html">%s КБ</a></td><td>%s</td></tr>'
                 % (r["hue"], r["t"].replace(".", "_"), r["t"], r["t"], "{:,}".format(r["tok"]).replace(",", " "),
                    "{:,}".format(r["think"]).replace(",", " "), r["sec"] // 60, r["sec"] % 60, sp, r["t"], r["t"],
                    str(r["kb"]).replace(".", ","), "лимит" if r["fin"] == "length" else "сам"))

model = "Swift 1.5 Flash-Next NVFP4 + Q8"
out = page % dict(prompt=html.escape(PROMPT), model=model, date=date, n=len(runs), maxi=max(len(runs) - 1, 0),
                  title=html.escape(META["title"]), title_cap=html.escape(META["title"][:1].upper() + META["title"][1:]),
                  lede=html.escape(META.get("lede", "")),
                  ticklabels="".join("<span>%s</span>" % r["t"] for r in runs), cards="".join(cards),
                  W=W, H=H, ticks="".join(ticks), bars="".join(bars), rows="".join(trows),
                  runs=json.dumps(runs, ensure_ascii=False),
                  obshee=('<div class="box" style="margin-top:18px"><h3>Что видно</h3><p style="margin:0">%s</p></div>'
                          % html.escape(OBSHEE)) if OBSHEE else "")
io.open(os.path.join(D, "index.html"), "w", encoding="utf-8", newline="\n").write(out)
print("index.html:", len(out.encode("utf-8")), "байт, прогонов", len(runs))
