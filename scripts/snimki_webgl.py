# -*- coding: utf-8 -*-
# Shots and a working check of the temperature runs' WebGL files (the Tidehaven prompt, 19.10.2026): every t<T>.html of
# the folder in Chromium (playwright, software WebGL), 1280x860:
#   - the shot t<T>.png after 8 s of the scene (the libraries come from their CDN, as for anyone who opens the file);
#   - "drawn": after one click the big WebGL canvas is on top (under the centre and a side point - not a loading screen
#     left up by a script error) and the shot is not one flat colour (a scene that failed to start leaves a blank page);
#   - "errors": the page's script errors (pageerror) from the load and from the clicks below;
#   - "buttons" / "clicked": the visible <button>s, each clicked once (a click that throws or times out is not counted);
#   - "ranges": the range inputs, each moved to the middle (input + change events).
# The result goes to proverka.json {"<T>": {...}} beside the files; gen_temp_page.py shows it as «Проверка».
#     python snimki_webgl.py <temperature/slug folder> [--redo]
import json
import os
import statistics
import sys

from PIL import Image
from playwright.sync_api import sync_playwright

D = sys.argv[1]
REDO = "--redo" in sys.argv
PROV = os.path.join(D, "proverka.json")
prov = json.load(open(PROV, encoding="utf-8")) if os.path.exists(PROV) else {}
files = sorted(f for f in os.listdir(D) if f.startswith("t") and f.endswith(".html") and f[1:-5].replace(".", "").isdigit())


def drawn(png):
    # not one flat colour: the grey levels of a 64x43 copy spread out
    im = Image.open(png).convert("L").resize((64, 43))
    return statistics.pstdev(list(im.getdata())) > 6.0


with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    for f in files:
        t = f[1:-5]
        png = os.path.join(D, "t%s.png" % t)
        if t in prov and os.path.exists(png) and not REDO:
            continue
        pg = b.new_page(viewport={"width": 1280, "height": 860})
        errs = []
        pg.on("pageerror", lambda e, errs=errs: errs.append(str(e)[:200]))
        url = "file:///" + os.path.abspath(os.path.join(D, f)).replace("\\", "/")
        try:
            pg.goto(url, wait_until="load", timeout=90000)
        except Exception as e:  # a page that never finishes loading still gets its shot and its errors
            errs.append("load: " + str(e)[:150])
        pg.wait_for_timeout(8000)
        # the first user gesture (the prompt starts audio only after one; a start screen goes away on it), then is the
        # scene really on top: a big canvas under the centre and two side points - a loading screen left up by a script
        # error (or any other overlay) is not a scene, however colourful (19.10.2026: t0.1 stuck on «Filling the lagoon…»)
        try:
            pg.mouse.click(640, 430)
        except Exception:
            pass
        pg.wait_for_timeout(2500)
        scene = pg.evaluate("""() => {
            const cs = [...document.querySelectorAll('canvas')].filter(c => c.width > 200 && c.height > 150);
            const pts = [[0.5, 0.5], [0.35, 0.55], [0.65, 0.55]];
            const on = pts.filter(([x, y]) => { const el = document.elementFromPoint(innerWidth * x, innerHeight * y);
                return !!el && cs.some(c => c === el || c.contains(el)); }).length;
            return {canvas: cs.length, on_top: on >= 2}; }""")
        pg.screenshot(path=png)
        load_errs = len(errs)
        btns = [h for h in pg.query_selector_all("button") if h.is_visible()]
        clicked = 0
        for h in btns:
            try:
                h.click(timeout=2000)
                pg.wait_for_timeout(400)
                clicked += 1
            except Exception:
                pass
        ranges = pg.evaluate("""() => { const r = [...document.querySelectorAll('input[type=range]')];
            r.forEach(e => { const lo = +e.min || 0, hi = e.max === '' ? 100 : +e.max; e.value = (lo + hi) / 2;
              e.dispatchEvent(new Event('input', {bubbles: true})); e.dispatchEvent(new Event('change', {bubbles: true})); });
            return r.length; }""")
        pg.wait_for_timeout(2000)
        prov[t] = {"drawn": bool(scene["on_top"]) and drawn(png), "canvas": scene["canvas"],
                   "errors": len(errs), "load_errors": load_errs, "buttons": len(btns),
                   "clicked": clicked, "ranges": ranges, "first_error": errs[0] if errs else ""}
        print(t, prov[t])
        pg.close()
        json.dump(prov, open(PROV, "w", encoding="utf-8"), ensure_ascii=False, indent=1, sort_keys=True)
    b.close()
print("proverka.json:", len(prov), "прогонов")
