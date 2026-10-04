# -*- coding: utf-8 -*-
"""Проверка чтения промта на двух картах (DACAN, 29.09.2026): свежий документ ~19K токенов (его нет в кеше движка),
жадно, без размышления, длинный ответ. Запуск: py pf2_test.py <метка> [сдвиг]  -> pf2_<метка>.txt (ответ) и строка итога.
Ответ на старом движке (одна карта), на новом (две) и с пакетом 8K (--prefill 8192) совпал знак в знак;
строка «фактов верно» сверяет каждую строку ответа с документом (имя, материал, место, вес)."""
import json
import sys
import time
import urllib.request

U = "http://192.168.1.214:8000/v1/chat/completions"
SLOVA = ["river", "copper", "lantern", "meadow", "violet", "harbor", "granite", "whistle", "orchard", "falcon",
         "saddle", "ember", "quartz", "willow", "thunder", "pepper", "canyon", "marble", "velvet", "comet"]
metka = sys.argv[1] if len(sys.argv) > 1 else "run"
sdvig = int(sys.argv[2]) if len(sys.argv) > 2 else 9   # другой сдвиг = другой свежий документ (эталон ответа - при 9)
lines = ["Reference document D. Each line describes one numbered item."]
for i in range(1, 601):
    a, b, c = (SLOVA[(i * 7 + sdvig) % 20], SLOVA[(i * 3 + 2 * sdvig) % 20], SLOVA[(i * 11 + sdvig) % 20])
    lines.append("Line %d: the item %s-%d is made of %s, kept near the %s, and weighs %d grams." %
                 (i, a, i, b, c, (i * 37 + sdvig) % 1000))
msgs = [{"role": "system", "content": "\n".join(lines)},
        {"role": "user", "content": "For items 1 to 25, write one line each: the item's name, what it is made of, "
                                    "where it is kept and its weight."}]
body = {"model": "x", "messages": msgs, "temperature": 0, "max_tokens": 1500,
        "chat_template_kwargs": {"enable_thinking": False}}
t0 = time.time()
req = urllib.request.Request(U, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=1800))
txt = r["choices"][0]["message"]["content"] or ""
u = r.get("usage", {})
open("pf2_%s.txt" % metka, "w", encoding="utf-8").write(txt)
print("%s: %.1f с, промт %s, ответ %s, %s; первые знаки: %r" % (metka, time.time() - t0, u.get("prompt_tokens"),
      u.get("completion_tokens"), r["choices"][0]["finish_reason"], txt[:80]))
verno = 0
for i in range(1, 26):
    a, b, c = (SLOVA[(i * 7 + sdvig) % 20], SLOVA[(i * 3 + 2 * sdvig) % 20], SLOVA[(i * 11 + sdvig) % 20])
    ves = (i * 37 + sdvig) % 1000
    stroki = [s for s in txt.splitlines() if ("%s-%d" % (a, i)) in s]
    if stroki and b in stroki[0] and c in stroki[0] and str(ves) in stroki[0]:
        verno += 1
print("фактов верно: %d из 25" % verno)
