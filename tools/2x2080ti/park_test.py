# -*- coding: utf-8 -*-
"""Проверка стоянки разговоров DACAN (--park, 29.09.2026).

Жадно (temperature 0), без размышления. Разговор A (~12K токенов) идёт три хода подряд: ответ A3 — эталон, он
продолжает живую сессию. Потом разговор B (другой документ) — A уезжает на стоянку. Потом тот же запрос A3 ещё
раз — A возвращается со стоянки. Ответ обязан совпасть с эталоном знак в знак, а прочитать заново DACAN должен
считанные токены. Строки «parked» / «brought back» — в журнале движка.
"""
import json
import time
import urllib.request

U = "http://192.168.1.214:8000/v1/chat/completions"
SLOVA = ["river", "copper", "lantern", "meadow", "violet", "harbor", "granite", "whistle", "orchard", "falcon",
         "saddle", "ember", "quartz", "willow", "thunder", "pepper", "canyon", "marble", "velvet", "comet"]


def dokument(nazvanie, sdvig, strok):
    out = ["Reference document %s. Each line describes one numbered item." % nazvanie]
    for i in range(1, strok + 1):
        a, b, c = (SLOVA[(i * 7 + sdvig) % 20], SLOVA[(i * 3 + 2 * sdvig) % 20], SLOVA[(i * 11 + sdvig) % 20])
        out.append("Line %d: the item %s-%d is made of %s, kept near the %s, and weighs %d grams." %
                   (i, a, i, b, c, (i * 37 + sdvig) % 1000))
    return "\n".join(out)


def sprosit(msgs, metka):
    body = {"model": "x", "messages": msgs, "temperature": 0, "max_tokens": 300,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    req = urllib.request.Request(U, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    txt = r["choices"][0]["message"]["content"] or ""
    u = r.get("usage", {})
    print("%-44s %6.1f с  промт %6s  ответ %4s  %s: %r" % (metka, time.time() - t0, u.get("prompt_tokens"),
          u.get("completion_tokens"), r["choices"][0]["finish_reason"], txt[:90]), flush=True)
    return txt


A = dokument("A", 1, 520)
B = dokument("B", 5, 430)
a1 = [{"role": "system", "content": A}, {"role": "user", "content": "In one sentence: what is item 17 made of?"}]
r1 = sprosit(a1, "A1")
a2 = a1 + [{"role": "assistant", "content": r1}, {"role": "user", "content": "And item 250? One sentence."}]
r2 = sprosit(a2, "A2 (продолжение A)")
a3 = a2 + [{"role": "assistant", "content": r2}, {"role": "user", "content": "Where is item 333 kept? One sentence."}]
r3 = sprosit(a3, "A3 — эталон (из живой сессии)")
b1 = [{"role": "system", "content": B}, {"role": "user", "content": "In one sentence: what is item 9 made of?"}]
sprosit(b1, "B1 (другой разговор: A на стоянку)")
r3b = sprosit(a3, "A3 ещё раз (A возвращается со стоянки)")
print("ИТОГ:", "СОВПАЛО знак в знак" if r3b == r3 else "РАЗОШЛОСЬ:\n  эталон %r\n  после  %r" % (r3, r3b))
