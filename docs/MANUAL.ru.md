# DACAN вручную: все варианты моделей и как их настроить

Установщик (`./setup.sh`, `START-HERE.bat`) ставит только кванты GSQ-RCO в GGUF — от ISTA-DASLab (оригинальная модель) и
от UkisAI (Swift 1.5). Всё остальное — наши кванты **NVFP4 + Q8 (DACAN)** и **Q8_0** с Hugging Face — ставится руками.
Этот документ — для них, но и установщиковые варианты здесь описаны так же подробно: какие файлы, какая сборка, какой
конфиг, какие ключи и сколько памяти.

Порядок одинаковый для любого варианта:

1. выбрать вариант модели и посчитать память;
2. собрать движок DACAN;
3. скачать файлы и разложить их по папкам;
4. написать конфиг сервера (JSON) под своё железо;
5. запустить сервер и проверить журнал.

---

## 1. Варианты моделей

Одна архитектура (`qwen4exp`, 125 млрд параметров, 512 экспертов) — две модели: оригинальная **Qwen3.8-Flash-Next** и
дообученная **Swift 1.5** от UkisAI (думает заметно короче). Другие модели DACAN не загружает.

| # | вариант | откуда | скачать | ОЗУ под модель* | ставит установщик |
|---|---|---|---:|---:|:---:|
| 1 | Qwen, GSQ-RCO Q2_0 / IQ2_XS / IQ3_XXS / IQ3_S | [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 66–84 ГБ | 48–62 ГБ | да |
| 2 | Swift 1.5, GSQ-RCO Q2_0 / IQ2_XS / IQ3_XXS | [UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 66–76 ГБ | 48–60 ГБ | да |
| 3 | **Qwen, NVFP4 + Q8 (DACAN)** — эксперты NVFP4 от NVIDIA | [tirex2001/Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Qwen3.8-Flash-Next-DACAN) | 131 ГБ | 63 + 51 ГиБ | нет |
| 4 | Qwen, эксперты Q8_0 (из FP8 Qwen) | там же | 191 ГБ | 120 + 51 ГиБ | нет |
| 5 | **Swift 1.5, NVFP4 + Q8 (DACAN)** | [tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN) | 131 ГБ | 63 + 51 ГиБ | нет |
| 6 | Swift 1.5, Q8_0 (один GGUF) | там же | 191 ГБ | 120 + 51 ГиБ | нет |

\* Для 3–6: эксперты + таблица n-грамм. Эксперты всегда лежат в ОЗУ целиком (видеокарта держит у себя копии самых
нужных). Таблица n-грамм (51 ГиБ в Q8_0) с ключом `--numa` закрепляется в ОЗУ; без него она живёт в файловом кеше
(подробности — в разделе 7). Для 1–2 число — требование установщика ко всей ОЗУ компьютера.

**Что брать.** Наша боевая служба работает на NVFP4 + Q8 (DACAN): 4-битные эксперты, всё, через что проходит каждый
токен, — в 8 битах. Q8_0 точнее по весам, но вдвое тяжелее и медленнее в ответе (на Swift 1.5: 49 против 58 т/с,
медиана 9 прогонов). GSQ-RCO 2–3 бита — когда ОЗУ меньше 100 ГБ.

### Из чего состоит модель («что за слои докачиваются»)

У Qwen3.8-Flash-Next, кроме обычных весов, есть две собственные части, которых нет у большинства моделей: таблица
n-грамм на 51 млрд параметров и слой MTP. Поэтому модель для DACAN — не один файл, а несколько частей:

| часть | что это | ключ движка | у вариантов 3–6 |
|---|---|---|---|
| **эксперты** | 512 экспертов в каждом из 48 слоёв, около 95 % всех весов; на токен работают 10 | берутся по пакету | файл `*-experts.bin` (у варианта 6 — внутри GGUF) |
| **плотная часть** | внимание, DeltaNet, общий эксперт, маршрутизаторы, эмбеддинги, выходная голова | `--native` | `*-dense-Q8_0.gguf` (у варианта 6 — весь GGUF) |
| **таблица n-грамм** | часть самой модели: эмбеддинги биграмм и триграмм (51 ГиБ в Q8_0) | `--ple-gguf` | отдельный GGUF (вариант 3–4) или внутри плотного GGUF (5–6) |
| **слой MTP** | собственный черновой слой модели: предлагает следующие токены, ответ идёт быстрее | `--mtp` | папка `mtp/` |
| **пакет** | служебная упаковка под движок: индекс экспертов, плотный блоб, токенизатор | `--pack` | папка `pack-nvfp4/` или `pack-q8_0/` |
| **таблицы маршрутов** | какие эксперты держать на видеокарте в первую очередь | `--expert-profile`, `--second-card-usage` | папка `data/` |

GGUF от ISTA и UkisAI (варианты 1–2) слоя MTP не содержат — его установщик «докачивает» отдельно (`tools/mtp_fetch.py`)
и упаковывает; пакет он строит из первого шарда. В наших репозиториях на HF все части уже лежат готовыми.

### Другие кванты (unsloth, lmstudio и т.п.)

- **NVFP4 из других источников.** `tools/nvfp4_experts.py` переупаковывает экспертов из чекпойнта в формате NVIDIA
  ModelOpt (как [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)): берутся
  **только эксперты**, плотная часть по-прежнему идёт из Q8_0 GGUF (`--native`). Чекпойнты NVFP4 в других форматах мы
  не проверяли. У unsloth NVFP4-версии этой модели на Hugging Face нет (есть GGUF, BF16 и FP8).
- **Свои кванты из BF16.** `tools/2x2080ti/swift_q8.py` и `swift_nvfp4.py` — ими сделаны кванты Swift 1.5 (вариант 5–6);
  `tools/q8_experts.py` — эксперты Q8_0 из FP8-чекпойнта Qwen.
- **Чужие GGUF** (Q4_K_M и подобные) на DACAN мы не проверяли.

---

## 2. Сборка движка

Установщик без ключа `--build` ставит **готовый движок апстрима Strata**. Он подходит для вариантов 1–2, но не знает
наших ключей (`--numa`, `--second-card`) и наших пакетов экспертов. Для вариантов 3–6 и для двух карт или двух сокетов
движок собирается из этого репозитория.

### Linux (проверено на нём)

Нужно: драйвер NVIDIA 580+, CUDA Toolkit 13.0, `cmake`, `ninja`, `gcc`, Python 3.10+ с `venv`, `git`.

```bash
git clone https://github.com/tirex999/DACAN && cd DACAN
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp && (cd ../llama.cpp && git checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON \
      -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=<архитектура> -DSTRATA_GGML_DIR=$(realpath ../llama.cpp)
cmake --build build --target strata -j
python3 -m venv .venv && .venv/bin/pip install numpy jinja2 regex pyyaml tqdm requests pillow psutil
```

`<архитектура>` — по видеокарте (если карт несколько и они разных поколений, перечислите через `;`, например `86;89`):

| карты | значение |
|---|---|
| RTX 20xx, Titan RTX, Quadro RTX (Turing) | `75` — так собрана наша служба |
| RTX 30xx, A2000–A6000 (Ampere) | `86` |
| RTX 40xx, RTX 4000–6000 Ada | `89` |
| RTX 50xx (Blackwell) | `120` (значение по умолчанию) |

Если `-DSTRATA_GGML_DIR` не указать, сборка сама скачает нужный коммит ggml. То же самое делает `./setup.sh --build`:
он ставит инструменты, собирает движок под вашу карту (10–20 минут, один раз) и кладёт его в `engine/strata`.

### Картинки (необязательно)

Энкодер картинок — отдельная программа `strata-vision`:

```bash
cmake -S tools/vision -B build-vision -DLLAMA_DIR=$(realpath ../llama.cpp) -DSTRATA_VISION_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=<архитектура>
cmake --build build-vision --target strata-vision -j
```

### Windows

`START-HERE.bat --build` ставит Visual Studio Build Tools и CUDA 13.0 через winget и собирает движок. Варианты 3–6 на
Windows мы не проверяли.

---

## 3. Файлы: что скачать и куда положить

Скачивать удобнее всего утилитой Hugging Face: `pip install -U huggingface_hub`, потом `hf download …` (в старых версиях
— `huggingface-cli download …`, те же ключи).

### Вариант 3: Qwen, NVFP4 + Q8 (DACAN)

```bash
hf download tirex2001/Qwen3.8-Flash-Next-DACAN --local-dir /models/qwen-dacan \
  --include "pack-nvfp4/*" "mtp/*" "data/*" \
            "Qwen3.8-Flash-Next-NVFP4-experts.bin" "Qwen3.8-Flash-Next-dense-Q8_0.gguf" \
            "Qwen3.8-Flash-Next-PLE-FP8-Q8_0.gguf"
```

| ключ движка | файл |
|---|---|
| `--pack` | `/models/qwen-dacan/pack-nvfp4` |
| `--native` | `/models/qwen-dacan/Qwen3.8-Flash-Next-dense-Q8_0.gguf` |
| `--ple-gguf` | `/models/qwen-dacan/Qwen3.8-Flash-Next-PLE-FP8-Q8_0.gguf` |
| `--mtp` | `/models/qwen-dacan/mtp` |
| токенизатор (конфиг сервера) | `/models/qwen-dacan/pack-nvfp4/tokenizer` |

**Важно:** файл экспертов (`*-NVFP4-experts.bin`) должен лежать **в одной папке с GGUF из `--native`**: пакет
(`pack-nvfp4/native_experts.txt`) называет его без пути, и движок ищет его рядом с `--native`. Если держите эксперты на
другом диске — поставьте рядом символическую ссылку.

### Вариант 4: Qwen, эксперты Q8_0

То же, что вариант 3, но `pack-q8_0/*` и `Qwen3.8-Flash-Next-Q8_0-experts.bin` вместо `pack-nvfp4/*` и NVFP4-файла;
`--pack …/pack-q8_0`, токенизатор `…/pack-q8_0/tokenizer`. Плотный GGUF и таблица — те же.

### Вариант 5: Swift 1.5, NVFP4 + Q8 (DACAN)

```bash
hf download tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN --local-dir /models/swift-dacan \
  --include "pack-nvfp4/*" "mtp/*" "data/*" \
            "Swift-1.5-Qwen3.8-Flash-Next-NVFP4-experts.bin" "Swift-1.5-Qwen3.8-Flash-Next-dense-Q8_0.gguf"
```

Здесь таблица n-грамм уже внутри плотного GGUF, поэтому `--native` и `--ple-gguf` указывают на **один и тот же**
файл `Swift-1.5-Qwen3.8-Flash-Next-dense-Q8_0.gguf`. Эксперты — рядом с ним, `--pack …/pack-nvfp4`.

### Вариант 6: Swift 1.5, Q8_0

```bash
hf download tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN --local-dir /models/swift-dacan \
  --include "pack-q8_0/*" "mtp/*" "data/*" "Swift-1.5-Qwen3.8-Flash-Next-Q8_0.gguf"
```

Один GGUF на всё: `--native` и `--ple-gguf` — `Swift-1.5-Qwen3.8-Flash-Next-Q8_0.gguf`, `--pack …/pack-q8_0`.
Эксперты движок читает из самого GGUF.

### Варианты 1–2: GSQ-RCO (руками, без установщика)

Установщик делает это сам; если хочется руками — те же шаги, что в нём:

```bash
# пример: оригинальная модель, IQ3_S (2 шарда)
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --local-dir /models/ista --include "IQ3_S/*"
S1=/models/ista/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
.venv/bin/python tools/strata_pack.py build --gguf $S1 --out /models/ista/pack-iq3s --skip-hash
.venv/bin/python tools/pack_index.py --pack /models/ista/pack-iq3s
.venv/bin/python tools/strata_tokenizer.py --gguf $S1 --out /models/ista/pack-iq3s
.venv/bin/python tools/iq_pack.py --gguf $S1 --out /models/ista/pack-iq3s
# слой MTP: скачать и упаковать
.venv/bin/python tools/mtp_fetch.py fetch --out /models/mtp
.venv/bin/python tools/mtp_pack.py --src /models/mtp --experts q2_0 --out /models/mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf /models/mtp/mtp-q2_0.gguf --out /models/mtp/rt
cp data/draft_vocab.bin /models/mtp/rt/
```

`--native` — первый шард, `--ple-gguf` — шард, где лежит тензор `per_layer_token_embd.weight` (у оригинальной модели —
второй, у Swift — первый), `--mtp /models/mtp/rt`. Таблица здесь в IQ4_NL, её можно читать прямо с диска (`--ple-io`
по умолчанию `direct`), и тогда в ОЗУ она не нужна.

### Таблицы маршрутов

`data/profile_other_2609.bin` и `data/usage_other_2609.bin` есть и в репозитории DACAN, и в папке `data/` на HF.
Для одной карты установщик берёт `data/expert-profile.bin` (8 000 экспертов — столько держит карта на 12 ГБ) или
`data/expert-profile-full.bin` (все эксперты по рейтингу — для карт от 20 ГБ).

---

## 4. Конфиг сервера

Сервер (`serve/server.py`) читает JSON. Минимум — путь к движку, его ключи и токенизатор:

```json
{
 "exe": "/opt/DACAN/run-qwen-dacan.sh",
 "args": ["--pack", "...", "--native", "...", "--ple-gguf", "..."],
 "cwd": "/opt/DACAN",
 "tokenizer": "/models/qwen-dacan/pack-nvfp4/tokenizer",
 "model_name": "qwen3.8-flash-next-dacan",
 "log": "/opt/DACAN/strata-qwen-dacan.log",
 "port": 8080
}
```

Необязательные поля:

| поле | что делает |
|---|---|
| `"host"` | `"127.0.0.1"` (по умолчанию, только этот компьютер) или `"0.0.0.0"` (вся сеть) |
| `"api_key"` | ключ, без которого сервер не отвечает (обязательно при `0.0.0.0`) |
| `"lib_dirs"` | папки с библиотеками CUDA, если их нет в системных путях |
| `"sampling"` | выборка по умолчанию, если клиент её не прислал: `{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0}` |
| `"min_max_tokens"` | если клиент просит меньший лимит ответа, сервер поднимает его до этого числа (но не дальше остатка контекста); полезно для агентов, которые шлют `max_tokens: 32000`. У нас `100000` |
| `"fit_max_tokens"` | `true`: слишком большой `max_tokens` обрезается до остатка контекста вместо ошибки 400 |
| `"vision"` | картинки: `{"exe": ".../strata-vision", "mmproj": ".../mmproj-Qwen3.8-Flash-Next-BF16.gguf", "model": "<GGUF из --native>", "gpu": true, "max_tokens": 1024}`; к ключам движка добавить `--vision --vram-reserve-mib 700` |

Переменные окружения движку проще всего дать через короткий скрипт, указанный в `"exe"` (так делает
`tools/2x2080ti/run-fast.sh`):

```sh
#!/bin/sh
export CUDA_VISIBLE_DEVICES=1,0          # главная карта — первой
export STRATA_HELPER=zqhe STRATA_COMMIT_OVERLAP=1
exec /opt/DACAN/build/strata "$@"
```

---

## 5. Ключи движка под ваше железо

Ниже — готовые наборы `"args"`. Везде `--pack/--native/--ple-gguf/--mtp` — из раздела 3.

### А. Две карты и два сокета (наша схема: 2× RTX 2080 Ti 22 ГБ + 2× Xeon с AVX-512)

```
--numa --pack <pack> --native <native.gguf> --ple-gguf <ple.gguf> --ple-io mmap
--expert-profile data/profile_other_2609.bin --expert-cache auto --adapt-every 0
--second-card 1 --second-card-usage data/usage_other_2609.bin --pcie-frac 0
--prefill 8192 --park 16 --park-mib 16384 --spec 4 --spec-min-p 0.5 --mtp <mtp>
--max-context 262144 --kv int8 --vram-reserve-mib 1024
```

Наша служба с 05.10.2026 работает с длинным окном: вместо последней строки — `--max-context 524288 --kv int8
--kv-resident 32768 --vram-reserve-mib 1024` и `--prefill 16384` (см. раздел 6).

- `--second-card 1` — вторая карта (номер внутри `CUDA_VISIBLE_DEVICES`). NVLink не обязателен: у пользователя с
  RTX 4070 Ti SUPER + RTX 5060 Ti (по 16 ГБ) работает и без него.
- `--numa` — оба сокета; не запускайте его вместе с `numactl --membind`. На одном сокете ключ просто не нужен.
- `--vram-reserve-mib 1024` — для пакетов с выходной головой в Q8_0 (варианты 3–6).

### Б. Одна большая карта (24–48 ГБ)

```
--pack <pack> --native <native.gguf> --ple-gguf <ple.gguf> --ple-io mmap
--expert-profile data/expert-profile-full.bin --expert-cache auto
--prefill 8192 --spec 4 --spec-min-p 0.5 --mtp <mtp>
--max-context 131072 --kv int8 --vram-reserve-mib 1024
```

- `expert-profile-full.bin` — чтобы карта заполнилась экспертами целиком (иначе она остановится на 8 000 и половина
  видеопамяти будет пустой). Подробно — [ONE_BIG_CARD.md](ONE_BIG_CARD.md); на 4090 48 ГБ это дало 95 т/с вместо ~50.
- На PCIe 3.0 добавьте `--pcie-frac 0.15` (значение по умолчанию 0.55 рассчитано на PCIe 4.0).
- Для вариантов 1–2 (таблица IQ4_NL) уберите `--ple-io mmap`.

### В. Одна карта 12–16 ГБ

```
--pack <pack> --native <native.gguf> --ple-gguf <ple.gguf>
--expert-profile data/expert-profile.bin --expert-cache auto
--prefill 2048 --spec 4 --spec-min-p 0.5 --mtp <mtp>
--max-context 65536 --kv int8 --kv-resident 32768
```

Это набор, который пишет установщик. Для вариантов 3–6 добавьте `--ple-io mmap --vram-reserve-mib 1024`.

### Процессор без AVX-512

Работает: эксперты, не попавшие на карту, считаются по пути AVX2 из ggml, и темп задаёт процессор (у пользователя с
2× Xeon E5-2678 v3 — около 45 т/с на размышлении). Эксперты NVFP4 на таком процессоре мы не мерили.

---

## 6. Тонкая настройка

### Контекст и KV-кеш

| ключ | значения | что важно |
|---|---|---|
| `--max-context N` | до 262 144 — родной предел модели; до 524 288 — см. ниже | KV занимает видеопамять, которую иначе взяли бы эксперты |
| `--kv` | `int8` (рекомендуем), `fp16`, `q4_0` | `int8` — 1 056 байт на токен на слой (около 3.6 ГБ на 262K), `fp16` — вдвое больше. **`q4_0` для длинного контекста не советуем**: точность падает с длиной |
| `--kv-resident N` | от 20 480; установщик ставит 32 768 | весь KV-кеш живёт в закреплённой ОЗУ, в видеопамяти — только N ячеек на слой; освободившаяся видеопамять уходит экспертам. Цена: около 13.7 КБ ОЗУ на токен контекста при `int8` |

**Контекст 524 288.** `--max-context 524288 --kv int8 --kv-resident 32768`: KV-кеш займёт 6.2 ГиБ ОЗУ. Это вдвое
дальше позиций, на которых модель учили. На наших проверках (05.10.2026, вариант 5) всё держится:

- 7 «иголок» в стоге из исходников на 461 866 токенов найдены все, включая 4 из 4 за отметкой 262K;
- функцию, спрятанную на позиции ~340K в стоге на 452K токенов, модель объяснила по шагам и вручную посчитала верно —
  тот же ответ, что на 32K; повторов и чужих алфавитов не больше, чем на коротком тексте.

Чтение промта при этом: 235K токенов — 350 с (около 670 т/с), 462K — 833 с (около 555 т/с). Один раз: дальше
разговор берётся из кеша.

С `--kv-resident` стоянка других разговоров (`--park`) не работает (в журнале: `parking other conversations is off`):
если к движку по очереди ходят несколько чатов, каждый при переключении перечитает свой контекст.

### Чтение промта

| ключ | что делает |
|---|---|
| `--prefill N` | промт читается кусками по N токенов. Больше кусок — быстрее, но нужна видеопамять: 2 048 для 12–16 ГБ, 8 192 или 16 384 для 22 ГБ и больше |
| `--prompt-cache N` | сколько точек разговора держать между запросами (по умолчанию 6, ~118 МБ ОЗУ каждая): следующий ход перечитывает только новое |
| `--park N`, `--park-mib M`, `--park-min T` | до N других разговоров ждут в ОЗУ (до M МиБ), а не перечитываются; паркуется сессия, у которой запрос переписал от T токенов |

### Ответ

| ключ | что делает |
|---|---|
| `--spec 4 --spec-min-p 0.5 --mtp <папка>` | спекулятивный вывод собственным слоем MTP модели: 4 токена черновика за проход. Без `--mtp` медленнее |
| `--expert-cache auto` | заполнить свободную видеопамять экспертами (по порядку из `--expert-profile`) |
| `--vram-reserve-mib M` | оставить M МиБ видеопамяти свободными (по умолчанию 700); больше — если карта делится с рабочим столом или браузером |
| `--temp-first` | порядок выборки как у vLLM (top-p после температуры) |

### Таблица n-грамм и ОЗУ

| случай | как |
|---|---|
| таблица Q8_0 (варианты 3–6) | только `--ple-io mmap` |
| таблица IQ4_NL (варианты 1–2) | `--ple-io direct` (по умолчанию): читается с SSD мимо ОЗУ |
| с `--numa` | таблица закрепляется в ОЗУ; `STRATA_PLE_LOCK=0` оставляет её в файловом кеше |

**ОЗУ меньше, чем эксперты + таблица** (например, 96 ГБ под NVFP4 + Q8: 63 ГиБ экспертов + 51 ГиБ таблицы): запускайте
без `--numa` (или с `STRATA_PLE_LOCK=0`), тогда таблица живёт в файловом кеше и подчитывается с диска, если не
поместилась. Держите её на NVMe. Насколько это медленнее, мы не мерили. Сами эксперты должны помещаться в ОЗУ целиком.

### Сеть и ключ

```bash
.venv/bin/python serve/server.py --engine strata --config strata-qwen-dacan.json --port 8080 --host 0.0.0.0 --api-key <ключ>
```

---

## 7. Сколько нужно ОЗУ: считаем

```
эксперты (63 ГиБ NVFP4 / 120 ГиБ Q8_0 / 40–50 ГиБ GSQ-RCO)
+ таблица n-грамм, если в ОЗУ (51 ГиБ Q8_0; IQ4_NL с --ple-io direct — 0)
+ KV при --kv-resident (13.7 КБ × контекст при int8: 1.7 ГБ на 128K, 3.6 ГБ на 262K, 6.2 ГиБ на 512K)
+ стоянка (--park-mib) и точки разговора (--prompt-cache × ~118 МБ)
+ 4–8 ГБ на систему и сервер
```

Пример: NVFP4 + Q8, таблица в ОЗУ, 262K с потоковым KV, `--park-mib 8192` — 63 + 51 + 3.6 + 8 + 6 ≈ 132 ГБ.

---

## 8. Запуск и проверка

```bash
cd /opt/DACAN
.venv/bin/python -u serve/server.py --engine strata --config strata-qwen-dacan.json --port 8080
```

Первая загрузка — от минут до получаса: эксперты читаются в ОЗУ (NVFP4 + Q8 у нас грузится ~25 минут с сетевого
хранилища, с локального NVMe быстрее). Готовность: `curl http://127.0.0.1:8080/health` отвечает `"status": "ok"` и
показывает `max_context`.

Что посмотреть в журнале (`"log"` из конфига):

| строка | значит |
|---|---|
| `NUMA: ...` с двумя группами рабочих | оба сокета в работе (`--numa`) |
| `n-gram table: 50.7 of 50.7 GiB in memory ...; locked` | таблица в ОЗУ |
| `expert cache auto: <N> GiB free, ... -> <slots> slots` | сколько экспертов поместилось на карту |
| `KV streaming: 32768 of 524288 cells per QSA layer in VRAM, the K/V in 6.19 GiB of pinned RAM` | потоковый KV включён |
| `strata serve: prompt N tokens = R reused + X read in ... (… tok/s), M generated in ... (… tok/s)` | после каждого запроса: сколько взято из кеша, скорость чтения и ответа |

Адреса: веб-чат и монитор — `http://<адрес>:8080/`; OpenAI — `http://<адрес>:8080/v1` (`/v1/chat/completions`);
Anthropic — `http://<адрес>:8080/v1/messages`. Движок обслуживает один запрос за раз.

---

## 9. Неполадки

| что видно | что сделать |
|---|---|
| загрузчик отказывает «geometry» | это не Qwen3.8-Flash-Next / Swift 1.5 той же формы (REAP и урезанные варианты не идут) |
| не найден файл экспертов | `*-experts.bin` должен лежать в папке GGUF из `--native` (или ссылка на него) |
| `cudaMalloc` при запуске | карту занимает другой процесс или мало резерва: `--vram-reserve-mib`, меньше `--max-context`, `--kv-resident` |
| `cannot pin ... GiB of RAM for a layer's KV copy` | не хватает ОЗУ под потоковый KV: меньше контекст или без `--kv-resident` |
| половина видеопамяти пустая, «8 000 experts cached» | `data/expert-profile-full.bin` (раздел 5Б) |
| ответ сильно медленнее, чем в README | ядра процессора заняты чем-то ещё (сборка, ВМ); карта перегрелась (у нас главная карта доходит до 84 °C и сбрасывает частоты) |
| `prompt (N tokens) leaves no room to answer in the context` | промт больше `--max-context`: увеличить контекст (с `--kv-resident`) или сжать историю на стороне клиента |

Вопросы и отчёты о своём железе — в issues репозитория.
