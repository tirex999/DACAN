# DACAN (Дацан)

🇬🇧 [English](README.md) | 🇷🇺 **Русский** · [❤️ Поддержать проект](#поддержать-проект)

**DACAN** — движок для запуска **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** (125 млрд
параметров, смесь экспертов) и дообученных моделей того же устройства, например
[Swift 1.5](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next), на обычном железе: одна-две игровые карты NVIDIA
и оперативная память ПК. В комплекте установщик, сервер с API, совместимым с OpenAI и Anthropic, и веб-чат.

**Движок выкладывается готовым** (`dacan` / `dacan.exe` в [релизах](https://github.com/tirex999/DACAN/releases), для
Windows x64 и Linux x64). Его исходники не публикуются; установщик и сервер открыты (MIT, [LICENSE](LICENSE)).

## Что нужно

- **Видеокарта NVIDIA начиная с серии RTX 20** (Turing, Ampere, Ada, Blackwell), **драйвер 580 или новее**:
  - RTX 20 (Turing): RTX 2060 / 2070 / 2080 / 2080 Ti (в том числе 2080 Ti на 22 ГБ), Titan RTX, Quadro RTX;
  - майнинговые **CMP 30HX / 40HX / 50HX** (Turing) и **CMP 70HX / 90HX** (Ampere) — установщик принимает их так же,
    как карты выше; сами мы на них не запускали. Шина PCIe у них урезана, поэтому загрузка модели и подкачка экспертов
    на карту идут дольше;
  - RTX 30, RTX 40, RTX 50 и профессиональные карты этих поколений.

  Старые карты (GTX 10, Volta), AMD и Intel не поддерживаются.
- **Windows 10/11 x64** или **Linux x64 с glibc 2.35 и новее** (Ubuntu 22.04 и новее, Debian 12, Fedora 36 и новее, ...).
- **Процессор с AVX2.** Объём ОЗУ зависит от модели: установщик скажет, что поместится, и предложит только это.

## Установка

**Windows**

1. Возьмите папку: `git clone https://github.com/tirex999/DACAN` или
   [ZIP](https://github.com/tirex999/DACAN/archive/refs/heads/main.zip) (распакуйте куда угодно).
2. Запустите **`START-HERE.bat`**. Он спросит модель, её размер, контекст и картинки, скачает всё (движок DACAN из
   релиза, модели с Hugging Face) и запустит сервер. Python поставит сам, если его нет.

**Linux**

```sh
git clone https://github.com/tirex999/DACAN
cd DACAN
./setup.sh
```

Вопросы те же, что на Windows; движок `dacan-linux-x64.zip` скачивается из релиза. Заранее нужен только драйвер
NVIDIA: Python 3.10+ с venv ставится через apt / dnf / pacman, если его нет (спросит пароль sudo), библиотеки CUDA 13
приходят pip-пакетами в собственный `.venv` папки. Следующие запуски `./setup.sh` просто стартуют модель.

## Обновление

**`UPDATE.bat`** (Windows) или **`./update.sh`** (Linux): свежие файлы (`git pull`), новый движок DACAN, если вышел,
настройки моделей. Модели заново не скачиваются. Дальше запуск как обычно (`START-HERE.bat` / `./setup.sh`).

Установку прежней версии DACAN обновление переносит само: папка со скачанными моделями переименовывается в
`DACAN-data`, конфиги моделей — в `dacan-<модель>.json`, старый движок заменяется.

Если в вашей копии ещё нет `UPDATE.bat` / `update.sh`, один раз выполните `git pull` в её папке (или распакуйте ZIP
поверх), затем обновление.

## Параметры выборки

Как выбирается следующее слово. По умолчанию стоят рекомендованные для модели значения, их DACAN и использует:

| параметр | по умолчанию | что делает |
|---|---|---|
| `temperature` | **0.6** | насколько свободно выбираются слова: 0 = всегда самое вероятное слово (на тот же вопрос — тот же ответ), выше = разнообразнее |
| `top_p` | **0.95** | рассматриваются только самые вероятные слова, вместе дающие 95 % вероятности |
| `top_k` | **20** | рассматривается не больше стольких самых вероятных слов (DACAN принимает от 1 до 64) |
| `presence_penalty`, `repetition_penalty`, `frequency_penalty` | **выкл.** | отталкивают модель от уже сказанных слов; включайте, только если ответы повторяются |

- **Веб-чат** по умолчанию отправляет именно эти значения (⚙ настройки: Temperature, Top-p, Top-k). Переключатель
  «Use for other apps too» делает настройки веб-чата значениями по умолчанию для всех клиентов API.
- **Клиенты API** (Open WebUI, Claude Code, свои скрипты) получают значения по умолчанию для всех полей, которые сами
  не прислали: они лежат в блоке `"sampling"` конфига модели `dacan-<модель>.json` рядом с `setup.py`, менять их можно
  там (после правки перезапустите модель). Поля из самого запроса всегда главнее.
- **Скорость.** Быстрее всего `temperature` 0: движок угадывает несколько токенов вперёд и проверяет их за один шаг, а
  когда каждый раз берётся самое вероятное слово, угаданных принимается больше. Со случайной выборкой (temperature
  выше 0) их принимается меньше.

## Веса

- **[tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN)**:
  Swift 1.5 в **NVFP4 + Q8 (DACAN)** и 8 бит (Q8_0).
- **[tirex2001/Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Qwen3.8-Flash-Next-DACAN)**: исходная модель в
  тех же форматах.
- Кванты GSQ-RCO (2-3.5 бита) от [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) и
  [UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) ставит установщик.

Словарь черновиков (слова, которые движок угадывает наперёд) выбирает установщик: по умолчанию кириллический — для
чата на русском; английский и французский тоже есть (`--draft-vocab en` / `fr`).

## Документы

[docs/](docs/) (на английском): [DETAILS](docs/DETAILS.md) (флаги, API, картинки, как устроен движок),
[MULTI_GPU](docs/MULTI_GPU.md) и [SECOND_GPU](docs/SECOND_GPU.md) (две карты), [BATCHING](docs/BATCHING.md) (несколько
запросов сразу), [UNSLOTH_Q4](docs/UNSLOTH_Q4.md), [MCP_SERVER](docs/MCP_SERVER.md) (DACAN как инструмент для
ИИ-ассистентов), [TROUBLESHOOTING](docs/TROUBLESHOOTING.md). Вопросы и проблемы:
[issues](https://github.com/tirex999/DACAN/issues).

## Лицензии

Установщик и сервер: MIT ([LICENSE](LICENSE)). [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT) внутри
движка и кодировщика картинок. У весов свои лицензии: Qwen Community License 1.0 у Qwen3.8-Flash-Next, NVIDIA Open
Model License у NVFP4-экспертов NVIDIA, Swift Open License v1.0 у Swift 1.5.

---

## Поддержать проект

Всё здесь — движок, кванты, замеры — сделано и выложено бесплатно. Если это помогло вам запустить модель на своём
железе, можно сказать спасибо донатом.

| | адрес | QR |
|---|---|---|
| **ЮMoney** (рубли: кошелёк ЮMoney или любая банковская карта) | `4100119356331418` · [перевести](https://yoomoney.ru/to/4100119356331418) | <img src="docs/donate/yoomoney.svg" width="130" alt="QR ЮMoney"> |
| **USDT** (TRC-20, сеть Tron) | `TBvoJHi7uyonSpvdH9Y6RAAXGWYVR2jeqw` | <img src="docs/donate/usdt-trc20.svg" width="130" alt="QR USDT TRC-20"> |
| **Ethereum** (ETH и токены сети Ethereum, ERC-20) | `0x66CA7c683fbaF030b2300c918A3751209eA30dEa` | <img src="docs/donate/eth.svg" width="130" alt="QR Ethereum"> |

> ⚠️ **Сеть важна.** На адрес Tron отправляйте только активы сети Tron (USDT TRC-20, TRX), на адрес Ethereum — только
> активы сети Ethereum. Отправленное не в той сети теряется.

Спасибо! 🙏
