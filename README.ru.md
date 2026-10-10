# DACAN (Дацан)

🇬🇧 [English](README.md) | 🇷🇺 **Русский** · [❤️ Поддержать проект](#поддержать-проект)

**DACAN** — движок для запуска **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** (125 млрд
параметров, смесь экспертов) и дообученных версий той же формы, например
[Swift 1.5](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next), на обычном железе. Начиная с этой версии он
построен на [Strata](https://github.com/Niko1221/Strata) 0.1.42 и добавляет к ней своё:

- **RTX 20 (Turing)** — промт на этих картах читается быстрее, без ключей;
- **наши кванты NVFP4 + Q8 (DACAN)** — эксперты NVFP4 считаются и на карте, и на процессоре (AVX-512);
- **`--numa`** — для серверов с двумя сокетами (Linux).

Всё остальное — установщик, сервер, веб-приложение, API OpenAI и Anthropic, деление модели на несколько карт — как в
Strata 0.1.42.

**Движок выкладываем готовым** (`strata.exe` в [релизах](https://github.com/tirex999/DACAN/releases)); исходники
наших изменений не публикуем. Установщик и сервер — открытые, из Strata (MIT).

## Установка (Windows)

Нужно: видеокарта NVIDIA от RTX 20 и новее, драйвер 580 или новее, Windows 10/11 x64, процессор с AVX2. Python установщик поставит сам.

1. Скачайте папку: `git clone https://github.com/tirex999/DACAN` или
   [ZIP](https://github.com/tirex999/DACAN/archive/refs/heads/main.zip) (распакуйте куда угодно).
2. Запустите **`START-HERE.bat`**. Он спросит модель, размер, контекст и картинки, всё скачает (движок DACAN — из
   релиза, модели — с Hugging Face) и запустит сервер.

## Обновление

**`UPDATE.bat`** — свежие файлы (`git pull`), новый движок DACAN, если он вышел, и настройки моделей. Модели заново не
качаются. Потом запуск — как обычно, `START-HERE.bat`.

**Если у вас стоял прежний DACAN** (без `UPDATE.bat`): один раз выполните в его папке `git pull` (или скачайте ZIP
заново и распакуйте рядом), затем `UPDATE.bat`. Скачанные модели установщик сам перенесёт в папку `Strata-data` рядом и
найдёт их там. Движок апстрима, если он стоял, заменится на движок DACAN.

## Linux

Готовый движок DACAN для Linux выйдет следом. Пока на Linux работает прежняя версия — ветка
[`dacan-1`](https://github.com/tirex999/DACAN/tree/dacan-1) (две RTX 2080 Ti, `--second-card`, `--park`, её README).

## Карты старше RTX 20 и AMD

DACAN выпускает движок только для NVIDIA RTX 20 и новее. Для Pascal / Volta (CUDA 12) и AMD установщик берёт готовый
движок Strata той же версии.

## Веса

- **[tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Swift-1.5-Qwen3.8-Flash-Next-DACAN)** —
  Swift 1.5: **NVFP4 + Q8 (DACAN)** и 8 бит (Q8_0).
- **[tirex2001/Qwen3.8-Flash-Next-DACAN](https://huggingface.co/tirex2001/Qwen3.8-Flash-Next-DACAN)** — исходная модель
  в тех же форматах.
- Кванты GSQ-RCO (2–3.5 бита) от [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) и
  [UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) ставит установщик.

## Документы

Руководство Strata (установка, ключи, API, картинки, неполадки) — в [docs/](docs/), главное —
[docs/DETAILS.md](docs/DETAILS.md).

## Благодарности и лицензии

[Strata](https://github.com/Niko1221/Strata) от Niko1221 и её участников (MIT, [LICENSE](LICENSE)) — основа движка,
установщик и сервер. [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT). У весов свои лицензии: Qwen
Community License 1.0 у Qwen3.8-Flash-Next, NVIDIA Open Model License у экспертов NVFP4 от NVIDIA, Swift Open License
v1.0 у Swift 1.5.

---

## Поддержать проект

Всё, что здесь лежит, — движок, кванты, замеры — мы делаем и выкладываем бесплатно. Если это помогло вам запустить
модель на своём железе, можно поблагодарить донатом.

| | адрес | QR |
|---|---|---|
| **ЮMoney** (рубли: кошелёк ЮMoney или карта любого банка) | `4100119356331418` · [перевести](https://yoomoney.ru/to/4100119356331418) | <img src="docs/donate/yoomoney.svg" width="130" alt="QR ЮMoney"> |
| **USDT** (TRC-20, сеть Tron) | `TBvoJHi7uyonSpvdH9Y6RAAXGWYVR2jeqw` | <img src="docs/donate/usdt-trc20.svg" width="130" alt="QR USDT TRC-20"> |
| **Ethereum** (ETH и токены сети Ethereum, ERC-20) | `0x66CA7c683fbaF030b2300c918A3751209eA30dEa` | <img src="docs/donate/eth.svg" width="130" alt="QR Ethereum"> |

> ⚠️ **Сеть важна.** На адрес Tron отправляйте только активы сети Tron (USDT TRC-20, TRX), на адрес Ethereum — только
> активы сети Ethereum. Отправленное не в той сети пропадёт.

Спасибо! 🙏
