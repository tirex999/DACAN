# 28.09.2026: плотная часть Qwen3.8-Flash-Next из шарда нашего кванта A (IQ4X-Q8A-00001-of-00002.gguf) в отдельный GGUF
# для выкладки DACAN на Hugging Face: DACAN берёт из --native только плотные проекции и голову, маршрутные эксперты
# идут из файла экспертов пакета (NVFP4 или Q8_0). Копия метаданных и тензоров по образцу
# gguf-py/gguf/scripts/gguf_new_metadata.py (copy_with_new_metadata), без тензоров *_exps и без ключей split.*.
#   GGUF_PY=<llama.cpp>/gguf-py python dense_gguf.py <src.gguf> <dst.gguf> [--dry]
# 03.10.2026: проверено — движок с --native на выходном файле дал три жадных ответа знак в знак как на исходном шарде.
import os
import re
import sys

sys.path.insert(0, os.environ.get("GGUF_PY", "/q/llama-up/gguf-py"))   # gguf-py из llama.cpp (или pip install gguf)
import gguf  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
dry = "--dry" in sys.argv
reader = gguf.GGUFReader(src, "r")
arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
EXPS = re.compile(r"\.ffn_(gate|up|down|gate_up)_exps\.")
keep = [t for t in reader.tensors if not EXPS.search(t.name)]
drop = [t for t in reader.tensors if EXPS.search(t.name)]
print(f"архитектура {arch}; тензоров {len(reader.tensors)}: оставляю {len(keep)} "
      f"({sum(t.n_bytes for t in keep) / 2**30:.2f} ГиБ), убираю {len(drop)} ({sum(t.n_bytes for t in drop) / 2**30:.2f} ГиБ)")
print("примеры убранных:", [t.name for t in drop[:4]])
print("ключи split:", [f.name for f in reader.fields.values() if f.name.startswith("split.")])
if dry:
    sys.exit(0)

writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF.") or field.name.startswith("split."):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
for t in keep:
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for t in keep:
    writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
writer.close()
print("записан", dst)
