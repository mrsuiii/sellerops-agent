# Eval seller agent (multi-turn)

Eval ini mengukur seller agent pada **simulator seller**: data penjualan,
stok, dan review sintetis toko Kala Studio (`data/seller_sim/`). Polanya
mengikuti τ-bench: agent bercakap dengan *user simulator* yang berperan
sebagai penjual, lalu verifier menilai **hasil akhir percakapan**, bukan
urutan langkah agent.

Harness lama (task offer/diskon `sellerops-sft-v0.2`, router Qwen 0.5B,
runner LangSmith) sudah dihapus pada 17 Nov 2026. Polanya tetap jadi acuan.

## Komponen

| Modul | Isi |
| --- | --- |
| `simulator/generator.py` | Generator 90 hari order, stok, review (seed tetap) + skenario tertanam |
| `simulator/store.py` | Read model: `sales_summary`, `product_performance`, `inventory_status`, `reviews`, `product`, `propose_caption_update` |
| `simulator/episode.py` | `SellerEpisode`: proposal milik episode sendiri + riwayat tool call |
| `evaluation/seller_tools.py` | Enam tool seller production (`harness/tools/seller.py`) dengan `SellerEpisode` sebagai client |
| `evaluation/terminal_tools.py` | `final(message, data)` dan `ask_user(question)` |
| `evaluation/prompts.py` | `SELLER_SYSTEM_PROMPT` |
| `evaluation/seller_tasks.py` | Task dev/test dengan kunci jawaban yang dihitung dari data |
| `evaluation/user_sim.py` | `ScriptedUser` (deterministik) dan `LLMUser` (model chat) |
| `evaluation/conversation.py` | Loop giliran agent → user → agent pada satu `thread_id` |
| `evaluation/seller_verifier.py` | Cek `finished`, `answer`, `grounded`, `caption_flow` |
| `evaluation/metrics.py` | pass@k dan pass^k |

## Data dan skenario tertanam

`python scripts/generate_seller_sim.py` (atau `make sim`) menulis ulang
`data/seller_sim/`. Hasilnya byte-identik untuk katalog dan seed yang sama;
`manifest.json` dan `tasks.manifest.json` menyimpan checksum, dan test gagal
kalau file yang di-commit tidak sesuai generator.

"Hari ini" simulator adalah `AS_OF = 2026-11-15`; periode tool adalah jendela
tetap (`last_7_days`, `previous_7_days`, `last_30_days`, `previous_30_days`,
`last_90_days`).

| Skenario | Produk | Yang terlihat di data |
| --- | --- | --- |
| `spike` | FS163 Girl Summer Dress | ~4x lebih laku 7 hari terakhir |
| `stockout` | FS186 Calvin Klein Heel Shoes | Habis 20 hari terakhir; omzet sepatu wanita turun ~50% |
| `low_stock` | FS180 Dress Pea | Laris, stok tinggal 6 (habis < 3 hari) |
| `slow_movers` | FS174 Prada Women Bag, FS157 Party Glasses | Tidak terjual 45 hari terakhir |
| `negative_reviews` | FS188 Pampi Shoes | 6 review 1–2 bintang "ukuran kekecilan" dalam 3 minggu |

Stok konsisten: `initial_stock = units_sold + stock`. Rumus restock:
`days_to_stockout = stock / (unit 30 hari / 30)`; saran restock = cukup untuk
`target_days` (default 14) dikurangi stok.

## Task

Setiap task berisi `opening_message`, `persona`, `goal`, `hidden_info`
(detail yang baru diberikan kalau ditanya), `expected` (kunci jawaban, dihitung
dari `SellerStore`), dan `needs_clarification`. Family: `sales_summary`,
`top_category`, `top_products`, `slow_movers`, `restock`, `review_insight`,
`sales_spike`, `sales_drop`, `caption`. Split `dev` (10) untuk iterasi,
`test` (9) **frozen**: jangan dipakai untuk tuning prompt atau graph.

## Percakapan

```python
from langgraph.checkpoint.memory import InMemorySaver
from sellerops.evaluation.conversation import run_conversation
from sellerops.evaluation.prompts import SELLER_SYSTEM_PROMPT
from sellerops.evaluation.seller_tasks import load_tasks
from sellerops.evaluation.seller_tools import make_seller_tools
from sellerops.evaluation.seller_verifier import verify_conversation
from sellerops.evaluation.terminal_tools import TERMINAL_TOOLS
from sellerops.evaluation.user_sim import ScriptedUser
from sellerops.harness.graph import build_agent_graph
from sellerops.harness.models import get_chat_model
from sellerops.simulator import SellerEpisode

task = load_tasks("dev")[0]
episode = SellerEpisode()
graph = build_agent_graph(get_chat_model(), make_seller_tools(episode), system_prompt=SELLER_SYSTEM_PROMPT,
                          terminal_tools=TERMINAL_TOOLS, checkpointer=InMemorySaver())
result = run_conversation(graph, task, ScriptedUser())
report = verify_conversation(task, episode.snapshot(), result, store=episode.store)
```

- Satu percakapan = satu `thread_id`; tiap `graph.invoke` = satu giliran agent.
- `ask_user` mengakhiri **giliran**. Balasan penjual dikirim sebagai
  `ToolMessage` untuk call `ask_user` tersebut (Messages API mewajibkan setiap
  `tool_use` diikuti `tool_result`), lalu graph dilanjutkan pada thread yang sama.
- Input setiap giliran membawa `final_action=None, stop_reason=None` agar field
  per giliran tidak terbawa (`turn_reset` bisa ditambah, mis. counter step).
- `final` mengakhiri percakapan; stop lain (`max_steps`, `no_tool_call`, …)
  berarti percakapan selesai tanpa jawaban.

## Penilaian

- `finished`: percakapan berakhir dengan `final` yang valid.
- `answer`: `final.data` cocok dengan kunci jawaban (angka persis; ranking
  urut; himpunan produk tanpa urutan; keluhan lewat kata kunci).
- `grounded`: setiap angka di `final.data` muncul di hasil tool yang sukses
  pada episode itu.
- `caption_flow`: ada proposal `pending` untuk produk yang benar, hanya field
  yang diizinkan, sesuai permintaan penjual, dan produk tidak berubah.
- Metrik (bukan syarat lulus): jumlah giliran, jumlah tool call, apakah agent
  bertanya saat task butuh klarifikasi.

`message` bebas belum dinilai; `judge.py` (Gemini) masih memakai rubrik lama
dan perlu rubrik analis seller sebelum dipakai lagi.

Agregasi: `pass_metrics` (pass@k, minimal satu dari k percobaan lulus) dan
`pass_hat_metrics` (pass^k τ-bench, semua k percobaan lulus) dari hasil
`decision` per task dan repetisi.
