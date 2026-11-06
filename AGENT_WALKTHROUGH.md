# Contoh pertama: router → action → tools → router

Ini potongan pembelajaran yang bisa dieksekusi, belum runtime final. Dua model
punya tugas berbeda: `qwen-base` memutuskan delegasi atau merespons percakapan,
sedangkan `sellerops-sft` memilih action SellerOps. Nama tersebut adalah alias
di `src/sellerops/agent/config.py`.

## Urutan membaca kode

1. `agent/state.py`: messages digabung menggunakan reducer; field lain diganti
   oleh update terbaru. `route`, `status`, dan `response` boleh belum ada pada
   input awal karena node yang akan mengisinya.
2. `agent/nodes.py`, `router_node`: model menghasilkan JSON `decision/text`.
   `act` meneruskan pekerjaan tanpa menambahkan JSON routing ke history.
   `reply` menambahkan jawaban dan `ask` menambahkan pertanyaan.
3. `call_model`: SFT membaca percakapan asli dan observation, bukan instruksi
   subtask baru buatan router. Action JSON tetap mengikuti format latihan.
4. `tool_node`: mengambil tepat satu tool call, mengonversinya, mengeksekusi
   lewat EpisodeService, lalu mengembalikan ToolMessage dengan ID yang cocok.
5. `agent/graph.py`: conditional edges memilih node berikutnya. `partial`
   mengikat service ke tool_node; LangGraph cukup memasukkan state.
6. `_model_step`: satu attempt model, pencatatan counter, dan keputusan retry
   schema. Edge `retry` kembali ke node yang sama; output invalid tidak ke tool.

```text
START → router ── reply / ask ──→ END
           │
           act
           ↓
         action ── final / ask_user ──→ router
           │
           tool_call
           ↓
         tools ─────────────────────→ router

router/action ── schema invalid + budget tersedia ──→ node yang sama
router/action ── budget habis / model error ─────────→ END (failed)
```

SFT `final` merupakan kandidat hasil dan `ask_user` merupakan kebutuhan
klarifikasi. Keduanya kembali ke router untuk disampaikan ke user. Router
memiliki prompt sendiri; system prompt routing tidak disimpan dalam state.

## Mencoba dengan model asli

Dari folder proyek dengan Conda `llm-post-training` aktif dan `src` tersedia
pada Python import path:

```python
from langchain_core.messages import HumanMessage
from sellerops.agent.graph import build_graph
from sellerops.services.episode_service.episode_service import EpisodeService

service = EpisodeService()
episode = service.create_episode("belajar-router")
graph = build_graph(service)

config = {
    "configurable": {
        "router_model_name": "qwen-base",
        "action_model_name": "sellerops-sft",
        "max_new_tokens": 256,
        "max_model_calls": 8,
        "max_retries": 2,
    },
    "recursion_limit": 40,
}
state = {
    "messages": [HumanMessage(content="Cek stok P1001")],
    "session_id": episode.session_id,
    "model_calls": 0,
    "retry_count": 0,
}
result = graph.invoke(state, config)
print(result["status"], result["response"])
```

Pemanggilan nyata memuat dua model; qwen-base mungkin perlu diunduh jika belum
ada di cache. Tidak ada model yang diunduh/dijalankan saat pengembangan contoh
ini. Cache adapter menampung dua model sehingga pergantian router/SFT tidak
memuat ulang bobot setiap langkah.

## Mempelajari tanpa mengunduh model

Jalankan `conda run -n llm-post-training pytest -q tests/test_agent_graph.py`.
Tes memakai graph dan EpisodeService asli; hanya generation model yang diganti
output terprogram. Mulai membaca `test_router_sft_tool_router_round_trip` untuk
melihat tiga completion, urutan alias model, dan history hasilnya. Tes ini
membuktikan alur kode, bukan kemampuan model merencanakan atau menjawab benar.

Untuk klarifikasi, lihat `test_clarification_continues_with_manual_history`:
setelah `waiting_user`, tambahkan HumanMessage ke hasil state dan invoke lagi.
Graph ini belum menggunakan checkpointer, sehingga seluruh state dikirim ulang.
Counter tetap berlaku untuk tugas yang sama. Lifecycle tugas baru dan pemisahan
riwayat antar-episode belum diimplementasikan.

## Retry yang sudah tersedia dan latihan berikutnya

`max_retries=2` membatasi percobaan ulang schema untuk seluruh episode, dibagi
router dan action. `retry_count` naik saat retry dimulai; semua percobaan juga
menambah `model_calls`. Jika budget call habis sebelum retry dimulai, retry
tidak dihitung. Feedback validasi hanya masuk prompt privat untuk perbaikan.
Lihat `test_schema_retries_recover_on_same_node_and_count_every_attempt` di
`tests/evaluation/test_langsmith.py` untuk contoh recovery kedua model.

- Tambahkan pemeriksaan hasil untuk tugas terstruktur sebelum jawaban diterima.
  `completed` sekarang hanya berarti router memilih menjawab, bukan verifier
  menyatakan jawaban benar. Jangan berikan oracle atau isi environment tersembunyi
  ke model. Jawaban bebas router masih bisa salah meskipun prompt meminta bukti.
- Pisahkan percakapan dan episode tugas, lalu tambahkan penyimpanan/resume.
  History percakapan bebas dari router belum ada dalam distribusi latihan SFT;
  perilaku follow-up model harus diuji pada development tasks.

Batas `max_model_calls` menghentikan loop dengan `status=failed`. Batas tool
tetap milik EpisodeService; pelanggaran batas/exception tool keluar sebagai
error, belum ditangani menjadi respons percakapan oleh graph langsung. Evaluation
target menangkapnya sebagai failed episode. Belum ada replan, memory persisten,
timeout, atau integrasi verifier di dalam loop runtime.

## Router berbasis aturan

`configurable.router_mode="rule"` mengganti router model dengan `rule_route`
di `agent/nodes.py`: pesan terakhir berupa `final` SFT menjadi reply,
`ask_user` menjadi pertanyaan, dan selain itu selalu `act`. Tidak ada panggilan
model untuk router, sehingga `model_calls` hanya menghitung SFT. Pada mode
`llm`, `act` yang membawa teks kini dinormalisasi (teks dibuang) kecuali
`strict_router_schema=True`. Lihat `test_rule_router_*` di
`tests/test_agent_graph.py`.

## Evaluasi melalui LangSmith SDK

Runner di `evaluation/runner.py` menjalankan graph ini per task dev, lalu
memanggil verifier setelah episode berhenti. Lihat
[panduan evaluation](src/sellerops/evaluation/README.md) untuk mode offline,
cloud, dan commented code pembanding loop lokal.

`call_model` kini menyimpan kandidat `final_action` dalam state, dan
mengosongkannya saat menghasilkan tool call berikutnya. Routing tetap sama;
router yang langsung menjawab tanpa kandidat tersebut dicatat sebagai
`missing_structured_outcome`. Ini akan terlihat sebagai kegagalan terukur.
Pencatatan attempts dan counter state menghitung generation yang gagal parsing.
Evaluasi dapat mengulang task pada episode baru dan mengagregasi pass@1/pass@k;
pengulangan episode terpisah dari retry schema di dalam graph.
