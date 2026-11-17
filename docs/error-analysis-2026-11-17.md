# Error analysis — seller agent, dev split (17 Nov 2026)

Session A3. Metode ala Hamel Husain: baca trajectory gagal satu per satu, beri
label akar masalah, perbaiki penyebabnya (bukan gejalanya), ukur ulang dengan
konfigurasi yang sama, dan jadikan kegagalan penting regression test.

## Setup

| | |
|---|---|
| Agent | `claude-haiku-5-5`, effort `medium`, graph A1/A2 (`harness/graph.py`) |
| User simulator | `LLMUser` (Haiku 5.5, effort `low`) dari persona + hidden info |
| Task | 10 task dev (`data/seller_sim/tasks.dev.jsonl`), k = 3 → 30 percakapan per run |
| Penilaian | verifier (`finished`, `answer`, `grounded`, `caption_flow`) + judge `message` NVIDIA `nemotron-3-ultra-550b-a55b` (hanya untuk yang lolos verifier) |
| Alat | `scripts/show_failures.py`, `scripts/replay_eval_thread.py`, trace LangSmith project `sellerops-eval` |

Split test (frozen) **tidak** dipakai untuk analisis atau tuning.

## Hasil

| Metrik | Baseline `…155835…-0437` | RCA-1..3 `…162626…-83b3` | Final (+RCA-5) `…163243…-deb2` |
|---|---|---|---|
| pass@1 | 0.83 | 0.97 | **1.00** |
| pass@3 | 0.90 | 1.00 | **1.00** |
| pass^3 (lolos di 3/3 percobaan) | 0.80 | 0.90 | **1.00** |
| Gagal verifier | 5 / 30 (caption 3, slow_movers 2) | 1 / 30 (sales_drop 1) | 0 / 30 |
| Bertanya saat task butuh klarifikasi | 9 / 12 | 12 / 12 | 12 / 12 |
| Judge `message` lolos (dari yang lolos verifier) | 24 / 25 | 27 / 29 | 29 / 30 |
| Rata-rata tool call / giliran agent | 1.6 / 1.3 | 1.3 / 1.4 | 1.4 / 1.4 |
| Latency p50 / p95 | 6.0 s / 8.7 s | 5.7 s / 10.3 s | 5.9 s / 10.1 s |
| Token agent per percakapan (input / cache read / output) | 10.7k / 8.3k / 960 | 10.6k / 9.1k / 830 | 11.5k / 10.1k / 870 |

Semua run memakai konfigurasi identik; hanya prompt, tool, dan validasi
`final` yang berubah. Satu
run k=3 per task masih kecil: selisih satu percakapan = 0.03 pada pass@1.

## Kegagalan dan akar masalah

### RCA-1 — Caption: mengubah field yang tidak diminta (caption 3/3 gagal)

- **Gejala.** "Perbaiki caption Dress Pea dong." → agent langsung
  `propose_caption_update` dengan `name`, `description`, dan `tags`.
  Hidden info: hanya deskripsi yang boleh diubah. `caption_flow` gagal:
  *changed fields ['description', 'name', 'tags'], seller asked for
  ['description']*. Agent tidak pernah bertanya.
- **Reproduksi.** `replay_eval_thread.py --task caption --times 2` dari
  checkpoint sebelum keputusan terakhir: gagal 2/2 dengan alasan sama
  (deterministik secara perilaku, bukan kebetulan sampling).
- **Akar masalah (prompt).** Aturan klarifikasi menyebut "period, category,
  product" sebagai detail yang ditanyakan, tidak menyebut field caption.
  Tidak ada aturan "ubah seminimal mungkin". Model menganggap "perbaiki
  caption" = boleh mengubah semua bagian.
- **Perbaikan.** Prompt: *Change only the fields the seller asked for (name,
  description, tags); if they did not say which, ask first.*
- **Hasil.** caption 3/3 lolos, agent bertanya field mana sebelum mengusulkan.

### RCA-2 — Klarifikasi tidak lengkap: hanya menanyakan periode (slow_movers 2/3 gagal)

- **Gejala.** "Produk mana yang nggak laku sama sekali?" → agent bertanya
  periode saja, lalu mengembalikan 7 produk nol penjualan dari semua
  kategori. Seller (hidden info) maksudnya kategori tas → kunci `FS174`.
  Percobaan yang lolos (t1) kebetulan menanyakan periode *dan* kategori.
- **Akar masalah (prompt).** Aturan "Ask one short question" ditafsirkan
  sebagai satu detail per pertanyaan; setelah satu jawaban agent langsung
  mengerjakan. Cakupan (semua produk vs satu kategori) tidak disebut sebagai
  detail yang perlu ditanyakan. `ScriptedUser` di smoke run menyembunyikan
  masalah ini karena ia membuka fakta berikutnya walau tidak ditanya;
  `LLMUser` hanya menjawab yang ditanya (lebih realistis).
- **Pertimbangan.** Pertanyaan pembuka memang ambigu; jawaban "semua
  kategori" masuk akal secara literal. Tetapi split test punya pola yang sama
  (restock: "Produk mana yang perlu restock segera?" dengan hidden info
  kategori dress), jadi benchmark ini memang mengharapkan agent menanyakan
  cakupan. Task tidak diubah; perilaku agent yang diperbaiki.
- **Perbaikan.** Prompt: *Ask once, covering every missing detail in one
  short question: the period, and for questions about products whether they
  mean all products or one category (or which product).*
- **Hasil.** slow_movers 3/3 lolos; klarifikasi saat dibutuhkan 12/12.
  Risiko: pertanyaan tambahan pada task yang tidak membutuhkannya (rata-rata
  giliran agent naik 1.3 → 1.4); tidak ada task yang gagal karenanya.

### RCA-3 — Klaim tren dikarang dari perbandingan di kepala model (judge: sales_drop)

- **Gejala.** Lolos verifier (FS186 + stok habis benar), tapi `message`
  menyatakan "Pampi Shoes dan Golden Shoes naik", padahal data tool
  menunjukkan Pampi turun. Judge: *faithful = false*. Temuan sejenis di A1:
  "turun sekitar 34%" yang dihitung model sendiri.
- **Akar masalah (tool).** Tool hanya mengembalikan angka per periode; untuk
  menjawab "kenapa turun" model harus memanggil dua periode dan
  membandingkan puluhan baris sendiri. Perbandingan antarbaris adalah
  aritmetika di kepala model → rawan salah, dan verifier tidak memeriksa
  `message`.
- **Perbaikan.** Tool `get_sales_summary` dan `get_product_performance`
  mendapat `compare_to_previous` (last_7 vs previous_7, last_30 vs
  previous_30): `previous_*`, `*_change_pct`, `units_change`, `trend`
  dihitung oleh simulator (dan nanti oleh service production). Prompt: *For
  any rise, fall or comparison, call the tool with `compare_to_previous:
  true` and report its change fields; never compute changes yourself.*
- **Hasil.** Klaim tren di sales_drop kini sesuai data di 3/3 percobaan
  (contoh: "Pampi Shoes turun 11,3%"). Persentase di `message` kini
  berasal dari tool sehingga bisa diverifikasi.

### RCA-4 — Bug runner: giliran kedua dst. tidak berjalan (infra, smoke run pertama)

- **Gejala.** Tiga task multi-turn mengulang pertanyaan yang sama 5 kali;
  state graph hanya berisi 2 pesan setelah 5 giliran.
- **Akar masalah (eval harness).** Config invoke membawa
  `metadata={"run_id": ...}`. LangGraph membaca key itu; setiap invoke
  setelah yang pertama mengembalikan state lama tanpa menjalankan node.
  Diisolasi dengan model palsu (hanya key `run_id` yang memicu).
- **Perbaikan.** Key diganti `eval_run_id`; regression test
  `test_multi_turn_trial_carries_the_reply_to_the_next_turn`.
- **Pelajaran.** Hasil eval pertama selalu dicurigai dulu sebagai bug harness
  sebelum disimpulkan sebagai kelemahan model.

### RCA-5 — Kontrak `final.data` tidak diikuti (sales_drop 1/3 setelah perbaikan)

- **Gejala.** Analisis benar (FS186, stok habis, tren lain sesuai tool),
  tetapi `final.data` = `{"product_ids": ["FS186", "FS188"], ...}`, padahal
  kontrak family sales_drop meminta satu `product_id`. Verifier `answer`
  gagal.
- **Akar masalah (kontrak output).** Data perbandingan yang lebih kaya
  membuat model ingin melaporkan dua produk; skema `final.data` adalah
  `dict` bebas yang hanya dijelaskan di docstring, jadi tidak ada validasi
  bentuk per family. Muncul setelah RCA-3 → efek samping perbaikan.
- **Perbaikan.** Validator bentuk di `FinalAnswer` (tanpa pengetahuan
  task): jawaban yang memuat `cause` atau `complaint` adalah tentang satu
  produk → wajib `product_id` (string), tanpa `product_ids`; `cause` harus
  salah satu nilai kontrak. Pelanggaran → `ValidationError` → node `tools`
  mengembalikan `invalid_arguments` sehingga model memanggil `final` lagi.
  Solusi referensi 19 task tetap lolos (validator tidak menolak jawaban
  benar).
- **Hasil.** sales_drop 3/3 lolos (run khusus dan run final).

### Kegagalan judge setelah perbaikan (RCA-1..3: 2 dari 29; final: 1 dari 30)

| Percakapan | Klaim | Penilaian |
|---|---|---|
| review_insight t2 | "beberapa pembeli harus tukar ukuran" | hanya satu ulasan menyebut tukar ukuran → berlebihan; judge benar |
| slow_movers t1 | "Women Handbag Black? Tidak, itu bukan yang paling sedikit…" | model "berpikir keras" di `message` dan salah (25 unit memang paling sedikit); judge benar |

Run final: sales_drop t2 menyebut "7 ulasan Pampi, rata-rata 1,86", padahal 7
adalah jumlah ulasan semua sepatu wanita (Pampi 6) — agregat yang tidak cermat
di `message`; judge benar.

Ketiganya masalah kualitas prosa, bukan angka di `final.data`. Belum
diperbaiki: butuh lebih banyak contoh sebelum mengubah prompt (risiko
overfitting ke dua percakapan). Kandidat: aturan "jangan menulis proses
berpikir di message" dan "kuantitas (beberapa/banyak) harus sesuai jumlah di
tool".

## Pembanding: agent NVIDIA `nemotron-3-ultra-550b-a55b`

Konfigurasi sama (`LLMUser` Claude, k=3, prompt dan tool final);
`--agent-provider openai_compatible`. Run `…163424…-2e25`.

| Metrik | Haiku 5.5 (final) | Nemotron 3 Ultra |
|---|---|---|
| pass@1 / pass@3 / pass^3 | 1.00 / 1.00 / 1.00 | 0.83 / 0.90 / 0.80 |
| Gagal | – | review_insight 3/3 (`answer`), caption 2/3 (`finished`) |
| Latency p50 / p95 | 5.9 s / 10.1 s | 19.0 s / 48.8 s |
| Prompt cache | ~10k token/percakapan dibaca dari cache | tidak ada |

- review_insight: isi benar (FS188, ukuran kekecilan) tetapi `data` berupa
  list `complaints: [...]`, bukan `product_id` + `complaint` → kontrak.
- caption: agent meminta seller memberi product ID ("sistem saya hanya bisa
  mencari berdasarkan Product ID") alih-alih mencarinya lewat
  `get_product_performance(category="dresses")` seperti Haiku; seller awam
  tidak tahu ID, percakapan berputar dan berakhir teks (`no_tool_call`).
  Rekomendasi tool: `find_product(name)` mengurangi beban penalaran ini untuk
  model yang lebih lemah.
- Judge di run ini memakai model yang sama dengan agent (self-preference),
  jadi angka judge-nya kurang bisa dibandingkan.

## Regression test

- `tests/evaluation/test_eval_regressions.py`: tiga record asli
  (`fixtures/eval_regressions.jsonl`) — caption RCA-1, slow_movers RCA-2,
  sales_drop RCA-5 — harus tetap gagal dengan check yang sama; aturan prompt
  RCA-1..3 dan opsi `compare_to_previous` dijaga keberadaannya.
- `tests/evaluation/test_seller_eval.py`: regression RCA-4 (`run_id`).
- `tests/simulator/test_seller_store.py`: angka `compare_to_previous`
  dihitung dari kedua jendela.

## Keterbatasan

- k = 3 dan 10 task: interval kepercayaan lebar; perbaikan di atas diukur
  pada data yang sama yang dipakai untuk menemukannya (dev). Angka final
  untuk laporan harus dari split test, dijalankan sekali (`--allow-test`).
- `LLMUser` juga Haiku: variasi perilaku seller terbatas pada satu model.
- Judge belum divalidasi dengan label manusia
  (`scripts/judge_labels.py export` → labeli → `score`). Sampai itu
  dilakukan, angka judge adalah indikasi, bukan ukuran.
- Pola RCA-2 di split test (restock + kategori) berarti skor test juga
  sensitif terhadap cara agent menanyakan cakupan.
