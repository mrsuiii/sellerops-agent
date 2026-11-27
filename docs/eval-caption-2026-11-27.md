# Eval workflow caption (C3/C3b) — 27 November 2026

Graph approval caption dijalankan dengan model asli pada 11 skenario. Setiap
skenario adalah satu sesi seller di database baru, dengan seller scripted
yang menjawab pertanyaan agent berdasarkan kata kunci. Skenario, cek, dan
rubric judge ada di `src/sellerops/evaluation/caption_eval.py`; perintahnya
`scripts/evaluate_caption.py`.

## Hasil

| Agent | k | Judge | cek kode pass@1 | kode + judge pass@1 | pass^k | bertanya dengan benar |
|---|---|---|---|---|---|---|
| **Claude Haiku 5.5** (vision) | 3 | Nemotron 3 Ultra | **1.00** | **0.91** | pass^3 = **0.73** | 33/33 |
| Nemotron 3 Ultra (tanpa vision) | 1 | Claude Haiku 5.5 | 0.82 | 0.64 | — | 10/11 |

- Haiku: median 3,3 detik per sesi, rata-rata 3.300 token input / 537 output.
- Nemotron: median 11,9 detik per sesi.
- Judge berasal dari keluarga model lain dari agent (*cross-judge*), untuk
  menghindari self-preference. Judge hanya menilai percobaan yang lolos cek
  kode.

Run:
- Haiku: `results/evaluation/caption-20261127-021804-claude-haiku-5-5-4be9-rejudged`
  (agent dari run `…-4be9`, cek dan judge dijalankan ulang).
- Nemotron: `results/evaluation/caption-20261127-022824-nvidia_nemotron-3-ultra-550b-a55b-b198-rejudged`.

## Yang diuji

| Kelompok | Skenario | Dicek |
|---|---|---|
| Draft pertama | `request-plain`, `request-name-tags` | tidak bertanya tanpa alasan, hanya field yang diminta |
| Bertanya ke seller | `request-material-asks`, `reject-care-asks` | bertanya, lalu memakai jawabannya ("linen", "air dingin") |
| Seller tidak tahu | `request-material-unknown` | tidak mengarang bahan, `reason` menyebut data kurang |
| Memori (`product_facts`) | `request-material-remembered`, `request-unknown-remembered` | tidak bertanya ulang; fakta lama dipakai, "tidak tahu" dihormati |
| Foto | `request-photo-colour`, `reject-photo-and-material` | warna/motif dari foto (putih, polkadot) |
| Revisi | `reject-shorter`, `reject-no-reason` | ≤ 2 kalimat; tanpa alasan → versi yang jelas berbeda |

Judge (rubric `caption-v2`) menilai tiga hal: `grounded` (setiap klaim ada di
caption sekarang, data produk, fakta seller, atau deskripsi foto),
`does_request`, dan `honest_reason`.

## Temuan

**Haiku**
- Perilaku bertanya sempurna (33/33): bertanya hanya ketika fakta tidak ada di
  sumber mana pun, tidak pernah bertanya ulang untuk fakta yang sudah diingat,
  termasuk yang dijawab "tidak tahu".
- Tiga kegagalan judge, semuanya kecil: lokasi tepi bergelombang disebut "rok
  atas dan bawah" (di foto: garis dada dan ujung rok), "potongan klasik" tanpa
  sumber, dan `reason` yang keliru menyebut ada klaim yang dihapus.
- Haiku menulis 27 dari 33 caption dalam Bahasa Indonesia walaupun caption
  sekarang berbahasa Inggris (mengikuti bahasa instruksi seller, bertentangan
  dengan aturan prompt "bahasa caption sekarang"). Belum dicek oleh eval;
  perlu keputusan produk.

**Nemotron**
- Tanpa vision, jadi dua skenario foto gagal seperti yang diharapkan. Di
  `request-photo-colour` model bertanya soal warna ke seller alih-alih
  mengarang; itu perilaku yang aman, tapi skenario mengharapkan jawaban dari
  foto.
- Dua kegagalan grounding: "loose-fit" tanpa sumber, dan `reason` yang tidak
  menyebut semua perubahan.
- Tidak ada crash: `tool_choice="any"` (→ `required`) berjalan di endpoint
  NVIDIA.

## Bug eval yang ditemukan dan diperbaiki (bukan bug agent)

Ditemukan lewat error analysis pada run pertama, lalu cek dan judge dijalankan
ulang (`--rejudge`) tanpa memanggil agent lagi:

1. `LLM_MODEL` di `.env` menunjuk model yang menolak permintaan, sehingga
   semua panggilan judge gagal (`APIStatusError`). Sekarang model judge bisa
   dipilih dengan `--judge-model`.
2. Pemecah kalimat menghitung titik di "Rp655.000" sebagai akhir kalimat.
3. Judge tidak menerima caption yang sedang tampil, sehingga nama produk
   ("Blue & Black Check Shirt", "pea pattern") dianggap karangan.
4. Deskripsi foto tertulis kurang lengkap (tepi bergelombang merah tidak
   disebut). Ditulis ulang dari foto 1000 px yang dilihat model.
5. Di run langsung, judge membaca fakta seller dari database trial, padahal
   graph tidak menyimpan jawaban (yang menyimpan adalah runner API).
   Sekarang memakai jawaban yang tercatat.

## Batasan

- Judge **belum dikalibrasi** terhadap label manusia (ditunda, keputusan
  18 Nov). Angka kode + judge dibaca sebagai sinyal.
- 11 skenario, dua produk, seller scripted: cukup untuk regresi dan demo,
  belum untuk klaim umum.
- Untuk Nemotron hanya k=1, jadi pass^k tidak dihitung.
