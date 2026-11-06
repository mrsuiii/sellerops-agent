# Step-by-Step Plan — SellerOps Agent

Terakhir diperbarui: 19 Agustus 2026  
Dokumen ini adalah panduan eksekusi untuk sesi Codex berikutnya. Sasaran dan
scope proyek berada di `../../TARGET.md`; dokumen ini mengatur **cara
mengerjakannya secara bertahap**.

## Kontrak belajar untuk Codex

Arahan user 13 September 2026: bagian agentic (model adapter/node, graph/state,
routing, retry/replan, dan agent tool orchestration) dikerjakan user sendiri dari
nol. Codex tidak membuat scaffold atau TODO untuk bagian tersebut; cukup
menjelaskan konsep atau mereview saat diminta. Implementasi fondasi episode,
snapshot/replay, outcome verifier, dan test pendukung telah diizinkan penuh.

Pada setiap sesi berikutnya, Codex harus:

1. Membaca `../../AGENTS.md`, `../../TARGET.md`, `PLAN.md`,
   `../../environment.yml`, dan bagian terbaru `../../LEARNING_LOG.md` sebelum
   mengubah proyek.
2. Memeriksa file dan hasil yang sudah ada, lalu melanjutkan dari milestone
   pertama yang belum selesai. Jangan mengulang pekerjaan yang sudah lolos
   test.
3. Mengerjakan hanya **satu session milestone** dalam satu waktu, kecuali user
   secara eksplisit meminta lebih.
4. Menjelaskan tujuan, teori minimum, bentuk input/output, dan definition of
   done sebelum mulai menulis kode.
5. Membuat scaffold, fixture, boilerplate, dan test yang membantu, tetapi
   menyisakan bagian inti sebagai `TODO` atau `FILL_ME` untuk dikerjakan user.
6. Memberikan maksimal 3–5 TODO inti per sesi agar latihan tetap fokus.
7. Tidak langsung mengisi TODO milik user. Review percobaan user terlebih
   dahulu, tunjukkan error atau invariant yang gagal, lalu beri hint bertahap.
   Isi solusi lengkap hanya jika user memintanya secara eksplisit.
8. Menjalankan test yang relevan setelah user/scaffold mengubah kode dan
   menjelaskan hubungan failure test dengan konsep yang dipelajari.
9. Menandai checkbox sesi selesai hanya setelah definition of done benar-benar
   terverifikasi.
10. Mengakhiri sesi dengan ringkasan singkat: yang selesai, bukti test/metric,
    TODO user berikutnya, dan session milestone selanjutnya.

Codex boleh mengerjakan penuh bagian mekanis seperti konfigurasi, serialization,
CLI plumbing, test fixtures, dan visualisasi. User sebisa mungkin mengerjakan
bagian yang membawa learning value, khususnya:

- schema validation dan decision rules;
- oracle/tool-policy logic;
- formatting dan masking dataset SFT;
- LangGraph routing serta retry/replan conditions;
- trajectory-to-preference conversion;
- DPO objective dan interpretasi metric.

Jika user sedang mengejar deadline dan meminta implementasi penuh, instruksi
terbaru user mengalahkan kontrak belajar ini.

Progress 5 Oktober 2026: atas permintaan eksplisit user, bounded schema retry
untuk router/action dan evaluasi repeated episodes dengan pass@1/pass@k sudah
diimplementasikan. Ini belum mencakup replan atau verifier di dalam runtime.
Detail perilaku, perintah, dan batas penilaian berada di
`src/sellerops/evaluation/README.md`; bukti pengujian ada di `../../LEARNING_LOG.md`.

## Aturan environment dan data

- Selalu gunakan Conda environment `llm-post-training`.
- Jangan install, remove, atau upgrade dependency sebelum menjelaskan
  perubahan.
- Catat dependency di `../../environment.yml` dan perubahan penting di
  `../../LEARNING_LOG.md`.
- Audit ulang OS, Conda environment, dan MPS sebelum memilih package/model yang
  memengaruhi acceleration.
- Jangan gunakan frozen test tasks untuk SFT, DPO-pair generation, prompt
  tuning, atau failure-specific patching.
- Simpan `task_id`, seed, split, environment version, dan provenance.
- Jangan menyimpan hidden chain-of-thought dalam trajectory. Simpan action,
  observation, evaluator result, dan alasan singkat yang diperlukan untuk
  debugging.

## Pipeline final

```text
schemas + deterministic simulator
    → frozen evaluation split
    → isolated episode state + deterministic replay + verifier
    → base-model evaluation
    → LoRA SFT
    → LangGraph runtime v1 dengan model SFT
    → collect and replay trajectories
    → failure attribution
    → [optional] targeted preference pairs + DPO dari checkpoint SFT
    → base vs SFT (+ DPO jika dijalankan) held-out evaluation
```

DPO bersifat conditional dan hanya dilakukan **setelah runtime v1 menghasilkan
preference failures yang cukup**, bukan langsung setelah SFT. Runtime/tool bugs
harus diperbaiki pada komponennya, bukan diubah menjadi DPO pairs.

## Struktur proyek yang dituju

Struktur boleh tumbuh bertahap; jangan membuat semua file kosong sekaligus.

```text
src/sellerops/
├── schemas.py
├── environment.py
├── services.py
├── scenarios.py
├── faults.py
├── evaluation/
│   ├── dataset.py
│   ├── target.py
│   ├── evaluators.py
│   ├── metrics.py
│   ├── answer_keys.py
│   ├── judge.py
│   ├── telemetry.py
│   ├── runner.py
│   └── verifier.py
├── model_adapter.py
├── graph.py
├── memory.py
└── trajectory.py

data/
├── environment/
├── tasks/
├── sft/
├── trajectories/
└── preferences/

scripts/
├── generate_sellerops_dataset.py
├── evaluate_agent.py
├── train_sft.py
└── train_dpo.py

notebooks/
└── sft.ipynb

tests/
configs/
results/
```

## Session 1 — Data contracts dan action schema

Status: [x]

Tujuan: menentukan bahasa bersama antara model, runtime, tools, dan evaluator
sebelum menghasilkan dataset.

Kerjakan:

- [x] Definisikan schema `SellerTask`, `ToolCall`, `AskUser`, `FinalAnswer`,
  `ToolObservation`, dan `EvaluationResult`.
- [x] Gunakan satu tagged action format: `tool_call`, `ask_user`, atau `final`.
- [x] Tentukan field wajib, batas retry/step, dan aturan JSON serialization.
- [x] Buat test valid/invalid untuk setiap action type.
- [x] Dokumentasikan tiga seller-task families dan contoh masing-masing.

Bagian inti untuk user:

- Melengkapi validator yang memastikan setiap action hanya membawa field yang
  sesuai dengan `type`-nya.
- Menjelaskan mengapa schema invalid harus ditolak sebelum tool dieksekusi.

Definition of done:

- Semua schema tests lolos.
- Tiga task family dan structured action dapat dijelaskan tanpa bergantung pada
  framework/model tertentu.

## Session 2 — Deterministic environment dan tools

Status: [x]

Tujuan: membuat dunia kecil yang menjadi source of truth untuk SFT, replay, dan
evaluation.

Kerjakan:

- [x] Buat catalog, inventory, dan discount-policy fixtures.
- [x] Implementasikan `search_products`, `get_inventory`,
  `get_discount_policy`, dan `calculate_offer`.
- [x] Tambahkan structured success/error result.
- [x] Tambahkan skenario stock habis, produk ambigu, dan invalid discount.
- [x] Test bahwa tool tidak mengubah state secara tidak sengaja.

Bagian inti untuk user:

- Melengkapi policy validation pada `calculate_offer`.
- Menentukan expected behavior untuk missing product dan unavailable stock.

Definition of done:

- Tools deterministic dan seluruh business-rule tests lolos.
- Dynamic inventory/policy selalu diperoleh dari tool, bukan dihafal model.

Deterministic transient fault injection dikerjakan setelah isolated episode
lifecycle tersedia agar fault dapat dikaitkan dengan session, call number, dan
seed secara reproducible.

## Session 3 — Oracle, task generator, dan frozen split

Status: [x]

Tujuan: membuat seed demonstrations serta evaluation set tanpa data production.

Kerjakan:

- [x] Tulis 20–30 canonical scenario templates.
- [x] Buat deterministic oracle yang menghasilkan expected action/tool
  sequence dan verifiable constraints.
- [x] Generate 150–250 training tasks, sekitar 30 development tasks, dan 40–50
  frozen test tasks.
- [x] Split berdasarkan scenario/template family, bukan random row saja.
- [x] Simpan manifest, seed, environment version, dan checksum frozen test set.

Bagian inti untuk user:

- Melengkapi minimal tiga oracle decision branches.
- Memeriksa manual sampel dari setiap split dan menemukan potensi leakage.

Definition of done:

- Task generation reproducible.
- Frozen test manifest tidak digunakan oleh pipeline training.
- Oracle trajectories lolos schema validation dan frozen split tetap
  terisolasi.

## Session 4 — Episode simulator, state isolation, dan replay

Status: [ ]

Tujuan: mengembangkan environment read-only menjadi service simulator berbasis
episode yang dapat dipakai identik oleh base, SFT, dan agent runtime.

Kerjakan:

- [ ] Definisikan `EpisodeState` dengan `session_id`, `scenario_id`, seed,
  isolated inventory/policy snapshot, tool history, step count, dan terminal
  flag.
- [ ] Implementasikan `reset(scenario_id, seed)` tanpa shared-state leakage.
- [x] Implementasikan `execute_tool(session_id, tool_call)` dengan structured
  observation dan step limit.
- [x] Implementasikan snapshot serta deterministic replay untuk tool history
  yang sama.
- [x] Tambahkan outcome verifier yang menerima beberapa tool sequence valid dan
  memeriksa product, stock, discount-policy, serta final-answer constraints.

Progres 13 September 2026: service mendukung `create_episode(task_id)`,
`reset(session_id)` ke snapshot awal, snapshot JSON, dan replay tool history pada
episode sementara. Replay melaporkan mismatch pertama dan tidak mengubah input
atau episode asli. Outcome verifier tersedia melalui `verify_outcome(task,
snapshot, final_action)`, memeriksa data terstruktur dan bukti tool untuk ketiga
task family tanpa mewajibkan satu urutan oracle. Integrasi `scenario_id`/seed
belum selesai, sehingga Session 4 tetap terbuka. Wording bebas tidak dinilai
secara semantik; audit train/dev menemukan satu label proposal diskon negatif
yang keliru pada oracle lama, dan dataset tidak diubah.

Bagian inti untuk user:

- Melengkapi state-isolation dan replay invariant.
- Menentukan outcome constraints tanpa memaksa satu exact oracle trajectory.

Definition of done:

- Dua episode tidak saling mengubah state.
- Scenario dan seed yang sama menghasilkan observation history yang sama.
- Verifier menerima dua urutan tool berbeda jika outcome keduanya valid.

## Session 5 — Base-model evaluation dan model selection

Status: [ ]

Tujuan: memperoleh baseline sebelum SFT.

Kerjakan:

- [ ] Audit kembali MPS/memory dan dependency environment.
- [ ] Pilih satu instruct model open-weight sekitar 0.5B–1.5B.
- [ ] Buat `ModelAdapter` yang menghasilkan tagged structured action.
- [ ] Jalankan base model pada development dan frozen baseline tasks.
- [ ] Ukur schema validity, tool-choice accuracy, dan exact argument accuracy.

Bagian inti untuk user:

- Melengkapi parser/validator output model.
- Menganalisis minimal lima base-model failures.

Definition of done:

- Baseline tersimpan dengan model revision, config, seed, dan metrics.
- Frozen baseline tidak ditimpa oleh eksperimen berikutnya.

## Session 6 — SFT dataset dan preprocessing

Status: [ ]

Tujuan: mengubah oracle demonstrations menjadi chat/tool-use training records.

Kerjakan:

- [x] Serialisasikan task, assistant action, tool observation, dan final answer
  ke chat records model-independent; model-specific chat template diterapkan
  setelah model dipilih.
- [x] Target sekitar 400–700 assistant decision points dari training tasks.
- [ ] Mask loss agar hanya token assistant target yang dilatih.
- [x] Tambahkan schema/provenance validation untuk setiap record.
- [ ] Visual-inspect tokenization dan label mask pada beberapa contoh.

Bagian inti untuk user:

- Melengkapi fungsi formatting satu trajectory.
- Melengkapi atau memverifikasi assistant-only label masking.

Definition of done:

- Tidak ada frozen test task di SFT records.
- Contoh yang telah di-decode kembali memiliki prompt dan target yang benar.

## Session 7 — LoRA SFT smoke test dan training

Status: [ ]

Tujuan: menghasilkan adapter SFT pertama secara reproducible.

Kerjakan:

- [ ] Buat config kecil untuk smoke test.
- [ ] Verifikasi trainable parameter count dan LoRA target modules.
- [ ] Jalankan smoke test pada beberapa batch sebelum full small run.
- [ ] Log train/eval loss, schema validity, dan tool-choice accuracy.
- [ ] Bandingkan base vs SFT pada frozen evaluation.

Bagian inti untuk user:

- Membaca satu batch, loss, dan gradient sanity check.
- Menjelaskan apakah improvement berasal dari format adherence, decision
  quality, atau keduanya.

Definition of done:

- Adapter, config, dataset version, dan result table tersimpan.
- SFT dinilai dengan task metrics, bukan training loss saja.

## Session 8 — LangGraph state dan graph skeleton

Status: [ ]

Tujuan: membangun runtime yang dimiliki sendiri, bukan sekadar API wrapper.

Kerjakan:

- [ ] Definisikan `AgentState` dengan goal, plan, observations, tool calls,
  evaluation, retry count, dan final answer.
- [ ] Buat node `load_context`, `plan`, `execute_tool`, `evaluate`, dan `final`.
- [ ] Buat conditional edges untuk continue, retry, replan, dan stop.
- [ ] Gunakan SFT adapter dari Session 7 sebagai policy model awal.
- [ ] Tambahkan maximum step dan retry protection.

Bagian inti untuk user:

- Melengkapi routing function berdasarkan state/evaluation.
- Menjelaskan kapan retry berbeda dari replan.

Definition of done:

- Satu happy-path task berjalan end-to-end.
- Graph berhenti dengan aman pada invalid output atau repeated failure.

## Session 9 — Evaluator dan self-correction

Status: [ ]

Tujuan: menambahkan verifikasi serta perbaikan dalam satu episode.

Kerjakan:

- [ ] Implementasikan deterministic verifier untuk schema, product ID, stok,
  diskon, dan tool arguments.
- [ ] Evaluator mengembalikan `pass`, `retry`, atau `replan` beserta reason
  code.
- [ ] Tambahkan failure observations kembali ke state tanpa hidden
  chain-of-thought.
- [ ] Test loop termination dan policy-violation prevention.

Bagian inti untuk user:

- Melengkapi mapping failure reason ke retry/replan.
- Menulis minimal dua adversarial/edge-case tests.

Definition of done:

- Agent memperbaiki minimal satu recoverable failure.
- Unrecoverable failure berhenti secara eksplisit, bukan loop tanpa akhir.

## Session 10 — Context, memory, trajectory, dan replay

Status: [ ]

Tujuan: membuat multi-turn runtime dapat di-debug dan direplay.

Kerjakan:

- [ ] Tambahkan short-term context budget dan summary sederhana.
- [ ] Simpan structured seller preferences di SQLite.
- [ ] Implementasikan retrieve-then-load berdasarkan task.
- [ ] Log trajectory JSONL dengan `task_id`, seed, model revision, action,
  observation, evaluator result, latency, dan steps.
- [ ] Replay trajectory pada environment version yang sama.

Bagian inti untuk user:

- Melengkapi aturan memory relevance atau compaction.
- Memverifikasi bahwa replay menghasilkan environment observations yang sama.

Definition of done:

- Satu multi-turn task menggunakan memory yang relevan.
- Satu failed trajectory berhasil direproduksi.

## Session 11 — Trajectory collection dan failure analysis

Status: [ ]

Tujuan: menghasilkan data nyata dari perilaku agent SFT.

Kerjakan:

- [ ] Jalankan 3–4 rollouts per development task dengan seed tercatat.
- [ ] Kelompokkan failure: planner, tool choice, arguments, context, policy,
  final answer, atau runtime.
- [ ] Pisahkan model failure dari environment/runtime failure.
- [ ] Buat RCA report dan tambahkan failure penting sebagai regression task.

Bagian inti untuk user:

- Mengaudit failure clusters dan memperbaiki attribution yang salah.
- Memilih failure yang layak menjadi DPO pair versus runtime fix.

Definition of done:

- Setiap candidate DPO example memiliki traceable source trajectory.
- Runtime/tool bugs tidak salah dimasukkan sebagai preference-model problem.

## Session 12 — Conditional preference-pair builder

Status: [ ]

Tujuan: hanya jika failure audit mendukung, membentuk DPO data yang bersih dari
first divergence point.

Kerjakan:

- [ ] Gunakan prompt/history yang identik untuk `chosen` dan `rejected`.
- [ ] Ambil `rejected` dari realistic SFT-agent failure.
- [ ] Ambil `chosen` dari successful rollout, verifier/human repair, atau
  audited oracle output.
- [ ] Lanjutkan hanya jika tersedia pasangan berkualitas yang cukup; target
  100–250 adalah panduan, bukan kewajiban.
- [ ] Audit length bias, duplicate pairs, ambiguous preference, dan leakage.

Bagian inti untuk user:

- Melengkapi first-divergence extraction.
- Mengaudit manual minimal 20–30% preference pairs.

Definition of done:

- Pair yang keduanya salah atau tidak jelas sudah dibuang.
- Frozen test tasks tidak ada dalam preference dataset.

## Session 13 — Conditional DPO objective dan training

Status: [ ]

Tujuan: jika Session 12 menghasilkan dataset yang layak, melatih preference
behavior dari checkpoint SFT.

Kerjakan:

- [ ] Implementasikan DPO loss pada dummy tensors dan unit test-nya.
- [ ] Jelaskan policy/reference log-ratio dan peran beta.
- [ ] Jalankan DPO smoke test dari SFT adapter/checkpoint.
- [ ] Jalankan small DPO training dengan config reproducible.
- [ ] Periksa chosen/rejected margin, loss, schema validity, dan tool behavior.

Bagian inti untuk user:

- Melengkapi perhitungan DPO logits/loss pada latihan kecil.
- Menginterpretasikan minimal satu contoh ketika loss membaik tetapi behavior
  tidak membaik.

Definition of done:

- DPO adapter dan seluruh metadata tersimpan.
- Tidak ada klaim improvement sebelum frozen replay evaluation.

## Session 14 — Final replay, report, dan application package

Status: [ ]

Tujuan: menghasilkan bukti akhir yang dapat ditunjukkan saat melamar.

Kerjakan:

- [ ] Replay frozen tasks dengan base dan SFT, serta DPO jika tersedia, pada
  graph yang sama.
- [ ] Bandingkan task success, valid-tool-call rate, policy violations, schema
  validity, steps, retry count, dan latency.
- [ ] Analisis minimal lima failure cases dan limitations.
- [ ] Tambahkan diagram, setup, demo command, evaluation command, serta result
  table ke README.
- [ ] Siapkan penjelasan proyek versi 3 menit dan 15 menit.
- [ ] Isi resume bullet hanya dengan angka yang telah diverifikasi.

Definition of done:

- Repository dapat dijalankan ulang dari environment/config yang tercatat.
- Candidate hanya dipromosikan jika melewati regression dan safety gates.
- README membedakan fitur yang sudah selesai dari future work.

## Current next session

Mulai dari **Session 4 — Episode simulator, state isolation, dan replay**.

Codex pada sesi berikutnya jangan langsung membuat model adapter, SFT pipeline,
LangGraph runtime, dan DPO trainer sekaligus. Audit environment yang sudah ada,
buat scaffold serta tests untuk isolated episode lifecycle, lalu berikan 3–5
TODO inti kepada user dan berhenti pada batas Session 4.
