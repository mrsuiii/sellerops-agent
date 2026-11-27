# Caption workflow eval caption-20261127-022824-nvidia_nemotron-3-ultra-550b-a55b-b198

agent `nvidia/nemotron-3-ultra-550b-a55b` (vision False), k=1, judge `claude-haiku-5-5`, commit `30a948b`

| metric | pass@1 | pass^1 |
|---|---|---|
| **code + judge** | 0.55 | 0.55 |
| code checks only | 0.82 | 0.82 |

| scenario | pass | code | asked | failed checks | judge fails |
|---|---|---|---|---|---|
| request-plain | 1/1 | 1/1 | [0] |  |  |
| request-material-asks | 0/1 | 1/1 | [1] |  | Bahan 'kain linen' dan sifat 'breathable' tidak ada di current_caption, product_data, seller_facts (kosong) maupun photo |
| request-material-unknown | 1/1 | 1/1 | [1] |  |  |
| request-material-remembered | 0/1 | 1/1 | [0] |  | Caption adds 'soft cotton fabric', which matches seller_facts material 'katun'. The 2-year warranty, overnight shipping  |
| request-unknown-remembered | 1/1 | 1/1 | [0] |  |  |
| request-photo-colour | 0/1 | 0/1 | [2] | ask_behaviour, includes_polka_1, includes_white |  |
| request-name-tags | 1/1 | 1/1 | [0] |  |  |
| reject-shorter | 1/1 | 1/1 | [0] |  |  |
| reject-care-asks | 0/1 | 1/1 | [1] |  | The caption adds 'Care: Wash with cold water, do not wring.' but no source holds care instructions (seller_facts is empt |
| reject-no-reason | 1/1 | 1/1 | [0] |  |  |
| reject-photo-and-material | 0/1 | 0/1 | [2] | includes_white_1 |  |

- ask behaviour correct: 10/11
- judge: 6/9 pass, 0 errors (only on code passes)
- crashed: 0, median 11.9s per trial, avg tokens in 2054 / out 467
