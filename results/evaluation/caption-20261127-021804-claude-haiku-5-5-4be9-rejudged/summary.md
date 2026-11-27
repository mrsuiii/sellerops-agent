# Caption workflow eval caption-20261127-021804-claude-haiku-5-5-4be9-rejudged

agent `claude-haiku-5-5` (vision True), k=3, judge `nvidia/nemotron-3-ultra-550b-a55b`, commit `30a948b`

| metric | pass@1 | pass@3 | pass^1 | pass^3 |
|---|---|---|---|---|
| **code + judge** | 0.91 | 1.00 | 0.91 | 0.73 |
| code checks only | 1.00 | 1.00 | 1.00 | 1.00 |

| scenario | pass | code | asked | failed checks | judge fails |
|---|---|---|---|---|---|
| request-plain | 2/3 | 3/3 | [0, 0, 0] |  | Reason falsely states 'suitable for casual and semi-formal' had no source and was deleted; current_caption supports it a |
| request-material-asks | 3/3 | 3/3 | [1, 1, 1] |  |  |
| request-material-unknown | 3/3 | 3/3 | [1, 1, 1] |  |  |
| request-material-remembered | 3/3 | 3/3 | [0, 0, 0] |  |  |
| request-unknown-remembered | 3/3 | 3/3 | [0, 0, 0] |  |  |
| request-photo-colour | 2/3 | 3/3 | [0, 0, 0] |  | Caption states 'tepi atas dan bawah rok bermotif lengkung' (skirt top and bottom edges scalloped), but photo only shows  |
| request-name-tags | 3/3 | 3/3 | [0, 0, 0] |  |  |
| reject-shorter | 3/3 | 3/3 | [0, 0, 0] |  |  |
| reject-care-asks | 2/3 | 3/3 | [2, 2, 2] |  | Caption claims 'potongan klasik' (classic cut) but no source (current_caption, product_data, seller_facts, photo) mentio |
| reject-no-reason | 3/3 | 3/3 | [0, 0, 0] |  |  |
| reject-photo-and-material | 3/3 | 3/3 | [2, 1, 1] |  |  |

- ask behaviour correct: 33/33
- judge: 30/33 pass, 0 errors (only on code passes)
- crashed: 0, median 3.32s per trial, avg tokens in 3300 / out 537
