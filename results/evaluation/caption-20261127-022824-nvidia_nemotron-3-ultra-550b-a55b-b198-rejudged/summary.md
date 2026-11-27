# Caption workflow eval caption-20261127-022824-nvidia_nemotron-3-ultra-550b-a55b-b198-rejudged

agent `nvidia/nemotron-3-ultra-550b-a55b` (vision False), k=1, judge `claude-haiku-5-5`, commit `30a948b`

| metric | pass@1 | pass^1 |
|---|---|---|
| **code + judge** | 0.64 | 0.64 |
| code checks only | 0.82 | 0.82 |

| scenario | pass | code | asked | failed checks | judge fails |
|---|---|---|---|---|---|
| request-plain | 1/1 | 1/1 | [0] |  |  |
| request-material-asks | 0/1 | 1/1 | [1] |  | Caption claims 'potongan loose-fit', which no source states (current_caption, product_data, seller_facts and photo are s |
| request-material-unknown | 1/1 | 1/1 | [1] |  |  |
| request-material-remembered | 0/1 | 1/1 | [0] |  | Material 'soft cotton' is supported by seller_facts (katun). But the reason only mentions adding the fabric, while the c |
| request-unknown-remembered | 1/1 | 1/1 | [0] |  |  |
| request-photo-colour | 0/1 | 0/1 | [2] | ask_behaviour, includes_polka_1, includes_white |  |
| request-name-tags | 1/1 | 1/1 | [0] |  |  |
| reject-shorter | 1/1 | 1/1 | [0] |  |  |
| reject-care-asks | 1/1 | 1/1 | [1] |  |  |
| reject-no-reason | 1/1 | 1/1 | [0] |  |  |
| reject-photo-and-material | 0/1 | 0/1 | [2] | includes_white_1 |  |

- ask behaviour correct: 10/11
- judge: 7/9 pass, 0 errors (only on code passes)
- crashed: 0, median 11.9s per trial, avg tokens in 2054 / out 467
