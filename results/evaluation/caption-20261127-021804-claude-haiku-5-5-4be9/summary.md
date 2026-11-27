# Caption workflow eval caption-20261127-021804-claude-haiku-5-5-4be9

agent `claude-haiku-5-5` (vision True), k=3, judge `meta/muse-glimmer-30b`, commit `f81b107`

| metric | pass@1 | pass@3 | pass^1 | pass^3 |
|---|---|---|---|---|
| **code + judge** | n/a | n/a | n/a | n/a |
| code checks only | 0.94 | 1.00 | 0.94 | 0.91 |

| scenario | pass | code | asked | failed checks | judge fails |
|---|---|---|---|---|---|
| request-plain | 0/3 | 3/3 | [0, 0, 0] |  |  |
| request-material-asks | 0/3 | 3/3 | [1, 1, 1] |  |  |
| request-material-unknown | 0/3 | 3/3 | [1, 1, 1] |  |  |
| request-material-remembered | 0/3 | 3/3 | [0, 0, 0] |  |  |
| request-unknown-remembered | 0/3 | 3/3 | [0, 0, 0] |  |  |
| request-photo-colour | 0/3 | 3/3 | [0, 0, 0] |  |  |
| request-name-tags | 0/3 | 3/3 | [0, 0, 0] |  |  |
| reject-shorter | 0/3 | 1/3 | [0, 0, 0] | short_enough |  |
| reject-care-asks | 0/3 | 3/3 | [2, 2, 2] |  |  |
| reject-no-reason | 0/3 | 3/3 | [0, 0, 0] |  |  |
| reject-photo-and-material | 0/3 | 3/3 | [2, 1, 1] |  |  |

- ask behaviour correct: 33/33
- judge: 0/0 pass, 31 errors (only on code passes)
- crashed: 0, median 3.32s per trial, avg tokens in 3300 / out 537
