# GPU timing study (Tesla T4, torch 2.11.0+cu130)

Target: excess loss at most 0.0001 x the initial excess; epoch cap 60. Convergence checks are excluded from the timed region. SVRG and Adam hyper-parameters are fixed by a simple rule, not tuned.

| n | d | method | seconds | epochs | reached target | peak extra memory (MB) |
|---|---|---|---|---|---|---|
| 100000 | 256 | direct_normal_equations | 0.163 | - | yes | 1 |
| 100000 | 256 | direct_lstsq | 0.173 | - | yes | 294 |
| 100000 | 256 | svrg_jacobi | 1.911 | 3 | yes | 9 |
| 100000 | 256 | adam | 21.262 | 60 | NO | 1 |
| 200000 | 1024 | direct_normal_equations | 0.153 | - | yes | 8 |
| 200000 | 1024 | direct_lstsq | 0.724 | - | yes | 1764 |
| 200000 | 1024 | svrg_jacobi | 0.943 | 3 | yes | 2 |
| 200000 | 1024 | adam | 10.413 | 60 | NO | 2 |
| 200000 | 2048 | direct_normal_equations | 0.490 | - | yes | 32 |
| 200000 | 2048 | direct_lstsq | 1.758 | - | yes | 3337 |
| 200000 | 2048 | svrg_jacobi | 0.488 | 3 | yes | 2 |
| 200000 | 2048 | adam | 5.165 | 60 | NO | 2 |

Engine step cost in its default adaptive mode (microseconds per SVRG step, 200 steps after warm-up):

| size | telemetry on (eager) | telemetry lazy | telemetry off | eager / off |
|---|---|---|---|---|
| 100000x256 | 2004 | 1887 | 1654 | 1.2x |
| 200000x1024 | 1937 | 1943 | 1557 | 1.2x |
| 200000x2048 | 2046 | 1875 | 1614 | 1.3x |
