# Goal2 Docker BERT Comparison

This run compares the current reusable Goal2 Docker artifacts:

- CrypTen Docker native baseline: `experiments/20260628_223612_docker_bert_full/summary.csv`
- MCU Docker three-role real_io run: `experiments/20260629_180301_mcu_bert_session_docker/summary.csv`
- MCU accuracy report against plaintext: `experiments/20260629_180736_mcu_bert_accuracy/summary.csv`

Main result:

- MCU Docker now completes 12-layer SST-2 BERT inference for 10 samples over real p0/p1/hp Docker TCP communication.
- MCU top-1 match with plaintext is `1.0`; mean JS is `2.49e-4`.
- MCU critical role time is `66.80s` total, or `6.68s/sample`.
- Existing CrypTen Docker native baseline is `117.80s` rank-level total, or `11.66s/sample`, with top-1 match `0.90`.
- Current MCU/CrypTen latency ratio is about `0.57x` by average per-sample latency.

Security note:

The MCU path is still an HP-clear numerical baseline for rescale, fixed/real conversion, LayerNorm, tanh, and final softmax reveal. It is suitable for Goal2 end-to-end Docker performance and numerical comparison, but it is not yet a final secure BERT implementation.
