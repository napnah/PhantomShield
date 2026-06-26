# Docker real-communication summary

- Output: `F:\AI_Agent\MCU-transformer\experiments\20260624_154028_docker_real_comm`
- Rows: 18
- Median ratio mcu/crypten: 7.481x
- Min ratio: 0.728x
- Max ratio: 34.005x
- MCU timing columns split socket send/recv time from local role time.
- Torch plaintext timing is measured in the CrypTen container on rank 0.

MCU uses three role containers over TCP. CrypTen uses two rank containers over Gloo/TCP.