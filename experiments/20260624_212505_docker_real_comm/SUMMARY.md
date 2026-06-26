# Docker real-communication summary

- Output: `F:\AI_Agent\MCU-transformer\experiments\20260624_212505_docker_real_comm`
- Rows: 18
- Median ratio mcu/crypten: 6.721x
- Min ratio: 0.734x
- Max ratio: 31.855x
- MCU timing columns split socket send/recv time from local role time.
- Torch plaintext timing is measured in the CrypTen container on rank 0.

MCU uses three role containers over TCP. CrypTen uses two rank containers over Gloo/TCP.