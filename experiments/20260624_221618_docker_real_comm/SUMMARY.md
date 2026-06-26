# Docker real-communication summary

- Output: `F:\AI_Agent\MCU-transformer\experiments\20260624_221618_docker_real_comm`
- Rows: 18
- Median ratio mcu/crypten: 3.950x
- Min ratio: 0.336x
- Max ratio: 10.344x
- MCU timing columns split socket send/recv time from local role time.
- Torch plaintext timing is measured in the CrypTen container on rank 0.

MCU uses three role containers over TCP. CrypTen uses two rank containers over Gloo/TCP.