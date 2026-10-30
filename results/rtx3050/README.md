# Measured outcome and memory

Evidence comes from saved local reports. Display checkpoint429526 completes616 steps/12.32s without failure or mid-clip reset. Final checkpoint452376 retains learning state and is not separately claimed to be the best.

| Memory scope | Recorded peak |
|---|---:|
| PyTorch allocated, final training |2181.42 MiB|
| PyTorch reserved, final training |2386 MiB|
| Total GPU sampled during final training |3727 MiB|
| Total GPU sampled during selected429526 evaluation |2958 MiB|

Under3GB describes the PyTorch allocator or that recorded evaluation; it does not describe the whole-card maximum throughout training. Total GPU accounting includes Warp, graphics and other processes; sampled peaks may miss instantaneous spikes.

This is pretrained fine-tuning: public291000 -> final452376, not452376 newly trained iterations from random weights. Strict stage335263->452376 took51733.99seconds. Selected metrics and thresholds are in adjacent JSON files. User visually accepted the demo; strict heading/tilt targets and final3-seed robustness were not all passed.
