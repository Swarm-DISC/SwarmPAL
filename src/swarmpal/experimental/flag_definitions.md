# Swarm Product Flags — Bitfield Reference

https://swarmhandbook.earth.esa.int/catalogue/sw_magx_lr_1b

Each flag field is a **bitfield**: every bit position is an independent flag, and every
"combination (sum)" value in the original documentation is just those bits added together.
So instead of memorizing hundreds of values, you only read the individual bits.

Three of the four fields (`Flags_F`, `Flags_B`, `Flags_Platform`) are clean bitfields.
`Flags_q` is **not** — it is mostly an enumerated code and is handled separately below.

**To decode a clean value:** split it into powers of two and look up each set bit.
Example: `Flags_F = 22` → `00010110` → bits 1 + 2 + 4 → outlier/gap in ASM filtering
**and** a suspicious ASM sample **and** an ASM/VFM discrepancy.

---

## Flags_F — magnetic field intensity (ASM)

| Bit | Value | Binary | Bit set (=1) means | Bit clear (=0) means |
|-----|-------|--------------|--------------------------------------------------------------|----------------------------|
| 0   | 1     | `00000001`   | ASM running in vector mode                                   | ASM nominal (scalar mode)  |
| 1   | 2     | `00000010`   | Outlier / gap / insufficient ASM freq-calibration data      | ok                         |
| 2   | 4     | `00000100`   | ≥1 of 4 nearest ASM samples suspicious                      | ok                         |
| 3   | 8     | `00001000`   | Within 8 s of ASM restart, field-lock loss, or telemetry gap | ok                        |
| 4   | 16    | `00010000`   | Discrepancy between ASM and VFM                             | ok                         |
| 5   | 32    | `00100000`   | Gap in 4 nearest ASM samples                               | ok                         |
| 6   | 64    | `01000000`   | VFM off → no stray-field corrections                       | ok                         |
| 7   | 128   | `10000000`   | (unused)                                                    | —                          |
| —   | 255   | `11111111`   | **Sentinel**: not enough ASM samples to generate F         | —                          |

---

## Flags_B — magnetic field vector (VFM)

| Bit | Value | Binary | Bit set (=1) means |
|-----|-------|--------------|-----------------------------------------------------------|
| 0   | 1     | `00000001`   | ASM instrument turned off                                 |
| 1   | 2     | `00000010`   | Outlier / gap / insufficient VFM temperature data         |
| 2   | 4     | `00000100`   | >5 suspicious VFM samples within 2 s of record           |
| 3   | 8     | `00001000`   | Discrepancy between ASM and VFM                          |
| 4   | 16    | `00010000`   | Gap in VFM samples within surrounding 2 s                |
| 5–7 | —     | —            | (unused)                                                  |
| —   | 255   | `11111111`   | **Sentinel**: not enough VFM samples                     |

> Bits 0 and 3 are never set together (if ASM is off, there is nothing to disagree with) —
> which is why the original lists 10 / 12 / 14 but never 9 / 11 / 13.

---

## Flags_Platform — platform telemetry (9 bits, overflows a byte)

| Bit | Value | Binary | Bit set (=1) means |
|-----|-------|--------------|-----------------------------------------------------------|
| 0   | 1     | `000000001`  | Thruster latch valves open, thrusters *not* activated     |
| 1   | 2     | `000000010`  | Thrusters activated                                       |
| 2   | 4     | `000000100`  | Gap in Bus telemetry (1–2 samples missing)                |
| 3   | 8     | `000001000`  | Outlier detected in Bus currents                          |
| 4   | 16    | `000010000`  | Not enough data to filter Bus currents                    |
| 5   | 32    | `000100000`  | Change in instrument state (per Bus telemetry)            |
| 6   | 64    | `001000000`  | No Bus telemetry (extended period)                        |
| 7   | 128   | `010000000`  | Gap in AOCS telemetry                                     |
| 8   | 256   | `100000000`  | Position from on-board GPSR solution *(else ground-derived orbit)* |

---

## Flags_q — attitude (STR): **not a clean bitfield**

Only **bit 3 is a true independent flag**. The rest is a structured code:
bits 4–5 select a *category*, and bits 0–2 are a small enumerated *sub-code* whose
meaning depends on that category (e.g. sub-code 3 means "CHU3", not "CHU1 + CHU2").
So these cannot simply be OR-ed together.

### The one real bit

| Bit | Value | Set means |
|-----|-------|-----------------------------------------------------------------------------|
| 3   | 8     | On-ground aberrational correction applied to ≥1 of the nearest attitude samples |

### Category — bits 5,4

| Bits 5,4 | Base | Category |
|----------|------|--------------------------------------------|
| `00`     | 0    | Partial CHU attitude lack                  |
| `01`     | 16   | Obscuration / multi-CHU degradation        |
| `10`     | 32   | Attitude from a single CHU (1–2 samples)   |
| `11`     | 48   | Missing samples / single CHU (3–4 samples) |

### Sub-code — bits 2,1,0, read *within* the category

| Sub | Cat 0 (partial lack) | Cat 16 (obscuration/multi) | Cat 32 (single CHU, 1–2) | Cat 48 (missing / single 3–4) |
|-----|----------------------|----------------------------|--------------------------|-------------------------------|
| 1   | CHU1 lacks 1–2       | CHU1 obscured              | CHU1 alone               | 1 sample missing              |
| 2   | CHU2 lacks 1–2       | CHU2 obscured              | CHU2 alone               | 2 samples missing             |
| 3   | CHU3 lacks 1–2       | CHU3 obscured              | CHU3 alone               | 3+ samples missing            |
| 4   | CHU1 lacks 3–4       | CHU1 & CHU2 lack           | intermittent single CHU (2 att.) | CHU1 alone (3–4)      |
| 5   | CHU2 lacks 3–4       | CHU1 & CHU3 lack           | —                        | CHU2 alone (3–4)              |
| 6   | CHU3 lacks 3–4       | CHU2 & CHU3 lack           | —                        | CHU3 alone (3–4)              |
| 7   | —                    | all three lack             | —                        | intermittent single CHU (3–4) |

Add **8** to any value for the aberrational-corrected version. `255` is the
"not enough STR data" sentinel.

### To decode a Flags_q value `v`

```
aberration = v & 8      # correction applied?
category   = v & 48     # 0, 16, 32, or 48
sub-code   = v & 7      # 1–7, look up in the matrix under the category
```