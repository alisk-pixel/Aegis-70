# AEGIS-70: Autonomous Edge Guardian & Behavioral Supervision System

> **A private, always-on accountability layer that notices when a night is quietly being lost, and intervenes before it is.**  
> Built solo for the **Nebius x NVIDIA Global AI Hackathon (Personal AI Track)**.

`pure-python edge math` + `Nebius Token Factory` + `nvidia/Llama-3.1-Nemotron-70B-Instruct` + `zero-command sensing` + `runs on any standard PC (No GPU required)`

---

## The Story

I'm 25. I build software by myself, and most of my real work happens at night, in one room, at one desk.

That is also where the loop lives. It starts as "just five more minutes." Then it's 11:45 PM, the screen is still on, nothing is being built, and I haven't moved in forty-five minutes. The room gets smaller and the night gets shorter. The next morning I pay for it, and then I do it again.

Productivity apps never helped, because they all wait for me to open them. The person stuck in the loop is the one least likely to type a command. Willpower fails at exactly the moment you need it, so I stopped trying to out-willpower the loop and built something that could see it.

**AEGIS-70** is a Personal AI in the literal sense. It runs locally in my room, on my own computer, around my habits. It watches presence rather than content: where I am, how long I've been still, and what hour it is. When the pattern says "drift," a calm, JARVIS-style sovereign voice tells me one small physical thing to do. It is never shaming and never a lecture.

While teams of professionals build enterprise middleware, I'm using open infrastructure to do something smaller and more personal: **reclaim my own nights, my discipline, and my craft.** A lot of people carry some version of this struggle, which is why I built it well.

---

## What It Does

1. **Senses with zero commands.** Simulates ambient mmWave radar for presence and micro-motion energy, paired with an on-device pose classifier (boolean *engaged or not*). No private video or audio frames ever leave the device.
2. **Learns locally without a GPU.** A heuristic Bayesian Markov chain learns your transition habits per time window (daytime, evening, late night) using pure Python on standard CPU hardware.
3. **Scores risk mathematically.** A Behavioral Risk Score from 0 to 100 combines room, hour, idleness, isolation, and the chain's forecast of whether you are trapped in a doomloop.
4. **Thinks big only when it matters.** At ORANGE or RED risk, after sustained idleness, an abstract report goes to **nvidia/Llama-3.1-Nemotron-70B-Instruct on Nebius Token Factory**, which returns a structured, empathetic intervention.
5. **Acts through physical skills.** The model selects from a closed set of locally executed skills (dim the workstation, lock the screen, light the hallway, start a breathing reset, draft a text to a friend).
6. **Remembers.** Persistent on-device state tracks learned habits, relapse streaks, interventions, and outcomes.

---

## Architecture

```text
  mmWave radar (LD2410/LD2450)        Local camera / edge classifier
        │ presence, motion energy              │ engaged? (boolean only)
        └───────────────┬──────────────────────┘
                        ▼
              ┌───────────────────┐
              │   SensorFusion    │  dropouts · ghost rejection · confidence
              └─────────┬─────────┘
                        ▼   FusedObservation
              ┌───────────────────┐        ┌──────────────────────┐
              │  BehaviorEngine   │◄──────►│  StateStore (local)  │
              │  Markov + risk    │        │  state.json · ledger │
              └─────────┬─────────┘        └──────────────────────┘
                        │ tier ≥ ORANGE and idle ≥ 40 min
                        ▼
              ┌───────────────────┐  allowlisted abstract payload only
              │  Privacy guard    │──────────────────────────────┐
              └───────────────────┘                              ▼
                                            Nebius Token Factory (serverless)
                                            nvidia/Llama-3.1-Nemotron-70B-Instruct
                        ┌────────────────────────────────────────┘
                        ▼   strict JSON plan -> validated
              ┌───────────────────┐
              │ Skill executor    │──► lights · display · chime · notify · timer
              └─────────┬─────────┘
                        ▼
                AEGIS intervention protocol