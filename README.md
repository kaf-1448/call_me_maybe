*This project has been created as part of the 42 curriculum by ykaf*

# 💻 Codexion - Multithreaded Concurrency & Scheduler Simulation

<p align="center">
  <img src="https://img.shields.io/badge/Language-C-blue.svg" alt="Language C" />
  <img src="https://img.shields.io/badge/School-42_Network-black.svg" alt="42 Network" />
  <img src="https://img.shields.io/badge/Threads-POSIX_pthread-orange.svg" alt="POSIX Threads" />
  <img src="https://img.shields.io/badge/Norminette-Passed-brightgreen.svg" alt="Norminette" />
</p>

> **Codexion** is a high-performance multithreaded simulation addressing synchronization, concurrency, and real-time scheduling (FIFO and Earliest Deadline First) in C. It models engineers (Coders) competing for shared compilation hardware (Dongles) around a circular table.

---

##  Table of Contents
- [Project Overview](#project-overview)
- [Simulation Lifecycle](#simulation-lifecycle)
- [Concurrency & Synchronization](#concurrency--synchronization)
- [How to Run Project](#how-to-run-project)
- [Examples](#examples)
- [AI usage](#ai-usage)

---

##  Project Overview

In **Codexion**, $N$ Coders sit at a circular table with $N$ shared Dongles placed between them. To compile their project, each Coder must simultaneously acquire **two adjacent dongles** (left and right). 

### Challenges Solved:
1. **Deadlock Elimination**: Breaking circular wait dependencies using resource hierarchy ordering.
2. **Starvation Prevention**: Dynamic priority queue (Min-Heap) and fair scheduling.
3. **Data-Race Freedom**: Granular mutex protection for every critical section.
4. **Zero Busy-Waiting**: Event-driven thread suspension with POSIX Condition Variables.

---

##  Simulation Lifecycle

Each Coder runs inside a separate POSIX thread and transitions through the following states:

```
            ┌──────────────────────────────┐
            │                              │
            ▼                              │
    [ REQUEST DONGLES ]                    │
            │                              │
            ▼                              │
    [ COMPILING ] (Holding 2 Dongles)      │
            │                              │
            ▼                              │
    [ DEBUGGING ] (Dongles Released)       │
            │                              │
            ▼                              │
      [ REFACTORING ] ─────────────────────┘
```

- **Compile**: Requires both left and right dongles. Resets the burnout timer.
- **Debug**: Releases both dongles; sleeps for `time_to_debug`.
- **Refactor**: Sleeps for `time_to_refactor`; updates the next request timestamp.
- **Burnout**: If `time_to_burnout` passes without a compile, the monitor halts the entire simulation in $< 10\text{ms}$.

---

## Concurrency & Synchronization

### Deadlock Prevention (Resource Hierarchy)
Every coder compares the IDs of their left and right dongles and **always acquires the dongle with the smaller ID first**:
```c
if (coder->left_dongle->id < coder->right_dongle->id)
{
    first_dongle = coder->left_dongle;
    second_dongle = coder->right_dongle;
}
else
{
    first_dongle = coder->right_dongle;
    second_dongle = coder->left_dongle;
}
```

---

##  How to Run Project

### Compilation
```bash
make        # Compile the executable
make clean  # Remove object files
make fclean # Remove executable and object files
make re     # Clean rebuild
```

### Usage Syntax
```bash
./codexion [coders] [burnout] [compile] [debug] [refactor] [nb_compiles] [cooldown] [scheduler]
```

---

## Examples

#### 1. Standard FIFO Simulation (4 coders, all complete 5 compiles)
```bash
./codexion 4 800 200 200 200 5 0 fifo
```

#### 2. Dynamic Priority Scheduling with EDF
```bash
./codexion 4 800 200 200 200 5 0 edf
```

#### 3. Single Coder Edge Case (Burns out at 800ms)
```bash
./codexion 1 800 200 200 200 0 0 fifo
```

---


## AI usage

AI was utilized strictly as an educational and code-review assistant for:
- "Concept Deep-Dives: POSIX condition variable mechanics, spurious wakeups, and Coffman's conditions."
- "Concurrency Verification: Validating thread interaction patterns against race conditions."






