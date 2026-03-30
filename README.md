# Multi-Robot Warehouse Path Planning & Collision Avoidance Simulator

A Python-based **multi-robot warehouse navigation simulator** that models autonomous robots operating in a fulfillment-center-style environment. The system uses **A\* path planning**, **time-step collision detection**, and **dynamic replanning** to coordinate multiple robots navigating through obstacle-dense warehouse layouts.

This project demonstrates core robotics and systems concepts including **multi-agent coordination, path planning, collision avoidance, and simulation-driven evaluation**.

---

## Project Overview

Modern automated warehouses rely on fleets of robots to transport items efficiently. These robots must:

- navigate complex warehouse layouts
- avoid collisions with other robots
- resolve path conflicts dynamically
- maintain high task completion rates

This simulator models those challenges in a simplified environment and evaluates the performance of **multi-robot navigation algorithms**.

---

## Key Features

- **Multi-Robot Coordination**  
  Simulates multiple autonomous warehouse robots navigating simultaneously.

- **A\* Path Planning**  
  Computes efficient routes through obstacle-filled warehouse environments.

- **Collision Detection**  
  Prevents robots from occupying the same cell or swapping positions.

- **Dynamic Replanning**  
  Recomputes routes when a robot is blocked.

- **Warehouse Environment Simulation**  
  Uses structured shelves and random obstacles to mimic warehouse layouts.

- **Performance Benchmarking**  
  Evaluates navigation performance across hundreds of simulation runs.

- **Real-Time Visualization**  
  Uses Pygame to animate robot movement and fleet behavior.

---

## Benchmark Results

The system was evaluated across **196 simulation runs**.

| Metric | Result |
|------|------|
| Task Success Rate | **97.45%** |
| Average Completion Steps | **40.46** |
| Average Path Conflicts | **7.45 per run** |
| Average Wait Steps | **7.45** |
| Average Dynamic Replans | **0.03 per run** |
| Average Route Computation Time | **0.30 ms** |

These results show efficient multi-robot coordination and fast route computation for real-time navigation scenarios.

---

## System Architecture

The simulator consists of four main components:

### 1. Warehouse Environment
- 30 × 30 grid representing a fulfillment center
- 100+ obstacles representing shelves and racks
- structured aisles and navigation corridors

### 2. Robot Fleet
- 3 autonomous robots
- individual start and goal locations
- independent path planning per robot

### 3. Navigation Logic
1. environment initialization
2. A\* path planning
3. collision detection
4. conflict resolution
5. dynamic replanning
6. step-by-step simulation

### 4. Visualization Engine
- Pygame-based simulation
- robot paths, starts, goals, and live movement

---

## Collision Avoidance Rules

The simulator enforces several constraints to ensure safe navigation:

- robots cannot occupy the **same grid cell** at the same time
- robots cannot **swap positions** in the same timestep
- lower-priority robots **wait** when conflicts occur
- repeated blocking can trigger **path replanning**

---

## Tech Stack

- **Python**
- **A\* Path Planning**
- **Multi-Agent Coordination**
- **Pygame**

---

## Project Structure

```text
multi-robot-warehouse-simulator/
├── multi_robot_warehouse_sim.py
└── README.md
