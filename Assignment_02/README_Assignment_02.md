# TTL Assignment 02 — Genetic Algorithm for a Lane-Driving Agent

## Overview
This notebook uses a **Genetic Algorithm (GA)** to evolve a simple rule-based driving agent that learns to avoid obstacles across a 5-lane simulated road, without any gradient-based learning.

## What the notebook does
1. **Environment & genome setup** — Defines a 5-lane road (`LANE_WIDTH=5`), a 3-position forward sensor (`SENSOR_RANGE=3`), and encodes an agent's policy as a genome: one action (`left`, `stay`, `right`) for every possible (position, sensed-obstacle-pattern) combination.
2. **Episode simulation** — `run_episode()` simulates one 40-step drive for a given genome against randomly generated obstacles, returning how many steps the agent survived before crashing.
3. **Fitness, crossover, mutation, and GA loop**:
   - `fitness()` averages survival time over 5 seeded episodes for a stable score.
   - `crossover()` combines two parent genomes at a random split point.
   - `mutate()` randomly perturbs genes at a 3% rate.
   - The main loop runs 40 generations over a population of 80 genomes, keeping the top 12 as elites each generation.
4. **Visualization** — Plots the best average survival time per generation to show the agent's learning curve, with a reference line for the maximum possible survival time (40).

## Key libraries
`random`, `numpy`, `matplotlib`

## Outcome
The plotted learning curve shows the agent's survival performance improving over generations as the GA selects and recombines better-performing genomes.

## How to run
Run all cells top to bottom. No external data files are needed — everything is simulated with `random.seed(42)` / `np.random.seed(42)` for reproducibility.
