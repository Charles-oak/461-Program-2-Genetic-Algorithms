"""

Run:
    python main.py

The initial mutation rate is fixed at the assignment's required value of 0.01.
The program automatically adjusts it during the run; no manual rate switching is needed.

Algorithm outline:
    1. Generate a random population of at least 250 schedules.
    2. Score every schedule using the Appendix A fitness function.
    3. Convert fitness scores to Softmax parent-selection probabilities.
    4. Select two parents and apply single-point crossover.
    5. Mutate the two children and place them in the next generation, still containing 250 schedules.
    6. After at least 100 generations, stop when the average-fitness improvement from one generation to the immediately previous generation is less than 1%.

Adaptive mutation rate:
    Start at 1%. After each generation, compare its average fitness with the immediately previous generation. If improvement is at least 1%, halve the mutation rate used to create the NEXT generation. If improvement is below 1%, do not halve it. Once at least 100 generations have been evaluated, stop when the adjacent-generation improvement is below 1%.

Dependencies: numpy, matplotlib 
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# ============================================================================
# 1. INPUT DATA AND CHROMOSOME DOMAINS
# ============================================================================
# Each activity is one gene in a chromosome. A gene contains three integer indices: (room, time slot, facilitator). The human-readable names and scoring preferences are stored here and converted to indices below.
ACTIVITIES = {
    'SLA101A': (40, ('Glen', 'Lock', 'Banks'), ('Numen', 'Richards', 'Shaw', 'Singer')),
    'SLA101B': (35, ('Glen', 'Lock', 'Banks'), ('Numen', 'Richards', 'Shaw', 'Singer')),
    'SLA191A': (45, ('Glen', 'Lock', 'Banks'), ('Numen', 'Richards', 'Shaw', 'Singer')),
    'SLA191B': (40, ('Glen', 'Lock', 'Banks'), ('Numen', 'Richards', 'Shaw', 'Singer')),
    'SLA201': (60, ('Glen', 'Banks', 'Zeldin', 'Lock', 'Singer'), ('Richards', 'Uther', 'Shaw')),
    'SLA291': (50, ('Glen', 'Banks', 'Zeldin', 'Lock', 'Singer'), ('Richards', 'Uther', 'Shaw')),
    'SLA303': (25, ('Glen', 'Zeldin'), ('Banks',)),
    'SLA304': (20, ('Singer', 'Uther'), ('Richards',)),
    'SLA394': (15, ('Tyler', 'Singer'), ('Richards', 'Zeldin')),
    'SLA449': (30, ('Tyler', 'Zeldin', 'Uther'), ('Zeldin', 'Shaw')),
    'SLA451': (90, ('Lock', 'Banks', 'Zeldin'), ('Tyler', 'Singer', 'Shaw', 'Glen')),
}
# Each room maps to its seating capacity. The assignment's Appendix A explicitly lists these room options and capacities.
ROOMS = {'Beach 201': 18, 'Beach 301': 25, 'Frank 119': 95,
         'Loft 206': 55, 'Loft 310': 48, 'James 325': 110,
         'Roman 201': 40, 'Roman 216': 80, 'Slater 003': 32}
# The six legal time slots, in chronological order.
TIMES = ('10 AM', '11 AM', '12 PM', '1 PM', '2 PM', '3 PM')
# All facilitators are legal random choices. Preferred/other lists are used only when scoring a schedule, not to restrict random generation or mutation.
FACILITATORS = ('Lock', 'Glen', 'Banks', 'Richards', 'Shaw', 'Singer',
                'Uther', 'Tyler', 'Numen', 'Zeldin')
# The assignment's prescribed starting mutation probability. The GA changes this value automatically during execution; users do not switch rates manually.
INITIAL_MUTATION_RATE = 0.01

NAMES = tuple(ACTIVITIES)
ROOM_NAMES = tuple(ROOMS)
ROOM_CAP = np.array(tuple(ROOMS.values()))
ENROLLMENT = np.array([ACTIVITIES[n][0] for n in NAMES])
PREFERRED = tuple(frozenset(ACTIVITIES[n][1]) for n in NAMES)
OTHER = tuple(frozenset(ACTIVITIES[n][2]) for n in NAMES)
A101 = (NAMES.index('SLA101A'), NAMES.index('SLA101B'))
A191 = (NAMES.index('SLA191A'), NAMES.index('SLA191B'))


# ============================================================================
# 2. FITNESS SCORE DATA STRUCTURE
# ============================================================================
@dataclass(frozen=True)
class Scores:
    total: float
    detail: dict[str, float]


# ============================================================================
# 3. FITNESS FUNCTION
# ============================================================================
# This function computes the total score and a category-by-category breakdown.
# It does not reject a schedule: penalties lower its score, but the chromosome remains eligible for Softmax selection.
def fitness(schedule: np.ndarray) -> Scores:
    """11 x 3 int array: indices (room, time, facilitator).
    Count pairwise special adjustments ONCE per pair; other penalties per activity.
    """
    if schedule.shape != (len(NAMES), 3):
        raise ValueError('Schedule must have shape (11, 3)')
    # Each row is one activity. The three columns hold room, time, and facilitator indices respectively.
    rooms, times, facs = schedule[:, 0], schedule[:, 1], schedule[:, 2]

    # Count repeated room/time and facilitator/time assignments. These counts let the scoring loop identify conflicts for each affected activity.
    roomtime = Counter(zip(rooms.tolist(), times.tolist()))
    factime = Counter(zip(facs.tolist(), times.tolist()))

    # Count the total number of activities assigned to each facilitator.
    facload = Counter(facs.tolist())

    # Store scores by rubric category so the final total can also be explained.
    detail = defaultdict(float)

    # Evaluate the activity-level rubric separately for each of the 11 activities.
    for i, name in enumerate(NAMES):
        r, t, f = map(int, schedule[i])
        # Room conflict: if more than one activity occupies this room at this time, each affected activity receives the -0.5 penalty.
        if roomtime[(r, t)] > 1:
            detail['room_conflict'] -= .5

        # Room-size score: only the first matching condition is applied.
        cap = ROOM_CAP[r]
        enrollment = ENROLLMENT[i]
        if cap < enrollment:
            detail['room_size'] -= .5
        elif cap > 3 * enrollment:
            detail['room_size'] -= .4
        elif cap > 1.5 * enrollment:
            detail['room_size'] -= .2
        else:
            detail['room_size'] += .3

        # Facilitator preference: Preferred takes precedence over Other.
        # This matters when a name appears in both lists for an activity.
        teacher = FACILITATORS[f]
        if teacher in PREFERRED[i]:
            detail['facilitator_preference'] += .5
        elif teacher in OTHER[i]:
            detail['facilitator_preference'] += .2
        else:
            detail['facilitator_preference'] -= .1

        # Same-time facilitator rule: a facilitator assigned to exactly one activity in this slot earns +0.2; simultaneous multiple assignments cause -0.2 for each affected activity.
        detail['same_time_facilitator'] += .2 if factime[(f, t)] == 1 else -.2

        # Total facilitator workload: the penalty is applied to each activity assigned to a facilitator whose total load falls outside the range.
        if facload[f] > 4:
            detail['facilitator_load'] -= .5
        elif facload[f] < 3 and not (teacher == 'Tyler' and facload[f] < 2):
            detail['facilitator_load'] -= .4

    # Generic consecutive-slot rule from Appendix A: it applies to ANY two activities taught by the same facilitator.
    # Each qualifying pair receives +0.5. If exactly one room is in Roman or Beach, the same pair also receives the -0.4 building-distance penalty.
    # The pair is examined once because j always starts at i + 1.
    for i in range(len(NAMES)):
        for j in range(i + 1, len(NAMES)):
            if facs[i] == facs[j] and abs(int(times[i]) - int(times[j])) == 1:
                detail['consecutive_facilitator'] += .5
                if far_buildings(int(rooms[i]), int(rooms[j])):
                    detail['consecutive_building'] -= .4

    # SLA101 and SLA191 each have their own A/B section rule. These are course-specific and are applied once to each course's A/B pair.
    for i, j in (A101, A191):
        gap = abs(int(times[i]) - int(times[j]))
        if gap > 4:
            detail['section_separation'] += .5
        elif gap == 0:
            detail['section_separation'] -= .5

    # Cross-course special rules: compare each SLA101 section with each SLA191 section (2 x 2 = 4 pairs). The rubric does not require the two activities to share a facilitator for these special time adjustments.
    for i in A101:
        for j in A191:
            gap = abs(int(times[i]) - int(times[j]))
            if gap == 1:
                detail['101_191_time'] += .5
                if far_buildings(int(rooms[i]), int(rooms[j])):
                    detail['101_191_building'] -= .4
            elif gap == 2:
                detail['101_191_time'] += .25
            elif gap == 0:
                detail['101_191_time'] -= .25
    return Scores(round(sum(detail.values()), 10), dict(detail))


# ============================================================================
# 4. SMALL HELPER FOR THE ROMAN/BEACH BUILDING RULE
# ============================================================================
def far_buildings(r1: int, r2: int) -> bool:
    """Exactly one room is in a Roman or Beach building."""
    def rb(r: int) -> bool:
        return ROOM_NAMES[r].split()[0] in ('Roman', 'Beach')
    return rb(r1) != rb(r2)


# ============================================================================
# 5. CONSTRAINT DIAGNOSTICS (REPORTING ONLY)
# ============================================================================
# These counts describe issues in a schedule. They are not hard filters and are not used by the genetic algorithm to discard a chromosome.
def violations(schedule: np.ndarray) -> dict[str, int]:
    rooms, times, facs = schedule.T
    rt = Counter(zip(rooms.tolist(), times.tolist()))
    ft = Counter(zip(facs.tolist(), times.tolist()))
    loads = Counter(facs.tolist())
    return {
        'colliding_room_pairs': sum(v * (v - 1) // 2 for v in rt.values()),
        'colliding_facilitator_pairs': sum(v * (v - 1) // 2 for v in ft.values()),
        'undersized_room_assignments': sum(ROOM_CAP[int(rooms[i])] < ENROLLMENT[i] for i in range(len(NAMES))),
        'overloaded_facilitators_5plus': sum(v > 4 for v in loads.values()),
        'low_load_non_Tyler': sum(v < 3 for k, v in loads.items() if FACILITATORS[k] != 'Tyler'),
        'same_slot_101_or_191_pairs': sum(times[i] == times[j] for i, j in (A101, A191)),
    }


# ============================================================================
# 6. INITIAL RANDOM POPULATION
# ============================================================================
# Shape: (population size, number of activities, 3 variables).
# Each variable is sampled from its own valid domain.
def random_population(n: int, rng: np.random.Generator) -> np.ndarray:
    """Create n random chromosomes, each containing 11 activities x 3 genes."""
    # Allocate the population array. int16 is sufficient for the small index values and keeps the population compact in memory.
    result = np.empty((n, len(NAMES), 3), dtype=np.int16)

    # Each chromosome column has its own legal domain: room index, time-slot index, and facilitator index.
    for col, size in enumerate((len(ROOMS), len(TIMES), len(FACILITATORS))):
        result[:, :, col] = rng.integers(
            0, size, size=(n, len(NAMES))
        )
    return result


# ============================================================================
# 7. SOFTMAX PARENT SELECTION
# ============================================================================
# Convert fitness values into probabilities. Subtracting the maximum first prevents numerical overflow while leaving the probability ratios unchanged.
def softmax(fitnesses: np.ndarray) -> np.ndarray:
    """Return parent-selection probabilities for the current population."""
    # Subtract the largest score before exponentiation. This does not change the resulting probability ratios, but helps avoid numeric overflow.
    shifted = fitnesses - np.max(fitnesses)
    exponentials = np.exp(shifted)

    # Normalize the exponentials so every probability is between 0 and 1 and all probabilities add up to 1.
    return exponentials / exponentials.sum()


# ============================================================================
# 8. OFFSPRING CREATION: CROSSOVER FIRST, THEN MUTATION
# ============================================================================
# Parent arrays are never changed. Crossover copies the parent segments into new child arrays; mutation is applied only to those children.
def offspring(parent_a: np.ndarray, parent_b: np.ndarray,
              rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Pick a crossover point between activity genes. The three values for an activity (room, time, facilitator) stay together as one inherited gene.
    cut = int(rng.integers(1, len(NAMES)))

    # Create two reciprocal children. .copy() ensures later mutation changes only the child arrays, never either parent array.
    a = np.concatenate((parent_a[:cut], parent_b[cut:])).copy()
    b = np.concatenate((parent_b[:cut], parent_a[cut:])).copy()

    # Mutation happens AFTER crossover. Each child and each chromosome column receives independent mutation tests.
    for child in (a, b):
        for col, domain_size in enumerate((len(ROOMS), len(TIMES), len(FACILITATORS))):
            # Generate one random number per activity. If it is below the mutation rate, replace this activity's value in the current column.
            mask = rng.random(len(NAMES)) < rate
            if np.any(mask):
                # Replacement is random within the correct variable domain.
                # It can coincidentally select the same value as before.
                child[mask, col] = rng.integers(
                    0, domain_size, size=int(mask.sum())
                )
    return a, b


# ============================================================================
# 9. ADJACENT-GENERATION AVERAGE-FITNESS IMPROVEMENT
# ============================================================================
# The result is a percentage: (current - previous) / |previous| * 100.
def improvement(current: float, previous: float) -> float:
    return 100.0 * (current - previous) / max(abs(previous), 1e-12)


def run_ga(pop_size: int = 250, seed: int = 461,
           min_gens: int = 100, quiet: bool = False):
    """Run the GA with automatic mutation-rate halving and no elitism.

    Stopping rule:
      * Evaluate at least 100 generations.
      * Beginning with Generation 100, compare average fitness with the
        immediately preceding generation.
      * Stop as soon as that adjacent-generation improvement is below 1%.

    Adaptive mutation-rate rule:
      * The initial rate defaults to 0.01.
      * If a generation's average fitness improves by at least 1% compared
        with the immediately previous generation, halve the rate that will be used to create the NEXT generation.
      * If improvement is below 1%, do not halve the rate.
      * This check uses the same adjacent-generation metric; there is no
        additional observation window, waiting period, or fixed adjustment interval.

    """
    # The assignment specifies a population of at least 250 and at least 100 generations. Reject smaller values rather than silently violating it.
    if pop_size < 250:
        raise ValueError('Population size must be at least 250.')
    if min_gens < 100:
        raise ValueError('The minimum generation count must be at least 100.')
    # Always begin at the assignment's prescribed rate. From this point on, the algorithm itself controls all mutation-rate changes automatically.
    mutation_rate = INITIAL_MUTATION_RATE
    initial_mutation_rate = INITIAL_MUTATION_RATE

    # Use NumPy's PCG64DXSM generator with a fixed seed so the same parameters can reproduce the same random sequence.
    rng = np.random.Generator(np.random.PCG64DXSM(seed))

    # Generation 1 is a completely random population of candidate schedules.
    population = random_population(pop_size, rng)

    # One row is added after each generation. This history records both fitness and the adaptive mutation-rate changes for later inspection.
    history: list[dict] = []
    adjustments: list[dict] = []

    # There is no previous average for Generation 1, so its improvement is blank.
    previous_avg = None

    # These variables are updated at every generation. The final output is specifically the best schedule from the FINAL evaluated generation.
    final_best_schedule = None
    final_best_fitness = float('-inf')
    best_ever_fitness = float('-inf')

    stop_reason = 'average fitness improvement fell below 1%'
    generation = 0

    while True:
        generation += 1

        # At this point, mutation_rate is the rate that was used to create the current population from its parents. Generation 1 is random, so no mutation was used to create it; we record that as N/A in the CSV.
        rate_used_for_current_generation = (
            '' if generation == 1 else mutation_rate
        )

        # ---------------------------------------------------------------
        # STEP 1: Evaluate every schedule in the current population.
        # ---------------------------------------------------------------
        fitness_values = np.array(
            [fitness(schedule).total for schedule in population],
            dtype=float
        )

        # ---------------------------------------------------------------
        # STEP 2: Calculate the required fitness statistics for this generation.
        # ---------------------------------------------------------------
        best_index = int(np.argmax(fitness_values))
        best_fitness = float(fitness_values[best_index])
        average_fitness = float(np.mean(fitness_values))
        worst_fitness = float(np.min(fitness_values))

        # Keep the best schedule from this generation for final output. This is only for reporting; it is NOT copied into the next generation.
        final_best_schedule = population[best_index].copy()
        final_best_fitness = best_fitness
        best_ever_fitness = max(best_ever_fitness, best_fitness)

        # ---------------------------------------------------------------
        # STEP 3: Compare this generation's average with the PREVIOUS one.
        # ---------------------------------------------------------------
        if previous_avg is None:
            generation_improvement = ''
        else:
            generation_improvement = improvement(
                average_fitness,
                previous_avg
            )

        # ---------------------------------------------------------------
        # STEP 4: Apply the stopping rule before creating another generation.
        # The algorithm must evaluate at least 100 generations. From Generation 100 onward, an improvement below 1% causes an immediate stop.
        # ---------------------------------------------------------------
        should_stop = (
            generation >= min_gens
            and generation_improvement != ''
            and generation_improvement < 1.0
        )

        # ---------------------------------------------------------------
        # STEP 5: Adapt the mutation rate for the NEXT generation.
        # If average fitness improved by at least 1%, halve the rate. This is exactly one comparison of adjacent generations; there is no 20-gen window, 30-gen waiting rule, or extra threshold.
        # ---------------------------------------------------------------
        old_mutation_rate = mutation_rate
        mutation_adjustment = 'unchanged'

        if should_stop:
            # No next generation will be created, so no rate adjustment is needed.
            mutation_adjustment = 'stopped; no next generation'
        elif (
            generation_improvement != ''
            and generation_improvement >= 1.0
        ):
            mutation_rate = mutation_rate / 2.0
            mutation_adjustment = 'halved'
            adjustments.append({
                'generation': generation,
                'improvement_pct': round(float(generation_improvement), 4),
                'old_rate': old_mutation_rate,
                'new_rate': mutation_rate
            })

        # Store the rate that created the current generation and the rate that will be used to create the next generation. This makes the adaptation trace explicit and easy to explain in the video report.
        history.append({
            'generation': generation,
            'best': round(best_fitness, 4),
            'average': round(average_fitness, 6),
            'worst': round(worst_fitness, 4),
            'improvement_pct': (
                round(float(generation_improvement), 4)
                if generation_improvement != ''
                else ''
            ),
            'mutation_rate_used': rate_used_for_current_generation,
            'mutation_rate_next': mutation_rate,
            'mutation_adjustment': mutation_adjustment
        })

        # Print normal progress every 10 generations, plus every rate change.
        if not quiet and (generation == 1 or generation % 10 == 0):
            improvement_text = (
                'N/A' if generation_improvement == ''
                else f'{generation_improvement:.4f}%'
            )
            print(
                f'Generation {generation:3d}: '
                f'best={best_fitness:7.3f} '
                f'avg={average_fitness:7.3f} '
                f'worst={worst_fitness:7.3f} '
                f'improvement={improvement_text} '
                f'mutation used={rate_used_for_current_generation if rate_used_for_current_generation != "" else "N/A"} '
                f'next={mutation_rate:.8f}'
            )

        if not quiet and mutation_adjustment == 'halved':
            print(
                f'  Adaptive mutation after Generation {generation}: '
                f'{old_mutation_rate:.8f} -> {mutation_rate:.8f} because '
                f'average fitness improved by {generation_improvement:.4f}%.'
            )

        # If the stop condition is met, the current generation has already been evaluated and recorded. Do not create an unused next generation.
        if should_stop:
            stop_reason = (
                f'Generation-to-generation average fitness improvement '
                f'was {generation_improvement:.4f}% (< 1.0%)'
            )
            if not quiet:
                print(f'  Stopping: {stop_reason}.')
            break

        # ---------------------------------------------------------------
        # STEP 6: Convert fitness scores into Softmax parent-selection probabilities. Higher-scoring schedules have a higher chance of selection, but lower-scoring schedules can still be selected.
        # ---------------------------------------------------------------
        selection_probabilities = softmax(fitness_values)

        # ---------------------------------------------------------------
        # STEP 7: Create the NEXT generation entirely from offspring.
        # IMPORTANT: next_population starts EMPTY. There is no elitism.
        # The current (possibly just-halved) mutation_rate is used below.
        # ---------------------------------------------------------------
        next_population = []

        while len(next_population) < pop_size:
            parent_1_index, parent_2_index = rng.choice(
                pop_size,
                size=2,
                replace=True,
                p=selection_probabilities
            )

            parent_1 = population[int(parent_1_index)]
            parent_2 = population[int(parent_2_index)]

            # offspring() performs crossover first and then mutation, using the updated mutation_rate for both newly created children.
            child_1, child_2 = offspring(
                parent_1,
                parent_2,
                mutation_rate,
                rng
            )

            next_population.extend((child_1, child_2))

        # Keep exactly pop_size offspring (including for odd population sizes).
        population = np.array(
            next_population[:pop_size],
            dtype=np.int16
        )

        # Save this generation's average for the next adjacent-generation check.
        previous_avg = average_fitness

    return {
        'schedule': final_best_schedule,
        'final_best': final_best_fitness,
        'best_ever': best_ever_fitness,
        'history': history,
        'adjustments': adjustments,
        'stop_reason': stop_reason,
        'mode': 'adaptive',
        'seed': seed,
        'initial_mutation_rate': initial_mutation_rate,
        'mutation_rate': mutation_rate
    }


# ============================================================================
# 11. OUTPUT FILES: SCHEDULE, FITNESS CSV, AND FITNESS PLOT
# ============================================================================
def write_results(result, output: Path, pop_size: int):
    """Export the final schedule, generation history, and fitness plot."""
    # Create the output folder if it does not already exist.
    output.mkdir(parents=True, exist_ok=True)
    s = result['schedule']
    # CSV: one row per evaluated generation, including best/average/worst fitness and the percentage change from the previous generation.
    with (output / 'fitness_history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=result['history'][0].keys())
        writer.writeheader()
        writer.writerows(result['history'])
    # Text schedule: sort activities chronologically, then alphabetically within the same time slot. Recalculate the breakdown for this schedule.
    ordered = sorted(range(len(NAMES)), key=lambda i: (int(s[i, 1]), NAMES[i]))
    breakdown = fitness(s)
    with (output / 'best_schedule.txt').open('w', encoding='utf-8') as f:
        f.write('CS 461 Program 2 — Final-generation best schedule\n')
        f.write(f"Seed: {result['seed']} | Population: {pop_size} | Generations: {len(result['history'])}\n")
        f.write(f"Initial mutation rate: {result['initial_mutation_rate']:.8f}\n")
        f.write(f"Final mutation rate: {result['mutation_rate']:.8f}\n")
        f.write(f"Adaptive mutation-rate changes: {len(result['adjustments'])}\n")
        f.write(f"Best fitness in final generation: {result['final_best']:.4f}\n")
        f.write(f"Best fitness observed across all generations: {result['best_ever']:.4f}\n")
        f.write(f"Stop: {result['stop_reason']}\n\n")
        if result['adjustments']:
            f.write('Mutation-rate adjustment log (after the listed generation):\n')
            for item in result['adjustments']:
                f.write(
                    f"  Gen {item['generation']}: {item['old_rate']:.8f} -> "
                    f"{item['new_rate']:.8f} (improvement {item['improvement_pct']:.4f}%)\n"
                )
            f.write('\n')
        f.write(f"{'Activity':<10} {'Enrollment':<10} {'Room':<14} {'Time':<8} Facilitator\n")
        f.write('-' * 65 + '\n')
        for i in ordered:
            room, time, fac = map(int, s[i])
            f.write(f'{NAMES[i]:<10} {ENROLLMENT[i]:<10} {ROOM_NAMES[room]:<14} {TIMES[time]:<8} {FACILITATORS[fac]}\n')
        f.write('\nFitness breakdown (additive categories):\n')
        for k, v in sorted(breakdown.detail.items()):
            f.write(f'  {k:<30} {v:+.4f}\n')
        f.write('\nConstraint diagnostics (counts):\n')
        for k, v in violations(s).items():
            f.write(f'  {k:<35} {v}\n')

    # Plot: use a non-interactive backend so the chart is saved directly to a PNG file even when the program runs in a terminal.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h = result['history']
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col, title in [('best', 'Best'), ('average', 'Average'), ('worst', 'Worst')]:
        ax.plot([r['generation'] for r in h], [r[col] for r in h], label=title, linewidth=1.8)
    ax.set(xlabel='Generation', ylabel='Fitness score', title='Genetic Algorithm — Fitness by Generation')
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output / 'fitness_plot.png', dpi=160)
    plt.close(fig)

    # Optional mutation-rate tracker: this shows the rate selected for the next generation after each generation's fitness comparison.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.step(
        [r['generation'] for r in h],
        [r['mutation_rate_next'] for r in h],
        where='post',
        label='Mutation Rate'
    )
    ax.set(
        xlabel='Generation',
        ylabel='Mutation probability',
        title='Adaptive Mutation Rate by Generation'
    )
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output / 'mutation_rate_plot.png', dpi=160)
    plt.close(fig)


# ============================================================================
# 12. COMMAND-LINE ENTRY POINT
# ============================================================================
def main():
    """Parse command-line options, run the GA, and save the output files."""
    parser = argparse.ArgumentParser(
        description='CS 461 Program 2 — Genetic Algorithm'
    )

    # At least 250 candidate schedules are required by the assignment.
    parser.add_argument('--population', type=int, default=250)

    # A fixed seed makes the run reproducible.
    parser.add_argument('--seed', type=int, default=461)

    # Each run writes to its own folder if a different output path is supplied.
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent / 'output'
    )

    args = parser.parse_args()

    # Run with adaptive mutation-rate halving. Evaluate at least 100 generations, then stop when adjacent-generation average-fitness improvement falls below 1%.
    result = run_ga(
        pop_size=args.population,
        seed=args.seed,
        min_gens=100
    )

    # Save the final-generation best schedule, the fitness history CSV, and the required best/average/worst fitness line chart.
    write_results(result, args.output, args.population)

    print(
        f"\nFINAL: best={result['final_best']:.4f}, "
        f"generations={len(result['history'])}"
    )
    print(f"Initial mutation rate: {result['initial_mutation_rate']:.8f}")
    print(f"Final mutation rate: {result['mutation_rate']:.8f}")
    print(f"Automatic mutation-rate reductions: {len(result['adjustments'])}")
    print('Violations:', violations(result['schedule']))
    print('Stop reason:', result['stop_reason'])
    print('Files:', args.output.resolve())


if __name__ == '__main__':
    main()
