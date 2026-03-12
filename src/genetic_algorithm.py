"""
Genetic algorithm for adversarial text optimization (Task 4).

The GA evolves a population of text variants to minimize the classifier's
confidence that the text is AI-generated. Mutations are linguistically
motivated: synonym substitution, punctuation insertion, sentence reordering.
"""

import random
import re
import copy
import numpy as np


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

def mutate_synonym(text, rate=0.1):
    """
    Randomly replaces words with simple synonym variants.
    Uses a small built-in synonym map targeting common AI-isms.
    """
    synonym_map = {
        "furthermore": ["besides", "also", "in addition"],
        "moreover": ["additionally", "what is more", "beyond that"],
        "important": ["significant", "notable", "consequential"],
        "crucial": ["vital", "essential", "critical"],
        "however": ["yet", "still", "nevertheless"],
        "therefore": ["thus", "hence", "consequently"],
        "regarding": ["concerning", "about", "touching upon"],
        "utilize": ["use", "employ", "make use of"],
        "implement": ["carry out", "execute", "put into practice"],
        "significant": ["considerable", "notable", "meaningful"],
        "demonstrate": ["show", "reveal", "illustrate"],
        "facilitate": ["help", "enable", "aid"],
        "comprehensive": ["thorough", "extensive", "complete"],
    }
    words = text.split()
    for i, w in enumerate(words):
        w_lower = w.lower().strip(".,;:!?")
        if w_lower in synonym_map and random.random() < rate:
            replacement = random.choice(synonym_map[w_lower])
            # Preserve original capitalization
            if w[0].isupper():
                replacement = replacement.capitalize()
            words[i] = replacement
    return " ".join(words)


def mutate_punctuation(text, rate=0.05):
    """
    Inserts semicolons or em-dashes at clause boundaries (after commas)
    to mimic Victorian punctuation patterns.
    """
    replacements = ["; ", " — ", "; "]
    words = text.split()
    result = []
    for w in words:
        result.append(w)
        if w.endswith(",") and random.random() < rate:
            w_base = w[:-1]
            result[-1] = w_base + random.choice(replacements).rstrip()
    return " ".join(result)


def mutate_sentence_shuffle(text, rate=0.1):
    """
    Randomly swaps two adjacent sentences with probability `rate`.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < 2:
        return text
    for i in range(len(sentences) - 1):
        if random.random() < rate:
            sentences[i], sentences[i+1] = sentences[i+1], sentences[i]
    return " ".join(sentences)


def mutate_insert_filler(text, rate=0.03):
    """
    Inserts Victorian-style filler phrases to disrupt AI-detection patterns.
    """
    fillers = [
        "indeed,", "it must be said,", "one might observe,",
        "as it were,", "to be sure,", "in truth,",
        "it seemed,", "curiously enough,",
    ]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for s in sentences:
        if random.random() < rate and len(s.split()) > 5:
            words = s.split()
            pos = random.randint(1, min(3, len(words) - 1))
            words.insert(pos, random.choice(fillers))
            result.append(" ".join(words))
        else:
            result.append(s)
    return " ".join(result)


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

def compute_fitness(text, classifier_fn, target_class=0):
    """
    Fitness = classifier's predicted probability of `target_class` (Human).
    Higher fitness means the text looks more human to the classifier.

    classifier_fn: callable that takes a string and returns a probability array [p_c0, p_c1, ...].
    """
    probs = classifier_fn(text)
    return probs[target_class]


# ---------------------------------------------------------------------------
# GA engine
# ---------------------------------------------------------------------------

def create_population(base_text, pop_size=20):
    """Creates initial population by applying light mutations to the base text."""
    population = [base_text]
    for _ in range(pop_size - 1):
        variant = mutate_synonym(base_text, rate=0.05)
        variant = mutate_punctuation(variant, rate=0.03)
        population.append(variant)
    return population


def crossover(parent1, parent2):
    """
    Sentence-level crossover: takes first half of sentences from parent1,
    second half from parent2.
    """
    sents1 = re.split(r'(?<=[.!?])\s+', parent1)
    sents2 = re.split(r'(?<=[.!?])\s+', parent2)
    mid = len(sents1) // 2
    child = sents1[:mid] + sents2[mid:]
    return " ".join(child)


def mutate(text, mutation_rate=0.1):
    """Applies all mutation operators with given rate."""
    text = mutate_synonym(text, rate=mutation_rate)
    text = mutate_punctuation(text, rate=mutation_rate * 0.5)
    text = mutate_sentence_shuffle(text, rate=mutation_rate * 0.3)
    text = mutate_insert_filler(text, rate=mutation_rate * 0.2)
    return text


def run_ga(
    base_text,
    classifier_fn,
    target_class=0,
    pop_size=20,
    generations=30,
    mutation_rate=0.1,
    elite_frac=0.2,
    seed=42,
    verbose=True,
):
    """
    Runs the genetic algorithm to evolve text that fools the classifier.

    Args:
        base_text: The AI-generated text to optimize
        classifier_fn: function(text) -> probability array
        target_class: class index to maximize (0 = Human)
        pop_size: population size
        generations: number of generations
        mutation_rate: base mutation rate
        elite_frac: fraction of population kept as elite
        seed: random seed
        verbose: print progress

    Returns:
        dict with best_text, best_fitness, history (fitness per generation)
    """
    random.seed(seed)
    np.random.seed(seed)

    population = create_population(base_text, pop_size)
    elite_count = max(2, int(pop_size * elite_frac))
    history = []

    for gen in range(generations):
        # Evaluate fitness
        fitnesses = [compute_fitness(t, classifier_fn, target_class) for t in population]

        # Sort by fitness (descending)
        ranked = sorted(zip(fitnesses, population), key=lambda x: -x[0])
        best_fitness = ranked[0][0]
        history.append(best_fitness)

        if verbose and (gen % 5 == 0 or gen == generations - 1):
            print(f"  Gen {gen:3d}: best_fitness={best_fitness:.4f}, "
                  f"avg={np.mean(fitnesses):.4f}")

        # Early stopping if nearly certain
        if best_fitness > 0.95:
            if verbose:
                print(f"  Converged at generation {gen} (fitness > 0.95)")
            break

        # Selection: elite + tournament
        elites = [text for _, text in ranked[:elite_count]]

        new_pop = list(elites)
        while len(new_pop) < pop_size:
            # Tournament selection (size 3)
            candidates = random.sample(list(zip(fitnesses, population)), min(3, len(population)))
            parent1 = max(candidates, key=lambda x: x[0])[1]
            candidates = random.sample(list(zip(fitnesses, population)), min(3, len(population)))
            parent2 = max(candidates, key=lambda x: x[0])[1]

            # Crossover
            child = crossover(parent1, parent2) if random.random() < 0.7 else parent1

            # Mutation
            child = mutate(child, mutation_rate)
            new_pop.append(child)

        population = new_pop

    # Final evaluation
    fitnesses = [compute_fitness(t, classifier_fn, target_class) for t in population]
    best_idx = np.argmax(fitnesses)

    return {
        "best_text": population[best_idx],
        "best_fitness": fitnesses[best_idx],
        "history": history,
        "original_fitness": compute_fitness(base_text, classifier_fn, target_class),
    }
