"""
Genetic algorithm for adversarial text optimization (Task 4).

Uses a Gemini-powered mutation strategy: the GA evolves a population of
AI-generated paragraphs to maximize the classifier's P(Human) score.
Two mutation types are tested independently:
  - Type A: Rhythm rewriting (change sentence cadence, keep vocabulary)
  - Type B: Archaic injection (add pre-1900 words + minor grammatical quirks)

The population is initialized by generating fresh paragraphs via Gemini,
not by mutating a single seed text.
"""

import os
import time
import random
import json
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv


class TextGeneticAlgorithm:
    """
    Gemini-powered genetic algorithm for adversarial text evolution.

    The fitness function is the classifier's P(Human) — higher means the
    text looks more human. The GA tries to push AI-generated text past a
    target threshold (default 0.90) using LLM-based mutations.
    """

    def __init__(
        self,
        classifier_fn,
        topic,
        author_style_prompt,
        model_name="gemini-2.5-flash",
        population_size=10,
        elite_size=3,
        max_generations=10,
        target_fitness=0.90,
        seed=42,
        rate_limit_sleep=1.5,
    ):
        """
        Args:
            classifier_fn: callable(text) -> np.array of class probabilities
                           Index 0 must be P(Human).
            topic: topic string for paragraph generation
            author_style_prompt: style prompt for Class 3 generation
            model_name: Gemini model to use for generation and mutation
            population_size: number of individuals per generation
            elite_size: number of top individuals preserved each generation
            max_generations: stopping criterion
            target_fitness: early-stop if best fitness exceeds this
            seed: random seed for reproducibility
            rate_limit_sleep: seconds between Gemini API calls
        """
        self.classifier_fn = classifier_fn
        self.topic = topic
        self.author_style_prompt = author_style_prompt
        self.model_name = model_name
        self.population_size = population_size
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.target_fitness = target_fitness
        self.seed = seed
        self.rate_limit_sleep = rate_limit_sleep

        random.seed(seed)
        np.random.seed(seed)

        # Configure Gemini
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        genai.configure(api_key=api_key)
        self.gemini = genai.GenerativeModel(model_name)

    def _call_gemini(self, prompt, max_retries=3):
        """Call Gemini API with retry logic and rate limiting."""
        for attempt in range(max_retries):
            try:
                response = self.gemini.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=300,
                        temperature=0.9,
                    ),
                )
                time.sleep(self.rate_limit_sleep)
                text = response.text.strip()
                # Clean up any markdown formatting Gemini might add
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
                return text
            except Exception as e:
                wait = 2 ** attempt * 2
                print(f"    Gemini API error (attempt {attempt+1}): {e}")
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
        print("    WARNING: All retries failed, returning original text")
        return None

    def initialize_population(self):
        """
        Generate initial population using Gemini with the Class 3 style prompt.
        Each individual is a fresh generation, not a mutation of a seed.
        """
        print(f"  Initializing population of {self.population_size} via Gemini...")
        population = []
        prompt = (
            f"{self.author_style_prompt}\n\n"
            f"Write a paragraph of 120-170 words on the topic of \"{self.topic}\"."
        )
        for i in range(self.population_size):
            text = self._call_gemini(prompt)
            if text:
                population.append(text)
                print(f"    Generated {i+1}/{self.population_size}")
            else:
                print(f"    Failed to generate {i+1}, using previous")
                if population:
                    population.append(population[-1])

        return population

    def evaluate_fitness(self, text):
        """
        Run text through the classifier, return P(Human).
        Index 0 of the probability array is the Human class.
        """
        probs = self.classifier_fn(text)
        return float(probs[0])

    def select_elites(self, population, fitnesses):
        """Return top-k individuals by fitness score."""
        ranked = sorted(zip(fitnesses, population), key=lambda x: -x[0])
        return [text for _, text in ranked[:self.elite_size]]

    def mutate(self, text, mutation_type="A"):
        """
        Apply Gemini-powered mutation.

        Type A — Rhythm rewriting: changes sentence structure and cadence
                 while preserving vocabulary and meaning.
        Type B — Archaic injection: introduces pre-1900 vocabulary and
                 minor grammatical irregularities (missing comma, inverted
                 verb phrase) to mimic human imperfection.
        """
        if mutation_type == "A":
            prompt = (
                "Rewrite the following paragraph to change the rhythm of the "
                "sentences while keeping the vocabulary and core meaning identical. "
                "Vary sentence lengths more dramatically — mix very short sentences "
                "with longer, more complex ones. Do not add new information.\n\n"
                f"Paragraph:\n{text}"
            )
        elif mutation_type == "B":
            prompt = (
                "Rewrite the following paragraph with these specific changes: "
                "introduce one subtle archaic word (pre-1900 English) and one "
                "minor grammatical inconsistency, like a missing comma or an "
                "inverted verb phrase. Keep everything else the same. "
                "The changes should be subtle, not obvious.\n\n"
                f"Paragraph:\n{text}"
            )
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}")

        result = self._call_gemini(prompt)
        return result if result else text

    def run(self, mutation_type="A", verbose=True):
        """
        Main GA loop for a single mutation type.

        Returns:
            dict with keys:
                best_text: str — highest-fitness text at termination
                best_fitness: float — its P(Human) score
                history: list[float] — best fitness per generation
                all_fitnesses: list[list[float]] — population fitnesses per gen
                mutation_type: str — "A" or "B"
        """
        label = "Rhythm" if mutation_type == "A" else "Archaic"
        if verbose:
            print(f"\n{'='*60}")
            print(f"  GA Run — Mutation Type {mutation_type} ({label})")
            print(f"  Topic: {self.topic}")
            print(f"  Target fitness: {self.target_fitness}")
            print(f"{'='*60}")

        # Initialize
        population = self.initialize_population()

        history = []
        all_fitnesses = []

        for gen in range(self.max_generations):
            # Evaluate
            fitnesses = [self.evaluate_fitness(t) for t in population]
            all_fitnesses.append(fitnesses)

            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            avg_fitness = np.mean(fitnesses)
            history.append(best_fitness)

            if verbose:
                print(f"  Gen {gen}: best={best_fitness:.4f}, "
                      f"avg={avg_fitness:.4f}, "
                      f"worst={min(fitnesses):.4f}")

            # Early stopping
            if best_fitness >= self.target_fitness:
                if verbose:
                    print(f"  Target reached at generation {gen}! "
                          f"(fitness={best_fitness:.4f} >= {self.target_fitness})")
                break

            # Select elites
            elites = self.select_elites(population, fitnesses)

            # Mutate non-elite individuals from the top half
            ranked = sorted(zip(fitnesses, population), key=lambda x: -x[0])
            parents = [text for _, text in ranked[:max(5, len(ranked)//2)]]

            new_pop = list(elites)
            if verbose:
                print(f"  Mutating {self.population_size - self.elite_size} "
                      f"individuals (Type {mutation_type})...")

            while len(new_pop) < self.population_size:
                parent = random.choice(parents)
                child = self.mutate(parent, mutation_type)
                new_pop.append(child)

            population = new_pop

        # Final evaluation
        fitnesses = [self.evaluate_fitness(t) for t in population]
        best_idx = np.argmax(fitnesses)

        return {
            "best_text": population[best_idx],
            "best_fitness": fitnesses[best_idx],
            "history": history,
            "all_fitnesses": all_fitnesses,
            "final_population": population,
            "final_fitnesses": fitnesses,
            "mutation_type": mutation_type,
            "converged": fitnesses[best_idx] >= self.target_fitness,
            "convergence_gen": next(
                (i for i, f in enumerate(history) if f >= self.target_fitness),
                None
            ),
        }


def run_comparative_ga(
    classifier_fn,
    topic,
    author_style_prompt,
    model_name="gemini-2.5-flash",
    population_size=10,
    elite_size=3,
    max_generations=10,
    target_fitness=0.90,
    seed=42,
    rate_limit_sleep=1.5,
    verbose=True,
):
    """
    Runs the GA with both mutation types (A and B) independently and
    returns results for comparison.

    This is the main entry point called from the notebook.

    Returns:
        dict with keys "type_a" and "type_b", each containing the GA results.
    """
    ga_kwargs = dict(
        classifier_fn=classifier_fn,
        topic=topic,
        author_style_prompt=author_style_prompt,
        model_name=model_name,
        population_size=population_size,
        elite_size=elite_size,
        max_generations=max_generations,
        target_fitness=target_fitness,
        seed=seed,
        rate_limit_sleep=rate_limit_sleep,
    )

    if verbose:
        print("Running GA with Type A (Rhythm) mutations...")
    ga_a = TextGeneticAlgorithm(**ga_kwargs)
    results_a = ga_a.run(mutation_type="A", verbose=verbose)

    if verbose:
        print("\n\nRunning GA with Type B (Archaic) mutations...")
    ga_b = TextGeneticAlgorithm(**ga_kwargs)
    results_b = ga_b.run(mutation_type="B", verbose=verbose)

    return {
        "type_a": results_a,
        "type_b": results_b,
        "topic": topic,
    }