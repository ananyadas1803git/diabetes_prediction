#tso_hba.py
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from typing import Dict, Any, List, Tuple
import logging
from tqdm import tqdm
class TSOHBAOptimizer:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.population_size = config['optimization']['population_size']
        self.max_iterations = config['optimization']['max_iterations']
        self.early_stopping_rounds = config['optimization']['early_stopping_rounds']
        self.convergence_threshold = config['optimization']['convergence_threshold']
        self.cv_folds = config['optimization']['cv_folds']
        self.cv_repeats = config['optimization'].get('repeated_cv_repeats', 1)
        self.search_space = config['optimization']['search_space']
        self.objective_metric = config['optimization'].get('objective_metric', 'roc_auc')
        self.param_names = list(self.search_space.keys())
        self.best_solution = None
        self.best_fitness = -np.inf
        self.convergence_history = []
        self.no_improvement_count = 0
    def _initialize_population(self) -> np.ndarray:
        population = []
        for _ in range(self.population_size):
            individual = []
            for param_name in self.param_names:
                param_config = self.search_space[param_name]
                if param_config['type'] == 'int':
                    value = np.random.randint(param_config['min'],param_config['max'] + 1)
                else:
                    value = np.random.uniform(param_config['min'],param_config['max'])
                individual.append(value)
            population.append(individual)
        return np.array(population)
    def _decode_solution(self, solution: np.ndarray) -> Dict[str, Any]:
        params = {}
        for i, param_name in enumerate(self.param_names):
            param_config = self.search_space[param_name]
            if param_config['type'] == 'int':
                params[param_name] = int(np.round(solution[i]))
            else:
                params[param_name] = float(solution[i])
        return params
    def _evaluate_fitness(self,solution: np.ndarray,X_train: np.ndarray,y_train: np.ndarray) -> float:
        params = self._decode_solution(self._clip_to_search_space(np.asarray(solution, dtype=float).reshape(1, -1))[0])
        pos_count = int(np.sum(y_train == 1))
        neg_count = int(np.sum(y_train == 0))
        if pos_count > 0:
            params.setdefault("scale_pos_weight", float(neg_count) / pos_count)
        else:
            params.setdefault("scale_pos_weight", 1.0)
        model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1)
        cv = RepeatedStratifiedKFold(n_splits=self.cv_folds, n_repeats=self.cv_repeats, random_state=42) if self.cv_repeats > 1 else StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=self.objective_metric, n_jobs=-1)
            fitness = float(np.mean(scores))
        except Exception as e:
            self.logger.warning(f"Fitness evaluation failed: %s", e)
            fitness = 0.0
        return fitness
    def _tuna_swarm_phase(self,population: np.ndarray,best_position: np.ndarray,iteration: int) -> np.ndarray:
        new_population = population.copy()
        for i in range(len(population)):
            r1 = np.random.random()
            r2 = np.random.random()
            if r1 < 0.5:
                a = 2 * (1 - iteration / self.max_iterations)
                b = 2 * np.random.random() - 1
                distance = np.linalg.norm(population[i] - best_position)
                new_population[i] = best_position + distance * np.exp(b * iteration) * np.cos(2 * np.pi * b)
            else:
                leader_idx = np.random.randint(0, len(population))
                new_population[i] = population[leader_idx] + r2 * (best_position - population[i])
        new_population = self._clip_to_search_space(new_population)
        return new_population
    def _honey_badger_phase(self,population: np.ndarray,best_position: np.ndarray,iteration: int) -> np.ndarray:
        new_population = population.copy()
        intensity = 1 - iteration / self.max_iterations
        for i in range(len(population)):
            r = np.random.random()
            if r < 0.5:
                C = intensity * np.cos(2 * np.pi * np.random.random())
                new_population[i] = best_position + C * (best_position - population[i])
            else:
                F = intensity * np.random.random()
                new_population[i] = population[i] + F * (best_position - population[i])
        new_population = self._clip_to_search_space(new_population)
        return new_population
    def _clip_to_search_space(self, population: np.ndarray) -> np.ndarray:
        for i, param_name in enumerate(self.param_names):
            param_config = self.search_space[param_name]
            population[:, i] = np.clip(population[:, i],param_config['min'],param_config['max'])
        return population
    def _adaptive_switching(self,iteration: int,fitness: float,previous_fitness: float) -> str:
        if iteration < self.max_iterations * 0.3:
            return 'tso'
        elif iteration > self.max_iterations * 0.7:
            return 'hba'
        else:
            if fitness - previous_fitness > self.convergence_threshold:
                return 'hba' 
            else:
                return 'tso' 
    def optimize(self,X_train: np.ndarray,y_train: np.ndarray) -> Dict[str, Any]:
        self.logger.info("Starting TSO-HBA optimization")
        self.no_improvement_count = 0
        self.convergence_history = []
        population = self._initialize_population()
        fitness_scores = []
        for individual in population:
            fitness = self._evaluate_fitness(individual, X_train, y_train)
            fitness_scores.append(fitness)
        fitness_scores = np.array(fitness_scores)
        best_idx = np.argmax(fitness_scores)
        self.best_solution = population[best_idx].copy()
        best_fitness = fitness_scores[best_idx]
        self.best_fitness = best_fitness
        previous_fitness = best_fitness
        self.logger.info(f"Initial best fitness: {best_fitness:.4f}")
        for iteration in tqdm(range(self.max_iterations), desc="Optimization Progress"):
            # Adaptive phase selection
            phase = self._adaptive_switching(iteration, best_fitness, previous_fitness)
            if phase == 'tso':
                population = self._tuna_swarm_phase(population, self.best_solution, iteration)
            else:
                population = self._honey_badger_phase(population, self.best_solution, iteration)
            new_fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, X_train, y_train)
                new_fitness_scores.append(fitness)
            new_fitness_scores = np.array(new_fitness_scores)
            current_best_idx = np.argmax(new_fitness_scores)
            current_best_fitness = new_fitness_scores[current_best_idx]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                self.best_fitness = best_fitness
                self.best_solution = population[current_best_idx].copy()
                self.no_improvement_count = 0
                self.logger.info(f"Iteration {iteration + 1}: New best fitness: {best_fitness:.4f}")
            else:
                self.no_improvement_count += 1
            fitness_scores = new_fitness_scores
            self.convergence_history.append(best_fitness)
            previous_fitness = best_fitness
            if self.no_improvement_count >= self.early_stopping_rounds:
                self.logger.info(f"Early stopping at iteration {iteration + 1}")
                break
        best_params = self._decode_solution(self.best_solution)
        self.logger.info(f"Optimization completed. Best fitness: {best_fitness:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        return best_params
    def get_convergence_history(self) -> List[float]:
        return self.convergence_history
    def get_best_params(self) -> Dict[str, Any]:
        if self.best_solution is None:
            raise ValueError("Optimization not performed yet")
        
        return self._decode_solution(self.best_solution)
