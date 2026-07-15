import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.optimization.tso_hba import TSOHBAOptimizer
from src.utils.config import load_config
@pytest.fixture
def config():
    config = load_config("config.yaml")
    # Reduce iterations for faster testing
    config['optimization']['max_iterations'] = 5
    config['optimization']['population_size'] = 10
    return config
@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(100, 8)
    y = np.random.randint(0, 2, 100)
    return X, y
@pytest.fixture
def optimizer(config):
    from src.utils.logger import setup_logger
    logger = setup_logger("test", level="ERROR")
    return TSOHBAOptimizer(config, logger)
def test_optimizer_initialization(optimizer):
    assert optimizer is not None
    assert optimizer.config is not None
    assert optimizer.population_size > 0
    assert optimizer.max_iterations > 0
def test_initialize_population(optimizer):
    population = optimizer._initialize_population()
    assert population.shape[0] == optimizer.population_size
    assert population.shape[1] == len(optimizer.param_names)
def test_decode_solution(optimizer):
    solution = np.random.randn(len(optimizer.param_names))
    params = optimizer._decode_solution(solution)
    assert isinstance(params, dict)
    assert len(params) == len(optimizer.param_names)
    for param_name in optimizer.param_names:
        param_config = optimizer.search_space[param_name]
        if param_config['type'] == 'int':
            assert isinstance(params[param_name], int)
        else:
            assert isinstance(params[param_name], float)
def test_evaluate_fitness(optimizer, sample_data):
    X, y = sample_data
    solution = np.random.randn(len(optimizer.param_names))
    fitness = optimizer._evaluate_fitness(solution, X, y)
    assert isinstance(fitness, float)
    assert 0 <= fitness <= 1
def test_clip_to_search_space(optimizer):
    population = optimizer._initialize_population()
    population[0, 0] = -1000
    population[0, 1] = 1000
    clipped = optimizer._clip_to_search_space(population)
    for i, param_name in enumerate(optimizer.param_names):
        param_config = optimizer.search_space[param_name]
        assert clipped[0, i] >= param_config['min']
        assert clipped[0, i] <= param_config['max']
def test_optimize(optimizer, sample_data):
    X, y = sample_data
    best_params = optimizer.optimize(X, y)
    assert isinstance(best_params, dict)
    assert len(best_params) == len(optimizer.param_names)
    assert len(optimizer.convergence_history) > 0
def test_get_best_params(optimizer, sample_data):
    X, y = sample_data
    optimizer.optimize(X, y)
    best_params = optimizer.get_best_params()
    assert isinstance(best_params, dict)
    assert len(best_params) == len(optimizer.param_names)