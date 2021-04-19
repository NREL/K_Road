from coodinated_optimizer.canonical_es_coordinated_optimizer import CanonicalESCoordinatedOptimizer
from coodinated_optimizer.cem_coordinated_optimizer import CEMCoordinatedOptimizer
from coodinated_optimizer.es_balanced_rl_coordinated_optimizer import ESBalancedRLCoordinatedOptimizer
from coodinated_optimizer.es_best_coordinated_optimizer import ESBestCoordinatedOptimizer
from coodinated_optimizer.es_best_spread_coordinated_optimizer import ESBestSpreadCoordinatedOptimizer
from coodinated_optimizer.es_ev_rl_coordinated_optimizer import ESEVRLCoordinatedOptimizer
from coodinated_optimizer.es_top_n_coordinated_optimizer import ESTopNCoordinatedOptimizer
from coodinated_optimizer.esrl_coordinated_optimizer import ESRLCoordinatedOptimizer
from coodinated_optimizer.ncem_coordinated_optimizer import NCEMCoordinatedOptimizer
from coodinated_optimizer.wcem_coordinated_optimizer import WCEMCoordinatedOptimizer

factory_table = {
    'ESRLCoordinatedOptimizer':         ESRLCoordinatedOptimizer,
    'CEMCoordinatedOptimizer':          CEMCoordinatedOptimizer,
    'WCEMCoordinatedOptimizer':         WCEMCoordinatedOptimizer,
    'NCEMCoordinatedOptimizer':         NCEMCoordinatedOptimizer,
    'CanonicalESCoordinatedOptimizer':  CanonicalESCoordinatedOptimizer,
    'ESBalancedRLCoordinatedOptimizer': ESBalancedRLCoordinatedOptimizer,
    'ESEVRLCoordinatedOptimizer':       ESEVRLCoordinatedOptimizer,
    'ESTopNCoordinatedOptimizer':       ESTopNCoordinatedOptimizer,
    'ESBestCoordinatedOptimizer':       ESBestCoordinatedOptimizer,
    'ESBestSpreadCoordinatedOptimizer': ESBestSpreadCoordinatedOptimizer,
    }


def get_optimizer_factory(name: str):
    optimizer_class = factory_table[name]
    return lambda *args, **kwargs: optimizer_class(*args, **kwargs)
