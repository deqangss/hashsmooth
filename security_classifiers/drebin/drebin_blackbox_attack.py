
from abc import ABC, abstractmethod
import random
import copy
import numpy as np

import torch
import torch.nn.functional as F
from deap import creator, base, tools

EXP_OVER_FLOW = 1e-30


class BlackBoxProblem(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_fitness(self):
        raise NotImplementedError

    @abstractmethod
    def init_starting_point(self):
        raise NotImplementedError


def _slice_sequence(slice_size, sequence, irregular=None):
    if irregular is not None:
        expanded_sequence = [
            sequence[: sum(irregular[:i + 1])] for i, _ in enumerate(irregular)
            # sequence[: sum(irregular[:i])] for i, val in enumerate(irregular)
        ]
    else:
        how_many = len(sequence) // slice_size
        expanded_sequence = [
            sequence[: (i + 1) * slice_size]
            for i in range(how_many)
        ]
    # expanded_sequence = np.array(expanded_sequence)
    return expanded_sequence


def _pad_sequence_with_last(sequence, until):
    how_many = until - len(sequence)
    if how_many <= 0:
        return sequence
    return sequence + [sequence[-1]] * how_many


class EABlackBoxEvasionProblem(BlackBoxProblem):
    def __init__(self,
                 classifier,
                 input_dim: int,
                 population_size: int,
                 penalty_regularizer: float,
                 constraints: tuple,
                 iterations: int,
                 stagnation: int,
                 benign_samples_as_seed: np.ndarray,
                 seed: int = 23456,
                 device='cpu'
                 ):
        super(EABlackBoxEvasionProblem, self).__init__()
        classifier.eval()
        self.classifier = classifier
        self.input_dim = input_dim
        self.population_size = population_size
        self.penalty_regularizer = penalty_regularizer
        self.insertion_perm, self.removal_perm = constraints
        self.iterations = iterations
        self.stagnation = stagnation
        self.benign_samples_as_seed = benign_samples_as_seed
        self.device = device

        random.seed(seed)
        np.random.seed(seed)

        # other intermediates
        self.original_x = None
        self.initial_adv_x = None
        self.confidences_ = []
        self.fitness_ = []
        self.sizes_ = []

    def clear_results(self):
        """
        Reset the internal state after computing an attack
        """
        self.confidences_ = []
        self.fitness_ = []
        self.sizes_ = []
        self.advx = []

    def init_starting_point(self, x: np.ndarray):
        self.original_x = x.copy()
        self.clear_results()
        lower_bound, upper_bound = self.get_bounds()

        adv_x_init = np.clip(self.original_x[None, ...] + self.benign_samples_as_seed, lower_bound, upper_bound)
        _conf = self.classifier.get_confidence(
            torch.from_numpy(adv_x_init).to(self.device).float()).detach().cpu().numpy()
        if _conf.shape[1] == 1:
            _conf = _conf[:, 0]
        elif _conf.shape[1] == 2:
            _conf = _conf[:, 1]  # targeted attack
        else:
            raise ValueError

        _toward_success = _conf < 0.5
        if np.any(_toward_success):
            diff_pertb_init = np.sum(np.abs(adv_x_init - x[None, ...]), axis=-1)[_toward_success]
            idx = diff_pertb_init.argsort()[0]
            self.init_adv_x = adv_x_init[_toward_success][idx]
        else:
            self.init_adv_x = adv_x_init[_conf.argsort()[0]]

    def get_bounds(self):
        # 1 & 0 -> 1; 1 & 1 -> 1; 0 & 1 -> 1; 0 & 0 -> 0
        upper_bound = np.maximum(self.original_x, self.insertion_perm)
        # 1 & 0 -> 0; 1 & 1 -> 0; 0 & 1 -> 1; 0 & 0 -> 0
        lower_bound = np.zeros_like(self.original_x)
        unmodif_flag = self.removal_perm == 0
        lower_bound[unmodif_flag] = self.original_x[unmodif_flag]
        return lower_bound, upper_bound

    def get_fitness(self, individual: np.ndarray):
        x_constrained = self.get_adv_x(individual)
        x_constrained_tensor = torch.from_numpy(x_constrained[None, ...]).to(self.device).float()
        confidence = self.classifier.get_confidence(x_constrained_tensor)
        if confidence.shape[1] == 1:
            confidence = confidence[0, 0]
        elif confidence.shape[1] == 2:
            confidence = confidence[0, 1]  # targeted attack
        else:
            raise ValueError
        l1_norm = torch.sum(
            torch.abs(x_constrained_tensor - torch.from_numpy(self.original_x).to(self.device).float()))

        # print(confidence, '---', l1_norm)
        fitness_value = confidence + self.penalty_regularizer * l1_norm
        # + self.penalty_regularizer * l1_norm
        self.confidences_.append(confidence.data.item())
        self.sizes_.append(l1_norm.data.item())
        self.fitness_.append(fitness_value.data.item())
        return [fitness_value.data.item()]

    def get_adv_x(self, individual):
        lower_bound, upper_bound = self.get_bounds()
        return np.clip(np.array(individual) + self.init_adv_x, lower_bound, upper_bound)

    def _export_internal_results(self, irregular=None) -> (list, list, list):
        """
        Exports the results of the attack

        Parameters
        ----------
        irregular : list, optional, default None
            Slices the internal results
        Returns
        -------

        """
        confidence = _slice_sequence(self.population_size, self.confidences_[1:], irregular)
        fitness = _slice_sequence(self.population_size, self.fitness_[1:], irregular)
        sizes = _slice_sequence(self.population_size, self.sizes_[1:], irregular)
        best_idx = [np.argmin(f) for f in fitness]
        fitness = [self.fitness_[0]] + [f[i] for f, i in zip(fitness, best_idx)]
        confidence = [self.confidences_[0]] + [f[i] for f, i in zip(confidence, best_idx)]
        sizes = [self.sizes_[0]] + [f[i] for f, i in zip(sizes, best_idx)]

        return confidence, fitness, sizes


class EvolutionAA(object):
    """
    Evolution algorithm attack
    """

    def __init__(self, attack_problem: BlackBoxProblem,
                 cx_prob: float = 0.5,
                 mut_prob: float = 0.3,
                 individual_flip_prob: float = 0.01,
                 tour_selection_k : int = 100,
                 n_repetition: int = 5,
                 ):
        self.attack_problem = attack_problem
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.individual_flip_prob = individual_flip_prob
        self.tour_selection_k = tour_selection_k
        self.n_repetition = n_repetition

    def perturb(self, x:np.array, label, verbose=True):
        """
        perturb feature vectors
        """
        if x is None or x.shape[0] <= 0:
            return []
        org_pred = self.attack_problem.classifier.predict(
            torch.from_numpy(x[None, ...]).to(self.attack_problem.device).float())
        if hasattr(self.attack_problem.classifier, 'ABSTAIN'):
            done = org_pred != label & org_pred != -1
        else:
            done = org_pred != label
        if torch.all(done):
            print("Attack success trivially")
            return x

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        # toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("attr_bool", np.random.choice, np.arange(2), None, True, [0.995, 0.005])
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.attack_problem.input_dim)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register('evaluate', self.attack_problem.get_fitness)
        toolbox.register('mate', tools.cxTwoPoint)
        toolbox.register('mutate', tools.mutFlipBit, indpb=self.individual_flip_prob)
        toolbox.register('select', tools.selTournament, tournsize=self.tour_selection_k)

        stats_res = tools.Statistics()
        stats_res.register('Min', np.min)

        adv_x = x.copy()
        hall_of_fame = tools.HallOfFame(1)
        for idx in range(self.n_repetition):
            self.attack_problem.init_starting_point(x)
            pop = toolbox.population(n=self.attack_problem.population_size)
            fitness = list(map(toolbox.evaluate, pop))
            slice_indices = [self.attack_problem.population_size]
            for ind, fit in zip(pop, fitness):
                ind.fitness.values = fit

            g, last_n_best_fits, all_best_ind = 0, [], []
            hall_of_fame.clear()
            all_best_ind.clear()
            while g < self.attack_problem.iterations:
                g += 1

                # select individuals using the selection strategy
                offspring = toolbox.select(pop, self.attack_problem.population_size)

                # clone the selected individuals in case of prohibiting the inplace modification
                offspring = list(map(toolbox.clone, offspring))

                # apply cross-over
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.cx_prob:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                for mutant in offspring:
                    if random.random() < self.mut_prob:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # re-evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                slice_indices.append(len(invalid_ind))
                fitness = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitness):
                    ind.fitness.values = fit

                this_gen_fitness = []
                for ind in offspring:
                    this_gen_fitness.append(ind.fitness.values[0])
                stats_of_this_gen = stats_res.compile(this_gen_fitness)
                stats_of_this_gen['Generation'] = g
                if verbose:
                    print(stats_of_this_gen)

                hall_of_fame.update(offspring)

                pop[:] = offspring

                # pop.extend(invalid_ind)
                all_best_ind.append(hall_of_fame[0])
                # print(np.sum(np.array(all_best_ind), axis=-1))

                fits = [ind.fitness.values[0] for ind in pop]
                best_fitness = min(fits)
                last_n_best_fits.insert(0, best_fitness)
                last_n_best_fits = last_n_best_fits[:self.attack_problem.stagnation]
                if len(last_n_best_fits) == self.attack_problem.stagnation and (
                        all(abs(np.array(last_n_best_fits) - best_fitness) < 1e-8) or all(
                    np.array(last_n_best_fits) == np.infty)):
                    print('Stagnating terminate!')
                    break
            # confidences, fitness, sizes = self.attack_problem._export_internal_results(slice_indices)
            # self.attack_problem.confidences_ = _pad_sequence_with_last(confidences, self.attack_problem.iterations)
            # self.attack_problem.fitness_ = _pad_sequence_with_last(fitness, self.attack_problem.iterations)
            # self.attack_problem.sizes_ = _pad_sequence_with_last(sizes, self.attack_problem.iterations)
            best_t = tools.selBest(all_best_ind, 1)[0]
            adv_x = self.attack_problem.get_adv_x(best_t)
            adv_pred = self.attack_problem.classifier.predict(
                torch.from_numpy(adv_x[None, ...]).to(self.attack_problem.device).float())
            if hasattr(self.attack_problem.classifier, 'ABSTAIN'):
                print("adv_pred:", adv_pred)
                done = adv_pred != label & adv_pred != -1
            else:
                done = adv_pred != label
            if torch.all(done):
                print("Attack success!")
                break
        del creator.FitnessMin
        del creator.Individual
        return adv_x


