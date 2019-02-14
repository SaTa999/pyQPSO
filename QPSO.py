import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from deap import base, benchmarks, creator, tools


def latin_hypercube_sampling(size, n):
    """standard latin hypercube sampling
    size: number of dimensions
    n: number of samples"""
    cut = np.linspace(0, 1, n + 1)   
    u = np.random.rand(n, size)
    a = cut[:n, np.newaxis]
    b = cut[1:n+1, np.newaxis] 
    rdpoints = u * (b - a) + a
    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(size):
        order = np.random.permutation(range(n))
        H[:, j] = rdpoints[order, j]
    return H


def lhs_swarm(size, n, pmin, pmax):
    """generate a swarm that is coposed of particles distributed in
    a latin hypercube"""
    H = latin_hypercube_sampling(size, n)
    H *= (pmax - pmin)
    H += pmin
    def gen_particle(position):
        part = creator.Particle(position)
        part.pmin=pmin
        part.pmax=pmax 
        return part
    population = map(gen_particle, H)
    return population


def generate(size, pmin, pmax, position=None):
    """generate a particle
    size: number of dimensions
    pmin, pmax: lower & upper bound of the position of the particle"""
    part = creator.Particle(np.random.uniform(pmin, pmax, size))
    part.pmin=pmin
    part.pmax=pmax 
    return part


def updateParticle(part, best, meanbest, beta):
    """update the particle's position (type1: standard QPSO)"""
    phi = np.random.random(len(part))
    p = phi * part.best + (1. - phi) * best
    k = np.random.randint(0,2,len(part)) * 2 - 1 # random array of -1 & 1
    u = np.random.random(len(part))
    part[:] = p + k * beta * np.abs(meanbest - part) * np.log(1./u)
    part[:] = np.clip(part, part.pmin, part.pmax)
    return part


def updateParticle2(part, best, meanbest, beta):
    """update the particle's position (type2: standard QPSO)"""
    c1 = np.random.random(len(part))
    c2 = np.random.random(len(part))
    p = (c1 * part.best + c2 * best) / (c1 + c2)
    k = np.random.randint(0,2,len(part)) * 2 - 1 # random array of -1 & 1
    u = np.random.random(len(part))
    part[:] = p + k * beta * np.abs(meanbest - part) * np.log(1./u)
    part[:] = np.clip(part, part.pmin, part.pmax)
    return part


def updateParticle3(part, best, meanbest, beta):
    """update the particle's position (type3: gaussian QPSO by Coelho 2007)"""
    phi = np.random.random(len(part))
    p = phi * part.best + (1. - phi) * best
    k = np.random.randint(0,2,len(part)) * 2 - 1 # random array of -1 & 1
    u = np.abs(np.random.normal(0, 1, len(part)))
    part[:] = p + k * beta * np.abs(meanbest - part) * np.log(1./u)
    part[:] = np.clip(part, part.pmin, part.pmax)
    return part


def updateParticle4(part, best, meanbest, beta):
    """update the particle's position (type4: gaussian QPSO by Sun 2011)
    increases diversity of particle position
    (lower chance of getting trapped in local optimum)"""
    phi = np.random.random(len(part))
    p = phi * part.best + (1. - phi) * best
    k = np.random.randint(0,2,len(part)) * 2 - 1 # random array of -1 & 1
    u = np.random.random(len(part))
    p += np.random.normal(0, 1, len(part)) * np.sqrt(np.abs(meanbest - part.best))
    part[:] = p + k * beta * np.abs(meanbest - part) * np.log(1./u)
    part[:] = np.clip(part, part.pmin, part.pmax)
    return part


INVALID = 0 # record the number of times the evaluation failed
LOWER = 2e-2 # lower bound of particle position
UPPER = 2e2 # upper bound of particle position
def evaluate(part):
    """evaluate function"""
    pass


# register creators
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, 
    pmin=None, pmax=None, best=None)


# register functions to the toolbox
toolbox = base.Toolbox()
toolbox.register("particle", generate, size=None, pmin=LOWER, pmax=UPPER)
# toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("population", lhs_swarm, size=None,
                 pmin=LOWER, pmax=UPPER)
toolbox.register("update", updateParticle)
toolbox.register("evaluate", evaluate)#benchmarks.rosenbrock)


def main():
    NPOP = 100 # number of particles
    pop = toolbox.population(n=NPOP)
    
    # register the functions used to calculate stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("mean", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    # initialize the logbook
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals", "invalidevals"] + stats.fields

    GEN = 40 # number of maximum generations
    best = None
    BETA_INIT = 1.0 # initial value of contraction-expansion coefficient
    BETA_FIN = 0.5 # final value of contraction-expansion coefficient
    """TIPS: large beta -> global search, small beta -> local search
             beta_init: 0.8 to 1.2
             beta_fin : below 0.6
             beta must be below e^gamma=1.781 to guarantee convergence of the particle"""
    betas = np.linspace(BETA_INIT, BETA_FIN, GEN)
    meanbest = np.zeros(len(pop[0])) # initialize the meanbest

    for g, beta in zip(range(GEN), betas):
        meanbest[:] = np.zeros(len(pop[0])) # reinitialize the meanbest
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            # reinitialize the particle if its initial fitness value is nan
            if part.best is None:
                while np.isnan(part.fitness.values):
                    part[:] = generate(size=None, pmin=part.pmin, pmax=part.pmax)
                    part.fitness.values = toolbox.evaluate(part)
            # update the particles best position
            if part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            # update the global best position
            if best is None or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
            meanbest += part.best
        meanbest /= NPOP
        for i, part in enumerate(pop):
            toolbox.update(part, best, meanbest, beta)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop)*(g+1), invalidevals=INVALID, **stats.compile(pop))
        print(logbook.stream)
        if logbook.select("std")[-1] <= 1e-5:
            break
    
    return pop, logbook, best


if __name__ == "__main__":
    pop, logbook, best = main()
    print "end of global optimization"
    print "best particle value is ", best.fitness.values[0]
    print "best particle location is ", best
    print 
    res = minimize(evaluate, best,
                   bounds=(LOWER, UPPER), tol=1e-10)
    print "end of local optimization"
    print "best particle value is ", - res.fun
    print "best particle location is ", res.x


# save logbook
dframe = pd.DataFrame.from_dict(logbook)
dframe.to_csv("logbook.csv", index=False)
# plot the stats
gen = logbook.select("gen")
fit_mins = logbook.select("max")
fit_avgs = logbook.select("mean")
fit_stds = logbook.select("std")


fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, fit_mins, "b-", label="Maximum Fitness")
line2 = ax1.plot(gen, fit_avgs, "r-", label="Mean Fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness (Max & Mean)")

ax2 = ax1.twinx()
ax2.set_ylabel("Fitness (Std)")
line3 = ax2.plot(gen, fit_stds, "g-", label="Std Fitness")

lns = line1 + line2 + line3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="center right")

#     plt.show()
plt.savefig("fitnees_log")
