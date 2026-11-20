import argparse
import itertools
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.stats import ranksums
from sklearn.ensemble import RandomForestClassifier

from pymoo.indicators.igd import IGD
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

# vars from assignment brief
problemNames = ["dtlz3", "dtlz4", "dtlz6", "dtlz7"]
varCounts = [5, 10, 100]
objCounts = [3, 5, 10]

# das-dennis partitions for the population (91 / 210 / 220)
popPartitions = {3: 12, 5: 6, 10: 3}
# denser set only used to build the igd reference front
igdPartitions = {3: 30, 5: 10, 10: 4}

# fixed algorithm settings
neighbourSize = 20
neighbourProb = 0.9
maxReplace = 2
diffWeight = 0.5
crossRate = 1.0
mutDist = 20
numTrees = 50
retrainEvery = 5
windowMult = 10
exploreRate = 0.1
minTrain = 100


def tchebycheff(objs, wts, ideal):

    return np.max(wts * np.abs(objs - ideal), axis=-1)


def polyMutation(vars):

    # simplified polynomial mutation on [0,1] variables
    r = rng.random(vars.shape)
    mutate = rng.random(vars.shape) < (1.0 / numVars)
    shift = np.where(r < 0.5,
                     (2 * r) ** (1 / (mutDist + 1)) - 1,
                     1 - (2 * (1 - r)) ** (1 / (mutDist + 1)))

    return np.clip(vars + mutate * shift, 0.0, 1.0)


def getRefFront():

    # tries a few call styles since pymoo versions differ
    dirs = get_reference_directions("das-dennis", numObjs,
                                    n_partitions=igdPartitions[numObjs])
    attempts = [lambda: problem.pareto_front(dirs),
                lambda: problem.pareto_front(ref_dirs=dirs),
                lambda: problem.pareto_front()]
    for attempt in attempts:
        try:
            front = attempt()
            if(front is not None):
                return front
        except Exception:
            continue

    raise RuntimeError("no pareto front available for " + str(problem))


def init(problemName, varsTotal, objsTotal, evalsTotal, modeIn, candsTotal, seed):

    global problem, numVars, numObjs, maxEvals, mode, numCands, candsPerSub
    global rng, weights, popSize, neighbours, refFront
    global pop, popObjs, idealPoint, evalsUsed, genNum
    global classifier, trainX, trainY, trainTime, numFits

    numVars = varsTotal
    numObjs = objsTotal
    maxEvals = evalsTotal
    mode = modeIn
    numCands = candsTotal
    rng = np.random.default_rng(seed)

    # plain moead makes one child per subproblem, rf makes several
    if(mode == "rf"):
        candsPerSub = numCands
    else:
        candsPerSub = 1

    # pymoo sets k = numVars - numObjs + 1
    problem = get_problem(problemName, n_var=numVars, n_obj=numObjs)
    refFront = getRefFront()

    # weight vectors and neighbourhoods
    weights = get_reference_directions("das-dennis", numObjs,
                                       n_partitions=popPartitions[numObjs])
    popSize = len(weights)
    weights = np.maximum(weights, 1e-6)
    neighbours = np.argsort(cdist(weights, weights), axis=1)[:, :neighbourSize]

    # random starting population
    pop = rng.random((popSize, numVars))
    popObjs = problem.evaluate(pop, return_values_of=["F"])
    idealPoint = popObjs.min(axis=0)
    evalsUsed = popSize
    genNum = 0

    # classifier state
    classifier = None
    trainX = np.empty((0, numVars + numObjs))
    trainY = np.empty(0, dtype=int)
    trainTime = 0.0
    numFits = 0


def generateCandidates():

    global candidates, genNum

    genNum += 1
    rows = np.arange(popSize)

    # picks three parents per candidate, from the neighbourhood or the whole pop
    useNeighbours = rng.random((popSize, candsPerSub, 1)) < neighbourProb
    fromNeighbours = neighbours[rows[:, None, None],
                                rng.integers(0, neighbourSize, (popSize, candsPerSub, 3))]
    fromAll = rng.integers(0, popSize, (popSize, candsPerSub, 3))
    parents = np.where(useNeighbours, fromNeighbours, fromAll)

    # de/rand/1 mutant
    mutant = pop[parents[..., 0]] + diffWeight * (pop[parents[..., 1]] - pop[parents[..., 2]])

    # binomial crossover with the subproblem's current solution
    cross = rng.random((popSize, candsPerSub, numVars)) < crossRate
    forced = rng.integers(0, numVars, (popSize, candsPerSub))
    cross[rows[:, None], np.arange(candsPerSub)[None, :], forced] = True
    trial = np.where(cross, mutant, pop[:, None, :])

    candidates = polyMutation(np.clip(trial, 0, 1))


def preselect():

    global children

    rows = np.arange(popSize)

    # classifier picks the candidate most likely to improve its subproblem
    if(mode == "rf" and classifier is not None):
        features = np.hstack([candidates.reshape(popSize * candsPerSub, numVars),
                              np.repeat(weights, candsPerSub, axis=0)])
        goodCol = list(classifier.classes_).index(1)
        goodProb = classifier.predict_proba(features)[:, goodCol]
        goodProb = goodProb.reshape(popSize, candsPerSub)
        pick = goodProb.argmax(axis=1)

        # some random picks so the training data is not only the classifier's choices
        explore = rng.random(popSize) < exploreRate
        pick[explore] = rng.integers(0, candsPerSub, explore.sum())
    # else: random candidate (only one exists in plain mode)
    else:
        pick = rng.integers(0, candsPerSub, popSize)

    children = candidates[rows, pick]


def evaluateChildren():

    global childObjs, idealPoint, evalsUsed

    childObjs = problem.evaluate(children, return_values_of=["F"])
    evalsUsed += popSize
    idealPoint = np.minimum(idealPoint, childObjs.min(axis=0))


def recordLabels():

    global trainX, trainY

    if(mode != "rf"):
        return

    # label is 1 if the child beat its parent on that subproblem
    label = (tchebycheff(childObjs, weights, idealPoint)
             < tchebycheff(popObjs, weights, idealPoint)).astype(int)

    # sliding window of recent evaluations
    window = windowMult * popSize
    trainX = np.vstack([trainX, np.hstack([children, weights])])[-window:]
    trainY = np.concatenate([trainY, label])[-window:]


def updatePop():

    # sequential neighbourhood update, same as standard moead
    for i in rng.permutation(popSize):
        if(rng.random() < neighbourProb):
            pool = neighbours[i]
        else:
            pool = np.arange(popSize)
        pool = rng.permutation(pool)

        childScore = tchebycheff(childObjs[i], weights[pool], idealPoint)
        oldScore = tchebycheff(popObjs[pool], weights[pool], idealPoint)
        better = np.where(childScore <= oldScore)[0][:maxReplace]

        pop[pool[better]] = children[i]
        popObjs[pool[better]] = childObjs[i]


def trainClassifier():

    global classifier, trainTime, numFits

    # only retrains every few gens and once both classes have been seen
    if(mode == "rf" and genNum % retrainEvery == 0
            and len(trainY) >= minTrain and trainY.min() != trainY.max()):
        startTime = time.perf_counter()
        classifier = RandomForestClassifier(n_estimators=numTrees,
                                            min_samples_leaf=5,
                                            class_weight="balanced",
                                            n_jobs=1)
        classifier.fit(trainX, trainY)
        trainTime += time.perf_counter() - startTime
        numFits += 1


def report():

    positiveRate = np.nan
    if(len(trainY) > 0):
        positiveRate = float(trainY.mean())

    return {"igd": float(IGD(refFront)(popObjs)),
            "gens": genNum,
            "trainTime": trainTime,
            "numFits": numFits,
            "positiveRate": positiveRate}


def main(problemName, varsTotal, objsTotal, evalsTotal, modeIn, candsTotal, seed):

    # sets up variables
    init(problemName, varsTotal, objsTotal, evalsTotal, modeIn, candsTotal, seed)

    # runs until the evaluation budget is used up
    while(evalsUsed + popSize <= maxEvals):

        # makes candidate offspring for every subproblem
        generateCandidates()

        # picks one candidate per subproblem
        preselect()

        # true evaluation of the chosen children, updates ideal point
        evaluateChildren()

        # stores good/bad labels for the classifier
        recordLabels()

        # replaces worse neighbours with the children
        updatePop()

        # retrains the classifier if it is due
        trainClassifier()

    return report()


def runOne(problemName, varsTotal, objsTotal, modeIn, seed, evalsTotal, candsTotal):

    startTime = time.perf_counter()
    results = main(problemName, varsTotal, objsTotal, evalsTotal, modeIn, candsTotal, seed)
    results.update({"problem": problemName, "numVars": varsTotal,
                    "numObjs": objsTotal, "algo": modeIn, "run": seed,
                    "time": time.perf_counter() - startTime})

    return results


def runAll(runsTotal, evalsTotal, jobs, candsTotal, outFile, quick):

    # numVars < numObjs gives k < 1 so those combinations are skipped
    configs = [(p, v, o) for p, v, o in itertools.product(problemNames, varCounts, objCounts)
               if(v >= o)]
    if(quick):
        configs = [("dtlz3", 10, 3)]

    tasks = [(p, v, o, m, s, evalsTotal, candsTotal)
             for (p, v, o) in configs for m in ("plain", "rf")
             for s in range(runsTotal)]
    print(len(configs), "valid configs,", len(tasks), "runs")

    # multiprocessing backend so each worker keeps its own copy of the globals
    rows = Parallel(n_jobs=jobs, backend="multiprocessing", verbose=5)(
        delayed(runOne)(*task) for task in tasks)
    results = pd.DataFrame(rows)
    results.to_csv(outFile, index=False)
    print("saved", outFile)

    return results


def summarise(results):

    # mean (std) igd with wilcoxon rank-sum symbol for rf vs plain
    tableRows = []
    tally = {"+": 0, "-": 0, "≈": 0}
    for (name, v, o), group in results.groupby(["problem", "numVars", "numObjs"]):
        plainIgd = group[group.algo == "plain"].igd.values
        rfIgd = group[group.algo == "rf"].igd.values
        pValue = ranksums(rfIgd, plainIgd)[1]

        symbol = "≈"
        if(pValue < 0.05):
            if(rfIgd.mean() < plainIgd.mean()):
                symbol = "+"
            else:
                symbol = "-"
        tally[symbol] += 1

        tableRows.append({"problem": name, "numVars": v, "numObjs": o,
                          "plain": "%.3e (%.1e)" % (plainIgd.mean(), plainIgd.std()),
                          "rf": "%.3e (%.1e)" % (rfIgd.mean(), rfIgd.std()),
                          "rfVsPlain": symbol,
                          "rfTime": group[group.algo == "rf"].time.mean(),
                          "plainTime": group[group.algo == "plain"].time.mean()})

    table = pd.DataFrame(tableRows)
    print(table.to_string(index=False))
    print("rf vs plain (+/-/≈):", tally)

    return table


if(__name__ == "__main__"):

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--evals", type=int, default=100000)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--cands", type=int, default=5)
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--summarise", metavar="CSV")
    args = parser.parse_args()

    if(args.summarise):
        summarise(pd.read_csv(args.summarise))
    else:
        if(args.quick):
            args.runs = 3
            args.evals = 10000
        summarise(runAll(args.runs, args.evals, args.jobs, args.cands, args.out, args.quick))