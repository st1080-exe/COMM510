#builds the tables for the numerical experiments section
#
#usage:
#   python makeTables.py results.csv
#
#writes each table as a .tex file (paste into latex with \input{...})
#and a .csv file (open in excel/word and copy across)

import sys

import pandas as pd
from scipy.stats import ranksums

import moeadRf

# set these to what you actually used
evalBudget = 100000
numCandsUsed = 5

symbolsLatex = {"+": "+", "-": "$-$", "≈": "$\\approx$"}


def formatValue(mean, std):

    return "%.2e (%.1e)" % (mean, std)


def writeTable(header, csvRows, latexRows, fileName, caption, label):

    # csv version for excel or word
    pd.DataFrame(csvRows, columns=header).to_csv(fileName + ".csv", index=False)

    # latex version
    lines = ["\\begin{table}[ht]",
             "\\centering",
             "\\caption{" + caption + "}",
             "\\label{" + label + "}",
             "\\begin{tabular}{" + "l" * len(header) + "}",
             "\\hline",
             " & ".join(header) + " \\\\",
             "\\hline"]
    for row in latexRows:
        lines.append(" & ".join(str(cell) for cell in row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]

    with open(fileName + ".tex", "w") as outFile:
        outFile.write("\n".join(lines))
    print("wrote", fileName + ".tex and", fileName + ".csv")


def buildProblemTable():

    # check these against deb et al. (2005) before submitting
    rows = [["DTLZ3", "Multimodal, many local fronts", "Concave (spherical)"],
            ["DTLZ4", "Biased solution density", "Concave (spherical)"],
            ["DTLZ6", "Biased variable-to-objective mapping", "Degenerate curve"],
            ["DTLZ7", "Many disconnected regions", "Disconnected"]]
    header = ["Problem", "Main difficulty", "Pareto front"]
    caption = ("Test problems. Each uses D decision variables and M objectives, "
               "with k = D - M + 1 distance variables.")
    writeTable(header, rows, rows, "tableProblems", caption, "tab:problems")


def buildParamTable(results):

    runsTotal = results.groupby(["problem", "numVars", "numObjs", "algo"]).run.nunique().max()
    rows = [["Population size N", "91 / 210 / 220 (M = 3 / 5 / 10)"],
            ["Evaluation budget", str(evalBudget)],
            ["Independent runs", str(runsTotal)],
            ["Neighbourhood size T", str(moeadRf.neighbourSize)],
            ["Neighbour selection probability", str(moeadRf.neighbourProb)],
            ["Max replacements per child", str(moeadRf.maxReplace)],
            ["DE scaling factor", str(moeadRf.diffWeight)],
            ["DE crossover rate", str(moeadRf.crossRate)],
            ["Mutation distribution index", str(moeadRf.mutDist)],
            ["Mutation probability", "1/D"],
            ["Candidates per subproblem (MOEA/D-RF)", str(numCandsUsed)],
            ["Random forest trees", str(moeadRf.numTrees)],
            ["Retraining interval (generations)", str(moeadRf.retrainEvery)],
            ["Training window", str(moeadRf.windowMult) + "N samples"],
            ["Exploration rate", str(moeadRf.exploreRate)],
            ["Minimum training samples", str(moeadRf.minTrain)]]
    header = ["Parameter", "Value"]
    caption = "Parameter settings used for MOEA/D and MOEA/D-RF."
    writeTable(header, rows, rows, "tableParams", caption, "tab:params")


def buildResultsTable(results):

    csvRows = []
    latexRows = []
    tally = {"+": 0, "-": 0, "≈": 0}

    for name in moeadRf.problemNames:
        for v in moeadRf.varCounts:
            for o in moeadRf.objCounts:
                label = name.upper()
                group = results[(results.problem == name)
                                & (results.numVars == v)
                                & (results.numObjs == o)]

                # undefined combination (k < 1)
                if(len(group) == 0):
                    csvRows.append([label, v, o, "N/A", "N/A"])
                    latexRows.append([label, v, o, "N/A", "N/A"])
                    continue

                plainIgd = group[group.algo == "plain"].igd.values
                rfIgd = group[group.algo == "rf"].igd.values

                # wilcoxon rank-sum, rf compared with plain
                pValue = ranksums(rfIgd, plainIgd)[1]
                symbol = "≈"
                if(pValue < 0.05):
                    if(rfIgd.mean() < plainIgd.mean()):
                        symbol = "+"
                    else:
                        symbol = "-"
                tally[symbol] += 1

                plainText = formatValue(plainIgd.mean(), plainIgd.std())
                rfText = formatValue(rfIgd.mean(), rfIgd.std())
                csvRows.append([label, v, o, plainText, rfText + " " + symbol])

                # bold the better mean in the latex version
                plainLatex = plainText
                rfLatex = rfText
                if(rfIgd.mean() < plainIgd.mean()):
                    rfLatex = "\\textbf{" + rfText + "}"
                else:
                    plainLatex = "\\textbf{" + plainText + "}"
                latexRows.append([label, v, o, plainLatex, rfLatex + " " + symbolsLatex[symbol]])

    header = ["Problem", "D", "M", "MOEA/D", "MOEA/D-RF"]
    caption = ("Mean (std) IGD over independent runs. Bold is the lower mean. "
               "Symbols mark the Wilcoxon rank-sum test at the 0.05 level for MOEA/D-RF "
               "against MOEA/D: + better, $-$ worse, $\\approx$ no significant difference. "
               "Summary (+/$-$/$\\approx$): " + str(tally["+"]) + "/" + str(tally["-"])
               + "/" + str(tally["≈"]) + ". N/A: undefined for D < M.")
    writeTable(header, csvRows, latexRows, "tableResults", caption, "tab:results")
    print("rf vs plain (+/-/≈):", tally)


def buildOverheadTable(results):

    rows = []
    for (v, o), group in results.groupby(["numVars", "numObjs"]):
        plain = group[group.algo == "plain"]
        rf = group[group.algo == "rf"]
        share = 100 * rf.trainTime.mean() / rf.time.mean()
        rows.append([v, o,
                     "%.1f" % plain.time.mean(),
                     "%.1f" % rf.time.mean(),
                     "%.1f" % rf.trainTime.mean(),
                     "%.0f" % share,
                     "%.0f" % rf.numFits.mean()])

    header = ["D", "M", "MOEA/D time (s)", "MOEA/D-RF time (s)",
              "RF training (s)", "Training share", "RF fits"]
    caption = ("Mean runtime per run, averaged over problems. Training share is the "
               "percentage of MOEA/D-RF runtime spent training the classifier.")
    writeTable(header, rows, rows, "tableOverhead", caption, "tab:overhead")


if(__name__ == "__main__"):

    results = pd.read_csv(sys.argv[1])

    buildProblemTable()
    buildParamTable(results)
    buildResultsTable(results)
    buildOverheadTable(results)