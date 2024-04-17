import numpy as np
import sympy
# import random
from scipy.stats import bernoulli
from fractions import Fraction
import matplotlib.pyplot as plt
import math
import time
import sys
from sympy.solvers import solve
from sympy import Symbol, nsolve


def Pr_function(x):
    f = decreasingfunction(x)
    orc = bernoulli.rvs(f)
    return orc


def decreasingfunction(x):
    #f = 1 - 0.8 * x
    #f = 0.5**x
    #f = 0.5 * (1 + math.sin(math.pi*2*x))
    #f = (math.e**x)/(1 + math.e**x)
    #f = (x**2) / (1 + x**2)
    #f = 1 - 1 / (1 + math.exp(-20 * (x-0.5)))
    f = 0.01**x
    #f = 1 - 0.8 * x
    return f


def decreasingfunctionTarget(x):
    f = 1 - 0.2 * x
    return f

def reducedfraction(d):
    # function that converts a rational number
    # to the reduced fraction
    b = Fraction(d).limit_denominator(100000)
    # b = d.as_integer_ratio()
    # reduced the list that contains the fraction
    return b.as_integer_ratio()
    # return d.as_integer_ratio()

def AgentA(N, loc, orc, Locations, step_up, step_down):

    # step_up=1
    # step_down=4
    # target=0.8, 1-target=0.2=1/4. 1 up, 4 down.
    # print("Up", step_up)
    # print("Down", step_down)
    Loc = Locations.index(loc)
    new_loc = loc

    if orc == 1 and Loc < N - step_up - 1:  # will change later
        # if (bernoulli.rvs(1-target_prob)==1):
        new_loc = Locations[Loc + step_up]
    elif orc == 0 and Loc > step_down + 1:
        # if (bernoulli.rvs(target_prob)==1):
        # print(Loc-step_down)
        new_loc = Locations[Loc - step_down]
    else:
        new_loc = loc
    return new_loc


def AgentR(N, loc, orc, Locations, target_prob):
    Loc = Locations.index(loc)
    new_loc = loc
    if orc == 1 and loc < 1 - 1 / N:  # will change later
        if (bernoulli.rvs(1 - target_prob) == 1):
            new_loc = Locations[Loc + 1]
    elif orc == 0 and loc > 1 / N:
        if (bernoulli.rvs(target_prob) == 1):
            new_loc = Locations[Loc - 1]
    else:
        new_loc = loc
    return new_loc


def AgentNoTarget(N, loc, orc, Locations):
    Loc = Locations.index(loc)
    new_loc = loc
    if orc == 1 and loc < 1 - 1 / N:
        new_loc = Locations[Loc + 1]
    elif orc == 0 and loc > 1 / N:
        new_loc = Locations[Loc - 1]
    else:
        new_loc = loc
    return new_loc

def main():
    # a=1
    # b=1
    N = 2048 #sys.argv[1]
    t = 15000 #sys.argv[2]
    nb_experiments = 100 #sys.argv[4]
    target = True
    target_prob = 0.8  # sys.argv[3]
    asymmetric = True
    iterationPlot = 50
    x = Symbol('x')
    ##function = 1 - 1 / (1 + sympy.exp(-20 * (x-0.5)))
    function = 0.01**x
    #function = 1 - 0.8 * x

    Locations = [i for i in np.linspace(0, 1, num=N)]


    loc = Locations[N // 2]

    horizon = t / 10
    lastiterations = t - horizon
    ff = "F1"
    type = ""
    if asymmetric:
        type = "Asymmetric"
    else:
        type = "Randomized"
    """
    if (target_prob>0.5):
        step_up=ratio[0]
        step_down=ratio[1]-ratio[0]
    else:
        step_up=ratio[1]
        step_down=ratio[0]-ratio[1]
    """
    if target:
        #lamdaOpt = math.log(target_prob,0.5)
        #lamdaOpt = math.asin(target_prob*2 -1) / 360
        #s = solve(0.5 * (1 + sympy.sin(math.pi*2*x)) - target_prob)

        s = solve(function - target_prob)
        lamdaOpt = s[0]
        #lamdaOpt = -1/20*(math.log(target_prob/(1-target_prob), math.e)) +0.5

    else:
        s = solve(function - 0.5)
        lamdaOpt = s[0]

    experiment = np.zeros((nb_experiments, t))
    errors = np.zeros((nb_experiments))

    for j in range(nb_experiments):
        start = time.process_time()
        loc = Locations[N // 2]
        #loc = Locations[0]
        #loc = random.choice(Locations)
        print("------------------------ EXP", j + 1, " ---------------------------")
        error = 0
        Chain = np.zeros((t))
        avgposition = loc
        avgposition2 = 0
        ratio = reducedfraction(target_prob)
        step_down = ratio[0]
        step_up = ratio[1] - ratio[0]
        print("Ratio ", ratio[0], "/", ratio[1])
        print("Lamda* = ", lamdaOpt)


        print("steps up = ", step_up)
        print("steps down = ", step_down)
        for i in range(t):
            Chain[i] = loc
            orc = Pr_function(loc)
            # print("orc = ", orc, " location = ", loc * N)
            if target:
                if asymmetric:
                    new_loc = AgentA(N, loc, orc, Locations, step_up, step_down)
                else:
                    new_loc = AgentR(N, loc, orc, Locations, target_prob)
            else:
                new_loc = AgentNoTarget(N, loc, orc, Locations)
            loc = new_loc
            # print(avgposition)
            avgposition2 = avgposition2 + loc

            if i > lastiterations:
                avgposition = avgposition + loc
                error = error + (lamdaOpt - loc)**2
            #error = error + (lamdaOpt - loc) ** 2
        # print(Chain)
        print("our aim is this target success prob:", target_prob)
        print("Average (only last 10% locations):", avgposition * 1.0 / horizon)
        print("we converge to (average only last 10% locations):", decreasingfunction(avgposition * 1.0 / horizon))
        experiment[j] = Chain
        errors[j] = error * 1.0/horizon
        #errors[j] = error * 1.0 / t
        print("MSE = ", errors[j])
        print("Time = ", time.process_time() - start)

    Avgs = np.zeros((t))
    for i in range(t):
        avg = 0.0
        for j in range(nb_experiments):
            avg = avg + experiment[j][i]
        Avgs[i] = avg * 1.0 / nb_experiments



    #print(Avgs)

    Avgs10 = np.zeros((t// iterationPlot))
    j = 0
    for i in range(0, t,  iterationPlot):
        Avgs10[j] = Avgs[i]
        j= j +1

    #plt.rcParams["figure.figsize"] = [7.50, 3.50]
    #plt.rcParams["figure.autolayout"] = True

    arrayT =  list(range(1, t+1))
    arrayT10 = list(range(1, t// iterationPlot+1))
    j = 0
    for i in range(0, t, iterationPlot):
        arrayT10[j] = arrayT[i]
        j = j + 1
    #x = np.array(arrayT)
    #y = np.array(Avgs)

    x = np.array(arrayT10)
    y = np.array(Avgs10)
    ax = plt.gca()
    #ax.set_yscale('log')
    #ax.set_ylim([0.22, 0.28])
    s = ""
    if target:
        s = "(Target = " + str((target_prob*100).__int__()) +"% , N = " + str(N) + ")"
    else:
        s = "N = " + str(N)
    plt.title(s)
    plt.xlabel("Time")
    plt.ylabel("λ")

    plt.plot(x, y, color="blue", marker="v", markevery= 0.2)
    plt.axhline(y=lamdaOpt, color='r', linestyle='-', label= r'$\lambda^{*}$')

    plt.legend()
    #s = "N" + str(N) + "T" + str(t) + "R" + str(nb_experiments) + "S" + str((target_prob*100).__int__())
    if target:
        s = "N" + str(N) + "T" + str(t) + "R" + str(nb_experiments) + "S" + str((target_prob*100).__int__()) + "Plot" + str(iterationPlot) + "Iterations" + type
        s = ff + str((target_prob*100).__int__()) + type
    else:
        s = "N" + str(N) + "T" + str(t) + "R" + str(nb_experiments) + "S" + str(50) + "Plot" + str(iterationPlot) + "Iterations" + type
        s = ff + "NoTarget"
    plt.savefig(s+".pdf")
    int_array = np.rint(arrayT)
    np.savetxt(s + ".csv", Avgs10, delimiter=',')
    np.savetxt("Errors" + s + ".csv", errors, delimiter=',')

    plt.show()

    """
    print("------------------------ All ---------------------------")
    print("Average (All locations):", avgposition2 * 1.0 / t)
    print("we converge to (average all locations):", decreasingfunction(avgposition2 * 1.0 / t))
    """
    # return Chain


main()
