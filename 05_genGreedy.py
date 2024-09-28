import sys
import re
import copy
import time
import math

# eg: python 05_genGreedy.py 3 m B

idx = int(sys.argv[1])
problem_size = sys.argv[2]
rule = sys.argv[3]

stime = time.time()
if problem_size == "s":
    f = open(f"./data/setcover_20r_20c_0.1d/instance_{idx}.txt")
    m = 21
elif problem_size == "sm":
    f = open(f"./data/setcover_50r_50c_0.1d/instance_{idx}.txt")
    m = 51
elif problem_size == "m":
    f = open(f"./data/setcover_100r_100c_0.1d/instance_{idx}.txt")
    m = 101
else:
    print("Wrong scale.")
    sys.exit(0)


line = f.readline()
data = [[] for i in range(m)]
cnt = 0
i = 0
c = []

while line:
    if cnt == 0:
        cnt = 1
        continue
    else:
        line = re.split(r"[\n\s]", line)
        data[i] = copy.deepcopy(line)
        i = i + 1

    line = f.readline()
f.close()

c = copy.deepcopy(data[0][:])
del c[-1]
data.remove(data[0][:])
# print(len(data[:][0]))
subsets = []
for j in range(len(data[0][:]) - 1):
    a = set()
    for i in range(len(data[:][0]) - 1):
        if j == len(data[0][:]) - 2:
            # print("get")
            del data[i][-1]
        if int(data[i][j]) == 1:
            # print(type(data[i][j]))
            # data[i][j] = int(1 + i)
            a.add(1 + i)
            # print(type(data[i][j]))
    b = [a, int(c[j])]
    subsets.append(b)


def coef(sub, gamma):
    val = 0
    # print(sub)
    for item in sub:
        if item in gamma:
            val += gamma[item]
    return val


def generalized_set_cover_A(universe, subsets):
    elements = set(e for s in subsets for e in s[0])
    if elements != universe:
        return None
    covered = set()
    cover = []
    while covered != elements:
        gam = {}
        other = elements - covered
        for it in other:
            cnt = 0
            for s in subsets:
                if it in s[0]:
                    cnt += 1
            gam[it] = 1 / cnt
        subset = max(subsets, key=lambda s: coef(s[0], gam) / s[1])
        cover.append(subset)
        covered |= subset[0]
        subsets.remove(subset)
    return cover


def generalized_set_cover_B(universe, subsets):
    elements = set(e for s in subsets for e in s[0])
    if elements != universe:
        return None
    covered = set()
    cover = []
    while covered != elements:
        gam = {}
        other = elements - covered
        for it in other:
            cnt = 0
            for s in subsets:
                if it in s[0]:
                    cnt += 1
            gam[it] = (1 / cnt) ** 2
        subset = max(subsets, key=lambda s: coef(s[0], gam) / s[1])
        cover.append(subset)
        covered |= subset[0]
        subsets.remove(subset)
    return cover


def generalized_set_cover_C(universe, subsets):
    elements = set(e for s in subsets for e in s[0])
    if elements != universe:
        return None
    covered = set()
    cover = []
    while covered != elements:
        gam = {}
        for it in elements - covered:
            cnt = 0
            for s in subsets:
                if it in s[0]:
                    cnt += 1
            gam[it] = math.floor((1 / cnt) ** 2)
        subset = max(subsets, key=lambda s: coef(s[0], gam) / s[1])
        cover.append(subset)
        covered |= subset[0]
        subsets.remove(subset)
    return cover


def generalized_set_cover_D(universe, subsets):
    elements = set(e for s in subsets for e in s[0])
    if elements != universe:
        return None
    covered = set()
    cover = []
    while covered != elements:
        gam = {}
        for it in elements - covered:
            cnt = 0
            for s in subsets:
                if it in s[0]:
                    cnt += 1
            gam[it] = math.sqrt(1 / cnt)
        subset = max(subsets, key=lambda s: coef(s[0], gam) / s[1])
        cover.append(subset)
        covered |= subset[0]
        subsets.remove(subset)
    return cover


def generalized_set_cover_E(universe, subsets):
    elements = set(e for s in subsets for e in s[0])
    if elements != universe:
        return None
    covered = set()
    cover = []
    while covered != elements:
        gam = {}
        for it in elements - covered:
            cnt = 0
            for s in subsets:
                if it in s[0]:
                    cnt += 1
            gam[it] = pow(1 / cnt, 1.5)
        subset = max(subsets, key=lambda s: coef(s[0], gam) / s[1])
        cover.append(subset)
        covered |= subset[0]
        subsets.remove(subset)
    return cover


def generalized_set_cover_F(universe, subsets):
    elements = set(e for s in subsets for e in s[0])
    if elements != universe:
        return None
    covered = set()
    cover = []
    while covered != elements:
        gam = {}
        for it in elements - covered:
            cnt = 0
            record = set()
            for s in subsets:
                if it in s[0]:
                    cnt += 1
                    record = s
            if cnt == 1:
                subset = record
                cover.append(subset)
                covered |= subset[0]
                subsets.remove(subset)
                break
            break

            gam[it] = 1 / (cnt - 1)
        subset = max(subsets, key=lambda s: coef(s[0], gam) / s[1])
        cover.append(subset)
        covered |= subset[0]
        subsets.remove(subset)
    return cover


def generalized_set_cover_G(universe, subsets):
    elements = set(e for s in subsets for e in s[0])
    if elements != universe:
        return None
    covered = set()
    cover = []
    while covered != elements:
        gam = {}
        for it in elements - covered:
            cnt = 0
            record = set()
            for s in subsets:
                if it in s[0]:
                    cnt += 1
                    record = s
            if cnt == 1:
                subset = record
                cover.append(subset)
                covered |= subset[0]
                subsets.remove(subset)
                break
            break
            gam[it] = (1 / (cnt - 1)) ** 2
        subset = max(subsets, key=lambda s: coef(s[0], gam) / s[1])
        cover.append(subset)
        covered |= subset[0]
        subsets.remove(subset)
    return cover


def generalized_set_cover_H(universe, subsets):
    elements = set(e for s in subsets for e in s[0])
    if elements != universe:
        return None
    covered = set()
    cover = []
    while covered != elements:
        gam = {}
        other = elements - covered
        for it in other:
            cnt = 0
            record = set()
            for s in subsets:
                if it in s[0]:
                    cnt += 1
                    record = s
            if cnt == 1:
                subset = record
                cover.append(subset)
                covered |= subset[0]
                subsets.remove(subset)
                break
            break
            gam[it] = math.sqrt(1 / (cnt - 1))
        subset = max(subsets, key=lambda s: coef(s[0], gam) / s[1])
        cover.append(subset)
        covered |= subset[0]
        subsets.remove(subset)
    return cover


def generalized_set_cover_I(universe, subsets):
    elements = set(e for s in subsets for e in s[0])
    if elements != universe:
        return None
    covered = set()
    cover = []
    while covered != elements:
        gam = {}
        for it in elements - covered:
            cnt = 0
            record = set()
            for s in subsets:
                if it in s[0]:
                    cnt += 1
                    record = s
            if cnt == 1:
                subset = record
                cover.append(subset)
                covered |= subset[0]
                subsets.remove(subset)
                break
            gam[it] = pow(1 / (cnt - 1), 1.5)
        subset = max(subsets, key=lambda s: coef(s[0], gam) / s[1])
        cover.append(subset)
        covered |= subset[0]
        subsets.remove(subset)
    return cover


def generalized_set_cover_J(universe, subsets):
    elements = set(e for s in subsets for e in s[0])
    if elements != universe:
        return None
    covered = set()
    cover = []
    while covered != elements:
        gam = {}
        for it in elements - covered:
            cnt = 0
            record = set()
            for s in subsets:
                if it in s[0]:
                    cnt += 1
                    record = s
            if cnt == 1:
                subset = record
                cover.append(subset)
                covered |= subset[0]
                subsets.remove(subset)
                break
            gam[it] = 1
        subset = max(subsets, key=lambda s: coef(s[0], gam) / s[1])
        cover.append(subset)
        covered |= subset[0]
        subsets.remove(subset)
    return cover


universe = set(range(1, m))
if rule == "A":
    cover = generalized_set_cover_A(universe, subsets)
elif rule == "B":
    cover = generalized_set_cover_B(universe, subsets)
elif rule == "C":
    cover = generalized_set_cover_C(universe, subsets)
elif rule == "D":
    cover = generalized_set_cover_D(universe, subsets)
elif rule == "E":
    cover = generalized_set_cover_E(universe, subsets)
elif rule == "F":
    cover = generalized_set_cover_F(universe, subsets)
elif rule == "G":
    cover = generalized_set_cover_G(universe, subsets)
elif rule == "H":
    cover = generalized_set_cover_H(universe, subsets)
elif rule == "I":
    cover = generalized_set_cover_I(universe, subsets)
elif rule == "J":
    cover = generalized_set_cover_J(universe, subsets)
else:
    print("Wrong algo")
    sys.exit(0)


sum = 0
for i in range(len(cover)):
    sum = sum + cover[i][1]
ftime = time.time()

print("instance %d: greedy_val = %d, time = %.4f" % (idx, sum, ftime - stime))
