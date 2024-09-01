import sys, os
import numpy as np
import re
import copy
import time


idx = int(sys.argv[1])
problem_size = sys.argv[2]

stime = time.time()
if problem_size == "s":
    f = open(f"/home/dingzhenxin/workspace/DPexp/data/setcover_20r_20c_0.1d/instance_{idx}.txt")
    m = 21
elif problem_size == "sm":
    f = open(f"/home/dingzhenxin/workspace/DPexp/data/setcover_50r_50c_0.1d/instance_{idx}.txt")
    m = 51
elif problem_size == "m":
    f = open(f"/home/dingzhenxin/workspace/DPexp/data/setcover_100r_100c_0.1d/instance_{idx}.txt")
    m = 101
else:
    print('Wrong scale.')
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
        line = re.split(r'[\n\s]', line)
        data[i] = copy.deepcopy(line)
        i = i + 1

    line = f.readline()
f.close()

c = copy.deepcopy(data[0][:])
del c[-1]
data.remove(data[0][:])
#print(len(data[:][0]))
subsets = []
for j in range(len(data[0][:])-1):
    a = set()
    for i in range (len(data[:][0]) - 1):
        if j == len(data[0][:]) - 2:
            #print("get")
            del data[i][-1]
        if int(data[i][j]) == 1:
            #print(type(data[i][j]))
            #data[i][j] = int(1 + i)
            a.add(1+i)
            #print(type(data[i][j]))
    b = [a, int(c[j])]
    subsets.append(b)


def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s[0])
    # Check the subsets cover the universe
    if elements != universe:
        return None
    covered = set()
    cover = []
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s[0] - covered)/s[1])
        cover.append(subset)
        covered |= subset[0]
        subsets.remove(subset)

    return cover

#print(len(subsets))
universe = set(range(1, m))
cover = set_cover(universe, subsets)
#print("subsets:", subsets)
#print("cover:", cover)
#print("c:", c)
sum = 0
for i in range(len(cover)):
    sum = sum + cover[i][1]
ftime=time.time()
print("instance %d: greedy_val = %d, time = %.4f" % (idx, sum, ftime-stime))

# In[ ]:




