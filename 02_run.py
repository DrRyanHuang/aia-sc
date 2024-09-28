# modified from ../DPexp/065_cyopt.py
import torch
import numpy as np
import random
import torch.nn.functional as F
import bisect
import sys
import time
import argparse
from utils.check_fixing import check_fixing_cy


class MyNet(torch.nn.Module):

    def __init__(self, m, n):
        super(MyNet, self).__init__()
        self.m = m
        self.n = n
        self.model = torch.nn.Sequential(
            torch.nn.Linear(m * n + m + n, m + n),
            # torch.nn.BatchNorm1d(m + n),
            torch.nn.Linear(m + n, 1),
        )

    def forward(self, x):
        instance, b = x
        m = self.m
        n = self.n
        data = torch.cat([instance.view(-1, m * n + n), F.relu(b.view(-1, m))], axis=1)
        out = self.model(data)
        return out


def calc_loss(nn, label, last_instance, instance, b, alphai, ci, GAMMA):
    left = nn([last_instance, b])
    item1 = nn([instance, b - alphai]) + ci
    item0 = nn([instance, b])
    if label == 3:
        loss = (left - GAMMA * min(item1, item0)) * (left - GAMMA * min(item1, item0))
    elif label == 2:
        loss = (left - GAMMA * item1) * (left - GAMMA * item1) + 100 * F.relu(
            item1 - item0
        )
    return loss


def weighted_choice(weights):
    totals = []
    running_total = 0
    for w in weights:
        running_total += w
        totals.append(running_total)
    rnd = random.random() * running_total
    idx = bisect.bisect_right(totals, rnd)
    return idx


def weighted_sample(weighted_data, n):
    op_weights = [i[0] for i in weighted_data]
    idxs = []
    for _ in range(n):
        i = weighted_choice(op_weights)
        idxs.append(i)
        op_weights[i] = 0
    return idxs


# Argument parsing
parser = argparse.ArgumentParser(description="Test options!")
parser.add_argument("--exp_rate", type=float, default=1, help="exploration rate(1)")
parser.add_argument("--batch_size", type=int, default=32, help="batchsize(32)")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate(0.1)")
parser.add_argument("--problem_size", type=str, default="s", help="size of problem(s)")
parser.add_argument("--ins_num", type=int, default=1, help="instance number(1)")
parser.add_argument(
    "--output", type=str, default="screen", help="screen or tmp, (default: screen)"
)
parser.add_argument("--gamma", type=float, default=1, help="discount for future(1)")
opts = parser.parse_args()


# cmd argv
exp_rate = opts.exp_rate
batchsize = opts.batch_size
learning_rate = opts.lr
GAMMA = opts.gamma
if opts.problem_size == "s":
    raw_instance = np.loadtxt(
        f"./data/setcover_20r_20c_0.1d/instance_{opts.ins_num}.txt"
    )
    T = 1000
elif opts.problem_size == "sm":
    raw_instance = np.loadtxt(
        f"./data/setcover_50r_50c_0.1d/instance_{opts.ins_num}.txt"
    )
    T = 10000
elif opts.problem_size == "m":
    raw_instance = np.loadtxt(
        f"./data/setcover_100r_100c_0.1d/instance_{opts.ins_num}.txt"
    )
    T = 25000
elif opts.problem_size == "b":
    raw_instance = np.loadtxt(
        f"./data/setcover_500r_1000c_0.05d/instance_{opts.ins_num}.txt"
    )
    T = 50000
else:
    print("Wrong scale")
    sys.exit(0)

if opts.output == "screen":
    f = sys.stdout
else:
    f = open(f"{opts.output}.log", "w")


# training preparation
nrow = raw_instance.shape[0] - 1
ncol = raw_instance.shape[1]
nn = MyNet(nrow, ncol).cuda()
optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.4)

# start training
t = 0
raw_instance = torch.from_numpy(raw_instance.astype(np.float32)).cuda()  # to tensor
datapool = []
stime = time.time()  # time

# terminating label
minobjval = float("inf")
noup_cnt = 0
while t < T:
    print("Iteration: %d" % t, file=f)
    current_datapool = []

    instance = raw_instance.clone()
    b_tmp = torch.ones((nrow, 1)).cuda()
    exp_sol = [0 for i in range(ncol)]  # some var x maybe randomly selected

    # print(torch.mean(instance))
    for i in range(ncol):
        alphai = instance[:, i][1:].unsqueeze(1).clone()
        ci = instance[0][i].unsqueeze(0).unsqueeze(0).clone()
        last_instance = instance.clone()

        # examine xi, xi+1, ..., xn
        label = check_fixing_cy(instance.cpu().numpy(), b_tmp.cpu().numpy(), i)
        # label=True, xi should be 1
        if label:
            exp_sol[i] = 1
            instance[:, i] = 0
            current_datapool.append(
                [
                    2,
                    last_instance.clone(),
                    instance.clone(),
                    b_tmp.clone(),
                    alphai.clone(),
                    ci.clone(),
                ]
            )
            # print(t, i, torch.mean(instance))
            b_tmp = b_tmp - alphai
        else:
            instance[:, i] = 0
            current_datapool.append(
                [
                    3,
                    last_instance.clone(),
                    instance.clone(),
                    b_tmp.clone(),
                    alphai.clone(),
                    ci.clone(),
                ]
            )
            item1 = nn([instance, b_tmp - alphai]) + ci
            item0 = nn([instance, b_tmp])
            rng = np.random.random()
            if rng > exp_rate:
                if item1 < item0:
                    exp_sol[i] = 1
                    b_tmp = b_tmp - alphai
                else:
                    exp_sol[i] = 0
            else:
                if np.random.random() < 0.5:
                    exp_sol[i] = 1
                    b_tmp = b_tmp - alphai
                else:
                    exp_sol[i] = 0
    t += 1

    weight = np.inner(raw_instance[0].cpu().numpy(), np.array(exp_sol))

    for data in current_datapool:
        datapool.append([weight, data])

    # maintain a datapool
    while len(datapool) > 20000:
        datapool.pop(0)

    # Starting point
    if len(datapool) >= 10000:
        training_idxs = weighted_sample(datapool, batchsize)
        training_batch = [datapool[i][1] for i in training_idxs]
    else:
        # not much samples
        continue
    # 20200813, emphasize current datapool more!
    current_samples = random.sample(current_datapool, int(batchsize / 4))
    training_batch.extend(current_samples)
    print(
        "datapoolsize = %d, training_batch_size = %d"
        % (len(datapool), len(training_batch)),
        file=f,
    )

    # calculate loss
    dp_loss = 0
    for i in range(len(training_batch)):
        label, last_instance, instance, b, alphai, ci = training_batch[i]
        # print(t, torch.mean(instance))
        dp_loss += calc_loss(nn, label, last_instance, instance, b, alphai, ci, GAMMA)

    dp_loss = dp_loss / batchsize
    print("training_loss = %.5f" % dp_loss.item(), file=f)
    optimizer.zero_grad()
    dp_loss.backward()
    for param in nn.parameters():
        param.grad.data.clamp_(-0.8, 0.8)  # 梯度裁剪
    optimizer.step()

    if True:
        instance = raw_instance.clone()
        b_tmp = torch.ones((ncol, 1)).cuda()
        true_sol = []
        for i in range(ncol):
            alphai = instance[:, i][1:].unsqueeze(1).clone()
            ci = instance[0][i].unsqueeze(0).unsqueeze(0).clone()

            label = check_fixing_cy(instance.cpu().numpy(), b_tmp.cpu().numpy(), i)

            if label:  # xi must be 1
                true_sol.append(1)
                instance[:, i] = 0
                b_tmp = b_tmp - alphai
            else:
                instance[:, i] = 0
                item1 = nn([instance, b_tmp - alphai]) + ci
                item0 = nn([instance, b_tmp])
                if item1 < item0:
                    true_sol.append(1)
                    b_tmp = b_tmp - alphai
                else:
                    true_sol.append(0)

    print(true_sol, file=f)
    nn_output = nn([raw_instance, torch.ones((ncol, 1)).cuda()]).item()
    objval = torch.matmul(raw_instance[0], torch.FloatTensor(true_sol).cuda()).item()
    print(
        "dp_loss: %.5f, nn_output: %.5f, objval = %.5f" % (dp_loss, nn_output, objval),
        file=f,
    )

    # exp_rate, learning_rate decay
    if t % 1000 == 0 and exp_rate > 0.01 and t > 2000:
        scheduler.step()
        exp_rate = exp_rate * 0.5
    print("------------------------------------------------------------------", file=f)

    # need terminate?
    # if min(objval, minobjval) == minobjval:
    #     noup_cnt += 1
    # minobjval = min(objval, minobjval)
    # if noup_cnt > 4000:
    #     break


ftime = time.time()
totaltime = ftime - stime
print(f"Totaltime: {totaltime:8.3f}s, terminate t = {t}. ", file=f)
if not f == sys.stdout:
    f.close()
