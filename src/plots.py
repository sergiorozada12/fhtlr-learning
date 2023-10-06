import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def plot_gridworld(W, mat_q_stationary, mat_q, mat_tlr):
    mat_r = np.zeros((W, W))
    mat_r[0, 0] = 50
    mat_r[-1, -1] = 100

    with plt.style.context(['science'], ['ieee']):
        matplotlib.rcParams.update({'font.size': 14})

        fig, axarr = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)

        vmin = 0.0
        vmax = 100.0

        cax1 = axarr[0, 0].imshow(mat_r, vmin=vmin, vmax=vmax, cmap='Reds')
        for i in range(W):
            for j in range(W):
                v = np.around(mat_r[i, j], 1)
                axarr[0, 0].text(j, i, v, ha='center', va='bottom', color='silver')
        axarr[0, 0].set_xlabel("(a)")

        axarr[0, 1].imshow(mat_q_stationary, vmin=vmin, vmax=vmax, cmap='Reds')
        for i in range(W):
            for j in range(W):
                v = np.around(mat_q_stationary[i, j], 1)
                axarr[0, 1].text(j, i, v, ha='center', va='bottom', color='silver')
        axarr[0, 1].set_xlabel("(b)")

        axarr[1, 0].imshow(mat_q, vmin=vmin, vmax=vmax, cmap='Reds')
        for i in range(W):
            for j in range(W):
                v = np.around(mat_q[i, j], 1)
                axarr[1, 0].text(j, i, v, ha='center', va='bottom', color='silver')
        axarr[1, 0].set_xlabel("(c)")

        axarr[1, 1].imshow(mat_tlr, vmin=vmin, vmax=vmax, cmap='Reds')
        for i in range(W):
            for j in range(W):
                v = np.around(mat_tlr[i, j], 1)
                axarr[1, 1].text(j, i, v, ha='center', va='bottom', color='silver')
        axarr[1, 1].set_xlabel("(d)")

        for ax in axarr.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        gap = 0.0
        width = 0.45
        height = 0.45

        axarr[0, 0].set_position([0, 0.5 + gap/2, width, height])
        axarr[0, 1].set_position([0.5 + gap, 0.5 + gap/2, width, height])
        axarr[1, 0].set_position([0, 0, width, height])
        axarr[1, 1].set_position([0.5 + gap, 0, width, height])

        fig.savefig('figures/fig_1.jpg', dpi=300)


def plot_wireless():
    dqn = np.load('results/dqn2.npy')
    dfhqn = np.load('results/dfhqn2.npy')
    fhtlr = np.load('results/fhtlr2.npy')
    ql = np.load('results/ql2.npy')

    mu_dqn = np.mean(dqn, axis=0)
    mu_dfhqn = np.mean(dfhqn, axis=0)
    mu_fhtlr = np.mean(fhtlr, axis=0)
    mu_ql = np.mean(ql, axis=0)

    w = 100

    mu_dqn_smt = [np.mean(mu_dqn[i - w:i]) for i in range(w, len(mu_dqn))]
    mu_dfhqn_smt = [np.mean(mu_dfhqn[i - w:i]) for i in range(w, len(mu_dfhqn))]
    mu_fhtlr_smt = [np.mean(mu_fhtlr[i - w:i]) for i in range(w, len(mu_fhtlr))]
    mu_ql_smt = [np.mean(mu_ql[i - w:i]) for i in range(w, len(mu_ql))]

    with plt.style.context(['science'], ['ieee']):
        matplotlib.rcParams.update({'font.size': 16})

        fig = plt.figure(figsize=[8, 3])
        plt.plot(mu_dqn_smt, c='b', label='DQN')
        plt.plot(mu_dqn, alpha=.2, c='b')
        plt.plot(mu_dfhqn_smt, c='orange', label='DFHQN')
        plt.plot(mu_dfhqn, alpha=.2, c='orange')
        plt.plot(mu_fhtlr_smt, c='g', label='FHTLR')
        plt.plot(mu_fhtlr, alpha=.2, c='g')
        plt.plot(mu_ql_smt, c='k', label='FHQ-learning')
        plt.plot(mu_ql, alpha=.2, c='k')
        plt.xlim(0, 100_000)
        plt.ylim(-3, 1.6)
        plt.grid()
        plt.legend()
        plt.xlabel("Episodes")
        plt.ylabel("Return")
        fig.savefig('figures/fig_2.jpg', dpi=300)
