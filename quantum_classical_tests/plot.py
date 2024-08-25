import matplotlib.pyplot as plt

def make_loss_enh_plot(vec_rk4, vec_rk4_enh):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        vec_rk4_enh[0],
        label="Classical RK4_ENH Training Loss",
        color="blue",
        linestyle="--",
    )
    ax.plot(
        vec_rk4_enh[1],
        label="Classical RK4_ENH Testing Loss",
        color="blue",
    )
    ax.plot(
        vec_rk4[0],
        label="Classical RK4 Training Loss",
        color="red",
        linestyle="--",
    )
    ax.plot(
        vec_rk4[1],
        label="Classical RK4 Testing Loss",
        color="red",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid()
    plt.savefig(
        "classical_transfomer_loss_cifar10_rk4_rk4_enh.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def make_roc_plot(vec_rk1, vec_rk2, vec_rk3, vec_rk4, vec_rk4_enh):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        vec_rk1[0],
        vec_rk1[1],
        label="Classical RK1 AUC = " + str(vec_rk1[2]),
        color="blue",
    )
    ax.plot(
        vec_rk2[0],
        vec_rk2[1],
        label="Classical RK2 AUC = " + str(vec_rk2[2]),
        color="green",
    )
    ax.plot(
        vec_rk3[0],
        vec_rk3[1],
        label="Classical RK3 AUC = " + str(vec_rk3[2]),
        color="orange",
    )
    ax.plot(
        vec_rk4[0],
        vec_rk4[1],
        label="Classical RK4 AUC = " + str(vec_rk4[2]),
        color="red",
    )
    ax.plot(
        vec_rk4_enh[0],
        vec_rk4_enh[1],
        label="Classical RK4_ENH AUC = " + str(vec_rk4_enh[2]),
        color="black",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid()
    plt.savefig(
        "classical_transfomer_roc_cifar10_rk1_rk2_rk3_rk4.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def make_auc_enh_plot(vec_rk4, vec_rk4_enh):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        vec_rk4_enh[0],
        label="Classical RK4_ENH Training AUC",
        color="blue",
        linestyle="--",
    )
    ax.plot(
        vec_rk4_enh[1],
        label="Classical RK4_ENH Testing AUC",
        color="blue",
    )
    ax.plot(
        vec_rk4[0],
        label="Classical RK4 Training AUC",
        color="red",
        linestyle="--",
    )
    ax.plot(
        vec_rk4[1],
        label="Classical RK4 Testing AUC",
        color="red",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend()
    ax.grid()
    plt.savefig(
        "classical_transfomer_auc_cifar10_rk4_rk4_enh.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def make_auc_plot(vec_rk1, vec_rk2, vec_rk3, vec_rk4):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        vec_rk1[0],
        label="Classical RK1 Training AUC",
        color="blue",
        linestyle="--",
    )
    ax.plot(
        vec_rk1[1],
        label="Classical RK1 Testing AUC",
        color="blue",
    )
    ax.plot(
        vec_rk2[0],
        label="Classical RK2 Training AUC",
        color="green",
        linestyle="--",
    )
    ax.plot(
        vec_rk2[1],
        label="Classical RK2 Testing AUC",
        color="green",
    )
    ax.plot(
        vec_rk3[0],
        label="Classical RK3 Training AUC",
        color="orange",
        linestyle="--",
    )
    ax.plot(
        vec_rk3[1],
        label="Classical RK3 Testing AUC",
        color="orange",
    )
    ax.plot(
        vec_rk4[0],
        label="Classical RK4 Training AUC",
        color="red",
        linestyle="--",
    )
    ax.plot(
        vec_rk4[1],
        label="Classical RK4 Testing AUC",
        color="red",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend()
    ax.grid()
    plt.savefig(
        "classical_transfomer_auc_cifar10_rk1_rk2_rk3_rk4.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def make_loss_plot(vec_rk1, vec_rk2, vec_rk3, vec_rk4):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        vec_rk1[0],
        label="Classical RK1 Training Loss",
        color="blue",
        linestyle="--",
    )
    ax.plot(
        vec_rk1[1],
        label="Classical RK1 Testing Loss",
        color="blue",
    )
    ax.plot(
        vec_rk2[0],
        label="Classical RK2 Training Loss",
        color="green",
        linestyle="--",
    )
    ax.plot(
        vec_rk2[1],
        label="Classical RK2 Testing Loss",
        color="green",
    )
    ax.plot(
        vec_rk3[0],
        label="Classical RK3 Training Loss",
        color="orange",
        linestyle="--",
    )
    ax.plot(
        vec_rk3[1],
        label="Classical RK3 Testing Loss",
        color="orange",
    )
    ax.plot(
        vec_rk4[0],
        label="Classical RK4 Training Loss",
        color="red",
        linestyle="--",
    )
    ax.plot(
        vec_rk4[1],
        label="Classical RK4 Testing Loss",
        color="red",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid()
    plt.savefig(
        "classical_transfomer_loss_cifar10_rk1_rk2_rk3_rk4.pdf",
        bbox_inches="tight",
        transparent=True,
    )

