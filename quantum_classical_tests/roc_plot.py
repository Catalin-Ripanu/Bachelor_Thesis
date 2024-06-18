import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 8))
ax.plot(
    [
    ],
    label="Quantum RK1 AUC=",
    color="blue",
)
ax.plot(
    [
    ],
    label="Quantum RK2 AUC=",
    color="orange",
)
ax.plot(
    [
    ],
    label="Quantum RK4 AUC=",
    color="red",
)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
ax.grid()
plt.savefig(
    "qtransfomer_loss_mnist_pos_learn_rk1_rk2_rk4.pdf",
    bbox_inches="tight",
    transparent=True,
)
