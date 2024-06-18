import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(
    [
        
    ],
    label="Quantum RK1 Training Loss",
    color="blue",
    linestyle="--",
)
ax.plot(
    [
        
    ],
    label="Quantum RK1 Testing Loss",
    color="blue",
)
ax.plot(
    [
        
    ],
    label="Quantum RK2 Training Loss",
    color="green",
    linestyle="--",
)
ax.plot(
    [
        
    ],
    label="Quantum RK2 Testing Loss",
    color="green",
)
ax.plot(
    [
        
    ],
    label="Quantum RK3 Training Loss",
    color="orange",
    linestyle="--",
)
ax.plot(
    [
        
    ],
    label="Quantum RK3 Testing Loss",
    color="orange",
)
ax.plot(
    [
        
    ],
    label="Quantum RK4 Training Loss",
    color="red",
    linestyle="--",
)
ax.plot(
    [
        
    ],
    label="Quantum RK4 Testing Loss",
    color="red",
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid()
plt.savefig(
    "qtransfomer_loss_mnist_pos_sincos_rk1_rk2_rk3_rk4.pdf",
    bbox_inches="tight",
    transparent=True,
)


# qtransfomer_loss_cifar10_pos_learn_encoder4_decoder1_90_acc.pdf -> 0.991 AUC
# qtransfomer_loss_cifar10_pos_learn_encoder4_decoder1_32_acc.pdf -> 0.776 AUC
#
#
