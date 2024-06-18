import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(
    [
        
    ],
    label="Quantum RK1 Training AUC",
    color="blue",
    linestyle="--",
)
ax.plot(
    [
        
    ],
    label="Quantum RK1 Testing AUC",
    color="blue",
)
ax.plot(
    [
        
    ],
    label="Quantum RK2 Training AUC",
    color="green",
    linestyle="--",
)
ax.plot(
    [
        
    ],
    label="Quantum RK2 Testing AUC",
    color="green",
)
ax.plot(
    [
        
    ],
    label="Quantum RK3 Training AUC",
    color="orange",
    linestyle="--",
)
ax.plot(
    [
        
    ],
    label="Quantum RK3 Testing AUC",
    color="orange",
)
ax.plot(
    [
        
    ],
    label="Quantum RK4 Training AUC",
    color="red",
    linestyle="--",
)
ax.plot(
    [
    
    ],
    label="Quantum RK4 Testing AUC",
    color="red",
)
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC")
ax.legend()
ax.grid()
plt.savefig(
    "qtransfomer_auc_mnist_pos_sincos_rk1_rk2_rk3_rk4.pdf",
    bbox_inches="tight",
    transparent=True,
)
