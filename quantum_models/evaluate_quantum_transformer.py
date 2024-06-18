from dataset_processing_dir.dataset import (
    get_mnist_dataloaders,
    get_cifar10_dataloaders,
    get_cifar100_dataloaders,
    get_imdb_dataloaders,
    get_image_net_dataloaders,
)
from utils.training import train_and_evaluate
from quantum_transformer import VisionTransformer
from utils.quantum_layer import get_circuit

import matplotlib.pyplot as plt


def make_roc_plot(vec_rk1, vec_rk2, vec_rk3, vec_rk4, vec_rk4_enh):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(
        vec_rk1[0],
        vec_rk1[1],
        label="Quantum RK1 AUC = " + str(vec_rk1[2]),
        color="blue",
    )
    ax.plot(
        vec_rk2[0],
        vec_rk2[1],
        label="Quantum RK2 AUC = " + str(vec_rk2[2]),
        color="green",
    )
    ax.plot(
        vec_rk3[0],
        vec_rk3[1],
        label="Quantum RK3 AUC = " + str(vec_rk3[2]),
        color="orange",
    )
    ax.plot(
        vec_rk4[0],
        vec_rk4[1],
        label="Quantum RK4 AUC = " + str(vec_rk4[2]),
        color="red",
    )
    ax.plot(
        vec_rk4_enh[0],
        vec_rk4_enh[1],
        label="Quantum RK4_ENH AUC = " + str(vec_rk4_enh[2]),
        color="black",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid()
    plt.savefig(
        "qtransfomer_roc_mnist_rk1_rk2_rk3_rk4.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def make_auc_plot(vec_rk1, vec_rk2, vec_rk3, vec_rk4, vec_rk4_enh):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(
        vec_rk1[0],
        label="Quantum RK1 Training AUC",
        color="blue",
        linestyle="--",
    )
    ax.plot(
        vec_rk1[1],
        label="Quantum RK1 Testing AUC",
        color="blue",
    )
    ax.plot(
        vec_rk2[0],
        label="Quantum RK2 Training AUC",
        color="green",
        linestyle="--",
    )
    ax.plot(
        vec_rk2[1],
        label="Quantum RK2 Testing AUC",
        color="green",
    )
    ax.plot(
        vec_rk3[0],
        label="Quantum RK3 Training AUC",
        color="orange",
        linestyle="--",
    )
    ax.plot(
        vec_rk3[1],
        label="Quantum RK3 Testing AUC",
        color="orange",
    )
    ax.plot(
        vec_rk4[0],
        label="Quantum RK4 Training AUC",
        color="red",
        linestyle="--",
    )
    ax.plot(
        vec_rk4[1],
        label="Quantum RK4 Testing AUC",
        color="red",
    )
    ax.plot(
        vec_rk4_enh[0],
        label="Quantum RK4_ENH Training AUC",
        color="black",
        linestyle="--",
    )
    ax.plot(
        vec_rk4_enh[1],
        label="Quantum RK4_ENH Testing AUC",
        color="black",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend()
    ax.grid()
    plt.savefig(
        "qtransfomer_auc_mnist_rk1_rk2_rk3_rk4.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def make_loss_plot(vec_rk1, vec_rk2, vec_rk3, vec_rk4, vec_rk4_enh):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(
        vec_rk1[0],
        label="Quantum RK1 Training Loss",
        color="blue",
        linestyle="--",
    )
    ax.plot(
        vec_rk1[1],
        label="Quantum RK1 Testing Loss",
        color="blue",
    )
    ax.plot(
        vec_rk2[0],
        label="Quantum RK2 Training Loss",
        color="green",
        linestyle="--",
    )
    ax.plot(
        vec_rk2[1],
        label="Quantum RK2 Testing Loss",
        color="green",
    )
    ax.plot(
        vec_rk3[0],
        label="Quantum RK3 Training Loss",
        color="orange",
        linestyle="--",
    )
    ax.plot(
        vec_rk3[1],
        label="Quantum RK3 Testing Loss",
        color="orange",
    )
    ax.plot(
        vec_rk4[0],
        label="Quantum RK4 Training Loss",
        color="red",
        linestyle="--",
    )
    ax.plot(
        vec_rk4[1],
        label="Quantum RK4 Testing Loss",
        color="red",
    )
    ax.plot(
        vec_rk4_enh[0],
        label="Quantum RK4_ENH Training Loss",
        color="black",
        linestyle="--",
    )
    ax.plot(
        vec_rk4_enh[1],
        label="Quantum RK4_ENH Testing Loss",
        color="black",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid()
    plt.savefig(
        "qtransfomer_loss_mnist_rk1_rk2_rk3_rk4_rk4enh.pdf",
        bbox_inches="tight",
        transparent=True,
    )


train_dataloader, val_dataloader, test_dataloader = get_mnist_dataloaders(batch_size=64)

hidden_size = 6
num_classes = 10

model4 = VisionTransformer(
    num_classes=num_classes,
    patch_size=14,
    hidden_size=hidden_size,
    num_heads=2,
    num_transformer_encoder_blocks=3,
    num_transformer_rk1_blocks=0,
    num_transformer_rk2_blocks=0,
    num_transformer_rk3_blocks=0,
    num_transformer_rk4_blocks=1,
    num_transformer_rk4_enhanced_blocks=0,
    num_transformer_decoder_blocks=1,
    quantum_attn_circuit = get_circuit(),
    quantum_mlp_circuit = get_circuit(),
    mlp_hidden_size=3,
)

info4 = train_and_evaluate(
    model4,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    src_mask_flag=False,
    trg_mask_flag=False,
    src=hidden_size,
    trg=hidden_size,
    num_classes=num_classes,
    num_epochs=65,
)
del model4

model4_enh = VisionTransformer(
    num_classes=num_classes,
    patch_size=14,
    hidden_size=hidden_size,
    num_heads=2,
    num_transformer_encoder_blocks=3,
    num_transformer_rk1_blocks=0,
    num_transformer_rk2_blocks=0,
    num_transformer_rk3_blocks=0,
    num_transformer_rk4_blocks=0,
    num_transformer_rk4_enhanced_blocks=1,
    num_transformer_decoder_blocks=1,
    quantum_attn_circuit = get_circuit(),
    quantum_mlp_circuit = get_circuit(),
    mlp_hidden_size=3,
)

info4_enh = train_and_evaluate(
    model4_enh,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    src_mask_flag=False,
    trg_mask_flag=False,
    src=hidden_size,
    trg=hidden_size,
    num_classes=num_classes,
    num_epochs=65,
)

del model4_enh

model1 = VisionTransformer(
    num_classes=num_classes,
    patch_size=14,
    hidden_size=hidden_size,
    num_heads=2,
    num_transformer_encoder_blocks=3,
    num_transformer_rk1_blocks=1,
    num_transformer_rk2_blocks=0,
    num_transformer_rk3_blocks=0,
    num_transformer_rk4_blocks=0,
    num_transformer_rk4_enhanced_blocks=0,
    num_transformer_decoder_blocks=1,
    quantum_attn_circuit = get_circuit(),
    quantum_mlp_circuit = get_circuit(),
    mlp_hidden_size=3,
)

info1 = train_and_evaluate(
    model1,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    src_mask_flag=False,
    trg_mask_flag=False,
    src=hidden_size,
    trg=hidden_size,
    num_classes=num_classes,
    num_epochs=65,
)

del model1

model2 = VisionTransformer(
    num_classes=num_classes,
    patch_size=14,
    hidden_size=hidden_size,
    num_heads=2,
    num_transformer_encoder_blocks=3,
    num_transformer_rk1_blocks=0,
    num_transformer_rk2_blocks=1,
    num_transformer_rk3_blocks=0,
    num_transformer_rk4_blocks=0,
    num_transformer_rk4_enhanced_blocks=0,
    num_transformer_decoder_blocks=1,
    quantum_attn_circuit = get_circuit(),
    quantum_mlp_circuit = get_circuit(),
    mlp_hidden_size=3,
)

info2 = train_and_evaluate(
    model2,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    src_mask_flag=False,
    trg_mask_flag=False,
    src=hidden_size,
    trg=hidden_size,
    num_classes=num_classes,
    num_epochs=65,
)

del model2

model3 = VisionTransformer(
    num_classes=num_classes,
    patch_size=14,
    hidden_size=hidden_size,
    num_heads=2,
    num_transformer_encoder_blocks=3,
    num_transformer_rk1_blocks=0,
    num_transformer_rk2_blocks=0,
    num_transformer_rk3_blocks=1,
    num_transformer_rk4_blocks=0,
    num_transformer_rk4_enhanced_blocks=0,
    num_transformer_decoder_blocks=1,
    quantum_attn_circuit = get_circuit(),
    quantum_mlp_circuit = get_circuit(),
    mlp_hidden_size=3,
)

info3 = train_and_evaluate(
    model3,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    src_mask_flag=False,
    trg_mask_flag=False,
    src=hidden_size,
    trg=hidden_size,
    num_classes=num_classes,
    num_epochs=65,
)

del model3

make_loss_plot(
    [info1["train_losses"], info1["val_losses"]],
    [info2["train_losses"], info2["val_losses"]],
    [info3["train_losses"], info3["val_losses"]],
    [info4["train_losses"], info4["val_losses"]],
    [info4_enh["train_losses"], info4_enh["val_losses"]],
)

make_auc_plot(
    [info1["train_aucs"], info1["val_aucs"]],
    [info2["train_aucs"], info2["val_aucs"]],
    [info3["train_aucs"], info3["val_aucs"]],
    [info4["train_aucs"], info4["val_aucs"]],
    [info4_enh["train_aucs"], info4_enh["val_aucs"]],
)

if num_classes == 2:
    make_roc_plot(
        [info1["test_fpr"], info1["test_tpr"], info1["best_auc"]],
        [info2["test_fpr"], info2["test_tpr"], info2["best_auc"]],
        [info3["test_fpr"], info3["test_tpr"], info3["best_auc"]],
        [info4["test_fpr"], info4["test_tpr"], info4["best_auc"]],
        [info4_enh["test_fpr"], info4_enh["test_tpr"], info4_enh["best_auc"]],
    )
