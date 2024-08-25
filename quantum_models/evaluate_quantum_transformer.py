from dataset_processing_dir.dataset import (
    get_mnist_dataloaders,
    get_cifar100_dataloaders,
    get_cifar10_dataloaders,
    get_imdb_dataloaders,
    get_image_net_dataloaders,
)

from utils.training import train_and_evaluate
from quantum_transformer import Transformer, VisionTransformer
from utils.quantum_layer import get_circuit


if __name__ == "__main__":

    train_dataloader, val_dataloader, test_dataloader = get_image_net_dataloaders(batch_size=64)
    hidden_size = 6
    num_classes = 1000
    model1 = VisionTransformer(
    num_classes=num_classes,
    patch_size=16,
    hidden_size=hidden_size,
    num_heads=2,
    num_transformer_encoder_blocks=3,
    num_transformer_rk1_blocks=0,
    num_transformer_rk2_blocks=0,
    num_transformer_rk3_blocks=0,
    num_transformer_rk4_blocks=0,
    num_transformer_rk4_enhanced_blocks=3,
    num_transformer_decoder_blocks=1,
    quantum_attn_circuit = get_circuit(),
    quantum_mlp_circuit = get_circuit(),
    mlp_hidden_size=6,
)
    
    info1 = train_and_evaluate(
    model1,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    src=hidden_size,
    trg=hidden_size,
    num_classes=num_classes,
    num_epochs=65,
)
