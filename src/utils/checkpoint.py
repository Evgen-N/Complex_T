import torch

def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, path="checkpoint.pth", freeze_encoder=False):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    # Если нужно заморозить слои до Residual-блока
    if freeze_encoder:
        for name, param in model.named_parameters():
            if "residual_block" not in name:  # Например, заморозим LSTM
                param.requires_grad = False
        print("Encoder (LSTM) frozen.")

    print(f"Checkpoint loaded from {path}, resuming from epoch {start_epoch}")
    return start_epoch