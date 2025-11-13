def get_scheduler(optimizer):
    scheduler = optimizer.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    return scheduler