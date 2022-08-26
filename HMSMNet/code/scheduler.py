def schedule(epoch):
    lr = 0.00005 * 0.5 ** int(epoch / 10)
    return lr
