import math

def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate

def a_cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps):
    base = float(current_step - warmup_steps) / float(decay_steps)
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
    return learning_rate

def dynamic_lr(config, rank_size=1, start_steps=0):
    """dynamic learning rate generator"""
    base_lr = config.base_lr

    base_step = (config.base_step // rank_size) + rank_size
    total_steps = int(base_step * config.total_epoch)
    warmup_steps = int(config.warmup_step)
    lr = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr.append(linear_warmup_learning_rate(i, warmup_steps, base_lr, base_lr * config.warmup_ratio))
        else:
            lr.append(a_cosine_learning_rate(i, base_lr, warmup_steps, total_steps))
    learning_rate = lr[start_steps:]
    return learning_rate