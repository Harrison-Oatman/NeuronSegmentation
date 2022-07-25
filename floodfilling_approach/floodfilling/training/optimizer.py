from keras.optimizers import Adam


def optimizer_from_config():
    """
    TODO: returns optimizer based on configs
    """
    return Adam(learning_rate=0.002)
