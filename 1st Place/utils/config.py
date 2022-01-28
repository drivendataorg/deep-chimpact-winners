def dict2cfg(cfg_dict):
    """Create config from dictionary.

    Args:
        cfg_dict (dict): dictionary with configs to be converted to config.

    Returns:
        cfg: python class object as config
    """
    class Config:
        def __init__(self, data):
            self.__dict__.update(**data)
    return Config(cfg_dict) # dict to cfg


def cfg2dict(cfg):
    """Create dictionary from config.

    Args:
        cfg (config): python class object as config.
    
        Returns:
            cfg_dict (dict): dictionary with configs.
    """
    return {k:v for k,v in dict(vars(cfg)).items() if '__' not in k}