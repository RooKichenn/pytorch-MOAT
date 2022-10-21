# --------------------------------------------------------
# MOAT
# Written by ZeChen Wu
# --------------------------------------------------------
from .moat import MOAT


def build_model(config):
    model_type = config.MODEL.TYPE
    print(f"Creating model: {model_type}")
    if model_type == "MOAT":
        model = eval(model_type)(
            in_channels=config.MODEL.MOAT.IN_CHANS,
            depths=config.MODEL.MOAT.DEPTHS,
            channels=config.MODEL.MOAT.CHANNELS,
            img_size=config.MODEL.MOAT.IMG_SIZE,
            use_window=config.MODEL.MOAT.USE_WINDOW,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.MOAT.EMBED_DIM,
            num_heads=config.MODEL.MOAT.DIM_HEAD,
            window_size=config.MODEL.MOAT.WINDOW_SIZE,
            global_pool=config.MODEL.MOAT.POOL,
            drop=config.MODEL.DROP_RATE,
            drop_path=config.MODEL.DROP_PATH_RATE,
        )
    return model
