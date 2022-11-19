
from cfg_unet_multilabel import multilabel_unet_cfg

cfg = multilabel_unet_cfg

cfg.wandb = False
# train
cfg.train = True
cfg.eval = True
cfg.eval_epochs = 1 #2
cfg.start_eval_epoch = 0 #200  # when use large lr, can set a large num
cfg.run_org_eval = False
cfg.run_tta_val = True
# lr
# warmup_restart, cosine
cfg.lr_mode = "cosine"
cfg.lr = 1e-4 #1e-4
cfg.min_lr = 1e-6
cfg.weight_decay = 1e-6
cfg.epochs = 20 # 1000 / 2000
cfg.restart_epoch = 100  # only for warmup_restart
