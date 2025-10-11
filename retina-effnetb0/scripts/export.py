import argparse, tensorflow as tf
from src.config import load_config


if __name__ == '__main__':
p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/efficientnet_b0.yaml')
p.add_argument('--out_dir', default='exports/effnet_b0_savedmodel')
args = p.parse_args()
cfg = load_config(args.config)


model = tf.keras.models.load_model(cfg.out['ckpt'])
model.save(args.out_dir)
print('Exported to', args.out_dir)