import h5py
import json
import shutil
import re
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

# --- Patch BatchNormalization to ignore unsupported args ---
_orig_bn_init = BatchNormalization.__init__

def _patched_bn_init(self, *args, **kwargs):
    kwargs.pop('renorm', None)
    kwargs.pop('renorm_clipping', None)
    kwargs.pop('renorm_momentum', None)
    _orig_bn_init(self, *args, **kwargs)

BatchNormalization.__init__ = _patched_bn_init


def patch_and_save(src_path, dst_path):
    print(f"Patching {src_path}...")

    # Check if source file exists
    import os
    if not os.path.exists(src_path):
        print(f"  ERROR: {src_path} not found. Skipping.")
        return False

    # Backup original
    backup_path = src_path.replace('.h5', '_backup.h5')
    shutil.copy(src_path, backup_path)
    print(f"  Backup saved to {backup_path}")

    # Patch model config inside HDF5
    with h5py.File(src_path, 'r+') as f:
        config_str = f.attrs['model_config']
        if isinstance(config_str, bytes):
            config_str = config_str.decode('utf-8')

        # Remove unsupported keys
        config_str = config_str.replace(',"quantization_config":null', '')
        config_str = config_str.replace('"quantization_config":null,', '')
        config_str = re.sub(r',"renorm":\s*(true|false)', '', config_str)
        config_str = re.sub(r'"renorm":\s*(true|false),', '', config_str)
        config_str = re.sub(r',"renorm_clipping":\s*\{[^}]*\}', '', config_str)
        config_str = re.sub(r'"renorm_clipping":\s*null,?', '', config_str)
        config_str = re.sub(r',"renorm_momentum":\s*[\d.]+', '', config_str)
        config_str = re.sub(r'"renorm_momentum":\s*[\d.]+,?', '', config_str)

        f.attrs['model_config'] = config_str.encode('utf-8')
        print(f"  Config patched!")

    # Load and re-save
    model = tf.keras.models.load_model(src_path, compile=False)
    model.save(dst_path)
    print(f"  Saved to {dst_path}")
    return True


# Patch both models
patch_and_save('model/acne_model.h5',  'model/acne_model_fixed.h5')
patch_and_save('model/skin_model.h5',  'model/skin_model_fixed.h5')

print("\nALL DONE!")