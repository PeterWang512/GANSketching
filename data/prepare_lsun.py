import os
import sys
import numpy as np
from PIL import Image


def create_lsun(save_dir, lmdb_dir, resolution=256, max_images=None):
    print('Loading LSUN dataset from "%s"' % lmdb_dir)
    import lmdb # pip install lmdb # pylint: disable=import-error
    import cv2 # pip install opencv-python
    import io
    with lmdb.open(lmdb_dir, readonly=True).begin(write=False) as txn:
            total_images = txn.stat()['entries']
            print("Total images: ", total_images)
            if max_images is None:
                max_images = total_images
            for _idx, (_key, value) in enumerate(txn.cursor()):
                if _idx == max_images:
                    break
                try:
                    try:
                        img = cv2.imdecode(np.fromstring(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.asarray(Image.open(io.BytesIO(value)))
                    crop = np.min(img.shape[:2])
                    img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
                    img = Image.fromarray(img, 'RGB')
                    img = img.resize((resolution, resolution), Image.ANTIALIAS)
                    img.save(save_dir + '/{:06d}.png'.format(_idx))
                except:
                    print(sys.exc_info()[1])


if __name__ == "__main__":
    save_dir = sys.argv[1]
    source = sys.argv[2]
    os.makedirs(save_dir, exist_ok=True)
    create_lsun(save_dir, source)
