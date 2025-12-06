import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_asymmetry_transform(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.Cutout(num_holes=1, max_h_size=int(img_size*0.12),
                 max_w_size=int(img_size*0.12), p=0.3),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])
