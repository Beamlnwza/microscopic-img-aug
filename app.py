import gradio as gr
import os
import imgaug.augmenters as iaa
import glob

image_dir = os.path.join(os.path.dirname(__file__), "images")
image_files = glob.glob(os.path.join(image_dir, "*.png"))

examples = []
for image_file in image_files:
    examples.append(image_file)

aug_seq = iaa.Sequential([
    iaa.Resize({"height": 224, "width": 224}),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.25))),
    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.025*255))),
    iaa.Sometimes(0.5, iaa.Multiply((0.95, 1.25))),
    iaa.Sometimes(0.5, iaa.LinearContrast((0.95, 1.25))),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Rotate((-45, 45)),
    iaa.Crop(percent=(0, 0.1)),
    iaa.CropToAspectRatio(1.0),
    iaa.PadToAspectRatio(1.0),
    iaa.Pad(percent=(0, 0.1)),
    iaa.PiecewiseAffine(scale=(0.01, 0.05), mode="reflect", cval=(0, 255)),
    iaa.ElasticTransformation(alpha=(0, 5.0), sigma=1.0, mode="reflect", cval=(0, 255)),
], random_order=True)

def aug_img(img_array):
    img_af_aug = aug_seq(image=img_array)
    return img_af_aug
    
demo = gr.Interface(
    aug_img,
    gr.Image(type="numpy"),
    "image",
    examples=examples,
).launch()