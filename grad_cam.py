import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms, models
from PIL import Image

# ================================================================
#  MODEL FACTORY (SAMA DENGAN KODE UTAMA KAMU)
# ================================================================
class ModelFactory:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def create(self, name):
        name = name.lower()

        if name == "convnext":
            model = models.convnext_tiny(weights="IMAGENET1K_V1")
            in_feat = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_feat, self.num_classes)

        elif name == "efficientnet_b3":
            model = models.efficientnet_b3(weights="IMAGENET1K_V1")
            in_feat = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_feat, self.num_classes)

        elif name == "efficientnet_b7":
            model = models.efficientnet_b7(weights="IMAGENET1K_V1")
            in_feat = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_feat, self.num_classes)

        elif name == "efficientnet_v2_s":
            model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
            in_feat = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_feat, self.num_classes)

        elif name == "efficientnet_v2_m":
            model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
            in_feat = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_feat, self.num_classes)

        elif name == "efficientnet_v2_l":
            model = models.efficientnet_v2_l(weights="IMAGENET1K_V1")
            in_feat = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_feat, self.num_classes)

        elif name == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V2")
            in_feat = model.fc.in_features
            model.fc = nn.Linear(in_feat, self.num_classes)

        elif name == "densenet":
            model = models.densenet121(weights="IMAGENET1K_V1")
            in_feat = model.classifier.in_features
            model.classifier = nn.Linear(in_feat, self.num_classes)

        else:
            raise ValueError("Unknown model: " + name)

        return model


# ================================================================
#  GRAD-CAM IMPLEMENTATION
# ================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze()

        cam = np.maximum(cam.cpu().numpy(), 0)
        cam = cv2.resize(cam, (224, 224))

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam


# ================================================================
#  IMAGE LOADING & PROCESSING
# ================================================================
def load_image(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0), img


def get_target_layer(model, model_name):
    model_name = model_name.lower()

    if "convnext" in model_name:
        return model.features[-1]

    if "efficientnet" in model_name:
        return model.features[-1][0]

    if "resnet" in model_name:
        return model.layer4[-1]

    if "densenet" in model_name:
        return model.features[-1]

    raise ValueError("Model not supported for Grad-CAM")


def apply_heatmap(cam, ori_img):
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    ori = np.array(ori_img)
    ori = cv2.resize(ori, (224, 224))

    overlay = heatmap * 0.4 + ori * 0.6
    return heatmap, overlay.astype(np.uint8)


# ================================================================
#  PROCESS FILE & FOLDER
# ================================================================
def process_image(path, cam_gen, model_name, output_dir):
    tensor, pil_img = load_image(path)
    cam = cam_gen.generate(tensor)

    heatmap, overlay = apply_heatmap(cam, pil_img)

    base = os.path.splitext(os.path.basename(path))[0]
    save1 = os.path.join(output_dir, base + "_heatmap.jpg")
    save2 = os.path.join(output_dir, base + "_overlay.jpg")

    cv2.imwrite(save1, heatmap)
    cv2.imwrite(save2, overlay)

    print("Saved:", save1)


def process_folder(folder_path, cam_gen, model_name, output_dir):
    for root, dirs, files in os.walk(folder_path):
        rel = os.path.relpath(root, folder_path)
        save_dir = os.path.join(output_dir, rel)
        os.makedirs(save_dir, exist_ok=True)

        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                process_image(
                    os.path.join(root, f),
                    cam_gen,
                    model_name,
                    save_dir
                )


# ================================================================
#  MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)  # file or folder
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    factory = ModelFactory(args.num_classes)
    model = factory.create(args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    target_layer = get_target_layer(model, args.model)
    cam_gen = GradCAM(model, target_layer)

    if os.path.isfile(args.image):
        process_image(args.image, cam_gen, args.model, args.output_dir)
    else:
        process_folder(args.image, cam_gen, args.model, args.output_dir)


if __name__ == "__main__":
    main()