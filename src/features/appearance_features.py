import torch
import torchvision.models as models
import torchvision.transforms as transforms

def extract_appearance_features(frames, device='cpu'):
    vgg = models.vgg16(pretrained=True).features.to(device).eval()  
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    features = []
    for frame in frames:
        inp = preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = vgg(inp)
            features.append(feat.flatten().cpu().numpy())
    return np.array(features)
