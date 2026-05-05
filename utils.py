from pathlib import Path
import torch
from torch.nn.functional import batch_norm
from torchvision import transforms, datasets

def save_model(model: torch.nn.Module,target_dir:str, model_name: str):
    model_path = Path(target_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    model_save_path = model_path / model_name

    torch.save(model.state_dict(), model_save_path)
def model_yukle(model: torch.nn.Module,target_dir:str, model_name: str):
    model_path = Path(target_dir)/model_name
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_stats(dataset_path,batch_size):
    temp_transforms=transforms.Compose([transforms.Resize(size=(64, 64)),transforms.ToTensor()])
    dataset=datasets.ImageFolder(dataset_path,transform=temp_transforms)
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False)
    sum_ = torch.tensor([0.0, 0.0, 0.0])
    sum_sq = torch.tensor([0.0, 0.0, 0.0])
    total_pixels = 0
    for images,_ in loader:
        batch_size=images.size(0)
        num_pixels=batch_size*64*64
        total_pixels += num_pixels
        sum_ += torch.sum(images,dim=[0,2,3])
        sum_sq += torch.sum(images**2,dim=[0,2,3])

    mean = sum_ / total_pixels

    # Genel standart sapma: sqrt( E[X^2] - (E[X])^2 )
    std = torch.sqrt((sum_sq / total_pixels) - (mean ** 2))
    return mean.tolist(), std.tolist()

if __name__ == '__main__':
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    HIDDEN_UNITS = 32
    LEARNING_RATE = 0.001

    train_dir = "data/desert101/train"
    test_dir = "data/desert101/test"

    mean, std = get_stats(train_dir, batch_size=BATCH_SIZE)

    mean,std=get_stats(train_dir,batch_size=BATCH_SIZE)
    print(f"mean: {mean}, std: {std}")