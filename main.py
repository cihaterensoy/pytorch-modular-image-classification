import torch
from torch.nn.functional import batch_norm
from torchvision import transforms, datasets
import setup_data,training_testing_engine,model_creation,utils


def main():

    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    HIDDEN_UNITS = 32
    LEARNING_RATE = 0.001

    train_dir="data/desert101/train"
    test_dir="data/desert101/test"


    mean, std = utils.get_stats(train_dir,batch_size=BATCH_SIZE)
    data_transforms = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),  # koyduğumuz görsellerin yüzde 40ını rastgele seç ve yatay çevirme yap
        transforms.TrivialAugmentWide(),  # bir veriye rastgele işlemlerden sadece birini yapar.
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    train_dataloader, test_dataloader,class_names=setup_data.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transforms,
        batch_size=BATCH_SIZE
    )
    model=model_creation.DesertClassifier(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    )

    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    results=training_testing_engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS
    )
    print("Final results:",results)
    utils.save_model(
        model=model,
        target_dir="models",
        model_name="desert_classifier.pth"
    )
if __name__=="__main__":
    #torch.multiprocessing.set_start_method("spawn",force=True)macte hata cıkarsa diye
    main()