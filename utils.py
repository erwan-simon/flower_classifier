import os
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import torchvision
import torch
import os.path
import shutil
from shutil import copyfile
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

original_data_path = "flowers_original/"
generated_data_path = "flowers_data/"

def generate_data(train_data_ratio=0.9):
    dest_dir = generated_data_path
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    if not os.path.exists(dest_dir + "Test/"):
        os.mkdir(dest_dir + "Test/")
    if not os.path.exists(dest_dir + "Train/"):
        os.mkdir(dest_dir + "Train/")

    directory = original_data_path

    for subdir, dirs, files in os.walk(directory):
        if len(dirs) != 0:
            category_list = dirs
            print("categories : " + str(category_list))
            for caregory_name in category_list:
                if not os.path.exists(dest_dir + "Test/" + caregory_name):
                    os.makedirs(dest_dir + "Test/" + caregory_name)
                if not os.path.exists(dest_dir + "Train/" + caregory_name):
                    os.makedirs(dest_dir + "Train/" + caregory_name)
            continue
        random.shuffle(files)
        train_pictures = files[:int(len(files) * train_data_ratio)]
        test_pictures = files[len(train_pictures):]
        print(subdir)
        for picture in train_pictures:
            filepath = subdir + os.sep + picture
            if filepath.endswith(".jpg"):
                if not os.path.exists(dest_dir + "Train/" + subdir.split("/")[-1] + "/" + picture):
                    copyfile(filepath, dest_dir + "Train/" + subdir.split("/")[-1] + "/" + picture)
                    print("Creating " + str(dest_dir + "Train/" + subdir.split("/")[-1] + "/" + picture))
            else:
                print("Something is not an image")
        for picture in test_pictures:
            filepath = subdir + os.sep + picture
            if filepath.endswith(".jpg"):
                if not os.path.exists(dest_dir + "Test/" + subdir.split("/")[-1] + "/" + picture):
                    copyfile(filepath, dest_dir + "Test/" + subdir.split("/")[-1] + "/" + picture)
                    print("Creating " + str(dest_dir + "Test/" + subdir.split("/")[-1] + "/" + picture))
            else:
                print("Something is not an image : " + filepath)

def get_category_list():
    for subdir, dirs, files in os.walk(original_data_path):
        if len(dirs) != 0:
            category_list = dirs
            return category_list

def construct_loaders(batch_size=64):
    def load_dataset():
        data_path = generated_data_path

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        
        transform_1 = transforms.Compose([
                transforms.Resize((200, 200)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation([-45, 45]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[mean[0], mean[1], mean[2]],
                                     std=[std[0], std[1], std[2]])
            ])
        transform_2 = transforms.Compose([
                transforms.Resize((200, 200)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[mean[0], mean[1], mean[2]],
                                     std=[std[0], std[1], std[2]])
            ])

        train_dataset = torchvision.datasets.ImageFolder(
            root=data_path + "Train/",
            transform=transform_1
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=data_path + "Test/",
            transform=transform_2
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True
        )
        return train_loader, test_loader


    train_loader, test_loader = load_dataset()
    
    """
    # show test dataset
    for data, target in test_loader:
        trans_to_pil = transforms.ToPILImage()
        plt.imshow(trans_to_pil(data[1,:]))
        plt.show()
        print(category_list[target[1]])
        break
    """
    return train_loader, test_loader

def compute_accuracy(model, dataloader, device='cuda:0', display_errors=False):
    category_list = get_category_list()
    training_before = model.training
    model.eval()

    all_predictions = []
    all_targets = []
    
    trans_to_pil = transforms.ToPILImage()
    
    for i_batch, batch in enumerate(dataloader):
        images, targets = batch
        images = images.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            predictions = model(images)
        
        if display_errors:
            for (prediction, target, image) in zip(predictions, targets, images):
                if prediction.argmax() != target:
                    plt.imshow(trans_to_pil(image.cpu()))
                    plt.show()
                    print("categories = " + str(category_list))
                    print("prediction = " + str(prediction))
                    print("target = " + str(target))
                    print()
        
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    if all_predictions[0].shape[-1] > 1:
        predictions_numpy = np.concatenate(all_predictions, axis=0)
        predictions_numpy = predictions_numpy.argmax(axis=1)
        targets_numpy = np.concatenate(all_targets, axis=0)
    else:
        predictions_numpy = np.ravel(all_predictions)
        targets_numpy = np.ravel(all_targets)
        predictions_numpy[predictions_numpy >= 0.5] = 1.0
        predictions_numpy[predictions_numpy < 0.5] = 1.0

    if training_before:
        model.train()

    return (predictions_numpy == targets_numpy).mean()

def predict_image(path_to_test_image, model):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    loader = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean[0], mean[1], mean[2]],
                             std=[std[0], std[1], std[2]])
        ])

    def image_loader(image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        plt.imshow(image)
        plt.show()
        image = loader(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        return image.cuda()  #assumes that you're using GPU

    image = image_loader(path_to_test_image)
    model.eval()
    output = model(image)
    category_list = get_category_list()
    print(category_list)
    print("Catgory found : " + str(category_list[output.argmax()]))
    print("Output : " + str(output))
    print()

def train_model(model, train_loader, test_loader, nb_epochs=10, learning_rate=0.01, momentum=0.9, device="cuda:0"):
    model.to(device)
    print("Test accuracy before training : {:.6f}%".format(compute_accuracy(model, test_loader, device) * 100))
    print()
    criterion = nn.CrossEntropyLoss()

    params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum)

    # Set to training mode
    model.train()
    train_accuracies = []
    test_accuracies = []
    for epoch in range(nb_epochs):
        epoch_start = time.time()

        # Loss within the epoch
        train_losses = []

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to('cuda:0')
            labels = labels.to('cuda:0')

            # Clean existing gradients
            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            train_losses.append(loss.item())

            # print("Batch number: {:03d}, Loss: {:.4f}".format(i, loss.item()))
        train_accuracies.append(compute_accuracy(model, train_loader, device))
        test_accuracies.append(compute_accuracy(model, test_loader, device))
        print("    Epoch number: {:03d}, Training: Loss: {:.4f}".format(epoch, np.mean(train_losses)))
        print("Train accuracy after training : {:.6f}%".format(train_accuracies[-1] * 100))
        print("Test accuracy after training : {:.6f}%".format(test_accuracies[-1] * 100))
        print('')
    return train_accuracies, test_accuracies

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model