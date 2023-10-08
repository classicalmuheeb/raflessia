import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

class EuroSATDataset(Dataset):
    def __init__(self, train_df, train_dir, is_train, transform=None):
        self.train_df = train_df
        self.train_dir = train_dir
        self.transform = transform
        self.is_train = is_train

        # Convert label to list
        labelArr = self.train_df['label'].unique()

        self.label2id = {}
        self.id2label = {}
        index = 0
        for class_name in labelArr:
            self.label2id[class_name] = str(index)
            self.id2label[str(index)] = class_name
            index = index + 1

        #self.targets = torch.tensor(self.train_df['label'].apply(lambda x: int(self.label2id[x])))
                
        self.targets = []
        for label in self.train_df['label']:
            self.targets.append(int(self.label2id[label]))

        self.targets = torch.tensor(self.targets)

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_name = os.path.join(self.train_dir, self.train_df.iloc[index, 0])
        image1 = cv2.imread(image_name)
        image = Image.fromarray(image1)

        if self.is_train:
            labelKey = self.train_df.iloc[index, 1]
            label = torch.tensor(int(self.label2id[labelKey]))
        else:
            label = torch.tensor(1)

        if self.transform:
            image = self.transform(image)

        return image, label