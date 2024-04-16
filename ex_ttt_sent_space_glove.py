
import numpy as np
from math import ceil
from strlearn.metrics import balanced_accuracy_score as bac
from tqdm import tqdm
from strlearn.metrics import balanced_accuracy_score as bac, recall, precision, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from cv2 import resize
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from gensim import downloader

X = np.load("fakeddit_stream/fakeddit_posts.npy", allow_pickle=True)
bias = np.load("fakeddit_stream/fakeddit_posts_y.npy")
# How many classes?
bias_id = 0
print(X.shape)
print(bias.shape)

# Only titles, without timestamp
# Binary problem
stream = X[:, 0]
y = np.array([1,0])[bias[:,bias_id]] if bias_id == 0 else bias[:,bias_id]
print(np.unique(y, return_counts=True))

chunk_size = 250
# All chunks
n_chunks = ceil(stream.shape[0]/chunk_size)
# Select dummies
classes = np.unique(y)
n_classes = len(classes)
dummies = stream[[np.where(y==label)[0][0] for label in classes]]

metrics=(recall, recall_score, precision, precision_score, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2, bac, balanced_accuracy_score)

"""
Model
"""
num_classes = 2
batch_size = 8
num_epochs = 1
# To transfer or not to transfer?
# weights = ResNet18_Weights.IMAGENET1K_V1
weights = None


model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("mps")
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# imb_weight = torch.from_numpy(np.array(imb_weights)).float().to(device)

criterion = nn.CrossEntropyLoss()

vectors = downloader.load('glove-wiki-gigaword-300')
# print(vectors)

# METHODS x CHUNKS x METRICS
# transformer = SentenceTransformer('all-MiniLM-L6-v2', device=device).to(device)
results = []
for chunk_id in tqdm(range(n_chunks)):
    chunk_X = stream[chunk_id*chunk_size:chunk_id*chunk_size+chunk_size]
    chunk_y = y[chunk_id*chunk_size:chunk_id*chunk_size+chunk_size]
    
    if len(np.unique(chunk_y)) != n_classes:
        chunk_X[:n_classes] = dummies
        chunk_y[:n_classes] = classes
    
    chunk_images = []
    for text_id, text in enumerate(tqdm(chunk_X, disable=True)):

        words = text.split(" ")

        # wordvecs = np.zeros((300,len(words)))
        wordvecs = np.zeros((len(words), 300))
        for idx, word in enumerate(words):
            try:
                # wordvecs[:, idx] = vectors[word]
                wordvecs[idx] = vectors[word]
            except KeyError:
                pass
            
        img = resize(wordvecs, (300, 200))
        rgb = np.stack((img, img, img), axis=0)
        chunk_images.append(rgb)
        # plt.imshow(img)
        # plt.title(text)
        # plt.tight_layout()
        # plt.savefig("bar.png")
        # exit()

    chunk_images = np.array(chunk_images)    
    
    chunk_X = torch.from_numpy(chunk_images).float()
    chunk_y = torch.from_numpy(chunk_y).long()
    
    stml_dataset = TensorDataset(chunk_X, chunk_y)
    data_loader = DataLoader(stml_dataset, batch_size=batch_size, shuffle=True)
    
    if chunk_id==0:
        model.train()
        for epoch in range(num_epochs):
            for i, batch in enumerate(data_loader, 0):
                inputs, labels = batch

                optimizer.zero_grad()

                outputs = model(inputs.to(device))
                loss = criterion(outputs.to(device), labels.to(device))
                loss.backward()
                optimizer.step()
                
    else:
        model.eval()
        logits = model(chunk_X.to(device))
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
        preds = np.argmax(probs, 1)
        scores = [metric(chunk_y.numpy(), preds) for metric in metrics]
        results.append(scores)
        
        model.train()
        for epoch in range(num_epochs):
            for i, batch in enumerate(data_loader, 0):
                inputs, labels = batch

                optimizer.zero_grad()

                outputs = model(inputs.to(device))
                loss = criterion(outputs.to(device), labels.to(device))
                loss.backward()
                optimizer.step()
results = np.array(results)
np.save("results/scores_sentence_space_glove_imgfixed_200_notransfer", results)