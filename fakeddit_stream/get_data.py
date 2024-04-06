import tarfile
from PIL import Image
import PIL
import io
import numpy as np
import pandas as pd
from cv2 import resize
from torchvision.models import ResNet18_Weights
from torch import from_numpy
import matplotlib.pyplot as plt
import traceback
from tqdm import tqdm
import sys
import datetime


root = "../../../../../Volumes/T7/Multimodal/fakeddit/"

# TEXT ------------------------

# Load posts
txt_files = ["multimodal_test_public.tsv", "multimodal_validate.tsv", "multimodal_train.tsv"]

post_data = None
for file in txt_files:
    tmp_data = pd.read_csv(root + file, delimiter='\t', on_bad_lines='skip', index_col=False)

    if post_data is None:
        post_data = tmp_data
    else:
        post_data = pd.concat([post_data, tmp_data], ignore_index=True)

# Sample data
post_data = post_data.sort_values('created_utc')
# post_data['created_utc'] = post_data['created_utc'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
print(post_data[['clean_title', 'created_utc', '2_way_label', '3_way_label', '6_way_label']])

# Post label
y_2 = post_data["2_way_label"]
y_3 = post_data["3_way_label"]
y_6 = post_data["6_way_label"]
y_post = np.vstack((y_2, y_3, y_6)).T

# Save to numpy
np.save(file='fakeddit_posts', arr=post_data[['clean_title', 'created_utc']].values)
np.save(file='fakeddit_posts_y', arr=y_post)