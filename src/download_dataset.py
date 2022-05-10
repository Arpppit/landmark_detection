import random
import re
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import sys
import requests
import shutil
import os
data = pd.read_csv('../Data/train.csv')
test_data = pd.read_csv('../Data/test.csv')
print(data.head(5))
landmark_list = [str(x) for x in list(range(1000, 3000))]
data_sample = data[data['landmark_id'].isin(landmark_list)]


# Check the distribution
# %matplotlib inline

colors = np.array(['#4285f4', '#34a853', '#fbbc05', '#ea4335'])
# Define the order in which to display the graph
order = ['1-5', '5-10', '10-50', '50-100', '100-200', '200-500', '>=500']
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))


def plot_distribution(data_f, data_k, axis):
    # data['landmark_id'].value_counts()
    x = data_f.landmark_id.value_counts().index
    y = pd.DataFrame(data_f.landmark_id.value_counts())

    # Create a variable to group the number of image sin each class
    y['Number of images'] = np.where(
        y['landmark_id'] >= 500, '>=500', y['landmark_id'])
    y['Number of images'] = np.where((y['landmark_id'] >= 200) & (
        y['landmark_id'] < 500), '200-500', y['Number of images'])
    y['Number of images'] = np.where((y['landmark_id'] >= 100) & (
        y['landmark_id'] < 200), '100-200', y['Number of images'])
    y['Number of images'] = np.where((y['landmark_id'] >= 50) & (
        y['landmark_id'] < 100), '50-100', y['Number of images'])
    y['Number of images'] = np.where((y['landmark_id'] >= 10) & (
        y['landmark_id'] < 50), '10-50', y['Number of images'])
    y['Number of images'] = np.where((y['landmark_id'] >= 5) & (
        y['landmark_id'] < 10), '5-10', y['Number of images'])
    y['Number of images'] = np.where((y['landmark_id'] >= 0) & (
        y['landmark_id'] < 5), '1-5', y['Number of images'])

    y['Number of images'].value_counts().loc[order].plot(
        kind='bar', color=colors, width=0.8, ax=axis)
    axis.set_xlabel('Number of images')
    axis.set_ylabel('Number of classes')
    axis.set_title(data_k)


plot_distribution(data, 'Original', ax1)
plot_distribution(data_sample, 'Sample', ax2)


TARGET_SIZE = 96  # imports images of resolution 96x96

'''change URLs to resize images to target size'''


def overwrite_urls(df):
    def reso_overwrite(url_tail, reso=TARGET_SIZE):
        pattern = 's[0-9]+'
        search_result = re.match(pattern, url_tail)
        if search_result is None:
            return url_tail
        else:
            return 's{}'.format(reso)

    def join_url(parsed_url, s_reso):
        parsed_url[-2] = s_reso
        return '/'.join(parsed_url)

    df = df[df.url.apply(lambda x: len(x.split('/')) > 1)]
    parsed_url = df.url.apply(lambda x: x.split('/'))
    train_url_tail = parsed_url.apply(lambda x: x[-2])
    resos = train_url_tail.apply(lambda x: reso_overwrite(x, reso=TARGET_SIZE))

    overwritten_df = pd.concat([parsed_url, resos], axis=1)
    overwritten_df.columns = ['url', 's_reso']
    df['url'] = overwritten_df.apply(
        lambda x: join_url(x['url'], x['s_reso']), axis=1)
    return df


data_sample_resize = overwrite_urls(data_sample)
print('1. URLs overwritten')

'''Split to test and train'''
data_test = pd.DataFrame(columns=['id', 'url', 'landmark_id'])
data_training_all = pd.DataFrame(columns=['id', 'url', 'landmark_id'])
percent_test = 0.01  # takes 1% from each class as holdout data

random.seed(42)
for landmark_id in set(data_sample_resize['landmark_id']):
    n = 1
    # get all images for a landmark id
    t = data_sample_resize[(data_sample_resize.landmark_id == landmark_id)]
    i = 0
    r = []
    while i < len(t.id):
        it = i
        r.append(t.id.iloc[it])  # create a list of all these images
        i += 1

    # randomly pick a sample of 1% images from list 'r'
    test = random.sample(r, int(percent_test*len(r)))
    training = list(set(r) - set(test))  # get the remaining images
    # holdout dataset
    data_t = data_sample_resize[data_sample_resize.id.isin(test)]
    data_tr = data_sample_resize[data_sample_resize.id.isin(
        training)]  # training dataset
    data_test = data_test.append(data_t)
    data_training_all = data_training_all.append(data_tr)
    n += 1

print('2. train and test set created')


'''Split into train and validation set'''
data_valid = pd.DataFrame(columns=['id', 'url', 'landmark_id'])
data_train = pd.DataFrame(columns=['id', 'url', 'landmark_id'])
percent_validation = 0.2  # takes 20% from each class as holdout data
random.seed(42)
for landmark_id in set(data_training_all['landmark_id']):
    n = 1
    t = data_training_all[(data_training_all.landmark_id == landmark_id)]
    i = 0
    r = []
    while i < len(t.id):
        it = i
        r.append(t.id.iloc[it])
        i += 1

    valid = random.sample(r, int(percent_validation*len(r)))
    train = list(set(r) - set(valid))
    data_v = data_training_all[data_training_all.id.isin(valid)]
    data_t = data_training_all[data_training_all.id.isin(train)]
    data_valid = data_valid.append(data_v)
    data_train = data_train.append(data_t)
    n += 1

print('3. train and validation set created')
print("Size of train dataset is :", len(data_train))
print("Size of valid dataset is :", len(data_valid))
print("Size of test dataset is :", len(data_test))


def fetch_image(path, folder):
    url = path
    response = requests.get(url, stream=True)
    with open('../Data/' + folder + '/image.jpg', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response


'''TRAIN SET - fetch images for the resized URLs and save in the already created directory train_images_model'''
i = 0
for link in data_train['url']:  # looping over links to get images
    if os.path.exists('../Data/train_images_model/'+str(data_train['id'].iloc[i])+'.jpg'):
        i += 1
        continue
    fetch_image(link, 'train_images_model')
    os.rename('../Data/train_images_model/image.jpg',
              '../Data/train_images_model/' + str(data_train['id'].iloc[i]) + '.jpg')
    i += 1
#     if(i==50):   #uncomment to test in your machine
#         break
print('4. train images fetched')


i = 0
for link in data_valid['url']:  # looping over links to get images
    if os.path.exists('../Data/validation_images_model/'+str(data_valid['id'].iloc[i])+'.jpg'):
        i += 1
        continue
    fetch_image(link, 'validation_images_model')
    os.rename('../Data/validation_images_model/image.jpg',
              '../Data/validation_images_model/' + str(data_valid['id'].iloc[i]) + '.jpg')
    i += 1
#     if(i==50):   #uncomment to test in your machine
#         break
print('5. Validation images fetched')

i = 0
for link in data_test['url']:  # looping over links to get images
    if os.path.exists('../Data/test_images_from_train/'+str(data_test['id'].iloc[i])+'.jpg'):
        i += 1
        continue
    fetch_image(link, 'test_images_from_train')
    os.rename('../Data/test_images_from_train/image.jpg',
              '../Data/test_images_from_train/' + str(data_test['id'].iloc[i]) + '.jpg')
    i += 1
#     if(i==50):   #uncomment to test in your machine
#         break
print('6. Test images fetched')
