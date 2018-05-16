import argparse
import random
import os
import shutil
from tqdm import tqdm
import pdb
from PIL import Image
import pandas as pd
import numpy as np

# Add arguments in case we change the name or location of the relevant directories
parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', default='dataset/images', help='Directory with CXR images')
parser.add_argument('--labels_dir', default='dataset/labels', help='Directory with csv containing CXR labels')
parser.add_argument('--output_dir', default='data/CXR', help='Directory containing two subdirectories, one for test and one for train')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Check to ensure that the folders we will be pulling the images and labels from exists
    assert os.path.isdir(args.images_dir), 'Could not find images folder at {}'.format(args.images_dir)
    assert os.path.isdir(args.labels_dir), 'Could not find labels folder at {}'.format(args.labels_dir)
    
    # Get all of the image filenames and ensure that we only get .png's
    image_filenames = os.listdir(args.images_dir)
    image_filenames = [os.path.join(args.images_dir, f) for f in image_filenames if f.endswith('.png')]
#     pdb.set_trace()
    
    # Get all the csv filenames and ensure that we only get .csv's
    label_filenames = os.listdir(args.labels_dir)
    label_filenames = [os.path.join(args.labels_dir, f) for f in label_filenames if f.endswith('.csv')]
#     pdb.set_trace()
    
    # Read Relevant Columns from CSV and convert to numpy arrays
    label_data = pd.read_csv(label_filenames[0], usecols=['Image Index', 'Finding Labels']) 
    filename_vec = label_data['Image Index'].values
    label_vec = label_data['Finding Labels'].values
#     pdb.set_trace()
    
    # Randomly split images into 95% train/val and 5% test -> subject to change. Store in a dictionary
    # Use a fixed seed for a reproducable split
    random.seed(230)
    image_filenames.sort()
    random.shuffle(image_filenames)
    
    split = int(0.95 * len(image_filenames))
    train_filenames = image_filenames[:split] # train is technically train and val. This will be split in build_dataset.py
    test_filenames = image_filenames[split:]
    
    filename_dict = {'train': train_filenames,
                     'test': test_filenames}
    
    # Check if the output directory already exists. If it does, delete it to avoid conflicts later. Then make a new directory
    if os.path.exists(args.output_dir):
        print("Warning: {} already exists. Deleting to avoid conflicts.".format(args.output_dir))
        try:
            shutil.rmtree(args.output_dir)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename,e.strerror))
    os.mkdir(args.output_dir)
    
    for split in ['train', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_cxr'.format(split))
        os.mkdir(output_dir_split) # rmtree should have removed this already
        
        print("Processing {} images and saving to {}".format(split, output_dir_split))
        for filename in tqdm(filename_dict[split]):
#             pdb.set_trace()
            curr_image = Image.open(filename)
            curr_file = filename.split('/')[-1]
#             pdb.set_trace()
            ix = np.where(filename_vec==curr_file)
            if len(ix[0]) > 1:
                print("The current file is {}.".format(curr_file))
            curr_ix = ix[0][0]
#             pdb.set_trace()
            if "neumonia" in label_vec[curr_ix]:
                val = '1_'
            else:
                val = '0_'
#             pdb.set_trace()
            curr_image.save(os.path.join(output_dir_split,val+filename.split('/')[-1]))