import numpy as np
import gzip
import os
import urllib

def download(filename, work_directory, source_url):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory, 0o755)
    filepath = os.path.join(work_directory,filename)
    if not os.path.exists(filepath):
        urllib.urlretrieve(source_url, filename)
        print("successfully download", filename)

def extract_images(filename):
    

        

if __name__=="__main__":
    source_url =  "http://yann.lecun.com/exdb/mnist"
    train_set = "train-images-idx3-ubyte.gz"
    download(train_set, './', source_url)

