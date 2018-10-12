#!/usr/bin/python
import sys
import urllib
import argparse

def main(url_list, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    i=0
    with open(url_list, 'r') as lines:
        for line in lines:
            sp = line.rstrip('\n').split('\t')
            url = sp[0]
            urllib.urlretrieve(url, save_dir + 'im_'+str(i)+'.png')
            i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--url_list', type=str, help='the url list file')
    parser.add_argument('--save_dir', type=str, default='./', help='the directory to save images')
    args = parser.parse_args()

    main(url_list=args.url_list, save_dir=args.save_dir)
