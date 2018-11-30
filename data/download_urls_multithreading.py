#!/usr/bin/env python
""" 
Tencent is pleased to support the open source community by making Tencent ML-Images available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and limitations under the License.
"""

import os
import sys
import urllib
import argparse
import threading,signal
import time
import socket
socket.setdefaulttimeout(10.0) 

def downloadImg(start, end, url_list, save_dir):
    global record,count,count_invalid,is_exit
    im_names = []
    with open(url_list, 'r')  as url_f:
        for line in url_f.readlines()[start:end]:
            sp = line.rstrip('\n').split('\t')
            url = sp[0]
            url_list = url.split('/')
            im_name = url_list[-2] + '_' + url_list[-1]
            try:
                urllib.urlretrieve(url, os.path.join(save_dir, im_name))
                record += 1
                im_file_Record.write(im_name + '\t' + '\t'.join(sp[1:]) + '\n')
                print('url = {} is finished and {} imgs have been downloaded of all {} imgs'.format(url, record, count))
            except IOError as e:
                print ("The url:{} is ***INVALID***".format(url))
                invalid_file.write(url + '\n')
                count_invalid += 1

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--url_list', type=str, help='the url list file')
    parser.add_argument('--im_list', type=str, default='img.txt',help='the image list file')
    parser.add_argument('--num_threads', type=int, default=8, help='the num of processing')
    parser.add_argument('--save_dir', type=str, default='./images', help='the directory to save images')
    args = parser.parse_args()

    url_list = args.url_list
    im_list = args.im_list
    num_threads = args.num_threads
    save_dir = args.save_dir
    # create savedir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    count = 0 # the num of urls
    count_invalid = 0 # the num of invalid urls
    record = 0
    with open(url_list,'r') as f:
        for line in f:
            count += 1
    part = int(count/num_threads)  
    with open(im_list, 'w') as im_file_Record,open('invalid_url.txt','w') as invalid_file: # record the downloaded imgs
        thread_list = []
        for i in range(num_threads):
            if(i == num_threads-1):
                t = threading.Thread(target = downloadImg, kwargs={'start':i*part, 'end':count, 'url_list':url_list, 'save_dir':save_dir})
            else:
                t = threading.Thread(target = downloadImg, kwargs={'start':i*part, 'end':(i+1)*part, 'url_list':url_list, 'save_dir':save_dir})
            t.setDaemon(True)
            thread_list.append(t)
            t.start()
        
        for i in range(num_threads):
            try:
                while thread_list[i].isAlive():
                    pass
            except KeyboardInterrupt:
                break

        if count_invalid==0:
            print ("all {} imgs have been downloaded!".format(count))
        else:
            print("{}/{} imgs have been downloaded, {} URLs are invalid".format(count-count_invalid, count, count_invalid))
