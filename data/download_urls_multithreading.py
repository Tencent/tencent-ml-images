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

debug = False

def downloadImg(i, url_file_mem,start, end, url_list, save_dir):
    print('Thread {} starting, set to download from {} to {}'.format(i, start, end))
    global record,count,count_invalid,count_already_done,is_exit
    im_names = []
    for line in url_file_mem[start:end]:
        sp = line.rstrip('\n').split('\t')
        url = sp[0]
        im_name = url.split('/')[-1]
        if os.path.isfile(os.path.join(save_dir, im_name)):
            count_already_done+=1
            if debug: print 'T', i, count_already_done, "already done" , os.path.join(save_dir, im_name)
        else:
            try:
                if debug: print 'T', i, "fetching", url
                urllib.urlretrieve(url, os.path.join(save_dir, im_name))
                record += 1
                im_file_Record.write(im_name + '\t' + '\t'.join(sp[1:]) + '\n')
                if debug: print('url = {} is finished and {} imgs have been downloaded of all {} imgs'.format(url, record, count))
            except:
                if debug: print ("The url:{} is ***INVALID***".format(url))
                invalid_file.write(url + '\n')
                count_invalid += 1
    if debug: print 'T', i, 'exiting'


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
    count_already_done = 0

    # Open url list in main thread, share it with other threads
    with open(url_list, 'r')  as url_f:
        url_file_mem = url_f.readlines()
        count = len(url_file_mem)
        print "loaded list of %d urls"%count

    part = int(count/num_threads)

    with open(im_list, 'w') as im_file_Record, open('invalid_url.txt','w') as invalid_file: # record the downloaded imgs
        thread_list = []

        t0 = time.time()
        for i in range(num_threads):
            if(i == num_threads-1):
                t = threading.Thread(name='Downloader %d'%i, target = downloadImg, kwargs={"i":i,"url_file_mem":url_file_mem,'start':i*part, 'end':count, 'url_list':url_list, 'save_dir':save_dir})
            else:
                t = threading.Thread(name='Downloader %d'%i, target = downloadImg, kwargs={"i":i,"url_file_mem":url_file_mem,'start':i*part, 'end':(i+1)*part, 'url_list':url_list, 'save_dir':save_dir})
            t.setDaemon(True)
            thread_list.append(t)
            t.start()
        
        for i in range(num_threads):
            try:
                t = thread_list[i]
                while t.is_alive():
                    t.join(.25)
                    t1 = time.time()
                    if t1 - t0 > 1:
                        print('{} threads. invalid={}, already={}, record={}'.format(threading.active_count(),count_invalid,count_already_done,record))
                        t0 = t1
                print 'Thread', t.name, ' is done'
            except KeyboardInterrupt:
                break

        if count_invalid==0:
            print ("all {} imgs have been downloaded!".format(count))
        else:
            print("{}/{} imgs have been downloaded, {} URLs are invalid".format(record, count, count_invalid))
