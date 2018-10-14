#!/usr/bin/python
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

def main(url_list, im_list, save_dir):
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    i=0
    with open(url_list, 'r') as url_lines, open(im_list, 'w') as im_lines:
        for line in url_lines:
            sp = line.rstrip('\n').split('\t')
            url = sp[0]
            urllib.urlretrieve(url, save_dir + 'im_'+str(i)+'.jpeg')
            
            im_line = 'im_'+str(i)+'.jpeg' + '\t' + '\t'.join(sp) + '\n'
            im_lines.write(im_line)
            i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--url_list', type=str, help='the url list file')
    parser.add_argument('--im_list', type=str, help='the image list file')
    parser.add_argument('--save_dir', type=str, default='./', help='the directory to save images')
    args = parser.parse_args()

    main(url_list=args.url_list, im_list=args.im_list, save_dir=args.save_dir)
