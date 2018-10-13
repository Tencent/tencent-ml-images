echo "Generating tfrecords ..."

./tfrecord.py -idx image_lists/ -tfs tfrecords/ -im images/ -cls 11166 -one True
