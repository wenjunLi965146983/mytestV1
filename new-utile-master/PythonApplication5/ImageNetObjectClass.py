import json
import os
import ImageNetObjectClass

def read_json(path):
    file=open(path)
    file_str=file.read()
    file.close()
    json_data=json.loads(file_str)
    return json_data





if __name__=='__main__':
    read_json('E:/github/imagenet_1000_labels.json')
