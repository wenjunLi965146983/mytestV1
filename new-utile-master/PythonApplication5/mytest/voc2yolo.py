import xml.etree.ElementTree as ET

import os
classes=["orange","lemon"]
import utile

path='C:/Users/rdpublic/Desktop/test/'

def convert_annotation(path, image_id, list_file):
    in_file = open('%s/annotation/%s.xml'%(path,image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))



imagepath="C:/Users/rdpublic/Desktop/test/image/"
image_ids = utile.get_datalist(imagepath)

list_file = open('yolo_annatation.txt', 'w')
for image_id in image_ids:
    xml_name=image_id.split('.')[0]
    list_file.write('%s.jpg'%( image_id))
    convert_annotation(path, xml_name, list_file)
    list_file.write('\n')
list_file.close()