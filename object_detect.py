import argparse
import os
import numpy as np
from yolo import yolo_detect
from SSD import SSD_detect
from mask_rcnn import mask_detect


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="input image dataset")
    ap.add_argument('-v', '--visualize', action='store_true', help="whether or not visualize each instance")
    ap.add_argument('-l', '--savelog', action='store_true', help="whether or not print results in a file")
    ap.add_argument('-i', '--input_path', default='MI3', help='input image path')
    ap.add_argument('-m', '--method', default=2, help='method to adopt(1=SSD, 2=yolo, 3=faster R-CNN, 4=mask R-CNN) ')
    args = vars(ap.parse_args())

    dataset = args['dataset']
    vis = args['visualize']
    log = args['savelog']
    method = args['method']
    model_path = 'models'


    MI3path = args['input_path']

    #    dataset = 'Pathway1_1'
    fo = open(method + '_' + dataset + ".txt", "w")
    channel_list = [2, 4, 6]

    for channel in channel_list:
        input_folder = os.path.join(MI3path, dataset, "ORIG/ch" + str(channel))
        for filename in os.listdir(input_folder):
            if method==1:
                SSD_detect(model_path,fo,dataset, input_folder, filename, channel, vis, log)
            elif method ==2:
                yolo_detect(model_path,fo,dataset, input_folder, filename, channel, vis, log)
            #elif method ==3:
            elif method == 4:
                mask_detect(model_path,fo,dataset, input_folder, filename, channel, vis, log)
        # detect(input_folder, filename,output_folder='output/'+dataset+'ch'+channel)
    fo.close()