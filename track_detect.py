import numpy as np
import yaml
import os
import cv2
import time
import torch
import argparse
import pickle
from PIL import Image
from read_detector import readFrameBBoxList
from modules.model import MDNet, BCELoss, set_optimizer
from modules.sample_generator import SampleGenerator
from tracking.bbreg import BBRegressor
from tracking.data_prov import RegionExtractor
from draw_result import draw_image
from modules.utils import overlap_ratio
opts = yaml.safe_load(open('tracking/options.yaml','r'))

class MDTrack:
    def __init__(self, frame_num, file_path, tracklet_num,frameBBoxList):
        self.frame_num = frame_num
        self.FILE_PATH = file_path
        self.TRACKLET_NUM = tracklet_num
        self.IOU_count_th = tracklet_num/2
        self.GLOBAL_TRACK_LIST = []
        self.frameBBoxList = frameBBoxList

    def forward_samples(self,model, image, samples, out_layer='conv3'):
        model.eval()
        extractor = RegionExtractor(image, samples, opts)
        for i, regions in enumerate(extractor):
            if opts['use_gpu']:
                regions = regions.cuda()
            with torch.no_grad():
                feat = model(regions, out_layer=out_layer)
            if i==0:
                feats = feat.detach().clone()
            else:
                feats = torch.cat((feats, feat.detach().clone()), 0)
        return feats


    def train(self,model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
        model.train()

        batch_pos = opts['batch_pos']
        batch_neg = opts['batch_neg']
        batch_test = opts['batch_test']
        batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

        pos_idx = np.random.permutation(pos_feats.size(0))
        neg_idx = np.random.permutation(neg_feats.size(0))
        while(len(pos_idx) < batch_pos * maxiter):
            pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
        while(len(neg_idx) < batch_neg_cand * maxiter):
            neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
        pos_pointer = 0
        neg_pointer = 0

        for i in range(maxiter):

            # select pos idx
            pos_next = pos_pointer + batch_pos
            pos_cur_idx = pos_idx[pos_pointer:pos_next]
            pos_cur_idx = pos_feats.new(pos_cur_idx).long()
            pos_pointer = pos_next

            # select neg idx
            neg_next = neg_pointer + batch_neg_cand
            neg_cur_idx = neg_idx[neg_pointer:neg_next]
            neg_cur_idx = neg_feats.new(neg_cur_idx).long()
            neg_pointer = neg_next

            # create batch
            batch_pos_feats = pos_feats[pos_cur_idx]
            batch_neg_feats = neg_feats[neg_cur_idx]

            # hard negative mining
            if batch_neg_cand > batch_neg:
                model.eval()
                for start in range(0, batch_neg_cand, batch_test):
                    end = min(start + batch_test, batch_neg_cand)
                    with torch.no_grad():
                        score = model(batch_neg_feats[start:end], in_layer=in_layer)
                    if start==0:
                        neg_cand_score = score.detach()[:, 1].clone()
                    else:
                        neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

                _, top_idx = neg_cand_score.topk(batch_neg)
                batch_neg_feats = batch_neg_feats[top_idx]
                model.train()

            # forward
            pos_score = model(batch_pos_feats, in_layer=in_layer)
            neg_score = model(batch_neg_feats, in_layer=in_layer)

            # optimize
            loss = criterion(pos_score, neg_score)
            model.zero_grad()
            loss.backward()
            if 'grad_clip' in opts:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
            optimizer.step()


    def search_track(self,track_num, frame_idx, init_bbox, previous_num,init_conf):
        # img_list, init_bbox

        # Init bbox

        # result = np.zeros((len(img_list), 4))
        # result_bb = np.zeros((len(img_list), 4))
        # result[0] = target_bbox
        # result_bb[0] = target_bbox

        # superorange params
        track_list = [(-1, -1)] * (self.TRACKLET_NUM+1)
        bbox_list = [(frame_idx, init_bbox,init_conf)]
        IOU_count = 0
        # last_bbox = -1
        # next_frame = -1
#        frameA_path = os.path.join(self.FILE_PATH, str(frame_idx) + "."+ext)

        frameA_path = os.path.join(self.FILE_PATH, str(frame_idx).zfill(5)+".png")

        target_bbox = np.array(init_bbox)
        # Init model
        model = MDNet(opts['model_path'])
        if opts['use_gpu']:
            model = model.cuda()

        # Init criterion and optimizer
        criterion = BCELoss()
        model.set_learnable_params(opts['ft_layers'])
        init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])
        update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])

        tic = time.time()
        # Load first image
        image = Image.open(frameA_path).convert('RGB')

        # Draw pos/neg samples
        pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
            target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

        neg_examples = np.concatenate([
            SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
            SampleGenerator('whole', image.size)(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
        neg_examples = np.random.permutation(neg_examples)

        # Extract pos/neg features
        if len(pos_examples)==0 or len(neg_examples)==0:
            print("!!pos_examples=0 skip!!")
            return
        pos_feats = self.forward_samples(model, image, pos_examples)
        neg_feats = self.forward_samples(model, image, neg_examples)

        # Initial training
        self.train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
        del init_optimizer, neg_feats
        torch.cuda.empty_cache()

        # Train bbox regressor
        bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'],
                                         opts['aspect_bbreg'])(
            target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
        bbreg_feats = self.forward_samples(model, image, bbreg_examples)
        bbreg = BBRegressor(image.size)
        bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
        del bbreg_feats
        torch.cuda.empty_cache()

        # Init sample generators for update
        sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
        pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
        neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

        # Init pos/neg features for update
        neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
        neg_feats = self.forward_samples(model, image, neg_examples)
        pos_feats_all = [pos_feats]
        neg_feats_all = [neg_feats]

        spf_total = time.time() - tic

        # fps = len(img_list) / spf_total
        # return result, result_bb, fps

        # Main loop
        for i in range(0, self.TRACKLET_NUM):  # next 10 frame
            frameB_idx = frame_idx + i + 1
        #    print("frameB_idx="+str(frameB_idx))
            if frameB_idx> self.frame_num:
                break
            else:


                frameB_path = os.path.join(self.FILE_PATH, str(frameB_idx).zfill(5)+".png")

                # ------------track by MDNet------------
                # Load image
                image = Image.open(frameB_path).convert('RGB')

                # Estimate target bbox
                samples = sample_generator(target_bbox, opts['n_samples'])
                sample_scores = self.forward_samples(model, image, samples, out_layer='fc6')

                top_scores, top_idx = sample_scores[:, 1].topk(5)
                top_idx = top_idx.cpu()
                target_score = top_scores.mean()
                target_bbox = samples[top_idx]
                if top_idx.shape[0] > 1:
                    target_bbox = target_bbox.mean(axis=0)
                success = target_score > 0

                # Expand search area at failure
                if success:
                    sample_generator.set_trans(opts['trans'])
                else:
                    sample_generator.expand_trans(opts['trans_limit'])

                # Bbox regression
                if success:
                    bbreg_samples = samples[top_idx]
                    if top_idx.shape[0] == 1:
                        bbreg_samples = bbreg_samples[None, :]
                    bbreg_feats = self.forward_samples(model, image, bbreg_samples)
                    bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
                    bbreg_bbox = bbreg_samples.mean(axis=0)
                else:
                    bbreg_bbox = target_bbox

                # Save result
                # result[i] = target_bbox
                # result_bb[i] = bbreg_bbox

         #       print(target_bbox)

                # Data collect
                if success:
                    pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
                    pos_feats = self.forward_samples(model, image, pos_examples)
                    pos_feats_all.append(pos_feats)
                    if len(pos_feats_all) > opts['n_frames_long']:
                        del pos_feats_all[0]

                    neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
                    neg_feats = self.forward_samples(model, image, neg_examples)
                    neg_feats_all.append(neg_feats)
                    if len(neg_feats_all) > opts['n_frames_short']:
                        del neg_feats_all[0]

                # Short term update
                if not success:
                    nframes = min(opts['n_frames_short'], len(pos_feats_all))
                    pos_data = torch.cat(pos_feats_all[-nframes:], 0)
                    neg_data = torch.cat(neg_feats_all, 0)
                    self.train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

                # Long term update
                elif i % opts['long_interval'] == 0:
                    pos_data = torch.cat(pos_feats_all, 0)
                    neg_data = torch.cat(neg_feats_all, 0)
                    self.train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

                torch.cuda.empty_cache()

                bboxT = bbreg_bbox
                anyIOU = False

                for bbox_id, bboxD in enumerate(frameBBoxList[frameB_idx]):
                    if bboxD.match == False:
                        IOU_ratio = overlap_ratio(bboxD.getRec(), bboxT)
                        print("IOUratio={}".format(IOU_ratio))
           #             print(IOU_ratio)
                        if IOU_ratio > 0.3:
            #                print("overlap")
                            track_list[i] = (frameB_idx, bbox_id)
                            IOU_count = IOU_count + 1
                            bbox_list.append((frameB_idx, bboxD.getRec(),bboxD.confidence))
                            anyIOU = True
                            break
                if not anyIOU:
                    bbox_list.append((frameB_idx, bboxT,0))

        print("bbox_list=====")
        print(bbox_list)

        print("track_list=====")
        print(track_list)

        ##debug   show track_list
        # for idx in range(0,10):
        #    print(track_list[idx])
        #    print(bbox_list[idx])

        if IOU_count >= self.IOU_count_th:  # add track
            for idx in range(0, self.TRACKLET_NUM+1):
                bbox_id = track_list[idx][1]
                if bbox_id != -1:
                    # print(frame_idx+idx+1)
                    # print(bbox_id)
                    frameBBoxList[frame_idx + idx + 1][bbox_id].setMatch()

            next_track_frame = -1
            init_bbox_next = []
            bbox_length = len(bbox_list)
            #!!! track_list length=10, bbox_list length=11
            for ii in range(1, bbox_length):
                
                idx = bbox_length - ii
                print("{} {} {}".format(ii,idx,track_list[idx][0]))
                if track_list[idx-1][0] != -1:
                    next_track_frame = (-1) * ii  # count from back
                    print("idx ={} ii={} next_track_frame={}".format(idx,ii,next_track_frame))
                    init_bbox_next = bbox_list[idx]
                    break

            # remove overlap range
            start_rm = -1
            while start_rm >= previous_num:
                del self.GLOBAL_TRACK_LIST[track_num][-1]  # rm 1 per move
                # GLOBAL_TRACK_LIST[track_num].remove(-1)
                start_rm = start_rm - 1

            if previous_num == 0:
                self.GLOBAL_TRACK_LIST.append(bbox_list)
            else:
                self.GLOBAL_TRACK_LIST[track_num].extend(bbox_list)

            next_start_frame = frame_idx +bbox_length + next_track_frame

            init_bbox_next = bbox_list[bbox_length + next_track_frame][1]
            init_conf_next = bbox_list[bbox_length + next_track_frame][2]
            # print(GLOBAL_TRACK_LIST)
            print("( " + str(track_num) + " " + str(next_start_frame) + " " + str(next_track_frame) + ")")
            print("init_bbox_next:"+str(init_bbox_next))
            self.search_track(track_num, next_start_frame, init_bbox_next,
                         next_track_frame,init_conf_next)  # next_start_frame = 310  next_track_frame=-3(will be deleted)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--channel', default=6, help='which channel')
    parser.add_argument('-t', '--tracklet_num', default=10, help='length of tracklet')
    parser.add_argument('-m', '--method',  default='faster-rcnn', help='algorithm')
    parser.add_argument('-d', '--dataset',  default='Pathway1_1', help='dataset')
    parser.add_argument('-n', '--frame_num', default=548, help='total frame')
    parser.add_argument('-s', '--savefig', action='store_true')
    parser.add_argument('-f', '--face', action='store_true', help="whether or not is face detection")

    parser.add_argument('-b', '--begin_frame', default = 0, help='set start frame')
    parser.add_argument('-i', '--input_path', default='MI3', help='input image path')
    parser.add_argument('--threshold',default = 0.8,help='confidence threshold') 
    args = parser.parse_args()
    
    dataset = args.dataset
    method = args.method
    channel = int(args.channel)
    tracklet_num = args.tracklet_num
    frame_num = int(args.frame_num)
    savefig = args.savefig
    detect_face = args.detect_face
    TH = args.threshold

    start_frame = int(args.begin_frame)
    print("dataset = {}, method = {}, channel = {}, frame_num = {}".format(dataset,method,channel,frame_num))
    #bbox file from detector
    filename = os.path.join('input_detections',method + "_" + dataset + ".txt")
    #source image
    input_path = os.path.join(args.input_path,dataset,'ORIG/ch'+str(channel))
    #output image with bbox
    output_folder = os.path.join('output',method + "_" + dataset + "_ch" + str(channel) + "_results")

    frameBBoxList = readFrameBBoxList(frame_num,filename,channel,detect_face,TH)
    #printFrameList(frameBBoxList)    
    MDNET = MDTrack(frame_num,input_path,tracklet_num,frameBBoxList)
    for i in range(start_frame, frame_num):
        if frameBBoxList[i]:
            for bboxD in frameBBoxList[i]:
                if bboxD.match == False:
                    # print(bboxD.getRec())
                    print(i)

                    track_num = len(MDNET.GLOBAL_TRACK_LIST)

                    MDNET.search_track(track_num, i, bboxD.getRec(), 0,bboxD.confidence)

    track_length = len(MDNET.GLOBAL_TRACK_LIST)
    print('track num = {}'.format(track_length))
    for i in range(track_length):
        print('track {} {}~{}'.format(i+1,MDNET.GLOBAL_TRACK_LIST[i][0][0],MDNET.GLOBAL_TRACK_LIST[i][-1][0]))


    output_pickle_name = 'track_list_'+method+'_'+dataset+'_ch'+str(channel)+'.pkl'
    afile = open(output_pickle_name, 'wb')
    pickle.dump(MDNET.GLOBAL_TRACK_LIST, afile)
    afile.close()

    if(args.savefig):
        draw_image(input_path, output_folder, MDNET.GLOBAL_TRACK_LIST)

