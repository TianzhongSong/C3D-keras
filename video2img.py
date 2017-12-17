import cv2
import os

video_path = '/home/tianz/datasets/ucf101/'
save_path = '/home/tianz/datsets/ucfimgs/'

action_list = os.listdir(video_path)

for action in action_list:
    if not os.path.exists(save_path+action):
        os.mkdir(save_path+action)
    video_list = os.listdir(video_path+action)
    for video in video_list:
        prefix = video.split('.')[0]
        if not os.path.exists(save_path+action+'/'+prefix):
            os.mkdir(save_path+action+'/'+prefix)
        save_name = save_path + action + '/' + prefix + '/'
        video_name = video_path+action+'/'+video
        cap = cv2.VideoCapture(video_name)
        fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_count = 0
        for i in range(fps):
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(save_name+str(10000+fps_count)+'.jpg',frame)
                fps_count += 1