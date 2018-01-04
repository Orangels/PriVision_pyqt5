import os
from easydict import EasyDict as edict

cur_pth = os.getcwd()

__C = edict()
cfg = __C

__C.GLOBAL = edict()
__C.GLOBAL.CAFFE_ROOT = '/home/user/workspace/openpose-priv-dev/3rdparty/caffe/python'
__C.GLOBAL.OPENPOSE_ROOT = '/home/user/workspace/openpose-priv-dev'
__C.GLOBAL.IM_SHOW_SIZE = (960, 640)
__C.GLOBAL.VIDEO1 = './files/news.avi'
__C.GLOBAL.VIDEO2 = './files/shanxi_traffic/0500320.avi'
__C.GLOBAL.VIDEO3 = './files/shanxi_traffic/0500320.avi'
__C.GLOBAL.F_MODEL1 = False
__C.GLOBAL.F_MODEL2 = False
__C.GLOBAL.F_MODEL3 = False
__C.GLOBAL.F_MODEL4 = False
__C.GLOBAL.SAVE_VIDEO_PATH = './files/save'
__C.GLOBAL.SAVE_VIDEO_MAX_SECOND = 1800 * 20
__C.GLOBAL.SAVE_VIDEO_FPS = 20
__C.GLOBAL.SAVE_VIDEO_SIZE = (960, 540)

__C.ICONS = edict()
__C.ICONS.LOGO = cur_pth + '/icons/pose_icon.png'
__C.ICONS.BACKGROUND = cur_pth + '/icons/back_large.jpg'
# top
__C.ICONS.TOP_SIZE = (110, 20)
__C.ICONS.TOP_LEFT1 = cur_pth + '/icons/icon-top/realtime-mode-bright.png'
__C.ICONS.TOP_LEFT2 = cur_pth + '/icons/icon-top/playback1-bright.png'
__C.ICONS.TOP_LEFT3 = cur_pth + '/icons/icon-top/playback2-bright.png'
__C.ICONS.TOP_LEFT4 = cur_pth + '/icons/icon-top/picture-mode-bright.png'
__C.ICONS.TOP_LEFT5 = cur_pth + '/icons/icon-top/loading-mode-bright.png'
# left
__C.ICONS.LEFT_SIZE = (110, 70)
__C.ICONS.LEFT_TOP1 = cur_pth + '/icons/icon-left/play-bright.png'
__C.ICONS.LEFT_TOP2 = cur_pth + '/icons/icon-left/pause-bright.png'
__C.ICONS.LEFT_TOP3 = cur_pth + '/icons/icon-left/record-bright.png'
__C.ICONS.LEFT_TOP4 = cur_pth + '/icons/icon-left/empty-bright.png'
__C.ICONS.LEFT_TOP5 = cur_pth + '/icons/icon-left/setting-bright.png'
__C.ICONS.LEFT_TOP6 = cur_pth + '/icons/icon-left/exit.png'
# right
__C.ICONS.RIGHT_SIZE = (110, 70)
__C.ICONS.RIGHT_TOP1 = cur_pth + '/icons/icon-right/human-pose-bright.png'
__C.ICONS.RIGHT_TOP2 = cur_pth + '/icons/icon-right/human-detection-bright.png'
__C.ICONS.RIGHT_TOP3 = cur_pth + '/icons/icon-right/object-detection-bright.png'
__C.ICONS.RIGHT_TOP4 = cur_pth + '/icons/icon-right/road-analysis-bright.png'
__C.ICONS.RIGHT_TOP5 = cur_pth + '/icons/icon-right/face-recog-bright.png'


# coco_obj_detection
__C.ObjDet = edict()
# fast mode
# __C.ObjDet.DEPLOY = cur_pth + '/models/coco_object_detection/' \
#                               'deploy_rfcn_coco_se-inception-v2-merge-multigrid-lite.prototxt'
# __C.ObjDet.WEIGHTS = cur_pth + '/models/coco_object_detection/' \
#                                'rfcn_coco_se-inception-v2_ms-ohem-multigrid-lite_iter_1000000.caffemodel'
# __C.ObjDet.PIXEL_MEANS = (104.0, 117.0, 123.0)
# __C.ObjDet.PIXEL_STDS = (1.0, 1.0, 1.0)
# accuracy mode
__C.ObjDet.DEPLOY = cur_pth + '/models/coco_object_detection/' \
                              'deploy_rfcn_coco_air152-merge-ohem-multigrid-deformpsroi-multicontext.prototxt'
__C.ObjDet.WEIGHTS = cur_pth + '/models/coco_object_detection/' \
                               'rfcn_coco_air152_ms-ohem-multigrid-deformpsroi-multicontext_iter_850000.caffemodel'
__C.ObjDet.PIXEL_MEANS = (103.52, 116.28, 123.675)
__C.ObjDet.PIXEL_STDS = (57.375, 57.12, 58.395)


# person_detection
__C.PersonDet = edict()
__C.PersonDet.DEPLOY = cur_pth + '/models/person_detection/' \
                                 'deploy_rfcn_cocoperson_se-resnet50-hik-merge-ohem-multigrid.prototxt'
__C.PersonDet.WEIGHTS = cur_pth + '/models/person_detection/' \
                                  'rfcn_cocoperson_se-resnet50_ms-ohem-multigrid_iter_400000.caffemodel'
__C.PersonDet.PIXEL_MEANS = (104.0, 117.0, 123.0)
__C.PersonDet.PIXEL_STDS = (58.82, 58.82, 58.82)


# face_detection
__C.FaceDet = edict()
__C.FaceDet.DEPLOY = cur_pth + '/models/face_detection/' \
                               'deploy_rfcn_wider_se-resnet50-hik-merge-ohem-multigrid.prototxt'
__C.FaceDet.WEIGHTS = cur_pth + '/models/face_detection/' \
                                'rfcn_wider_se-resnet50_ms-ohem-multigrid_iter_180000.caffemodel'
__C.FaceDet.PIXEL_MEANS = (104.0, 117.0, 123.0)
__C.FaceDet.PIXEL_STDS = (58.82, 58.82, 58.82)


# fast face_detection
__C.FastFace = edict()
__C.FastFace.DEPLOY = cur_pth + '/models/face_detection/' \
                                'deploy_rfcn_wider_se-air14-thin-merge-ohem.prototxt'
__C.FastFace.WEIGHTS = cur_pth + '/models/face_detection/' \
                                 'rfcn_wider_se-air14-thin_ms-ohem_iter_120000.caffemodel'
__C.FastFace.PIXEL_MEANS = (103.52, 116.28, 123.675)
__C.FastFace.PIXEL_STDS = (57.375, 57.12, 58.395)


# face_identity
__C.FaceID = edict()
__C.FaceID.DEPLOY = cur_pth + '/models/vggface2/' \
                              'senet50_ft.prototxt'
__C.FaceID.WEIGHTS = cur_pth + '/models/vggface2/' \
                               'senet50_ft.caffemodel'
__C.FaceID.PIXEL_MEANS = (91.49, 103.88, 131.09)
__C.FaceID.PIXEL_STDS = (1.0, 1.0, 1.0)
__C.FaceID.GALLERY = cur_pth + '/models/vggface2/songze_idbase.npy'
__C.FaceID.GALLERY_NAMES = cur_pth + '/models/vggface2/songze_idbase_namelist.txt'


# traffic_detection
__C.Traffic = edict()
__C.Traffic.DEPLOY = cur_pth + '/models/traffic_detection/' \
                               'deploy_rfcn_privshanxi_se-resnet50-hik-merge-ohem-multigrid.prototxt'
__C.Traffic.WEIGHTS = cur_pth + '/models/traffic_detection/' \
                                'rfcn_privshanxi_se-resnet50_ms-ohem-multigrid_iter_40000.caffemodel'
__C.Traffic.PIXEL_MEANS = (104.0, 117.0, 123.0)
__C.Traffic.PIXEL_STDS = (58.82, 58.82, 58.82)
