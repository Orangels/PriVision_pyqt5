import os
from easydict import EasyDict as edict

cur_pth = os.getcwd()

__C = edict()
cfg = __C

__C.GLOBAL = edict()
__C.GLOBAL.CAFFE_ROOT = '/home/user/workspace/openpose-priv-dev/3rdparty/caffe/python'
__C.GLOBAL.F_MODEL1 = False
__C.GLOBAL.F_MODEL2 = False
__C.GLOBAL.F_MODEL3 = False
__C.GLOBAL.F_MODEL4 = False

__C.ObjDet = edict()
# fast mode
# __C.ObjDet.DEPLOY = cur_pth + '/models/deploy_rfcn_coco_se-inception-v2-merge-multigrid-lite.prototxt'
# __C.ObjDet.WEIGHTS = cur_pth + '/models/rfcn_coco_se-inception-v2_ms-ohem-multigrid-lite_iter_1000000.caffemodel'
# __C.ObjDet.PIXEL_MEANS = (104.0, 117.0, 123.0)
# __C.ObjDet.PIXEL_STDS = (1.0, 1.0, 1.0)
# accuracy mode
__C.ObjDet.DEPLOY = cur_pth + '/models/deploy_rfcn_coco_air152-merge-ohem-multigrid-deformpsroi-multicontext.prototxt'
__C.ObjDet.WEIGHTS = cur_pth + \
                     '/models/rfcn_coco_air152_ms-ohem-multigrid-deformpsroi-multicontext_iter_850000.caffemodel'
__C.ObjDet.PIXEL_MEANS = (103.52, 116.28, 123.675)
__C.ObjDet.PIXEL_STDS = (57.375, 57.12, 58.395)


__C.PersonDet = edict()
__C.PersonDet.DEPLOY = cur_pth + '/models/deploy_rfcn_cocoperson_se-resnet50-hik-merge-ohem-multigrid.prototxt'
__C.PersonDet.WEIGHTS = cur_pth + '/models/rfcn_cocoperson_se-resnet50_ms-ohem-multigrid_iter_400000.caffemodel'
__C.PersonDet.PIXEL_MEANS = (104.0, 117.0, 123.0)
__C.PersonDet.PIXEL_STDS = (58.82, 58.82, 58.82)

__C.FaceDet = edict()
__C.FaceDet.DEPLOY = cur_pth + '/models/deploy_rfcn_wider_se-resnet50-hik-merge-ohem-multigrid.prototxt'
__C.FaceDet.WEIGHTS = cur_pth + '/models/rfcn_wider_se-resnet50_ms-ohem-multigrid_iter_180000.caffemodel'
__C.FaceDet.PIXEL_MEANS = (104.0, 117.0, 123.0)
__C.FaceDet.PIXEL_STDS = (58.82, 58.82, 58.82)

__C.FastFace = edict()
__C.FastFace.DEPLOY = cur_pth + '/models/deploy_rfcn_wider_se-air14-thin-merge-ohem.prototxt'
__C.FastFace.WEIGHTS = cur_pth + '/models/rfcn_wider_se-air14-thin_ms-ohem_iter_120000.caffemodel'
__C.FastFace.PIXEL_MEANS = (103.52, 116.28, 123.675)
__C.FastFace.PIXEL_STDS = (57.375, 57.12, 58.395)

__C.FaceID = edict()
__C.FaceID.DEPLOY = cur_pth + '/models/vggface2/senet50_ft.prototxt'
__C.FaceID.WEIGHTS = cur_pth + '/models/vggface2/senet50_ft.caffemodel'
__C.FaceID.PIXEL_MEANS = (91.49, 103.88, 131.09)
__C.FaceID.PIXEL_STDS = (1.0, 1.0, 1.0)
__C.FaceID.GALLERY = cur_pth + '/files/songze_idbase.npy'
__C.FaceID.GALLERY_NAMES = cur_pth + '/files/songze_idbase_namelist.txt'

__C.Traffic = edict()
__C.Traffic.DEPLOY = cur_pth + '/models/deploy_rfcn_privshanxi_se-resnet50-hik-merge-ohem-multigrid.prototxt'
__C.Traffic.WEIGHTS = cur_pth + '/models/rfcn_privshanxi_se-resnet50_ms-ohem-multigrid_iter_40000.caffemodel'
__C.Traffic.PIXEL_MEANS = (104.0, 117.0, 123.0)
__C.Traffic.PIXEL_STDS = (58.82, 58.82, 58.82)
