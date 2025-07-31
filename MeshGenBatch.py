import smplx
import numpy as np
import torch
import os
from os.path import join, basename, dirname, realpath
import sys
import time
from datetime import date, datetime
import yaml
from lib.config_parser import parse_config, parse_outfits
from lib.network import POP
from lib.utils_io import load_masks, load_barycentric_coords, load_latent_feats
from lib.utils_model import SampleSquarePoints
from lib.utils_model import gen_transf_mtx_from_vtransf
from os import path as osp


# FOLDER STRUCTURE:
# DATA_ROOT_DIR--
#              --FOLDER1--
#                       --BATCH1--
#                               --SEQUENCE1--
#                                          --NPZ1
#                                          --NPZ2
#                                          --...
#                               --SEQUENCE2
#                               --...
#                       --BATCH2
#                       --...
#              --FOLDER2
#              --...

DATA_ROOT_DIR = ' '
FOLDER_LIST = os.listdir(DATA_ROOT_DIR)
# Parameters for SMPL model creation
MODEL_FOLDER = '/dockerspace/Workspace/POPcustomized/POP-main/POP_app/smpl_models'
MODEL_TYPE = 'smpl'
GENDER = 'male'   # change gender here
EXT = 'pkl'
USE_FACE_CONTOUR = False
NUM_BETAS = 10
# create a SMPL model with random shape
batman = smplx.create(MODEL_FOLDER, model_type=MODEL_TYPE,
                      gender=GENDER, use_face_contour=USE_FACE_CONTOUR,
                      num_betas=NUM_BETAS, ext=EXT, create_expression=False, create_jaw_pose=False,
                      create_leye_pose=False, create_reye_pose=False, create_left_hand_pose=False,
                      create_right_hand_pose=False)
SHAPE_NUM = 10  # set the number of different figures with the same pose
# Parameters for POP
PROJECT_DIR = dirname(realpath(__file__))
LOGS_PATH = join(PROJECT_DIR, 'checkpoints')
sys.path.append(PROJECT_DIR)
torch.manual_seed(12345)
np.random.seed(12345)
DEVICE = torch.device('cuda')
args = parse_config()
body_model = 'smpl'
flist_uv, valid_idx, uv_coord_map = load_masks(PROJECT_DIR, args.query_posmap_size, body_model=body_model)
bary_coords = load_barycentric_coords(PROJECT_DIR, args.query_posmap_size, body_model=body_model)
outfits_num = 12
geom_featmap = torch.ones(outfits_num, args.c_geom, args.inp_posmap_size, args.inp_posmap_size).normal_(mean=0.,
                                                                                                        std=0.01).cuda()
geom_featmap.requires_grad = True
N_subsample = 1
subpixel_sampler = SampleSquarePoints(npoints=1)  # doesnt matter, keep it original
pq_samples = subpixel_sampler.sample_regular_points()
model_config = {
    'device': DEVICE,
    'flist_uv': flist_uv,
    'valid_idx': valid_idx,
    'uv_coord_map': uv_coord_map,
    'bary_coords_map': bary_coords,
    'transf_scaling': args.transf_scaling,
}
# build_model
model = POP(
    input_nc=3,
    c_pose=args.c_pose,
    c_geom=args.c_geom,
    inp_posmap_size=args.inp_posmap_size,  # default = 128
    hsize=args.hsize,
    nf=args.nf,
    up_mode=args.up_mode,
    use_dropout=bool(args.use_dropout),
    pos_encoding=bool(args.pos_encoding),
    num_emb_freqs=args.num_emb_freqs,
    posemb_incl_input=bool(args.posemb_incl_input),
    uv_feat_dim=2,
    geom_layer_type=args.geom_layer_type,
    gaussian_kernel_size=args.gaussian_kernel_size,
)
pretrained_resynth_pop = torch.load('POP_pretrained_ReSynthdata_12outfits_epoch00400_model.pt')
pretrained_resynth_geom = 'POP_pretrained_ReSynthdata_12outfits_epoch00400_geom_featmap.pt'
model.load_state_dict(pretrained_resynth_pop['model_state'])  # load the pretrained POP network
load_latent_feats(pretrained_resynth_geom, geom_featmap)  # load the learnt geometric feature maps for the outfit sets


###  Generate the positional maps for the POP network
def posmap_create(resolution, vertices):
    if not (resolution == 128 or resolution == 256):
        raise Exception("Resolution not supported, choose between 128 and 256")
    posmap = np.ndarray((resolution, resolution, 3), dtype=float)
    mask = np.load('/dockerspace/Workspace/POPcustomized/POP-main/POP_app/uv_mask'+str(resolution)+'_with_faceid_smpl.npy')
    v_UV = np.load('/dockerspace/Workspace/POPcustomized/POP-main/POP_app/UV_files/vertex_UV_map.npy')
    UV_pixel = np.load('/dockerspace/Workspace/POPcustomized/POP-main/POP_app/UV_files/UV_pixels.npy') * (resolution - 1)
    f_pair_rich = np.concatenate((v_UV, UV_pixel), axis=1)   # vertex No., UV pixel No, U coord, V coord
    for i in range(0, resolution):  # v is vertical
        for j in range(0, resolution):  # u is horizontal
            if mask[i, j] == -1:  # keep the mask and UV map aligned
                posmap[i, j, :] = posmap[i, j, :] * 0
            else:
                v_dist = resolution - i - 1 - f_pair_rich[:, 3]  # vertical axis reversed in UV mask
                u_dist = j - f_pair_rich[:, 2]
                v_dist2 = np.square(v_dist)
                c_dist2 = np.square(u_dist)
                dist = np.sqrt(v_dist2 + c_dist2)
                nearest4 = dist.argsort()[:4]  # distance in ascending order, get the first 4 indices
                inv_d0 = 1.0 / dist[nearest4[0]]
                inv_d1 = 1.0 / dist[nearest4[1]]
                inv_d2 = 1.0 / dist[nearest4[2]]
                inv_d3 = 1.0 / dist[nearest4[3]]
                inv_dist_sum = inv_d0 + inv_d3 + inv_d2 + inv_d1
                # print('sum' + str(inv_dist_sum))
                posmap[i, j, 0] = (inv_d0 * vertices[int(f_pair_rich[nearest4[0], 0] - 1), 0] +
                                  inv_d1 * vertices[int(f_pair_rich[nearest4[1], 0] - 1), 0] +
                                  inv_d2 * vertices[int(f_pair_rich[nearest4[2], 0] - 1), 0] +
                                  inv_d3 * vertices[int(f_pair_rich[nearest4[3], 0] - 1), 0]) / inv_dist_sum
                posmap[i, j, 1] = (inv_d0 * vertices[int(f_pair_rich[nearest4[0], 0] - 1), 1] +
                                  inv_d1 * vertices[int(f_pair_rich[nearest4[1], 0] - 1), 1] +
                                  inv_d2 * vertices[int(f_pair_rich[nearest4[2], 0] - 1), 1] +
                                  inv_d3 * vertices[int(f_pair_rich[nearest4[3], 0] - 1), 1]) / inv_dist_sum
                posmap[i, j, 2] = (inv_d0 * vertices[int(f_pair_rich[nearest4[0], 0] - 1), 2] +
                                  inv_d1 * vertices[int(f_pair_rich[nearest4[1], 0] - 1), 2] +
                                  inv_d2 * vertices[int(f_pair_rich[nearest4[2], 0] - 1), 2] +
                                  inv_d3 * vertices[int(f_pair_rich[nearest4[3], 0] - 1), 2]) / inv_dist_sum
    return posmap


for FOLDER in FOLDER_LIST:
    BATCH_LIST = os.listdir(osp.join(DATA_ROOT_DIR, FOLDER))
    for BATCH in BATCH_LIST:
        SEQUENCE_LIST = os.listdir(osp.join(DATA_ROOT_DIR, FOLDER, BATCH))
        for SEQUENCE in SEQUENCE_LIST:
            NPZ_LIST = os.listdir(osp.join(DATA_ROOT_DIR, FOLDER, BATCH, SEQUENCE))
            for NPZ in NPZ_LIST:
                pose_file = osp.join(DATA_ROOT_DIR, FOLDER, BATCH, SEQUENCE, NPZ)
                bdata = np.load(pose_file)
                traj1 = bdata['pose_body'].reshape((-1, 21, 3))  # shape [frame_num, 21, 3]
                betas_ms = torch.zeros([1, batman.num_betas], dtype=torch.float32)  # mean shape
                for shape_idx in range(0, SHAPE_NUM):
                    betas = torch.randn([1, batman.num_betas], dtype=torch.float32)  # random shape
                    body_pose = np.zeros((1, 69), dtype=float)
                    for i in range(1, traj1.shape[0]):  # will make cloud for each frame
                        tmp_arr = np.array([0, 0, 0, 0, 0, 0])
                        for j in reversed(range(0, 21)):  # 21 is the joint number, reverse so the concatenation is performed correctly
                            tmp_arr = np.concatenate((traj1[i, j, :], tmp_arr))
                        tmp_arr = np.array([tmp_arr])
                        body_pose = np.concatenate((body_pose, tmp_arr), axis=0)  # batch x 69

                    body_pose_tensor = torch.tensor(body_pose[1:], dtype=torch.float32)
                    output = batman(betas=betas, body_pose=body_pose_tensor)  # output include the vertices for point cloud
                    output_ms = batman(betas=betas_ms, body_pose=body_pose_tensor)  # mean shape SMPL

                    # Get the vertices and the points indicating skeletal joints of the SMPL model
                    vertices = output['vertices'].detach().cpu().numpy()  # B x V x 3
                    joints = output['joints'].detach().cpu().numpy()  # B x J x 3
                    pos_vtransf = output['vtransf'].detach().cpu().numpy()  # B x V x 4 x 4
                    vertices_ms = output_ms['vertices'].detach().cpu().numpy()
                    joints_ms = output_ms['joints'].detach().cpu().numpy()
                    pos_vtransf_ms = output_ms['vtransf'].detach().cpu().numpy()

                    posmap256 = posmap_create(256, vertices[0])
                    posmap128 = posmap_create(128, vertices[0])
                    posmap256_ms = posmap_create(256, vertices_ms[0])
                    posmap128_ms = posmap_create(128, vertices_ms[0])
                    for b in range(1, vertices.shape[0]):
                        posmap256 = np.concatenate((posmap256, np.array([posmap_create(256, vertices[b])])), axis=0)
                        posmap128 = np.concatenate((posmap128, np.array([posmap_create(128, vertices[b])])), axis=0)
                        posmap256_ms = np.concatenate((posmap256_ms, np.array([posmap_create(256, vertices_ms[b])])), axis=0)
                        posmap128_ms = np.concatenate((posmap128_ms, np.array([posmap_create(256, vertices_ms[b])])), axis=0)
                    query_posmap = torch.tensor(posmap256).float().permute([0, 3, 1, 2]).to('cuda', non_blocking=True)
                    inp_posmap = torch.tensor(posmap128_ms).float().permute([0, 3, 1, 2]).to('cuda', non_blocking=True)
                    bs, _, H, W = query_posmap.size()
                    pq_repeated = pq_samples.expand(bs, H * W, -1, -1)  # B, H*W, samples_per_pix, 2
                    uv_coord_map_batch = uv_coord_map.expand(bs, -1, -1).contiguous()
                    vtransf4 = torch.tensor(pos_vtransf).float()
                    transf_mtx_map = gen_transf_mtx_from_vtransf(vtransf, bary_coords, flist_uv,
                                                                 scaling=args.transf_scaling)
                    if vtransf4.shape[-1] == 4:
                        vtransf = vtransf4[:, :, :3, :3].to('cuda', non_blocking=True)
                    else:
                        vtransf = vtransf4.to('cuda', non_blocking=True)
                    for idx in range(0, outfits_num):
                        index = torch.tensor([idx]).cuda()  # pick outfit
                        geom_featmap_batch = geom_featmap[index, ...]
                    bp_locations = query_posmap.expand(N_subsample, -1, -1, -1, -1).permute(
                        [1, 2, 3, 4, 0])  # bs, C, H, W, N_sample
                    transf_mtx_map = transf_mtx_map.expand(N_subsample, -1, -1, -1, -1, -1).permute(
                        [1, 2, 3, 0, 4, 5])  # [bs, H, W, N_subsample, 3, 3]
                    model.to('cuda')
                    model.eval()
                    with torch.no_grad():
                        pred_res, pred_normals = model(inp_posmap,
                                                       # mean shape, posed body positional maps as input to the network
                                                       geom_featmap=geom_featmap_batch,
                                                       uv_loc=uv_coord_map_batch,
                                                       pq_coords=pq_repeated  # always 0 in POP
                                                       )

                        # local coords --> global coords
                        pred_res = pred_res.permute([0, 2, 3, 4, 1]).unsqueeze(-1)
                        pred_normals = pred_normals.permute([0, 2, 3, 4, 1]).unsqueeze(-1)

                        pred_res = torch.matmul(transf_mtx_map, pred_res).squeeze(-1)
                        pred_normals = torch.matmul(transf_mtx_map, pred_normals).squeeze(-1)
                        pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)

                        # residual to abosolute locations in space
                        full_pred = pred_res.permute([0, 4, 1, 2, 3]).contiguous() + bp_locations

                        # take the selected points and reshape to [N_valid_points, 3]
                        full_pred = full_pred.permute([0, 2, 3, 4, 1]).reshape(bs, -1, N_subsample, 3)[:, valid_idx,
                                    ...]
                        pred_normals = pred_normals.reshape(bs, -1, N_subsample, 3)[:, valid_idx, ...]
                        full_pred = full_pred.reshape(bs, -1, 3).contiguous()
                        pred_normals = pred_normals.reshape(bs, -1, 3).contiguous()


