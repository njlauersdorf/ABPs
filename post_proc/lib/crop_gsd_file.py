start = 0
end = 350
file_name = "/Volumes/EXTERNAL2/crop_gsd/random_init_pa500_pb500_xa50_ep1.0_phi60_pNum100000_aspect1.1"
import gsd.hoomd
pre_crop = gsd.hoomd.open(file_name + '.gsd', mode='rb')
post_crop = file_name + '_crop.gsd'
with gsd.hoomd.open(post_crop, mode='wb') as f:
    for i in range(start, end):
        f.append(pre_crop[i])