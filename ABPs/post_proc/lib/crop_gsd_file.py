start = 500
end = 515
file_name = "/Volumes/EXTERNAL2/n100000test/panel/xa50/new/random_pa80_pb500_phi60_eps1.0_xa0.5_pNum50000_dtau1.0e-06"
import gsd.hoomd
pre_crop = gsd.hoomd.open(file_name + '.gsd', mode='rb')
post_crop = file_name + '_crop.gsd'
with gsd.hoomd.open(post_crop, mode='wb') as f:
    for i in range(start, end):
        f.append(pre_crop[i])