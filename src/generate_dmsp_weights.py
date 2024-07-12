import geopandas as gpd
import shapefile
from osgeo import ogr,osr,gdal
from SwinIR_train_function import *
import geopandas as gpd
import shapefile
import numpy as np
import pandas as pd
import xarray as xr
import os
import time
from osgeo import ogr,osr,gdal
import sys
import io
import wxee
import matplotlib.pyplot as plt

# !earthengine authenticate
import ee
import os


# Set the path to the service account key file
service_account = 'editor@ee-manmeet20singh15-wbis.iam.gserviceaccount.com'
key_file = '/mnt/kaggle/ndui/ee-manmeet20singh15-wbis-fab7f1ca35e0.json'

# Use the service account for authentication
credentials = ee.ServiceAccountCredentials(service_account, key_file)
ee.Initialize(credentials)

upscale = 1
window_size = 5
height = 30 #(1024 // upscale // window_size + 1) * window_size
width = 30 #(720 // upscale // window_size + 1) * window_size
device = 'cuda'
model = SwinIR(upscale=1, img_size=(height, width),
               window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
               embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect').to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

F101992 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F101992').select('stable_lights')
F101993 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F101993').select('stable_lights')
F101994 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F101994').select('stable_lights')
F121994 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F121994').select('stable_lights')
F121995 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F121995').select('stable_lights')
F121996 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F121996').select('stable_lights')
F121997 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F121997').select('stable_lights')
F121998 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F121998').select('stable_lights')
F121999 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F121999').select('stable_lights')
F141997 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F141997').select('stable_lights')
F141998 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F141998').select('stable_lights')
F141999 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F141999').select('stable_lights')
F142000 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F142000').select('stable_lights')
F142001 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F142001').select('stable_lights')
F142002 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F142002').select('stable_lights')
F142003 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F142003').select('stable_lights')
F152000 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F152000').select('stable_lights')
F152001 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F152001').select('stable_lights')
F152002 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F152002').select('stable_lights')
F152003 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F152003').select('stable_lights')
F152004 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F152004').select('stable_lights')
F152005 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F152005').select('stable_lights')
F152006 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F152006').select('stable_lights')
F152007 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F152007').select('stable_lights')
F162004 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F162004').select('stable_lights')
F162005 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F162005').select('stable_lights')
F162006 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F162006').select('stable_lights')
F162007 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F162007').select('stable_lights')
F162008 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F162008').select('stable_lights')
F162009 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F162009').select('stable_lights')
F182010 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F182010').select('stable_lights')
F182011 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F182011').select('stable_lights')
F182012 = ee.Image('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F182012').select('stable_lights')

collections = [F101992, F101993, F101994, F121994, F121995, F121996, F121997, F121998, F121999, F141997, F141998,
                    F141999, F142000, F142001,F142002, F142003, F152000, F152001, F152002, F152003, F152004, F152005,
                    F152006, F152007,F162004, F162005, F162006, F162007, F162008, F162009, F182010, F182011, F182012]

c = [-3.06516, -2.0638, -1.68421, -1.71621, 0.530922, 0.303469, -0.18513, 0.490138, 1.800988, -0.6186,
         -0.91352, -1.37993, 0.061872, 0.249452, 1.127103, 0.866522,0, 0.005164,-0.04462, -0.27189, -0.06977, 0.449229,
         0.913485, 0.644785, -0.02563, -0.54115, -0.38377, 0.629564, 0.745403, -0.15161, 6.22332, 1.427157, 3.866698]

b = [-0.00698, -0.00726, -0.00695, -0.00454, 0.00011, -0.00176, -0.00057, 0.001236, 0.002969, -0.0094,
     -0.00929, -0.00889, -0.00469, -0.00452, -0.00221, -0.00351, 0, 8.94e-05, 0.000117, -0.0085, -0.00912, -0.00601,
     -0.00595, -0.00675, -0.00496, -0.0094, -0.0061, -0.00084, -0.00062, -0.00278, 0.014627,0.002877, 0.007962]

a = [1.519907, 1.516595, 1.491333, 1.331971, 0.984465, 1.111207, 1.034429, 0.905787, 0.761106, 1.603921,
      1.603648, 1.586457, 1.294471, 1.275902, 1.128708, 1.206319, 1, 1.002879, 0.987943, 1.555808, 1.591033, 1.401146,
      1.381139,1.448976, 1.317581, 1.613536, 1.41435, 1.040815, 1.037042, 1.193437, -0.08536, 0.774923, 0.355542]

#Image correction using coefficient
images = [0 for _ in range(33)]
count = 0
total_list = []
images_correct = [0 for _ in range(33)]

def fun3(raw,correct):
    out = correct.where(raw.lt(5.0),raw.float())
    return out

for i in range(33):
    images[i] = ee.Image(a[i]).multiply(collections[i].float()).add(ee.Image(b[i]).multiply(collections[i].float().pow(2))).add(ee.Image(c[i]))
    images_correct[i] = fun3(collections[i],images[i]).select('constant')

def unpatchify(patches, img_shape):
    patch_size = patches.shape[1]
    assert patches.shape[0] == (img_shape[0] // patch_size) * (img_shape[1] // patch_size), "Patches and image shape are not compatible"

    img = np.zeros(img_shape, dtype=patches.dtype)
    patch_idx = 0

    for i in range(0, img_shape[0], patch_size):
        for j in range(0, img_shape[1], patch_size):
            img[i:i + patch_size, j:j + patch_size] = patches[patch_idx]
            patch_idx += 1

    return img

def patchify(img, patch_size):
    img_shape = img.shape
    patches = np.array([img[i:i + patch_size, j:j + patch_size] for i in range(0, img_shape[0], patch_size) for j in range(0, img_shape[1], patch_size)])
    return patches

class ncDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).unsqueeze(0)
        y = torch.from_numpy(self.targets[index]).unsqueeze(0)
        # x = self.data[index]
        # y = self.targets[index]
        # x = x.to(dtype=torch.float32)
        # y = y.to(dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.data)
    
latc = sys.argv[1]
lonc = sys.argv[2]

buffer = 2.5

lats, late = float(latc)-buffer, float(latc)+buffer
lons, lone = float(lonc)-buffer, float(lonc)+buffer

print('lats, late, lons, lone ', lats, late, lons, lone)





aoi = ee.Geometry.Polygon(
        [[[lons, lats],
        [lone, lats],
        [lone, late],
        [lons, late]]])
coords = aoi.coordinates().getInfo()[0]

# Calling VIIRS data from GEE

dataset = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filter(ee.Filter.date('2017-01-01', '2017-12-31')).select('water').mean()
dataset = dataset.set('system:time_start', 0)
ds_water = dataset.wx.to_xarray(region=aoi.bounds(), scale=500)
# print(ds_water)

# sys.exit()


dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').filter(ee.Filter.date('2012-01-01', '2013-12-31'))
viirs_image_2012 = dataset#.select('avg_rad').mean()
viirs_image_2012 = viirs_image_2012.set('system:time_start', 0)
ds_viirs = viirs_image_2012.wx.to_xarray(region=aoi.bounds(), scale=463.83)

dmsp_image_2012 = images_correct[-1]
dmsp_image_2012 = dmsp_image_2012.set('system:time_start', 0)
ds_dmsp = dmsp_image_2012.wx.to_xarray(region=aoi.bounds(), scale=927.67)

ds_viirs_ = ds_viirs.sel(time=slice('2012','2012')).mean(dim='time').interp(x=ds_dmsp.x.values, y=ds_dmsp.y.values,method="cubic", kwargs={"fill_value": "extrapolate"})

x_train = ds_viirs_.avg_rad.values.astype(np.float32)
y_train = ds_dmsp.constant.values[0,:,:].astype(np.float32)

# Create patches from the image
patch_size = 30
img = x_train[:600,:600]
patches = patchify(img, patch_size)

x_train_max = x_train.max()
y_train_max = y_train.max()
x_train /= x_train_max
y_train /= y_train_max

x_train_patches = patchify(x_train[:600,:600], patch_size)
y_train_patches = patchify(y_train[:600,:600], patch_size)

x_val_patches = x_train_patches[200:300]
y_val_patches = y_train_patches[200:300]

x_test_patches = x_train_patches[300:400]
y_test_patches = y_train_patches[300:400]

x_train_patches = x_train_patches[:200]
y_train_patches = y_train_patches[:200]

train_dataset = ncDataset(x_train_patches, y_train_patches)
val_dataset = ncDataset(x_val_patches, y_val_patches)
test_dataset = ncDataset(x_val_patches, y_val_patches)

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True)

from copy import deepcopy

num_epochs = 1000
print_interval = 100
patience = 500
best_val_loss = float('inf')
counter = 0
best_model = None

writer = SummaryWriter("runs/swinir")
for epoch in range(1, num_epochs + 1):
    train_loss, val_loss = train(model, train_dataloader, val_dataloader, criterion, optimizer, device)
# Log losses to TensorBoard
    writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = deepcopy(model)
        counter = 0
    else:
        counter += 1

    if epoch % print_interval == 0:
        print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    if counter >= patience:
        print("Early stopping triggered.")
        break
writer.close()

model_save_path = "/best_model_lats_"+str(lats)+"_late_"+str(late)+"_lons_"+str(lons)+"_lone_"+str(lone)+".pth"
torch.save(best_model.state_dict(), model_save_path)
ds_water.to_netcdf('water_dw_2017_'+str(lats)+"_late_"+str(late)+"_lons_"+str(lons)+"_lone_"+str(lone)+".nc")
