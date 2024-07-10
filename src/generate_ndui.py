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

df = pd.read_excel('/mnt/kaggle/ndui/US_DOE_SW_IFL_cities.xlsx')
print(df)

city_names=[]
for i in range(len(df.Name)):
  city = str(df.Name[i])
  city_names.append(city)
print(city_names)

lats_=[]
for i in range(len(df.lat)):
  lat_city = (df.lat[i])
  lats_.append(lat_city)
print(lats_)

lons_=[]
for i in range(len(df.lon)):
  lon_city = (df.lon[i])
  lons_.append(lon_city)
print(lons_)

cities_folder = '/mnt/kaggle/ndui/'

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
    
#Lon: 98-113E
#Lat: 20-35N

# import os
# directory = 'drive/MyDrive/Shivam/Long_DMSP_NDUI/Test/'
# cities = ['la', 'albuquerque', 'denver', 'portland', 'louisville', 'washington_dc', 'kansas_city', 'columbus', 'minneapolis', 'seattle']
# lats_  = [34.0549, 35.0844, 39.7392, 45.5152, 38.2527, 38.9072, 39.0997, 39.9612, 44.9778, 47.6062]
# lons_  = [-118.2426, -106.6504, -104.9903, -122.6784, -85.7585, -77.0369, -94.5786, -82.9988, -93.2650, -122.3321]

buffer = 2.5

for i_city, city in enumerate(city_names):
    if not os.path.exists(cities_folder+city_names[i_city]+"/best_model_"+city_names[i_city]+".pth"):

        lats, late = lats_[i_city]-buffer, lats_[i_city]+buffer
        lons, lone = lons_[i_city]-buffer, lons_[i_city]+buffer

        aoi = ee.Geometry.Polygon(
                [[[lons, lats],
                [lone, lats],
                [lone, late],
                [lons, late]]])
        coords = aoi.coordinates().getInfo()[0]

        # Calling VIIRS data from GEE

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

        model_save_path = city_names[i_city]+"/best_model_"+city_names[i_city]+".pth"
        torch.save(best_model.state_dict(), cities_folder+model_save_path)

best_model_swinir_path =[]
for i in range(len(city_names)):
  path = cities_folder+city_names[i]+'/best_model_'+str(city_names[i])+'.pth'
  best_model_swinir_path.append(path)
print(best_model_swinir_path)

loaded_model_city= []
for i in range(len(city_names)):
  model_save_path = best_model_swinir_path[i]
  upscale = 1
  window_size = 5
  height = 30 #(1024 // upscale // window_size + 1) * window_size
  width = 30 #(720 // upscale // window_size + 1) * window_size
  device = 'cuda'
  loaded_model = SwinIR(upscale=1, img_size=(height, width),
               window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
               embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect').to(device)
  loaded_model.load_state_dict(torch.load(model_save_path))
  loaded_model.eval()
  loaded_model_city.append(loaded_model)

aoi_city = []
for i in range(len(df.lat)):
  lats, late = df.lat[i]-2.5, df.lat[i]+2.5
  lons, lone = df.lon[i]-2.5, df.lon[i]+2.5
  aoi = ee.Geometry.Polygon(
        [[[lons, lats],
          [lone, lats],
          [lone, late],
          [lons, late]]])
  aoi_city.append(aoi)
print(aoi_city[0])

coords = aoi_city[0].coordinates().getInfo()[0]
print(coords)

dmsp_image_2012 = images_correct[-1]
years = np.arange(2012,1991,-1)
#years
indices = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -14, -15, -16, -17, -22, -23, -24, -28, -29, -30, -32, -33]
dmsp_image = images_correct[indices[0]]
years[indices[0]]

dmsp_image_2012 = dmsp_image_2012.set('system:time_start', 0)
ds_dmsp_2012_city = []
for i in range(len(city_names)):
  ds_dmsp_c2012 = dmsp_image_2012.wx.to_xarray(region=aoi_city[i].bounds(), scale=927.67)
  ds_dmsp_2012_city.append(ds_dmsp_c2012)
ds_dmsp_2012_city[0]

# Calling VIIRS 2012 from GEE
dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').filter(ee.Filter.date('2012-01-01', '2013-12-31'))
viirs_image_2012 = dataset#.select('avg_rad').mean()

## clipping for cities in the list
viirs_image_2012 = viirs_image_2012.set('system:time_start', 0)
ds_viirs_2012_city=[]
for i in range(len(city_names)):
  ds_viirs_c2012 = viirs_image_2012.wx.to_xarray(region=aoi_city[i].bounds(), scale=463.83)
  ds_viirs_2012_city.append(ds_viirs_c2012)
ds_viirs_2012_city[0]

ds_viirs_i2012_city = []
for i in range(len(city_names)):
  ds_viirs_ic2012= ds_viirs_2012_city[i].sel(time=slice('2012','2012')).mean(dim='time').interp(x=ds_dmsp_2012_city[i].x.values, y=ds_dmsp_2012_city[i].y.values,method="cubic", kwargs={"fill_value": "extrapolate"})
  ds_viirs_i2012_city.append(ds_viirs_ic2012)
ds_viirs_i2012_city[0]

x_train_2012_city =[]
y_train_2012_city = []
for i in range(len(city_names)):
  x_train_2012 = ds_viirs_i2012_city[i].avg_rad.values.astype(np.float32)
  y_train_2012 = ds_dmsp_2012_city[i].constant.values[0,:,:].astype(np.float32)
  x_train_2012_city.append(x_train_2012)
  y_train_2012_city.append(y_train_2012)
x_train_2012_city[0], y_train_2012_city[0]

## Getting max value of training dataset for City
x_train_2012_max_city = []
y_train_2012_max_city = []
for i in range(len(city_names)):
  x_train_c2012_m = x_train_2012_city[i].max()
  y_train_c2012_m = y_train_2012_city[i].max()
  x_train_2012_max_city.append(x_train_c2012_m)
  y_train_2012_max_city.append(y_train_c2012_m)

patch_size = 30

patches_2012_city =[]
img_2012_city = []
for i in range(len(city_names)):
  img_c2012 = x_train_2012_city[i][:600,:600]
  patches_c2012 = patchify(img_c2012, patch_size)
  img_2012_city.append(img_c2012)
  patches_2012_city.append(patches_c2012)

reconstructed_predicted_sr_2012_city =[]

for i in range(len(city_names)):
  x_train_c2012 = x_train_2012_city[i]
  y_train_c2012 = y_train_2012_city[i]
  x_train_c2012_max = x_train_c2012.max()
  y_train_c2012_max = y_train_c2012.max()
  # Normalizing
  x_train_c2012 /= x_train_c2012_max       #sets x_train to x_train/x_train_max
  y_train_c2012 /= y_train_c2012_max
  x_train_patches_c2012 = patchify(x_train_c2012[:600,:600], patch_size)[:,np.newaxis,:,:]
  x_train_patches_c2012_tensor = torch.from_numpy(x_train_patches_c2012).to(device)
  with torch.no_grad():
    predicted_sr_c2012 = loaded_model_city[i](x_train_patches_c2012_tensor)
  predicted_sr_c2012_np = predicted_sr_c2012.cpu().numpy() * y_train_c2012_max  #why .cpu() used??
  predicted_sr_c2012_np[predicted_sr_c2012_np<0] = 0.0
  reconstructed_predicted_sr_c2012 = unpatchify(predicted_sr_c2012_np[:,0,:,:], img_2012_city[i].shape)
  reconstructed_predicted_sr_2012_city.append(reconstructed_predicted_sr_c2012)

ds_dmsp_vi_swin_2012_city = []
for i in range(len(city_names)):
  lats_c2012 = ds_dmsp_2012_city[i].y.values[:600]
  lons_c2012 =  ds_dmsp_2012_city[i].x.values[:600]
  dmsp_c2012_ = ds_dmsp_2012_city[i].constant.values[0,:600,:600]
  ds_dmsp_vi_swin_c2012 = xr.Dataset({
    'dmsp': xr.DataArray(
                data   = dmsp_c2012_,   # enter data here
                dims   = ['lat', 'lon'],
                coords = {'lat': lats_c2012, 'lon': lons_c2012},

                ),
    'viirs_swinir': xr.DataArray(
                data   = reconstructed_predicted_sr_2012_city[i],   # enter data here
                dims   = ['lat', 'lon'],
                coords = {'lat': lats_c2012, 'lon': lons_c2012},
                ),
    'viirs': xr.DataArray(
                data   = ds_viirs_i2012_city[i].avg_rad.values[:600,:600],   # enter data here
                dims   = ['lat', 'lon'],
                coords = {'lat': lats_c2012, 'lon': lons_c2012},
                )
            },
    )
  ds_dmsp_vi_swin_2012_city.append(ds_dmsp_vi_swin_c2012)

print(ds_dmsp_vi_swin_2012_city[0])

# setting date for annual data
dates = pd.date_range('1992', '2012', freq='YS')[::-1]
print(dates)

# Creating dmsp timeseries dataset for Cities
ds_dmsp_2012_1992_city = []
for j in range(len(city_names)):
  dmsp_c2012_1992 = []
  for i_ind,ind in enumerate(indices):
    dmsp_image = images_correct[ind]
    print(years[i_ind])
    dmsp_image = dmsp_image.set('system:time_start', 0)
    ds_dmsp_c = dmsp_image.wx.to_xarray(region=aoi_city[j].bounds(), scale=927.67)
    dmsp_c2012_1992.append(ds_dmsp_c.constant.values[0,:600,:600])
  dmsp_c2012_1992_np =  np.stack(dmsp_c2012_1992)
  print(dmsp_c2012_1992_np.shape)
  print(ds_dmsp_c.x[0])

  lats_c = ds_dmsp_c.y.values[:600]
  lons_c = ds_dmsp_c.x.values[:600]

  ds_dmsp_c2012_1992 = xr.Dataset({
        'dmsp': xr.DataArray(
                    data   = dmsp_c2012_1992_np,   # enter data here
                    dims   = ['time', 'lat', 'lon'],
                    coords = {'time':dates, 'lat': lats_c, 'lon': lons_c},

                    ),
               },
         )
  ds_dmsp_2012_1992_city.append(ds_dmsp_c2012_1992)
  print(ds_dmsp_c.constant.values.mean())

print(ds_dmsp_2012_1992_city[0])

for i in range(len(city_names)):
  ds_dmsp_2012_1992_city[i].to_netcdf(cities_folder+city_names[i]+'/dmsp_1992_2012_'+city_names[i]+'.nc')
# !ls drive/MyDrive/Shivam/Long_DMSP_NDUI/Cities/Baltimore

dates = pd.date_range('2013', '2022', freq='YS')
print(dates)

ds_dmsp_2013_2022_city = []
for j in range(len(city_names)):
  print(city_names[j])
  ds_dmsp_c2013_2022 = []
  for year_ in range(2013,2023):
    year = str(year_)#'2013'
    dataset_y = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').filter(ee.Filter.date(year+'-01-01', year+'-12-31'))
    viirs_image_y = dataset_y#.select('avg_rad').mean()
    viirs_image_y = viirs_image_y.set('system:time_start', 0)
    ds_viirs_c_y = viirs_image_y.wx.to_xarray(region=aoi_city[j].bounds(), scale=463.83)
    ds_viirs_interp_c_y = ds_viirs_c_y.sel(time=slice(year,year)).mean(dim='time').interp(x=ds_dmsp_2012_city[j].x.values, y=ds_dmsp_2012_city[j].y.values,method="cubic", kwargs={"fill_value": "extrapolate"})

    x_train_c_y = ds_viirs_interp_c_y.avg_rad.values.astype(np.float32)
    img_c_y = x_train_c_y[:600,:600]

    print('x_train_2012_cj_max:',x_train_2012_max_city[j],'y_train_2012_cj_max:', y_train_2012_max_city[j])

    x_train_c_y /= x_train_2012_max_city[j]
    x_train_c_y_patches = patchify(x_train_c_y[:600,:600], patch_size)[:,np.newaxis,:,:]
    x_train_c_y_patches_tensor = torch.from_numpy(x_train_c_y_patches).to(device)
    with torch.no_grad():
      predicted_sr_c_y = loaded_model_city[j](x_train_c_y_patches_tensor)
    predicted_sr_c_y_np = predicted_sr_c_y.cpu().numpy() * y_train_2012_max_city[j]
    predicted_sr_c_y_np[predicted_sr_c_y_np<0] = 0.0
    reconstructed_predicted_sr_c_y = unpatchify(predicted_sr_c_y_np[:,0,:,:], img_c_y.shape)
    ds_dmsp_c2013_2022.append(reconstructed_predicted_sr_c_y)

  ds_dmsp_c2013_2022_np =  np.stack(ds_dmsp_c2013_2022)
  print(ds_dmsp_c2013_2022_np.shape)

  lats_cj = ds_dmsp_2012_city[j].y.values[:600]
  lons_cj = ds_dmsp_2012_city[j].x.values[:600]
  print(lats_cj[0], lats_cj[0])

  ds_dmsp_cj_2013_2022 = xr.Dataset({
    'dmsp': xr.DataArray(
                data   = ds_dmsp_c2013_2022_np,   # enter data here
                dims   = ['time', 'lat', 'lon'],
                coords = {'time':dates, 'lat': lats_cj, 'lon': lons_cj},

                ),
            },
    )
  ds_dmsp_2013_2022_city.append(ds_dmsp_cj_2013_2022)
#   print(ds_dmsp_cj_2013_2022.dmsp.mean())


for i in range(len(city_names)):
  ds_dmsp_2013_2022_city[i].to_netcdf(cities_folder+city_names[i]+'/dmsp_2013_2022_'+str(city_names[i])+'.nc')
# !ls drive/MyDrive/Shivam/Long_DMSP_NDUI/Cities/Baltimore

ds_dmsp_1992_2022_city = []
for i in range(len(city_names)):
  ds_dmsp_c_1992_2022 = xr.concat([ds_dmsp_2012_1992_city[i], ds_dmsp_2013_2022_city[i]], dim='time').sortby('time')
  ds_dmsp_1992_2022_city.append(ds_dmsp_c_1992_2022)

print(ds_dmsp_1992_2022_city[0])

for i in range(len(city_names)):
  ds_dmsp_1992_2022_city[i].to_netcdf(cities_folder+city_names[i]+'/dmsp_swinIR_1992_2022_'+str(city_names[i])+'.nc')
# !ls drive/MyDrive/Shivam/Long_DMSP_NDUI/Cities/Baltimore

# Area of Interest
aoi_1_city = []
for i in range(len(df.lat)):
  lats, late = df.lat[i]-0.20, df.lat[i]+0.20
  lons, lone = df.lon[i]-0.20, df.lon[i]+0.20
  aoi = ee.Geometry.Polygon(
        [[[lons, lats],
          [lone, lats],
          [lone, late],
          [lons, late]]])
  aoi_1_city.append(aoi)
print(aoi_1_city[2])

coords = aoi_1_city[0].coordinates().getInfo()[0]
print(coords)

# ndui_1999_2022 for Listed Cities
dates = pd.date_range('1999', '2022', freq='YS')
ndui_city = []
for i in range(2):
    print(city_name[i])
    ndui_c_ = []
    for year_ in range(1999,2023):
        print(year_)
        year = str(year_)#'1999'
        # ds_dmsp_interp = ds_dmsp.sel(time=slice(year, year)).interp(lon=ds_ndvi.x, lat=ds_ndvi.y).dmsp.values[0,:,:]/63.0
        # L7 = ee.ImageCollection('LE7_L1T_TOA').filterDate(year+'-01-01', year+'-12-31')
        if year_ == 1999:
            L7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_TOA").filterDate(str(year_)+'-01-01', str(year_+2)+'-12-31')
            print(year_)
            print(str(year_)+'-01-01', str(year_+2)+'-12-31')
        elif year_ == 2000:
            L7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_TOA").filterDate(str(year_-1)+'-01-01', str(year_+1)+'-12-31')
            print(year_)
            print(str(year_-1)+'-01-01', str(year_+1)+'-12-31')
        else:
            L7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_TOA").filterDate(str(year_-2)+'-01-01', str(year_)+'-12-31')
            print(year_)
            print(str(year_-2)+'-01-01', str(year_)+'-12-31')

        def fun4(img):
            bad1 = img.select('B1').eq(0.0)
            bad2 = img.select('B2').eq(0.0)
            bad3 = img.select('B3').eq(0.0)
            bad4 = img.select('B4').eq(0.0)
            bad5 = img.select('B5').eq(0.0)
            bad7 = img.select('B7').eq(0.0)
            mask = img.mask().And(bad1.Or(bad2).Or(bad3).Or(bad4).Or(bad5).Or(bad7).Not())
            #var mask = img.select('10','20','30','40','50','70').mask().reduce('product').eq(1);
            masked = img.mask(mask);
            ndvi = masked.normalizedDifference(["B4","B3"])
            return ndvi

        NDVIs = L7.map(fun4)

        Mean_NDVI = NDVIs.median()
        Max_NDVI = NDVIs.max()
        Min_NDVI = NDVIs.min()
        mosaic = Mean_NDVI.where(Max_NDVI.gt(0.4), Max_NDVI)
        mosaic = mosaic.where(Min_NDVI.lt(-0.2), Min_NDVI)

        mosaic = mosaic.set('system:time_start', 0)
        ds_ndvi_c = mosaic.wx.to_xarray(region=aoi_city[i].bounds(), scale=30)
        ds_ndvi_c_ = ds_ndvi_c.nd.values[0,:,:]
        ds_dmsp_interp_c = ds_dmsp_1992_2022_city[i].sel(time=slice(year, year)).interp(lon=ds_ndvi_c.x, lat=ds_ndvi_c.y).dmsp.values[0,:,:]/63.0
        # ds_ndvi_c_ = mosaic.wx.to_xarray(region=aoi_city[i].bounds(), scale=30).nd.values[0,:,:]

        ndui_c = (ds_dmsp_interp_c - ds_ndvi_c_)/(ds_dmsp_interp_c + ds_ndvi_c_)
        ndui_c[ndui_c>1.0] = 1.0
        ndui_c[ndui_c<-1.0] = -1.0
        ndui_c_.append(ndui_c)
    print(len(ndui_c_))
    print(np.stack(ndui_c_).shape)

    ndui_c_stack = np.stack(ndui_c_)
    ds_ndui_c = xr.Dataset({
        'ndui': xr.DataArray(
            data   = ndui_c_stack,   # enter data here
            dims   = ['time', 'lat', 'lon'],
            coords = {'time':dates, 'lat': ds_ndvi_c.y.values, 'lon': ds_ndvi_c.x.values},
            ),
        },
                                               )
    print(ds_ndui_c)
    ndui_city.append(ds_ndui_c)
print(ndui_city[0])

for i in range(2):
  ndui_city[i].to_netcdf(cities_folder+city_names[i]+'/ndui_1992_2022_'+str(city_names[i])+'.nc')

