# %%
!pip install xarray boto3 rasterio


# %%
import subprocess

import boto3
from botocore import UNSIGNED
from botocore.client import Config

import xarray as xr
import numpy as np
from scipy.interpolate import griddata

import rasterio
from rasterio.transform import from_bounds


# %%
import ee

# Trigger the authentication flow.
ee.Authenticate()

PROJECT_ID = 'pyregence-ee'

# Initialize the library.
ee.Initialize(project=PROJECT_ID, opt_url='https://earthengine-highvolume.googleapis.com')

# %%
base_path = f'projects/{PROJECT_ID}/assets/'
# ee_path = 'wrf-data/wrf-data-etrans-sfc'
ee_path = 'wrf-data/relative-humidity'

# %%
# see variable names here: https://dept.atmos.ucla.edu/sites/default/files/alexhall/files/aws_tiers_dirstructure_nov22.pdf
# 'soil_m' # etrans_sfc # sh_sfc
wrf_variable_name = 'rh'

# %%
# original link to download: https://wrf-cmip6-noversioning.s3.amazonaws.com/index.html#downscaled_products/gcm/canesm5_r1i1p2f1_ssp370_bc/postprocess/d02/

# Initialize a session using Amazon S3 (unsigned access)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Define the bucket and file key for etrans_sfc file
bucket_name = 'wrf-cmip6-noversioning'
historical_wrf_variable_folder_path = 'downscaled_products/gcm/canesm5_r1i1p2f1_historical_bc/postprocess/d02'
historical_wrf_variable_file_key = wrf_variable_name+'.daily.canesm5.r1i1p2f1.hist.bias-correct.d02.{}.nc'
hist_years = range(1990, 2014)
gcp_bucket = f'wrf-{wrf_variable_name}'

future_wrf_variable_folder_path = 'downscaled_products/gcm/canesm5_r1i1p2f1_ssp370_bc/postprocess/d02'
future_wrf_variable_file_key = wrf_variable_name + '.daily.canesm5.r1i1p2f1.ssp370.bias-correct.d02.{}.nc'
future_years = range(2014, 2031)

# Define the file path for the coordinate file (wrfinput_d02_coord.nc)
coord_file_name = 'wrfinput_d02_coord.nc'
coord_file_key = f'downscaled_products/wrf_coordinates/{coord_file_name}'


# %% [markdown]
# ### convert netcdfs (historical)

# %%
# Download the wrfinput_d02_coord.nc file
s3.download_file(bucket_name, coord_file_key, coord_file_name)

# %%
wrfinput = xr.open_dataset(coord_file_name)

# %%
for year in hist_years:
    print(f'download historical {year}')
    f_name = historical_wrf_variable_file_key.format(year)
    # Download the etrans_sfc file
    s3.download_file(bucket_name, f'{historical_wrf_variable_folder_path}/{f_name}', f_name)



# %%
def get_saved_file_name(file_name, soil_m_level=None):
    save_file_name = file_name
    if soil_m_level:
        try:
            soil_m_mapping = {
                1: '5-cm',
                2: '25-cm',
                3: '70-cm',
                4: '150-cm',
            }
            save_file_name += f'_level_{soil_m_mapping[soil_m_level]}'
        except KeyError:
            raise ValueError(f'soil_m_level not supported for {wrf_variable_name}')

    save_file_name += f'.daily.canesm5.r1i1p2f1.{year}.tif'
    return save_file_name

# %%
def process_netcdf(file_name, soil_m_level=None):
    nc_file = xr.open_dataset(file_name)

    # Extract lat2d and lon2d
    lat2d = wrfinput['lat2d'].values
    lon2d = wrfinput['lon2d'].values

    if wrf_variable_name == 'soil_m' and soil_m_level:
        average_wrf_var = nc_file[wrf_variable_name].isel(soil_nz=soil_m_level-1).mean(dim='day').values
    else:
        average_wrf_var = nc_file[wrf_variable_name].mean(dim='day').values

    # Create a regular lon-lat grid for interpolation (note the swapped order)
    lon_interp = np.linspace(np.min(lon2d), np.max(lon2d), average_wrf_var.shape[1])
    lat_interp = np.linspace(np.min(lat2d), np.max(lat2d), average_wrf_var.shape[0])

    # Create a meshgrid for the new grid
    lon_mesh, lat_mesh = np.meshgrid(lon_interp, lat_interp)

    # Flatten the original lat/lon and data values for interpolation
    lat_lon_points = np.array([lat2d.flatten(), lon2d.flatten()]).T
    average_wrf_var_flat = average_wrf_var.flatten()


    # Perform interpolation to the regular grid
    grid_data = griddata(lat_lon_points, average_wrf_var_flat, (lat_mesh, lon_mesh), method='linear')

    # Flip the grid data vertically to correct the upside-down orientation
    grid_data_flipped = np.flipud(grid_data)

    # Save the interpolated data as a GeoTIFF with correct bounds
    transform = from_bounds(np.min(lon_interp), np.min(lat_interp), np.max(lon_interp), np.max(lat_interp), grid_data_flipped.shape[1], grid_data_flipped.shape[0])

    saved_file_name = get_saved_file_name(wrf_variable_name, soil_m_level=soil_m_level)

    with rasterio.open(saved_file_name, 'w', driver='GTiff',
                       height=grid_data_flipped.shape[0], width=grid_data_flipped.shape[1], count=1,
                       dtype=grid_data_flipped.dtype, crs='EPSG:4326', transform=transform) as dst:
        dst.write(grid_data_flipped, 1)
    return saved_file_name


# %%
wrf_variable_name

# %%
f_name

# %%
if wrf_variable_name != 'soil_m':
    soil_m_level = None
else:
    soil_m_level = 4 # soil_m_level index starts at 1 and ends at 4

# %%
soil_m_level

# %%
for year in hist_years:
    print(f'processed historical {year}')
    f_name = historical_wrf_variable_file_key.format(year)
    saved_file_name = process_netcdf(f_name, soil_m_level)


# %%
saved_file_name

# %%
with rasterio.open(saved_file_name) as dataset:
    # resolution (in degrees)
    resolution_deg = dataset.res

    lat_center = (dataset.bounds.top + dataset.bounds.bottom) / 2

    # Convert resolution in degrees to meters
    resolution_lat_m = resolution_deg[1] * 111320  # Resolution for latitude (constant ~111.32 km per degree)
    resolution_lon_m = resolution_deg[0] * 111320 * np.cos(np.radians(lat_center))  # Resolution for longitude

    print(f"Approximate resolution in meters: {resolution_lon_m:.2f} m x {resolution_lat_m:.2f} m")


# %% [markdown]
# ### make cogs

# %%
!apt-get install -y gdal-bin python3-gdal

# %%
!gdalinfo --version


# %%


# %%
# save cogs
def save_cogs(filename):
    if filename.endswith('.tif'):
        filename = filename[:-4]
    print('filename', filename)
    cog_cmd = f'gdal_translate {filename}.tif {filename}_cog.tif -co TILED=YES -co COPY_SRC_OVERVIEWS=YES -co COMPRESS=LZW'
    print(f"cog_cmd >> : {cog_cmd}")

    result = subprocess.check_output(cog_cmd, shell=True)
    print("result", result)


# %%
soil_m_level

# %%
for year in hist_years:
    print(f'uploaded historical {year}')
    f_name = get_saved_file_name(wrf_variable_name, soil_m_level=soil_m_level)
    # f_name = f'{wrf_variable_name}.daily.canesm5.r1i1p2f1.{year}'
    save_cogs(f_name)

# %% [markdown]
# ### upload to cloud

# %%
# create bucket if not exists
REGION = 'us-central1'

from google.cloud import storage
from google.api_core.exceptions import Conflict

def create_bucket_if_not_exists(bucket_name, location=REGION):
    # Initialize a storage client
    storage_client = storage.Client(project=PROJECT_ID)

    # Check if the bucket already exists
    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f'Bucket {bucket_name} already exists.')
    except Exception as e:
        if isinstance(e, Conflict):
            print(f'Bucket {bucket_name} already exists.')
        else:
            # If the bucket does not exist, create it
            bucket = storage_client.bucket(bucket_name)
            bucket.location = location
            bucket.create()
            print(f'Bucket {bucket_name} created at location {location}.')

create_bucket_if_not_exists(gcp_bucket)


# %%


# %%
def upload_to_gcp(filename):
    if filename.endswith('.tif'):
        filename = filename[:-4]
    # Copy the file
    cp = f"gsutil cp {filename}_cog.tif gs://{gcp_bucket}/{filename}.tif"
    print(f"cp >> : {cp}")

    result = subprocess.check_output(cp, shell=True)
    print("result", result)

# %%
for year in hist_years:
    print(f'uploaded historical {year}')
    # f_name = f'{wrf_variable_name}.daily.canesm5.r1i1p2f1.{year}'
    f_name = get_saved_file_name(wrf_variable_name, soil_m_level=soil_m_level)
    upload_to_gcp(f_name)


# %% [markdown]
# ### upload to ee

# %%
# Set default project
!earthengine set_project {PROJECT_ID}
!gcloud config set project {PROJECT_ID}

# %%
PROJECT_ID

# %%
f'{base_path}{ee_path}'

# %%
# create collection if not exists
try:
    ee.data.createAsset({'type': 'IMAGE_COLLECTION'}, f'{base_path}{ee_path}', False)
except ee.EEException:
    print('assets already exists')


# %%
for year in hist_years:
    # f_name = f'{wrf_variable_name}.daily.canesm5.r1i1p2f1.{year}'
    f_name = get_saved_file_name(wrf_variable_name, soil_m_level=soil_m_level)
    img = ee.Image.loadGeoTIFF(f'gs://{gcp_bucket}/{f_name}')
    img = img.set(
        'system:time_start', ee.Date(f'{year}-01-01').millis(),
        'system:time_end', ee.Date(f'{year}-12-31').millis(),
        'soil_m_level', soil_m_level,
        'wrf_variable_name', wrf_variable_name,
    )
    export_image = f'projects/{PROJECT_ID}/assets/{ee_path}/{"_".join(f_name.split("."))}'
    print(f'uploading to {export_image}')
    image_task = ee.batch.Export.image.toAsset(
        image=img,
        description=f'{f_name}',
        assetId=export_image,
        region=img.geometry().bounds(),
        scale=5000,
        maxPixels=1e13,
    )

    image_task.start()


# %% [markdown]
# ### convert netcdfs (future)

# %%
# Initialize a session using Amazon S3 (unsigned access)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# %%
for year in future_years:
    print(f'download future {year}')
    f_name = future_wrf_variable_file_key.format(year)
    # Download the etrans_sfc file
    s3.download_file(bucket_name, f'{future_wrf_variable_folder_path}/{f_name}', f_name)



# %%


# %%
for year in future_years:
    print(f'processed historical {year}')
    f_name = future_wrf_variable_file_key.format(year)
    saved_file_name = process_netcdf(f_name, soil_m_level)


# %% [markdown]
# ### make cogs

# %%
for year in future_years:
    print(f'uploaded historical {year}')
    f_name = get_saved_file_name(wrf_variable_name, soil_m_level=soil_m_level)
    # f_name = f'{wrf_variable_name}.daily.canesm5.r1i1p2f1.{year}'
    save_cogs(f_name)

# %% [markdown]
# ### upload to cloud

# %%
for year in future_years:
    print(f'uploaded future {year}')
    # f_name = f'{wrf_variable_name}.daily.canesm5.r1i1p2f1.{year}'
    f_name = get_saved_file_name(wrf_variable_name, soil_m_level=soil_m_level)
    upload_to_gcp(f_name)


# %% [markdown]
# ### upload to ee

# %%
for year in future_years:
    # f_name = f'{wrf_variable_name}.daily.canesm5.r1i1p2f1.{year}'
    f_name = get_saved_file_name(wrf_variable_name, soil_m_level=soil_m_level)
    img = ee.Image.loadGeoTIFF(f'gs://{gcp_bucket}/{f_name}')
    img = img.set(
        'system:time_start', ee.Date(f'{year}-01-01').millis(),
        'system:time_end', ee.Date(f'{year}-12-31').millis(),
        'soil_m_level', soil_m_level,
        'wrf_variable_name', wrf_variable_name,
        'year', year,
    )
    export_image = f'projects/{PROJECT_ID}/assets/{ee_path}/{"_".join(f_name.split("."))}'
    print(f'uploading to {export_image}')
    image_task = ee.batch.Export.image.toAsset(
        image=img,
        description=f'{f_name}',
        assetId=export_image,
        region=img.geometry().bounds(),
        scale=5000,
        maxPixels=1e13,
    )

    image_task.start()


# %%



