# Configuration for Image Processing Project

# Google Cloud / Earth Engine Project ID
PROJECT_ID = 'ivory-scion-476708-t5'

# Google Drive Paths
# Update these paths to match your Google Drive structure
DRIVE_MOUNT_PATH = '/content/drive'
DRIVE_PROJECT_ROOT = '/content/drive/MyDrive/Image processing course 001-2-9301/project/u-net/data'
DRIVE_IMAGES_PATH = f'{DRIVE_PROJECT_ROOT}/images'
DRIVE_LCS_PATH = f'{DRIVE_PROJECT_ROOT}/lcs'
CLASS_JSON_PATH = f'{DRIVE_LCS_PATH}/dict_land_cover.json'
EXPORT_DRIVE_FOLDER = '/content/drive/MyDrive/GEE_Exports_For_project'

# Earth Engine Asset Paths
ASSET_EXPORT_FOLDER = f'projects/{PROJECT_ID}/assets'
LC_ASSET_ID = f'{ASSET_EXPORT_FOLDER}/lc_map'
POINTS_ASSET_PATH = f'{ASSET_EXPORT_FOLDER}/sample_points'

# Area of Interest and Dates
AOI_NAME = "east_negev"
START_DATE = '2020-01-01'
END_DATE = '2020-12-31'
YEAR = int(START_DATE[:4])

# Generated Asset Paths
S2_ASSET_ID = f'{ASSET_EXPORT_FOLDER}/{AOI_NAME}_Sentinel2_{YEAR}'
S1_ASSET_ID = f'{ASSET_EXPORT_FOLDER}/{AOI_NAME}_Sentinel1_{YEAR}'

# Data Processing Parameters
PATCH_SIZE = 128
SCALE = 10
IMAGE_PER_LC = 20

# Band Definitions
S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
S1_BANDS = ['VV', 'VH']
FEATURE_NAMES = S2_BANDS + S1_BANDS
LABEL_NAME = 'label'
NUM_BANDS = len(FEATURE_NAMES)

# Model Training Parameters
BUFFER_SIZE = 1000
BATCH_SIZE = 16
EPOCHS = 100
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# Output Files
TFRECORD_FILE = f'S2_S1_patches_{YEAR}.tfrecord.gz'
MODEL_FILENAME = 'unet_s1_s2_model_v1.keras'
HISTORY_CSV_FILENAME = 'training_history.csv'
