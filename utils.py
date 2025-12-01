import ee
import numpy as np
import tensorflow as tf
import io
from google.api_core import retry
import config

# --- Earth Engine Helpers ---

def maskS2clouds(image):
    """
    Cloud masking function for Sentinel-2 SR.
    Keeps good pixels: 4 (veg), 5 (bare), 6 (water), 7 (unclassified), 11 (snow).
    Scales optical bands by dividing by 10000.
    """
    scl = image.select('SCL')
    good_qa = ee.List([4, 5, 6, 7, 11])
    mask = scl.remap(good_qa, ee.List.repeat(1, good_qa.length()), 0)

    optical_bands = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).divide(10000)

    return optical_bands.updateMask(mask).copyProperties(image, ["system:time_start"])

def load_coords_from_asset(asset_path):
    """
    Loads a FeatureCollection asset and extracts the [lon, lat]
    coordinates from each feature.
    Assumes the asset is already projected to EPSG:4326.
    """
    print(f"Loading coordinates from EE Asset: {asset_path}...")
    try:
        ee_fc = ee.FeatureCollection(asset_path)
    except Exception as e:
        print(f"Error loading asset: {e}")
        return []

    print("Fetching coordinates from server...")
    try:
        geometries = ee_fc.aggregate_array('.geo').getInfo()
    except Exception as e:
        print(f"Error fetching geometries: {e}")
        return []

    # The output is a list of {'type': 'Point', 'coordinates': [lon, lat]}
    lon_lat_list = [item['coordinates'] for item in geometries]

    print(f"Successfully fetched {len(lon_lat_list)} coordinates from asset.")
    return lon_lat_list

# --- Patch Extraction ---

# Pre-compute a geographic coordinate system.
# Note: This requires EE to be initialized.
def get_projection_info(scale=config.SCALE):
    proj = ee.Projection('EPSG:4326').atScale(scale).getInfo()
    scale_x = proj['transform'][0]
    scale_y = -proj['transform'][4]
    return proj, scale_x, scale_y

@retry.Retry()
def get_patch(coords, image, patch_size=config.PATCH_SIZE, scale=config.SCALE, all_bands=None):
    """Get a patch centered on the coordinates, as a numpy array."""
    if all_bands is None:
        # Default to config if not provided, but allow override
        all_bands = config.FEATURE_NAMES + [config.LABEL_NAME]

    proj, scale_x, scale_y = get_projection_info(scale)
    
    offset_x = -scale_x * patch_size / 2
    offset_y = -scale_y * patch_size / 2

    request = {
        'fileFormat': 'NPY',
        'grid': {
            'dimensions': {
                'width': patch_size,
                'height': patch_size
            },
            'affineTransform': {
                'scaleX': scale_x,
                'shearX': 0,
                'shearY': 0,
                'scaleY': scale_y,
                'translateX': coords[0] + offset_x,
                'translateY': coords[1] + offset_y,
            },
            'crsCode': proj['crs']
        },
        'expression': image.select(all_bands)
    }

    return np.load(io.BytesIO(ee.data.computePixels(request)))

# --- TFRecord Processing ---

def parse_tfrecord(example_proto):
    """Parses a single TFRecord example."""
    kernel_shape = [config.PATCH_SIZE, config.PATCH_SIZE]
    features_dict = {}

    for f in config.FEATURE_NAMES:
        features_dict[f] = tf.io.FixedLenFeature(shape=kernel_shape, dtype=tf.float32)
    features_dict[config.LABEL_NAME] = tf.io.FixedLenFeature(shape=kernel_shape, dtype=tf.int64)

    parsed_features = tf.io.parse_single_example(example_proto, features_dict)

    for f in config.FEATURE_NAMES:
        parsed_features[f] = tf.clip_by_value(parsed_features[f], 0.0, 1.0)

    return parsed_features

def augment_data(features, label):
    """Applies data augmentation (flips and rotations)."""
    if tf.random.uniform(()) > 0.5:
        features = tf.image.flip_left_right(features)
        label = tf.image.flip_left_right(label)

    if tf.random.uniform(()) > 0.5:
        features = tf.image.flip_up_down(features)
        label = tf.image.flip_up_down(label)

    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    features = tf.image.rot90(features, k)
    label = tf.image.rot90(label, k)

    return features, label

def format_data(parsed_features, num_classes):
    """Stacks features and one-hot encodes the label."""
    feature_tensors = [parsed_features[f] for f in config.FEATURE_NAMES]
    features = tf.stack(feature_tensors, axis=-1)

    label = parsed_features[config.LABEL_NAME]
    label_one_hot = tf.one_hot(label, depth=num_classes)

    if label_one_hot.shape.rank == 4:
        label_one_hot = tf.squeeze(label_one_hot, axis=-2)

    label_one_hot = tf.cast(label_one_hot, tf.float32)

    return features, label_one_hot

def create_dataset(tfrecord_path, num_classes, is_training=True):
    """Creates a full dataset pipeline from TFRecord."""
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP")
    
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Use a lambda to pass num_classes to format_data
    dataset = dataset.map(lambda x: format_data(x, num_classes), num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(config.BUFFER_SIZE)
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

# --- Inference Helpers ---

def spline_window_2d(window_size, power=2):
    """
    Generates a 2D window using a Hanning window for image blending.
    """
    win_1d = np.hanning(window_size)
    window = np.outer(win_1d, win_1d)
    window = window / window.max()
    return window

def _pad_img(img, window_size, subdivisions):
    """
    Adds reflection padding to the image.
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (0, 0))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    return ret

def _unpad_img(padded_img, window_size, subdivisions):
    """
    Removes the padding added before prediction.
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    ret = padded_img[
        aug : -aug,
        aug : -aug,
        :
    ]
    return ret

from tqdm import tqdm

def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func):
    """
    Apply the `pred_func` (model.predict) using a sliding window with overlap.
    Overlapping predictions are averaged using a weighted window to remove edge artifacts.
    """
    pad_img = _pad_img(input_img, window_size, subdivisions)
    pad_img_shape = pad_img.shape

    step = int(window_size/subdivisions)

    window_weights = spline_window_2d(window_size)
    window_weights = np.expand_dims(window_weights, axis=-1)
    window_weights = np.tile(window_weights, (1, 1, nb_classes))

    prediction_probs = np.zeros((pad_img_shape[0], pad_img_shape[1], nb_classes), dtype=float)
    prediction_weights = np.zeros((pad_img_shape[0], pad_img_shape[1], nb_classes), dtype=float)

    path_h = range(0, pad_img_shape[0] - window_size + 1, step)
    path_w = range(0, pad_img_shape[1] - window_size + 1, step)

    total_patches = len(path_h) * len(path_w)
    print(f"  - Smooth Windowing: Processing {total_patches} sub-windows...")

    for r in tqdm(path_h, desc="Rows", leave=False):
        for c in path_w:
            patch = pad_img[r:r+window_size, c:c+window_size, :]
            patch_batch = np.expand_dims(patch, axis=0)
            pred_batch = pred_func(patch_batch)
            pred_one = pred_batch[0]

            prediction_probs[r:r+window_size, c:c+window_size] += pred_one * window_weights
            prediction_weights[r:r+window_size, c:c+window_size] += window_weights

    prediction_weights[prediction_weights == 0] = 1
    final_prediction = prediction_probs / prediction_weights

    return _unpad_img(final_prediction, window_size, subdivisions)
