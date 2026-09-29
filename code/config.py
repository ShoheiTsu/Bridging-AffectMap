"""
Project settings for Bridging-AffectMap analyses and figures.

By default all paths are relative to this repository root. Point DATA_DIR /
RESULTS_DIR at your local OASIS tables and CLIP feature arrays if they live
elsewhere.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"

# Inputs (OASIS scores, images, frozen CLIP features, prior step outputs)
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_2DPRED = PROJECT_ROOT / "results_2dpred"

# Analysis / figure outputs for this project
RESULTS_GENDER = PROJECT_ROOT / "results"
FIG_INTEGRATED = PROJECT_ROOT / "fig_doc"
CVAE_CROSS_GENDER_DIR = RESULTS_GENDER / "cvae_cross_gender"

# Data files
OASIS_SCORES_CSV = DATA_DIR / "oasis_scores.csv"
IMAGES_DIR = DATA_DIR / "images"

# Step-wise result dirs (CLIP features, fusion models, etc.)
RESULTS_STEP1 = RESULTS_DIR / "step1_individual_models"
RESULTS_STEP2 = RESULTS_DIR / "step2_image_characteristics"
RESULTS_STEP3 = RESULTS_DIR / "step3_fusion_model"
RESULTS_LAYER = RESULTS_DIR / "step_layer_analysis"

# Evaluation
TARGET_COLUMNS = ["valence", "arousal"]
CATEGORY_COLUMN = "category"
IMAGE_ID_COLUMN = "image_id"
VALENCE_AROUSAL_SCALE_MIN = 1.0
VALENCE_AROUSAL_SCALE_MAX = 7.0

# Model
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
VIT_MODEL_NAME = "google/vit-base-patch16-224"

# Training / evaluation
RANDOM_SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
TRAIN_RATIO = 0.8

# Ridge alpha
ALPHA_SEARCH_LOW = 0.1
ALPHA_SEARCH_HIGH = 1000.0
ALPHA_SEARCH_LOW_VIT_LOTO = 50.0
LOTO_VIT_CONSERVATIVE_ALPHA_RATIO = 0.95
ALPHA_SEARCH_HIGH_LOCO = None
ALPHA_LOCO_OPTIMIZED_CLIP = 256.93
ALPHA_LOCO_OPTIMIZED_VIT = 802.60
ALPHA_FUSION_DEFAULT = 10.0

# Layer analysis
LAYERS_TO_EVAL = [3, 6, 9, 12]
ALPHA_LAYER_LOTO_CLIP = 0.23
ALPHA_LAYER_LOTO_VIT = 0.10
ALPHA_LAYER_LOCO_CLIP = 256.93
ALPHA_LAYER_LOCO_VIT = 802.60
ALPHA_LAYER_LOTO_FUSION = 0.23
ALPHA_LAYER_LOCO_FUSION = 256.93
CONCEPTUAL_CATEGORY_KEYWORDS = ["face", "pose", "woman", "man", "couple", "nude", "portrait", "expression", "person"]
VISUAL_CATEGORY_KEYWORDS = ["nature", "landscape", "sky", "water", "fire", "flower", "food", "animal", "object", "scene"]
