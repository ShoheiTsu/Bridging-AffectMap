"""
論文B（性別・男女差）用設定: EmotionPro2 出力先、データは EmotionPro1 を参照。
"""
from pathlib import Path

# EmotionPro2 をプロジェクトルートとする（出力先）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# データ・既存結果は EmotionPro1 を参照（EmotionPro2 と EmotionPro1 は project/ 直下の兄弟フォルダ想定）
EMOTIONPRO1 = PROJECT_ROOT.parent / "EmotionPro1"
DATA_DIR = EMOTIONPRO1 / "data"
RESULTS_DIR = EMOTIONPRO1 / "results"
RESULTS_2DPRED = EMOTIONPRO1 / "results_2dpred"
CODE_DIR = PROJECT_ROOT / "code"

# 論文B の出力先: EmotionPro2 内
RESULTS_GENDER = PROJECT_ROOT / "results"
FIG_INTEGRATED = PROJECT_ROOT / "fig_doc"
# クロス性別 cVAE（Encoder + Male/Female Decoder）の保存先
CVAE_CROSS_GENDER_DIR = RESULTS_GENDER / "cvae_cross_gender"

# データ
OASIS_SCORES_CSV = DATA_DIR / "oasis_scores.csv"
IMAGES_DIR = DATA_DIR / "images"

# 結果保存先（Step 別）— 読み取りは EmotionPro1
RESULTS_STEP1 = RESULTS_DIR / "step1_individual_models"
RESULTS_STEP2 = RESULTS_DIR / "step2_image_characteristics"
RESULTS_STEP3 = RESULTS_DIR / "step3_fusion_model"
RESULTS_LAYER = RESULTS_DIR / "step_layer_analysis"

# 評価
TARGET_COLUMNS = ["valence", "arousal"]
CATEGORY_COLUMN = "category"
IMAGE_ID_COLUMN = "image_id"
VALENCE_AROUSAL_SCALE_MIN = 1.0
VALENCE_AROUSAL_SCALE_MAX = 7.0

# モデル
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
VIT_MODEL_NAME = "google/vit-base-patch16-224"

# 学習・評価
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

# 層分析用
LAYERS_TO_EVAL = [3, 6, 9, 12]
ALPHA_LAYER_LOTO_CLIP = 0.23
ALPHA_LAYER_LOTO_VIT = 0.10
ALPHA_LAYER_LOCO_CLIP = 256.93
ALPHA_LAYER_LOCO_VIT = 802.60
ALPHA_LAYER_LOTO_FUSION = 0.23
ALPHA_LAYER_LOCO_FUSION = 256.93
CONCEPTUAL_CATEGORY_KEYWORDS = ["face", "pose", "woman", "man", "couple", "nude", "portrait", "expression", "person"]
VISUAL_CATEGORY_KEYWORDS = ["nature", "landscape", "sky", "water", "fire", "flower", "food", "animal", "object", "scene"]
