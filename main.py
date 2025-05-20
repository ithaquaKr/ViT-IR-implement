import json
import os
import pickle

import numpy as np
import tensorflow as tf
from scipy.spatial import distance as scipy_distance
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import normalize as sk_normalize
from tqdm import tqdm
from vit_keras import vit

IMAGE_SIZE = 384

def get_vit_keras_model(vit_model_name_paper):
    model_name_keras = ""
    if vit_model_name_paper == 'ViT-B16':
        model_func = vit.vit_b16
    elif vit_model_name_paper == 'ViT-L16':
        model_func = vit.vit_l16
    elif vit_model_name_paper == 'ViT-B32':
        model_func = vit.vit_b32
    elif vit_model_name_paper == 'ViT-L32':
        model_func = vit.vit_l32
    else:
        raise ValueError(f"Tên mô hình ViT không xác định từ bài báo: {vit_model_name_paper}")

    try:
        model = model_func(
            image_size=IMAGE_SIZE,
            include_top=False,
            pretrained_top=False,
            weights='imagenet21k+imagenet2012'
        )
    except TypeError as e:
        print(f"Lỗi TypeError khi khởi tạo {vit_model_name_paper}: {e}. Thử không có tham số 'weights' và tải sau, hoặc không có 'pretrained'.")
        try:
            model = model_func(
                image_size=IMAGE_SIZE,
                include_top=False,
                pretrained=True,
                pretrained_top=False
            )
        except Exception as e2:
            raise ValueError(f"Không thể tải mô hình {vit_model_name_paper} với vit-keras: {e2}")

    model.trainable = False
    return model

# MODELS = ['ViT-B16', 'ViT-L16', 'ViT-B32', 'ViT-L32']
# for i in MODELS:
#   a = get_vit_keras_model(i)
#   if a:
#     print("DONE")

def load_and_preprocess_image_keras(image_path):
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img) # Chuyển đổi sang mảng numpy, 0-255

        # Tiền xử lý: trừ 127.5, chia tỷ lệ bởi 255 để có giá trị [-1, 1]
        processed_img_array = (img_array - 127.5) / 127.5
        # Thêm chiều batch
        return tf.expand_dims(processed_img_array, axis=0)
    except Exception as e:
        print(f"Lỗi khi tải hình ảnh {image_path}: {e}")
        return None

# image_path = "/content/drive/MyDrive/paper/information_retrieval/dataset/inria-holidays/images/116001.jpg"
# a = load_and_preprocess_image_keras(image_path)

def extract_features_keras(model, image_tensor):
    if image_tensor is None:
        return None
    # model.predict sẽ trả về một mảng numpy
    # Đầu ra của ViT khi include_top=False thường là (batch_size, num_patches + 1, hidden_dim)
    # Đặc trưng của token [CLS] là token đầu tiên
    model_output = model.predict(image_tensor, verbose=0)

    if model_output.ndim == 3: # (batch, sequence, hidden_dim)
        features = model_output[:, 0, :]
    elif model_output.ndim == 2: # (batch, hidden_dim)
        features = model_output
    else:
        raise ValueError(f"Định dạng đầu ra đặc trưng không mong đợi: {model_output.shape}")

    return features.squeeze()


def apply_normalization(features, method="l2_axis1", axis=1):
    if features is None or features.ndim == 0:
        return features
    if features.ndim == 1:
        features_2d = features.reshape(1, -1)
    else:
        features_2d = features

    if method == "l1_axis1":
        return sk_normalize(features_2d, norm='l1', axis=1).squeeze()
    elif method == "l2_axis1":
        return sk_normalize(features_2d, norm='l2', axis=1).squeeze()
    elif method == "l1_axis0":
        return sk_normalize(features_2d, norm='l1', axis=0).squeeze()
    elif method == "l2_axis0":
        return sk_normalize(features_2d, norm='l2', axis=0).squeeze()
    elif method == "robust":
        scaler = RobustScaler(quantile_range=(25.0, 75.0))
        return scaler.fit_transform(features_2d).squeeze()
    else: # none
        return features.squeeze()

def calculate_distance(feat1, feat2, metric="cosine"):
    if feat1 is None or feat2 is None:
        return float('inf')
    feat1 = np.asarray(feat1).ravel()
    feat2 = np.asarray(feat2).ravel()

    if metric == "manhattan":
        return scipy_distance.cityblock(feat1, feat2)
    elif metric == "euclidean":
        return scipy_distance.euclidean(feat1, feat2)
    elif metric == "cosine":
        return scipy_distance.cosine(feat1, feat2)
    elif metric == "braycurtis":
        return scipy_distance.braycurtis(feat1, feat2)
    elif metric == "canberra":
        return scipy_distance.canberra(feat1, feat2)
    elif metric == "chebyshev":
        return scipy_distance.chebyshev(feat1, feat2)
    elif metric == "correlation":
        if np.all(feat1 == feat1[0]) or np.all(feat2 == feat2[0]):
            return 1.0 if not np.array_equal(feat1, feat2) else 0.0
        return scipy_distance.correlation(feat1, feat2)
    else:
        raise ValueError(f"Khoảng cách không xác định: {metric}")


def calculate_ap(ranked_db_indices, relevant_indices_set):
    precisions = []
    num_relevant_retrieved = 0
    for i, db_idx in enumerate(ranked_db_indices):
        if db_idx in relevant_indices_set:
            num_relevant_retrieved += 1
            precisions.append(num_relevant_retrieved / (i + 1))
    if not precisions:
        return 0.0
    return np.mean(precisions)

def calculate_ns_score(ranked_db_indices, relevant_indices_set, top_k=4):
    count = 0
    for i in range(min(top_k, len(ranked_db_indices))):
        if ranked_db_indices[i] in relevant_indices_set:
            count += 1
    return count

def load_dataset_paths_and_ground_truth(dataset_name, dataset_base_path):
    print(f"Đang tải tập dữ liệu: {dataset_name}")
    dataset_path = os.path.join(dataset_base_path, dataset_name)
    images_dir = os.path.join(dataset_path, "images")  # tất cả ảnh nằm trong thư mục này
    gt_file = os.path.join(dataset_path, "groundtruth.json")  # ground truth file JSON như bạn mô tả

    # Đọc ground truth từ JSON
    with open(gt_file, "r") as f:
        queries_json = json.load(f)

    query_image_paths = []
    db_image_paths = []
    ground_truth = {}

    # Tạo index cho ảnh trong database
    all_image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))])
    image_name_to_path = {f: os.path.join(images_dir, f) for f in all_image_files}
    image_name_to_idx = {f: idx for idx, f in enumerate(all_image_files)}

    for q_idx, entry in queries_json.items():
        q_name = entry["query"]
        similar_list = entry["similar"]
        # Đảm bảo ảnh tồn tại
        if q_name not in image_name_to_path:
            continue
        query_image_paths.append(image_name_to_path[q_name])
        # Thêm các ảnh similar vào db nếu chưa có
        for sim_img in similar_list:
            if sim_img in image_name_to_path and image_name_to_path[sim_img] not in db_image_paths:
                db_image_paths.append(image_name_to_path[sim_img])

    # Đảm bảo query cũng có trong database
    for q_path in query_image_paths:
        if q_path not in db_image_paths:
            db_image_paths.append(q_path)

    # Cập nhật lại index sau khi có danh sách final
    db_name_to_idx = {os.path.basename(p): idx for idx, p in enumerate(db_image_paths)}
    query_idx_map = {i: os.path.basename(p) for i, p in enumerate(query_image_paths)}

    # Tạo ground truth dạng: {query_idx: set of db indices}
    for q_i, q_name in query_idx_map.items():
        q_entry = [v for v in queries_json.values() if v["query"] == q_name]
        if q_entry:
            sim_images = q_entry[0]["similar"]
            gt_indices = {db_name_to_idx[sim] for sim in sim_images if sim in db_name_to_idx}
            ground_truth[q_i] = gt_indices

    return query_image_paths, db_image_paths, ground_truth


# dataset_name = "inria-holidays"
# dataset_base_path = "/content/drive/MyDrive/paper/information_retrieval/dataset"
# query_image_paths, db_image_paths, ground_truth = load_dataset_paths_and_ground_truth(dataset_name, dataset_base_path)


# --- Các phương pháp chuẩn hóa theo yêu cầu của bài báo ---
NORMALIZATION_METHODS = [
    "l1_axis1",  # Tương ứng L1-norm Axis-1 trong bài báo (chuẩn hóa từng descriptor)
    "l1_axis0",  # Tương ứng L1-norm Axis-0 trong bài báo (chuẩn hóa từng characteristic/feature)
    "l2_axis1",  # Tương ứng L2-norm Axis-1 trong bài báo
    "l2_axis0",  # Tương ứng L2-norm Axis-0 trong bài báo
    "robust",     # Tương ứng ROBUST scaling
    "none"
]

# --- Các thước đo khoảng cách ---
# Giữ nguyên, bạn có thể chọn chỉ chạy Cosine nếu muốn khớp hoàn toàn với yêu cầu ban đầu
DISTANCE_METRICS = ["cosine", "euclidean", "manhattan", "chebyshev", "braycurtis", "canberra", "correlation"]
# Hoặc chỉ Cosine:
# DISTANCE_METRICS = ["cosine"]


# --- Hàm trợ giúp ---

def apply_normalization(features_dict, method="l1_axis1"):
    if not features_dict: return {}
    indices = list(features_dict.keys())
    valid_features_list = [features_dict[idx] for idx in indices if features_dict[idx] is not None]
    if not valid_features_list: return {idx: None for idx in indices}

    all_features_array = np.array(valid_features_list)
    if all_features_array.ndim == 1 and all_features_array.size > 0:
        all_features_array = all_features_array.reshape(1, -1)
    elif all_features_array.size == 0:
        return {idx: None for idx in indices}

    normalized_features_array = None
    if all_features_array.shape[0] == 0: # Không có đặc trưng hợp lệ
        normalized_features_array = all_features_array
    elif method == "robust":
        # RobustScaler hoạt động trên từng đặc trưng (cột), nên nó tương ứng với axis=0
        # nhưng nó được fit trên toàn bộ dữ liệu.
        scaler = RobustScaler(quantile_range=(25.0, 75.0))
        normalized_features_array = scaler.fit_transform(all_features_array)
    elif method == "l1_axis1": # Chuẩn hóa từng mẫu/descriptor (row-wise)
        normalized_features_array = sk_normalize(all_features_array, norm='l1', axis=1)
    elif method == "l1_axis0": # Chuẩn hóa từng đặc trưng/characteristic (column-wise)
        normalized_features_array = sk_normalize(all_features_array, norm='l1', axis=0)
    elif method == "l2_axis1": # Chuẩn hóa từng mẫu/descriptor (row-wise)
        normalized_features_array = sk_normalize(all_features_array, norm='l2', axis=1)
    elif method == "l2_axis0": # Chuẩn hóa từng đặc trưng/characteristic (column-wise)
        normalized_features_array = sk_normalize(all_features_array, norm='l2', axis=0)
    elif method == "none":
        normalized_features_array = all_features_array
    else:
        raise ValueError(f"Phương pháp chuẩn hóa không xác định: {method}")

    normalized_features_dict = {}
    valid_feature_idx = 0
    for original_idx in indices:
        if features_dict[original_idx] is not None and \
           normalized_features_array is not None and \
           normalized_features_array.shape[0] > 0 and \
           valid_feature_idx < normalized_features_array.shape[0]:
            normalized_features_dict[original_idx] = normalized_features_array[valid_feature_idx]
            valid_feature_idx += 1
        else:
            normalized_features_dict[original_idx] = None
    return normalized_features_dict

# --- Các hàm calculate_distance, calculate_ap, load_data_from_pkl, average_query_expansion ---
# --- GIỮ NGUYÊN NHƯ PHIÊN BẢN TRƯỚC ---

def calculate_distance(feat1, feat2, metric="cosine"):
    if feat1 is None or feat2 is None: return float('inf')
    feat1 = np.asarray(feat1).ravel()
    feat2 = np.asarray(feat2).ravel()
    if metric in ["braycurtis", "canberra"] and (np.all(feat1 == 0) and np.all(feat2 == 0)): return 0.0
    if metric == "correlation":
        if np.std(feat1) < 1e-9 or np.std(feat2) < 1e-9 : # Sử dụng epsilon nhỏ để tránh lỗi chia cho 0
            return 1.0 if not np.array_equal(feat1, feat2) else 0.0
    try:
        if metric == "cosine": return scipy_distance.cosine(feat1, feat2)
        elif metric == "euclidean": return scipy_distance.euclidean(feat1, feat2)
        elif metric == "manhattan": return scipy_distance.cityblock(feat1, feat2)
        elif metric == "chebyshev": return scipy_distance.chebyshev(feat1, feat2)
        elif metric == "braycurtis": return scipy_distance.braycurtis(feat1, feat2)
        elif metric == "canberra": return scipy_distance.canberra(feat1, feat2)
        elif metric == "correlation": return scipy_distance.correlation(feat1, feat2)
        else: raise ValueError(f"Thước đo khoảng cách không xác định: {metric}")
    except Exception: return float('inf')

def calculate_ap(ranked_db_indices, relevant_indices_set):
    precisions = []
    num_relevant_retrieved = 0
    for i, db_idx in enumerate(ranked_db_indices):
        if db_idx in relevant_indices_set:
            num_relevant_retrieved += 1
            precisions.append(num_relevant_retrieved / (i + 1))
    if not precisions: return 0.0
    return np.mean(precisions)

def load_data_from_pkl(base_path, model_prefix):
    query_features_path = os.path.join(base_path, f"{model_prefix}_query_features.pkl")
    db_features_path = os.path.join(base_path, f"{model_prefix}_db_features.pkl")
    # Giả sử ground truth có tên theo model prefix hoặc một tên cố định
    # Nếu tên cố định, ví dụ: "INRIA_Holidays_ground_truth.pkl"
    # ground_truth_path = os.path.join(base_path, "INRIA_Holidays_ground_truth.pkl")
    ground_truth_path = os.path.join(base_path, f"{model_prefix}_ground_truth.pkl")


    if not all(os.path.exists(p) for p in [query_features_path, db_features_path, ground_truth_path]):
        print(f"LỖI: Thiếu một hoặc nhiều tệp dữ liệu cho model {model_prefix}.")
        print(f"  Kiểm tra: {query_features_path}, {db_features_path}, {ground_truth_path}")
        return None, None, None
    try:
        with open(query_features_path, 'rb') as f: query_features_raw = pickle.load(f)
        with open(db_features_path, 'rb') as f: db_features_raw = pickle.load(f)
        with open(ground_truth_path, 'rb') as f: ground_truth_map = pickle.load(f)
        print(f"Đã tải dữ liệu cho model {model_prefix}.")
        return query_features_raw, db_features_raw, ground_truth_map
    except Exception as e:
        print(f"Lỗi khi tải tệp PKL cho {model_prefix}: {e}")
        return None, None, None


def average_query_expansion(original_query_feat, top_k_db_feats, alpha=0.5):
    if not top_k_db_feats: return original_query_feat
    valid_top_k_feats = [feat for feat in top_k_db_feats if feat is not None]
    if not valid_top_k_feats: return original_query_feat
    mean_top_k_feat = np.mean(np.array(valid_top_k_feats), axis=0)
    expanded_query_feat = alpha * original_query_feat + (1 - alpha) * mean_top_k_feat
    return expanded_query_feat

# --- Hàm xử lý chính với nhiều cấu hình ---
def evaluate_all_configurations_ext(base_path, model_prefixes, norm_methods, dist_metrics, K_for_qe=3, alpha_qe=0.2):
    print("\n===== Bắt đầu Đánh giá Toàn diện Mở rộng =====")
    print(f"Tham số QE: K={K_for_qe}, alpha={alpha_qe}")

    results_summary = []

    for model_prefix in model_prefixes:
        print(f"\n--- Đang xử lý Model: {model_prefix} ---")
        query_features_raw, db_features_raw, ground_truth_map = load_data_from_pkl(base_path, model_prefix)

        if query_features_raw is None or db_features_raw is None or ground_truth_map is None:
            print(f"Bỏ qua model {model_prefix} do thiếu dữ liệu.")
            continue

        for norm_method in norm_methods:
            print(f"  -- Chuẩn hóa: {norm_method} --")

            # Áp dụng chuẩn hóa
            # Việc kết hợp query và db features để chuẩn hóa cùng lúc là quan trọng,
            # đặc biệt đối với RobustScaler và chuẩn hóa axis=0.
            # Chúng ta sẽ tạo một mảng lớn, chuẩn hóa, rồi tách ra.

            query_keys = list(query_features_raw.keys())
            db_keys_orig = list(db_features_raw.keys())

            # Lấy tất cả các đặc trưng hợp lệ vào một danh sách để tạo mảng numpy
            all_valid_feats_list = []
            # Map để theo dõi vị trí của từng đặc trưng gốc trong mảng lớn
            # key: ("query", q_idx) hoặc ("db", db_idx), value: index trong all_valid_feats_list
            feature_to_array_idx_map = {}
            current_array_idx = 0

            for q_key in query_keys:
                feat = query_features_raw.get(q_key)
                if feat is not None:
                    all_valid_feats_list.append(feat)
                    feature_to_array_idx_map[("query", q_key)] = current_array_idx
                    current_array_idx += 1
            
            for db_key in db_keys_orig:
                feat = db_features_raw.get(db_key)
                if feat is not None:
                    all_valid_feats_list.append(feat)
                    feature_to_array_idx_map[("db", db_key)] = current_array_idx
                    current_array_idx += 1
            
            if not all_valid_feats_list:
                print(f"    Không có đặc trưng hợp lệ nào cho model {model_prefix}. Bỏ qua chuẩn hóa {norm_method}.")
                continue
            
            # Tạo mảng numpy từ danh sách các đặc trưng hợp lệ
            combined_features_array = np.array(all_valid_feats_list)
            
            # Tạo một dict tạm thời để truyền vào apply_normalization
            # (apply_normalization hiện tại nhận dict, nhưng logic bên trong xử lý mảng)
            # Cách đơn giản hơn là sửa apply_normalization để nhận trực tiếp mảng
            # Hoặc, chúng ta có thể truyền một dict giả ở đây.
            # Để giữ apply_normalization như cũ, chúng ta tạo dict giả.
            temp_dict_for_norm = {i: combined_features_array[i] for i in range(combined_features_array.shape[0])}
            
            normalized_temp_dict = apply_normalization(temp_dict_for_norm, norm_method)
            
            # Tách trở lại
            current_query_features = {}
            current_db_features = {}

            for key_tuple, array_idx in feature_to_array_idx_map.items():
                normalized_feat = normalized_temp_dict.get(array_idx)
                if key_tuple[0] == "query":
                    current_query_features[key_tuple[1]] = normalized_feat
                elif key_tuple[0] == "db":
                    current_db_features[key_tuple[1]] = normalized_feat


            for dist_metric in dist_metrics:
                # ... (phần còn lại của vòng lặp dist_metrics và QE giữ nguyên như trước) ...
                print(f"    --- Khoảng cách: {dist_metric} ---")
                all_aps_initial = []
                all_aps_qe = []

                query_indices_sorted = sorted(current_query_features.keys()) # Dùng key từ dict đã chuẩn hóa
                db_indices_sorted = sorted(current_db_features.keys())

                for q_idx in tqdm(query_indices_sorted, desc=f"      QE {model_prefix}-{norm_method[:6]}-{dist_metric[:4]}", leave=False):
                    original_q_feat = current_query_features.get(q_idx)
                    if original_q_feat is None: continue
                    relevant_set = ground_truth_map.get(q_idx) # Giả sử q_idx trong GT khớp với query_features_raw
                    if relevant_set is None: continue

                    # Truy vấn ban đầu
                    distances_initial = []
                    for db_idx in db_indices_sorted:
                        db_feat = current_db_features.get(db_idx)
                        if db_feat is None: continue
                        dist = calculate_distance(original_q_feat, db_feat, dist_metric)
                        distances_initial.append((dist, db_idx))
                    if not distances_initial: continue
                    distances_initial.sort(key=lambda x: x[0])
                    ranked_db_initial_indices = [item[1] for item in distances_initial]
                    ap_initial = calculate_ap(ranked_db_initial_indices, relevant_set)
                    all_aps_initial.append(ap_initial)

                    # Mở rộng Truy vấn
                    expanded_q_feat = original_q_feat
                    if K_for_qe > 0 and ranked_db_initial_indices:
                        top_k_indices_for_qe = ranked_db_initial_indices[:K_for_qe]
                        top_k_db_feats_for_qe = [current_db_features.get(idx) for idx in top_k_indices_for_qe]
                        valid_top_k_db_feats_for_qe = [f for f in top_k_db_feats_for_qe if f is not None]
                        if valid_top_k_db_feats_for_qe:
                            expanded_q_feat = average_query_expansion(original_q_feat, valid_top_k_db_feats_for_qe, alpha=alpha_qe)

                    # Truy vấn lại
                    distances_qe = []
                    for db_idx in db_indices_sorted:
                        db_feat = current_db_features.get(db_idx)
                        if db_feat is None: continue
                        dist = calculate_distance(expanded_q_feat, db_feat, dist_metric)
                        distances_qe.append((dist, db_idx))
                    if not distances_qe: all_aps_qe.append(0.0); continue
                    distances_qe.sort(key=lambda x: x[0])
                    ranked_db_qe_indices = [item[1] for item in distances_qe]
                    ap_qe = calculate_ap(ranked_db_qe_indices, relevant_set)
                    all_aps_qe.append(ap_qe)

                mAP_initial = np.mean(all_aps_initial) if all_aps_initial else 0.0
                mAP_qe = np.mean(all_aps_qe) if all_aps_qe else 0.0

                results_summary.append({
                    "model": model_prefix,
                    "normalization": norm_method,
                    "distance_metric": dist_metric,
                    "mAP_initial": mAP_initial,
                    "mAP_qe_K3_alpha02": mAP_qe
                })
                print(f"        {model_prefix}-{norm_method}-{dist_metric}: mAP Ban đầu={mAP_initial:.4f}, mAP QE={mAP_qe:.4f}")

    print("\n\n===== Bảng Tổng hợp Kết quả Cuối cùng =====")
    header = f"{'Model':<10} | {'Normalization':<12} | {'Distance':<12} | {'mAP Initial':<12} | {'mAP QE (K=3, alpha=0.2)':<25}"
    print(header)
    print("-" * len(header))
    for res in results_summary:
        print(f"{res['model']:<10} | {res['normalization']:<12} | {res['distance_metric']:<12} | {res['mAP_initial']:.4f}{'':<7} | {res['mAP_qe_K3_alpha02']:.4f}")
    print("=" * len(header))


# --- Điểm vào chính ---
if __name__ == "__main__":
    DATA_PKL_BASE_PATH = "/kaggle/input/last-vit-retrieval"
    MODEL_PREFIXES_TO_TEST = ["ViT-B16", "ViT-B32", "ViT-L16", "ViT-L32"]
    # MODEL_PREFIXES_TO_TEST = ["ViT-B32"] # Để test nhanh

    K_QE_FIXED = 3
    ALPHA_QE_FIXED = 0.2

    # Để khớp với yêu cầu ban đầu, bạn có thể chỉ chạy với Cosine
    # DISTANCE_METRICS_TO_RUN = ["cosine"]
    DISTANCE_METRICS_TO_RUN = DISTANCE_METRICS # Chạy tất cả các distance metrics

    evaluate_all_configurations_ext(
        DATA_PKL_BASE_PATH,
        MODEL_PREFIXES_TO_TEST,
        NORMALIZATION_METHODS,
        DISTANCE_METRICS_TO_RUN,
        K_for_qe=K_QE_FIXED,
        alpha_qe=ALPHA_QE_FIXED
    )


