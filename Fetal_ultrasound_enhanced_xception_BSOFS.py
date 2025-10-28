import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Add, SeparableConv2D, ReLU, BatchNormalization, MaxPool2D, \
    GlobalAvgPool2D, Dropout, Layer
from tensorflow.keras.models import Model
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit as logsig
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.manifold import TSNE
import os
import pywt
import cv2
import seaborn as sns
import matplotlib.pyplot as plt


# Wavelet Denoising Function
def wavelet_denoising(image):
    image = image.astype(np.float32) / 255.0
    input_shape = image.shape[:2]  # Store original shape (299,299)
    denoised_image = np.zeros_like(image)

    for c in range(image.shape[2]):
        # Perform 2D wavelet decomposition (db4 wavelet, level=1)
        coeffs = pywt.wavedec2(image[:, :, c], 'db4', level=1)
        approx, (horizontal, vertical, diagonal) = coeffs

        # Estimate noise standard deviation from detail coefficients
        detail_coeffs = [horizontal, vertical, diagonal]
        valid_details = [c for c in detail_coeffs if c.size > 0]
        if not valid_details:
            denoised_image[:, :, c] = approx
            continue
        flat_details = np.concatenate([c.flatten() for c in valid_details])
        sigma = np.median(np.abs(flat_details)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(image.size))

        # Apply soft thresholding to detail coefficients
        denoised_horizontal = pywt.threshold(horizontal, threshold, mode='soft') if horizontal.size > 0 else horizontal
        denoised_vertical = pywt.threshold(vertical, threshold, mode='soft') if vertical.size > 0 else vertical
        denoised_diagonal = pywt.threshold(diagonal, threshold, mode='soft') if diagonal.size > 0 else diagonal

        # Reconstruct coefficients as [approx, (horizontal, vertical, diagonal)]
        denoised_coeffs = [approx, (denoised_horizontal, denoised_vertical, denoised_diagonal)]

        # Reconstruct the image and resize to match input shape
        reconstructed = pywt.waverec2(denoised_coeffs, 'db4')
        # Ensure the reconstructed image matches the input dimensions
        if reconstructed.shape != input_shape:
            reconstructed = cv2.resize(reconstructed, input_shape[::-1], interpolation=cv2.INTER_LINEAR)
        denoised_image[:, :, c] = reconstructed

    denoised_image = np.clip(denoised_image, 0, 1) * 255.0
    return denoised_image.astype(np.uint8)


# Adaptive Gamma Correction with Weighted Distribution (AGCWD)
def agcwd(image, alpha=0.5):
    image = image.astype(np.float32) / 255.0
    enhanced_image = np.zeros_like(image)

    for c in range(image.shape[2]):
        hist, bins = np.histogram(image[:, :, c].flatten(), bins=256, range=(0, 1))
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]
        w_cdf = np.power(cdf, alpha)
        w_cdf = w_cdf / w_cdf[-1]
        intensities = np.linspace(0, 1, 256)
        mapped_intensities = np.interp(intensities, w_cdf, intensities)
        gamma = 1.0 / (1.0 + mapped_intensities)
        enhanced_image[:, :, c] = np.power(image[:, :, c], gamma[np.searchsorted(intensities, image[:, :, c])])

    enhanced_image = np.clip(enhanced_image, 0, 1) * 255.0
    return enhanced_image.astype(np.uint8)


# Preprocessing Function
def preprocess_image(image):
    # Convert EagerTensor to NumPy array
    image = image.numpy()

    # Handle batched or single images
    if len(image.shape) == 4:  # Batched: (batch_size, 299, 299, 3)
        batch_size = image.shape[0]
        processed_images = np.zeros_like(image)
        for i in range(batch_size):
            single_image = image[i]
            if single_image.shape != (299, 299, 3):
                raise ValueError(f"Unexpected single image shape: {single_image.shape}, expected (299, 299, 3)")
            single_image = wavelet_denoising(single_image)
            single_image = agcwd(single_image, alpha=0.5)
            single_image = cv2.GaussianBlur(single_image, (3, 3), sigmaX=0)
            processed_images[i] = single_image
        return processed_images
    else:  # Single image: (299, 299, 3)
        if image.shape != (299, 299, 3):
            raise ValueError(f"Unexpected image shape: {image.shape}, expected (299, 299, 3)")
        image = wavelet_denoising(image)
        image = agcwd(image, alpha=0.5)
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=0)
        return image


# TensorFlow Wrapper for Preprocessing
def tf_preprocess_image(image, label):
    [image, ] = tf.py_function(
        func=preprocess_image,
        inp=[image],
        Tout=[tf.uint8]
    )
    # Handle dynamic batch size
    image = tf.ensure_shape(image, [None, 299, 299, 3])  # Allow variable batch size
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# Custom AvgTopKPooling Layer
class AvgTopKPooling(tf.keras.layers.Layer):
    def __init__(self, ksize=3, kk=5, stride=2):
        super(AvgTopKPooling, self).__init__()
        self.ksize = ksize
        self.kk = kk
        self.stride = stride

    def call(self, inputs):
        k_size = self.ksize
        stride = self.stride
        channel = inputs.shape[3]
        x_patches = tf.image.extract_patches(inputs,
                                             sizes=[1, k_size, k_size, 1],
                                             strides=[1, stride, stride, 1],
                                             rates=[1, 1, 1, 1],
                                             padding='SAME')
        output = tf.concat(
            [tf.reduce_mean(tf.math.top_k(x_patches[:, :, :, c::channel], k=self.kk).values, keepdims=True, axis=-1) for
             c in range(channel)], axis=-1)
        return output


# Custom CropToMatch Layer
class CropToMatch(tf.keras.layers.Layer):
    def call(self, inputs):
        tensor, target = inputs
        target_height = tf.shape(target)[1]
        target_width = tf.shape(target)[2]
        return tensor[:, :target_height, :target_width, :]


# Channel Attention Mechanism
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, filters, ratio=8):
        super(ChannelAttention, self).__init__()
        self.filters = filters
        self.ratio = ratio
        self.global_avg_pool = GlobalAvgPool2D()
        self.dense1 = Dense(filters // ratio, activation='relu')
        self.dense2 = Dense(filters, activation='sigmoid')

    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = tf.expand_dims(tf.expand_dims(x, 1), 1)
        x = self.dense1(x)
        x = self.dense2(x)
        return inputs * x


# Spatial Attention Mechanism
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention


# Conv-BatchNorm Block
def conv_bn(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


# SeparableConv-BatchNorm Block
def sep_bn(x, filters, kernel_size, strides=1):
    x = SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


# Dilated Convolution Block
def dilated_conv_bn(x, filters, kernel_size, dilation_rate=2):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate, use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


# Modified Entry Flow
def entry_flow(x):
    x = conv_bn(x, filters=32, kernel_size=3, strides=2)
    x = ReLU()(x)
    x = conv_bn(x, filters=64, kernel_size=3, strides=1)
    tensor = ReLU()(x)

    x = dilated_conv_bn(tensor, filters=128, kernel_size=3, dilation_rate=2)
    x = ReLU()(x)
    x = sep_bn(x, filters=128, kernel_size=3)
    x = AvgTopKPooling(ksize=3, kk=5, stride=2)(x)

    tensor = conv_bn(tensor, filters=128, kernel_size=1, strides=2)
    x = Add()([tensor, x])

    x = ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = ReLU()(x)
    x = ChannelAttention(filters=256)(x)
    x = AvgTopKPooling(ksize=3, kk=5, stride=2)(x)

    tensor = conv_bn(tensor, filters=256, kernel_size=1, strides=2)
    tensor = CropToMatch()([tensor, x])
    x = Add()([tensor, x])

    x = ReLU()(x)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = SpatialAttention()(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=728, kernel_size=1, strides=2)
    tensor = CropToMatch()([tensor, x])
    x = Add()([tensor, x])
    return x


# Middle Flow
def middle_flow(tensor):
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        tensor = Add()([tensor, x])
    return tensor


# Exit Flow
def exit_flow(tensor):
    x = ReLU()(tensor)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=1024, kernel_size=3)
    x = AvgTopKPooling(ksize=3, kk=5, stride=2)(x)

    tensor = conv_bn(tensor, filters=1024, kernel_size=1, strides=2)
    tensor = CropToMatch()([tensor, x])
    x = Add()([tensor, x])

    x = sep_bn(x, filters=1536, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=2048, kernel_size=3)
    x = GlobalAvgPool2D()(x)

    return x


# Build Feature Extraction Model
def build_feature_extractor():
    input = Input(shape=(299, 299, 3))
    x = entry_flow(input)
    x = middle_flow(x)
    x = exit_flow(x)
    model = Model(inputs=input, outputs=x)
    return model


# Grad-CAM Implementation
def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img


# Occlusion Sensitivity Implementation
def occlusion_sensitivity(model, scaler, selected_features, img, true_label, patch_size=30, stride=15):
    img_array = np.copy(img)
    height, width = img.shape[:2]
    output = np.zeros((height, width))
    original_features = model.predict(np.expand_dims(img, axis=0))
    original_features = scaler.transform(original_features[:, selected_features == 1])
    original_prob = grid_search.predict_proba(original_features)[0, true_label]

    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            occluded_img = np.copy(img_array)
            occluded_img[i:i + patch_size, j:j + patch_size, :] = 128  # Gray patch
            features = model.predict(np.expand_dims(occluded_img, axis=0))
            features = scaler.transform(features[:, selected_features == 1])
            prob = grid_search.predict_proba(features)[0, true_label]
            output[i:i + patch_size, j:j + patch_size] = original_prob - prob

    return output


# Visualization Function
def visualize_results(test_features_selected, test_labels, test_predictions, test_images, feature_extractor, scaler,
                      selected_features, class_names, grid_search):
    os.makedirs('results', exist_ok=True)

    # 1. Confusion Matrix
    cm = confusion_matrix(test_labels, test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('results/confusion_matrix.png')
    plt.close()

    # 2. t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=123)
    tsne_features = tsne.fit_transform(test_features_selected)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=test_labels, cmap='viridis', alpha=0.6)
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names, title="Classes")
    plt.title('t-SNE Visualization of Selected Features')
    plt.savefig('results/tsne_plot.png')
    plt.close()

    # 3. ROC Curve (One-vs-Rest)
    n_classes = len(class_names)
    y_score = grid_search.predict_proba(test_features_selected)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels == i, y_score[:, i])
        roc_auc[i] = roc_auc_score(test_labels == i, y_score[:, i])

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curve.png')
    plt.close()

    # 4. Grad-CAM Visualization
    last_conv_layer_name = 'separable_conv2d_25'  # Last conv layer in exit_flow
    for i in range(min(5, len(test_images))):  # Visualize 5 samples
        img = test_images[i]
        img_array = np.expand_dims(img, axis=0)
        heatmap = get_gradcam_heatmap(feature_extractor, img_array, last_conv_layer_name,
                                      pred_index=int(test_predictions[i]))
        superimposed_img = overlay_gradcam(img * 255.0, heatmap)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f'Original Image (Class: {class_names[int(test_labels[i])]})')
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Grad-CAM')
        plt.imshow(superimposed_img)
        plt.axis('off')
        plt.savefig(f'results/gradcam_sample_{i}.png')
        plt.close()

    # 5. Occlusion Sensitivity
    for i in range(min(5, len(test_images))):  # Visualize 5 samples
        img = test_images[i]
        sensitivity_map = occlusion_sensitivity(feature_extractor, scaler, selected_features, img, int(test_labels[i]),
                                                patch_size=30, stride=15)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f'Original Image (Class: {class_names[int(test_labels[i])]})')
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Occlusion Sensitivity')
        plt.imshow(sensitivity_map, cmap='hot')
        plt.colorbar(label='Drop in Confidence')
        plt.axis('off')
        plt.savefig(f'results/occlusion_sensitivity_sample_{i}.png')
        plt.close()


# Adapted BSO2O for Feature Selection
def binary_BSO2O(X_train, y_train, X_val, y_val, D):
    max_FEs = 1000 * D
    NP = 4 * D
    max_iter = int(max_FEs / NP)
    prob_one_cluster = 0.1
    Sr = 20
    k = int(NP / Sr)

    pop = np.random.randint(0, 2, size=(NP, D))
    iters = 0
    FEs = 0
    prob = np.zeros(k)

    def fitness_function(selected_features):
        if np.sum(selected_features) == 0:
            return 0
        X_train_selected = X_train[:, selected_features == 1]
        X_val_selected = X_val[:, selected_features == 1]
        clf = SVC(kernel='rbf', C=1.0)
        clf.fit(X_train_selected, y_train)
        return clf.score(X_val_selected, y_val)

    fit = np.array([fitness_function(pop[i]) for i in range(NP)])
    bestfit = np.max(fit)
    best_solution = pop[np.argmax(fit)].copy()
    temp_pop = np.zeros((NP, D))
    leaders = np.zeros(NP, dtype=int)
    cluster_idx = np.zeros(NP, dtype=int)

    FEs += NP

    while FEs < max_FEs:
        kr = int(np.ceil(k * (1 - iters / max_iter)))
        N_nbc = NP - kr * Sr
        kn = k - kr
        species = {}

        if kn > 0:
            sidx = np.argsort(-fit)
            fit = fit[sidx]
            pop = pop[sidx, :]
            pop_nbc = pop[:N_nbc, :]
            fit_nbc = fit[:N_nbc]
            matdis = cdist(pop_nbc, pop_nbc)
            species, leaders_nbc, cluster_idx_nbc = NBC(matdis, fit_nbc, pop_nbc, kn)
            cluster_idx[:N_nbc] = cluster_idx_nbc
            leaders[:N_nbc] = leaders_nbc

        N_rg = NP - N_nbc
        rp = N_nbc + np.random.permutation(N_rg)
        start_idx = 0
        for j in range(kn, k):
            if j == k - 1:
                species[j] = {'idx': rp[start_idx:], 'len': len(rp[start_idx:])}
            else:
                species[j] = {'idx': rp[start_idx:start_idx + Sr], 'len': Sr}
            cluster_idx[species[j]['idx']] = j
            start_idx += Sr

        for i in range(N_nbc, NP):
            c = cluster_idx[i]
            temp = np.where(fit[species[c]['idx']] > fit[i])[0]
            if len(temp) == 0:
                leaders[i] = i
            else:
                leaders[i] = species[c]['idx'][np.random.choice(temp)]

        for i in range(NP):
            temp_idx = np.random.randint(NP)
            if np.random.rand() < prob_one_cluster:
                r = np.random.rand(D)
                temp_pop[i, :] = (1 - r) * pop[temp_idx, :] + r * pop[leaders[temp_idx], :]
            else:
                c1 = np.random.randint(k)
                i1 = species[c1]['idx'][np.random.randint(species[c1]['len'])]
                c2 = np.random.randint(k)
                i2 = species[c2]['idx'][np.random.randint(species[c2]['len'])]
                r1 = np.random.rand(D)
                r2 = np.random.rand(D)
                temp_pop[i, :] = (1 - r1 - r2) * pop[temp_idx, :] + r1 * pop[i1, :] + r2 * pop[i2, :]

            step_size = logsig((0.5 * max_iter - iters) / 20) * np.random.rand(D)
            rn = np.random.normal(0, 1, D)
            temp_pop[i, :] = temp_pop[i, :] + step_size * rn
            temp_pop[i, :] = (temp_pop[i, :] > 0.5).astype(int)

        temp_fit = np.array([fitness_function(temp_pop[i]) for i in range(NP)])
        FEs += NP
        is_update = temp_fit > fit
        fit[is_update] = temp_fit[is_update]
        pop[is_update, :] = temp_pop[is_update, :]

        if np.max(temp_fit) > bestfit:
            bestfit = np.max(temp_fit)
            best_solution = temp_pop[np.argmax(temp_fit)].copy()

        iters += 1
        if iters % 100 == 0:
            print(f"Iter: {iters}, Best Accuracy: {bestfit:.6f}")

    return best_solution


def NBC(matdis, fit, pop, k):
    n = len(matdis)
    leader_node = [[] for _ in range(n)]
    cluster_idx = np.zeros(n, dtype=int)
    nbc = np.zeros((n, 3))
    nbc[:, 0] = np.arange(n)
    nbc[0, 1] = -1
    nbc[0, 2] = -1

    for i in range(1, n):
        u = np.min(matdis[i, :i])
        v = np.argmin(matdis[i, :i])
        nbc[i, 1] = v
        nbc[i, 2] = u

    sidx = np.argsort(-nbc[:, 2])
    divid = sidx[:k - 1]
    nbc[divid, 1] = -1
    nbc[divid, 2] = -1

    seeds = nbc[nbc[:, 1] == -1, 0]
    m = np.zeros((n, 2))
    m[:, 0] = np.arange(n)

    for i in range(n):
        j = int(nbc[i, 1])
        k = j
        while j != -1:
            if j != -1:
                leader_node[i].append(j)
            k = j
            j = int(nbc[j, 1])
        if k == -1:
            m[i, 1] = i
        else:
            m[i, 1] = k

    species = {}
    leaders = np.zeros(n, dtype=int)

    for i in range(len(seeds)):
        seed = int(seeds[i])
        species[i] = {
            'seed_idx': seed,
            'idx': np.where(m[:, 1] == seed)[0],
            'len': np.sum(m[:, 1] == seed),
            'seed': pop[seed, :],
            'seed_fit': fit[seed]
        }
        cluster_idx[species[i]['idx']] = i

    for i in range(n):
        temp = leader_node[i]
        if len(temp) > 0:
            leaders[i] = temp[np.random.randint(len(temp))]
        else:
            leaders[i] = i

    return species, leaders, cluster_idx


# Main Pipeline
def main():
    # Data Loading and Preprocessing
    data_dir = '/kaggle/input/javed-ultrasound-fetal/6_class_image'  # Update to your local dataset path
    batch_size = 16  # Adjusted for typical laptop GPUs
    img_size = (299, 299)

    # Data augmentation (applied only to training)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset='training',
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset='validation',
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    # Get class names
    class_names = train_ds.class_names

    # Check if validation dataset is empty
    val_size = sum(1 for _ in val_ds)
    if val_size == 0:
        raise ValueError("Validation dataset is empty. Check dataset path or increase dataset size.")

    # Apply preprocessing (denoising, contrast enhancement, smoothing) to all datasets
    train_ds = train_ds.map(tf_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(tf_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Collect all validation images and labels (after preprocessing)
    val_images, val_labels = [], []
    for images, labels in val_ds:
        val_images.append(images.numpy())
        val_labels.append(labels.numpy())

    val_images = np.concatenate(val_images)
    val_labels = np.concatenate(val_labels)

    # Split validation data into validation and test sets (50% each)
    num_samples = len(val_images)
    split_idx = num_samples // 2
    indices = np.random.permutation(num_samples)

    val_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    val_images_split = val_images[val_indices]
    val_labels_split = val_labels[val_indices]
    test_images = val_images[test_indices]
    test_labels = val_labels[test_indices]

    # Apply augmentation to training set
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Optimize data pipeline
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    # Feature Extraction
    feature_extractor = build_feature_extractor()
    train_features, train_labels = [], []
    for images, labels in train_ds:
        features = feature_extractor.predict(images)
        train_features.append(features)
        train_labels.append(labels.numpy())

    train_features = np.concatenate(train_features)
    train_labels = np.concatenate(train_labels)

    val_features = feature_extractor.predict(val_images_split)
    test_features = feature_extractor.predict(test_images)

    # Feature Selection with BSO2O
    D = train_features.shape[1]
    selected_features = binary_BSO2O(train_features, train_labels, val_features, val_labels_split, D)

    # Apply feature selection
    train_features_selected = train_features[:, selected_features == 1]
    val_features_selected = val_features[:, selected_features == 1]
    test_features_selected = test_features[:, selected_features == 1]

    # Scale features
    scaler = StandardScaler()
    train_features_selected = scaler.fit_transform(train_features_selected)
    val_features_selected = scaler.transform(val_features_selected)
    test_features_selected = scaler.transform(test_features_selected)

    # Train SVM with Grid Search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    }
    svm = SVC(kernel='rbf', probability=True)  # Enable probability for ROC and occlusion
    global grid_search  # Make grid_search global for occlusion_sensitivity
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(train_features_selected, train_labels)

    print(f"Best SVM Parameters: {grid_search.best_params_}")
    print(f"Best Validation Accuracy: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    test_predictions = grid_search.predict(test_features_selected)
    test_accuracy = grid_search.score(test_features_selected, test_labels)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Visualize Results
    visualize_results(
        test_features_selected, test_labels, test_predictions, test_images,
        feature_extractor, scaler, selected_features, class_names, grid_search
    )


if __name__ == '__main__':
    main()