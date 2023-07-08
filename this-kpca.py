import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from math import log10, sqrt
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as ticker
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt


# Display the original and compressed images side by side
def simple_plot(X_original,X_pca, X_kpca, image_shape):
    plt.subplot(131)
    plt.imshow(X_original.reshape(image_shape))
    plt.title('Original')

    plt.subplot(132)
    plt.imshow(X_pca.reshape(image_shape))
    plt.title('PCA')
    
    plt.subplot(133)
    plt.imshow(X_kpca.reshape(image_shape))
    plt.title('KPCA')

    plt.show()

def simple_measure(X_original, X_pca, X_kpca, image_shape):
    
    X_original = X_original.reshape(image_shape)
    X_pca = X_pca.reshape(image_shape)
    X_kpca = X_kpca.reshape(image_shape)
    
    pca_mse, pca_psnr = image_metrics(X_original, X_pca)
    kpca_mse, kpca_psnr = image_metrics(X_original, X_kpca)

    print("pca_mse ", pca_mse)
    print("pca_psnr",pca_psnr)
    print("kpca_mse ",kpca_mse)
    print("kpca_psnr",kpca_psnr) 

    plt.bar(0, pca_mse, width=0.25)
    plt.bar(0.25, kpca_mse, width=0.25)
    plt.show()
    
# Plot original, PCA reconstructed, and KPCA reconstructed images
def plot_images(X_original, X_pca, X_kpca, image_shape):
    fig, axes = plt.subplots(3, 10, figsize=(10, 4),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))

    for pic_ind in range(0, 10):
        i = pic_ind - 0
        axes[0, i].imshow(X_original[pic_ind].reshape(image_shape), cmap='gray')
        axes[1, i].imshow(X_pca[pic_ind].reshape(image_shape), cmap='gray')
        axes[2, i].imshow(X_kpca[pic_ind].reshape(image_shape), cmap='gray')

    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('PCA')
    axes[2, 0].set_ylabel('KPCA')

    plt.show()

# def measure_images(X_original, X_pca, X_kpca, image_shape):
#     pca_mse, pca_psnr, pca_ssim = [], [], []
#     kpca_mse, kpca_psnr, kpca_ssim = [], [], []
#     for pic_ind in range(0, 10):
#         ori_img = X_original[pic_ind].reshape(image_shape)
#         pca_img = X_pca[pic_ind].reshape(image_shape)
#         kpca_img = X_kpca[pic_ind].reshape(image_shape)
        
#         t_pca_metrics = image_metrics(ori_img, pca_img)
#         t_kpca_metrics = image_metrics(ori_img, kpca_img)
#         pca_mse.append(t_pca_metrics[0])
#         pca_psnr.append(t_pca_metrics[1])
#         pca_ssim.append(t_pca_metrics[2])

#         kpca_mse.append(t_kpca_metrics[0])
#         kpca_psnr.append(t_kpca_metrics[1])
#         kpca_ssim.append(t_kpca_metrics[2])

#     x = np.arange(len(pca_mse))
#     width = 0.25

#     fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx()

#     ax1.bar(x - width, pca_mse, width, label='PCA MSE')
#     ax1.bar(x, kpca_mse, width, label='KPCA MSE')
#     ax1.bar(x + width, pca_psnr, width, label='PCA PSNR')
#     ax1.bar(x + 2 * width, kpca_psnr, width, label='KPCA PSNR')
#     ax1.set_ylabel('MSE/PSNR')
#     ax1.set_xticks(x)
#     ax1.legend(loc='upper left')

#     ax2.plot(x, pca_ssim, label='PCA SSIM', color='r', marker='o')
#     ax2.plot(x, kpca_ssim, label='KPCA SSIM', color='g', marker='o')
#     ax2.set_ylabel('SSIM')
#     ax2.yaxis.set_major_locator(ticker.LinearLocator(numticks=6))
#     ax2.legend(loc='upper right')

#     for i in range(len(pca_mse)):
#         ax1.text(x[i] - width, pca_mse[i], f"{pca_mse[i]:.2f}", ha='center', va='bottom')
#         ax1.text(x[i], kpca_mse[i], f"{kpca_mse[i]:.2f}", ha='center', va='bottom')
#         ax1.text(x[i] + width, pca_psnr[i], f"{pca_psnr[i]:.2f}", ha='center', va='bottom')
#         ax1.text(x[i] + 2 * width, kpca_psnr[i], f"{kpca_psnr[i]:.2f}", ha='center', va='bottom')
#         ax2.text(x[i], pca_ssim[i], f"{pca_ssim[i]:.2f}", ha='center', va='bottom', color='r')
#         ax2.text(x[i], kpca_ssim[i], f"{kpca_ssim[i]:.2f}", ha='center', va='bottom', color='g')

#     plt.tight_layout()
#     plt.show()

def measure_images(X_original, X_pca, X_kpca, image_shape):
    pca_mse, pca_psnr, pca_ssim = [], [], []
    kpca_mse, kpca_psnr, kpca_ssim = [], [], []
    for pic_ind in range(0, 10):
        ori_img = X_original[pic_ind].reshape(image_shape)
        pca_img = X_pca[pic_ind].reshape(image_shape)
        kpca_img = X_kpca[pic_ind].reshape(image_shape)
        
        t_pca_metrics = image_metrics(ori_img, pca_img)
        t_kpca_metrics = image_metrics(ori_img, kpca_img)
        pca_mse.append(t_pca_metrics[0])
        pca_psnr.append(t_pca_metrics[1])
        pca_ssim.append(t_pca_metrics[2])

        kpca_mse.append(t_kpca_metrics[0])
        kpca_psnr.append(t_kpca_metrics[1])
        kpca_ssim.append(t_kpca_metrics[2])

    x = np.arange(len(pca_mse))
    width = 0.35

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].bar(x - width / 2, pca_mse, width, label='PCA MSE', color='tab:blue')
    axs[0].bar(x + width / 2, kpca_mse, width, label='KPCA MSE', color='tab:orange')
    axs[0].set_ylabel('MSE')
    axs[0].set_xticks(x)
    axs[0].legend(loc='upper left')

    axs[1].bar(x - width / 2, pca_psnr, width, label='PCA PSNR', color='tab:cyan')
    axs[1].bar(x + width / 2, kpca_psnr, width, label='KPCA PSNR', color='tab:olive')
    axs[1].set_ylabel('PSNR')
    axs[1].set_xticks(x)
    axs[1].legend(loc='upper left')

    axs[2].bar(x - width / 2, pca_ssim, width, label='PCA SSIM', color='tab:green')
    axs[2].bar(x + width / 2, kpca_ssim, width, label='KPCA SSIM', color='tab:red')
    axs[2].set_ylabel('SSIM')
    axs[2].set_xticks(x)
    axs[2].set_ylim(0.9, 1.005)
    axs[2].legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    

def image_metrics(X1, X2):
    mse = mean_squared_error(X1, X2)
    psnr = 20 * log10(255.0 / sqrt(mse))
    data_range = X1.max() - X1.min()
    ssim_val = ssim(X1, X2, data_range=data_range, multichannel=True)
    return mse, psnr, ssim_val

def perform_pca(X_std, n_components=2):
    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    X_pca_reconstructed = pca.inverse_transform(X_pca)
    return X_pca_reconstructed
    
def perform_kpca(X_std, n_components=2):
    # KPCA
    # linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'
    kpca = KernelPCA(n_components=n_components, kernel='poly', gamma=100, fit_inverse_transform=True)
    X_kpca = kpca.fit_transform(X_std)
    X_kpca_reconstructed = kpca.inverse_transform(X_kpca)
    return X_kpca_reconstructed

def perform_isomap(X_std, n_components=64):
    # 使用 Isomap 进行降维
    isomap = Isomap(n_components=n_components)
    X_transformed = isomap.fit_transform(X_std)
    # 训练一个线性回归模型，从低维表示恢复原始空间
    linear_regression = LinearRegression()
    linear_regression.fit(X_transformed, X_std)
    # 使用训练好的线性回归模型进行重构
    X_reconstructed = linear_regression.predict(X_transformed)

    return X_reconstructed

def perform_lle(X_std, n_components=64):
    # 使用 LLE 进行降维
    lle = LocallyLinearEmbedding(n_components=n_components)
    X_transformed = lle.fit_transform(X_std)
    # 训练一个线性回归模型，从低维表示恢复原始空间
    linear_regression = LinearRegression()
    linear_regression.fit(X_transformed, X_std)
    # 使用训练好的线性回归模型进行重构
    X_reconstructed = linear_regression.predict(X_transformed)

    return X_reconstructed


def fetch_faces():
    # Load 'olivetti_faces' dataset
    faces = fetch_openml("olivetti_faces")
    X = np.array(faces.data)  # Convert X to a NumPy array
    y = faces.target

    X = X[95:105]
    y = y[:10]
    return X

def fetch_101():
    categories = os.listdir("101_ObjectCategories")
    while True:
        category = random.choice(categories)
        image_path = os.path.join("101_ObjectCategories", category, random.choice(os.listdir(os.path.join("101_ObjectCategories", category))))
        # Load and display the image
        image = Image.open(image_path)
        X = np.asarray(image)
        if len(X.shape) == 2:
            ori_shape = X.shape
            break
    print(X.shape)
    X = X.reshape(X.shape[0], -1)
    # print(X.shape)
    return X, ori_shape

X = fetch_faces()
# X, image_shape = fetch_101()

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

flag = ['normal', 'faces'][1]
plot_mod = ['single', 'multiple'][1]

if flag == 'normal':
    k = 10
    # X_pca_reconstructed = compress_image_isomap(X_std, k)
    # X_kpca_reconstructed = compress_image_lle(X_std, k)
    X_pca_reconstructed = perform_pca(X_std, k)
    X_kpca_reconstructed = perform_kpca(X_std, k)
elif flag == 'faces':
    k = 7
    image_shape = (64, 64)
    X_pca_reconstructed = np.empty(((0,) + X_std.shape[1:]))
    for X_per_img in X_std:
        img_result = perform_pca(X_per_img.reshape(image_shape), k)
        img_result = img_result.flatten()
        img_result = img_result.reshape((1,) + img_result.shape)
        X_pca_reconstructed = np.concatenate((X_pca_reconstructed, img_result), axis=0)
        
    X_kpca_reconstructed = np.empty(((0,) + X_std.shape[1:]))
    for X_per_img in X_std:
        img_result = perform_kpca(X_per_img.reshape(image_shape), k)
        img_result = img_result.flatten()
        img_result = img_result.reshape((1,) + img_result.shape)
        X_kpca_reconstructed = np.concatenate((X_kpca_reconstructed, img_result), axis=0)

# Inverse standardization for reconstruction
X_std_reconstructed = scaler.inverse_transform(X_pca_reconstructed)
X_kpca_std_reconstructed = scaler.inverse_transform(X_kpca_reconstructed)

if plot_mod == 'single':
    simple_plot(X, X_std_reconstructed, X_kpca_std_reconstructed, image_shape)
    simple_measure(X, X_std_reconstructed, X_kpca_std_reconstructed, image_shape)
else:
    plot_images(X, X_std_reconstructed, X_kpca_std_reconstructed, image_shape)
    measure_images(X, X_std_reconstructed, X_kpca_std_reconstructed, image_shape)
