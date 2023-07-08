import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.linear_model import LinearRegression
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from math import log10, sqrt
from sklearn.metrics import pairwise_distances
['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']

def fetch_faces():
    # Load 'olivetti_faces' dataset
    faces = fetch_openml("olivetti_faces")
    X = np.array(faces.data)  # Convert X to a NumPy array
    y = faces.target

    X = X[20:30]
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


class DimensionalityReduction:
    def __init__(self, X, image_shape, method='pca', n_components=2, kernel='poly', n_neighbors=5, distance_metric='minkowski'):
        self.X = X
        self.image_shape = image_shape
        self.method = method
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.X_std = self.scaler.fit_transform(self.X)
        self.kpca_kernel = kernel
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric

        if self.method == 'pca':
            self.X_reconstructed = self.perform_pca()
        elif self.method == 'kpca':
            self.X_reconstructed = self.perform_kpca()
        elif self.method == 'isomap':
            self.X_reconstructed = self.perform_isomap()
        elif self.method == 'lle':
            self.X_reconstructed = self.perform_lle()
        else:
            raise ValueError(f"Invalid method: {self.method}")

        self.X_std_reconstructed = self.scaler.inverse_transform(self.X_reconstructed)

    def perform_pca(self):
        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(self.X_std)
        X_pca_reconstructed = pca.inverse_transform(X_pca)
        return X_pca_reconstructed

    def perform_kpca(self):
        kpca = KernelPCA(n_components=self.n_components, kernel=self.kpca_kernel, gamma=100, fit_inverse_transform=True)
        X_kpca = kpca.fit_transform(self.X_std)
        X_kpca_reconstructed = kpca.inverse_transform(X_kpca)
        return X_kpca_reconstructed

    def perform_isomap(self):
        isomap = Isomap(n_components=self.n_components, n_neighbors=self.n_neighbors, metric=self.distance_metric)
        X_transformed = isomap.fit_transform(self.X_std)
        linear_regression = LinearRegression()
        linear_regression.fit(X_transformed, self.X_std)
        X_reconstructed = linear_regression.predict(X_transformed)
        return X_reconstructed

    def perform_lle(self):
        lle = LocallyLinearEmbedding(n_components=self.n_components)
        X_transformed = lle.fit_transform(self.X_std)
        linear_regression = LinearRegression()
        linear_regression.fit(X_transformed, self.X_std)
        X_reconstructed = linear_regression.predict(X_transformed)
        return X_reconstructed

    def plot_images(self, ax):
        ax.imshow(self.X_std_reconstructed.reshape(self.image_shape))
        if self.method == 'kpca':
            ax.set_title((self.method + '_' + self.kpca_kernel).upper())
        else:
            ax.set_title(self.method.upper())
        
    # ! add this after convert this file to R code
    def plot_all_kpca_kernels(self):
        kernels = ['linear', 'poly', 'rbf', 'cosine']
        n_kernels = len(kernels)

        fig, axes = plt.subplots(1, n_kernels + 1, figsize=(n_kernels * 3, 3))

        # Plot original image
        axes[0].imshow(self.X.reshape(self.image_shape))
        axes[0].set_title('Original')

        # Plot images for all kpca methods
        for i, kernel in enumerate(kernels):
            dr = DimensionalityReduction(self.X, self.image_shape, method='kpca', kernel=kernel, n_components=self.n_components)
            dr.plot_images(axes[i + 1])

        plt.tight_layout()
        plt.show()
        
    # ! deprecated
    def plot_all_isomap(self):
        n_neighbors = np.array([2, 3, 4, 5, 6])*3
        len_n_neighbors = len(n_neighbors)

        fig, axes = plt.subplots(1, len_n_neighbors + 1, figsize=(len_n_neighbors * 3, 3))

        # Plot original image
        axes[0].imshow(self.X.reshape(self.image_shape))
        axes[0].set_title('Original')

        # Plot images for all kpca methods
        for i, neighbors in enumerate(n_neighbors):
            dr = DimensionalityReduction(self.X, self.image_shape, method='isomap', n_neighbors=neighbors, n_components=self.n_components)
            dr.plot_images(axes[i + 1])

        plt.tight_layout()
        plt.show()
        
    # ! add this after convert this file to R code
    def plot_kpca_n_components(self):
        n_components = np.array([2, 3, 4, 5, 6])*5
        len_n_components = len(n_components)

        fig, axes = plt.subplots(1, len_n_components + 1, figsize=(len_n_components * 3, 3))

        # Plot original image
        axes[0].imshow(self.X.reshape(self.image_shape))
        axes[0].set_title('Original')

        # Plot images for all kpca methods
        for i, components in enumerate(n_components):
            dr = DimensionalityReduction(self.X, self.image_shape, method='kpca', n_components=components)
            dr.plot_images(axes[i + 1])

        plt.tight_layout()
        plt.show()


    def plot_all_methods(self):
        methods = ['pca', 'kpca', 'isomap', 'lle']
        n_methods = len(methods)

        fig, axes = plt.subplots(1, n_methods + 1, figsize=(n_methods * 3, 3))

        # Plot original image
        axes[0].imshow(self.X.reshape(self.image_shape))
        axes[0].set_title('Original')

        # Plot images for all methods
        for i, method in enumerate(methods):
            dr = DimensionalityReduction(self.X, self.image_shape, method=method, n_components=self.n_components)
            dr.plot_images(axes[i + 1])

        plt.tight_layout()
        plt.show()

    def image_metrics(self, X1, X2):
        mse = mean_squared_error(X1, X2)
        psnr = 20 * log10(255.0 / sqrt(mse))
        data_range = X1.max() - X1.min()
        ssim_val = ssim(X1, X2, data_range=data_range, multichannel=True)
        return mse, psnr, ssim_val

    # ! add this after convert this file to R code
    def evaluate_kpca_kernels(self):
        kernels = ['linear', 'poly', 'rbf', 'cosine']
        evaluations = []

        # * deprecated because original PCA is identical to KPCA linear
        # # original PCA (identical to KPCA linear)
        # dr = DimensionalityReduction(self.X, self.image_shape, method='pca', n_components=self.n_components)
        # mse, psnr, ssim_val = self.image_metrics(self.X, dr.X_std_reconstructed)
        # evaluations.append(('pca', mse, psnr, ssim_val))
        # max_diff = np.abs(self.X - dr.X_std_reconstructed).max()
        # print(f"{'pca'.upper()} - Max difference between original and reconstructed image: {max_diff}")
            
        for kernel in kernels:
            dr = DimensionalityReduction(self.X, self.image_shape, method='kpca', kernel=kernel, n_components=self.n_components)
            mse, psnr, ssim_val = self.image_metrics(self.X, dr.X_std_reconstructed)
            evaluations.append(('kpca_' + kernel, mse, psnr, ssim_val))
            max_diff = np.abs(self.X - dr.X_std_reconstructed).max()
            print(f"{('kpca_' + kernel).upper()} - Max difference between original and reconstructed image: {max_diff}")

        return evaluations
    
    # !deprecated
    def evaluate_isomaps(self):
        n_neighbors = np.array([2, 3, 4, 5, 6])*3
        evaluations = []
            
        for neighbors in n_neighbors:
            dr = DimensionalityReduction(self.X, self.image_shape, method='isomap', n_neighbors=neighbors, n_components=self.n_components)
            mse, psnr, ssim_val = self.image_metrics(self.X, dr.X_std_reconstructed)
            evaluations.append((neighbors, mse, psnr, ssim_val))
            max_diff = np.abs(self.X - dr.X_std_reconstructed).max()
            print(f"{neighbors} - Max difference between original and reconstructed image: {max_diff}")

        return evaluations
    
    # ! add this after convert this file to R code
    def evaluate_n_components(self):
        n_components = np.array([2, 3, 4, 5, 6])*8
        evaluations = []
            
        for components in n_components:
            dr = DimensionalityReduction(self.X, self.image_shape, method='kpca', n_components=components)
            mse, psnr, ssim_val = self.image_metrics(self.X, dr.X_std_reconstructed)
            evaluations.append((components, mse, psnr, ssim_val))
            max_diff = np.abs(self.X - dr.X_std_reconstructed).max()
            print(f"{components} - Max difference between original and reconstructed image: {max_diff}")
            
        return evaluations

    def evaluate_methods(self):
        methods = ['pca', 'kpca', 'isomap', 'lle']
        evaluations = []

        for method in methods:
            dr = DimensionalityReduction(self.X, self.image_shape, method=method, n_components=self.n_components)
            mse, psnr, ssim_val = self.image_metrics(self.X, dr.X_std_reconstructed)
            evaluations.append((method, mse, psnr, ssim_val))
            max_diff = np.abs(self.X - dr.X_std_reconstructed).max()
            print(f"{method.upper()} - Max difference between original and reconstructed image: {max_diff}")

        return evaluations

    def plot_evaluation_barchart(self, evaluations):

        methods = [eval[0] for eval in evaluations]
        mse_values = [eval[1] for eval in evaluations]
        psnr_values = [eval[2] for eval in evaluations]
        ssim_values = [eval[3] for eval in evaluations]

        x = np.arange(len(methods))
        width = 0.3

        fig, ax = plt.subplots()
        ax2 = ax.twinx()  # Create a second y-axis

        rects1 = ax.bar(x - width, mse_values, width, label='MSE')
        rects2 = ax.bar(x, psnr_values, width, label='PSNR')

        # Plot SSIM values on the second y-axis
        rects3 = ax2.bar(x + width, ssim_values, width, label='SSIM', color='g', alpha=0.5)
        ax2.set_ylabel('SSIM')

        ax.set_ylabel('MSE / PSNR')
        ax.set_title('Evaluation metrics for dimensionality reduction methods')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        def autolabel(rects, axis=ax):
            for rect in rects:
                height = rect.get_height()
                axis.annotate('{}'.format(round(height, 2)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3, axis=ax2)

        fig.tight_layout()
        plt.show()

# # Load dataset and perform dimensionality reduction
# X, image_shape = fetch_101()
# dr_pca = DimensionalityReduction(X, image_shape, method='pca', n_components=10)
# dr_kpca = DimensionalityReduction(X, image_shape, method='kpca', n_components=10)
# dr_isomap = DimensionalityReduction(X, image_shape, method='isomap', n_components=64)
# dr_lle = DimensionalityReduction(X, image_shape, method='lle', n_components=64)

# # Plot images
# dr_pca.plot_images()
# dr_kpca.plot_images()
# dr_isomap.plot_images()
# dr_lle.plot_images()

# Load dataset and perform dimensionality reduction
X, image_shape = fetch_101()
dr = DimensionalityReduction(X, image_shape, n_components=15)

# Plot images for all methods


# dr.plot_all_kpca_kernels()
# dr.plot_evaluation_barchart(dr.evaluate_kpca_kernels())

# dr.plot_all_methods()
# dr.plot_evaluation_barchart(dr.evaluate_methods())

# dr.plot_all_isomap()
# dr.plot_evaluation_barchart(dr.evaluate_isomaps())

dr.plot_kpca_n_components()
dr.plot_evaluation_barchart(dr.evaluate_n_components())