
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


class FuzzyCMeans:

    def __init__(self, n_clusters=2, max_iter=100, m=2.0, error=1e-5, random_state=42):
        self.n_clusters = n_clusters  # K
        self.max_iter = max_iter
        self.m = m  # Paramètre de flou
        self.error = error
        self.random_state = random_state
        self.centers = None
        self.membership = None  
        
    def _initialize_membership(self, n_samples):
        np.random.seed(self.random_state)
        membership = np.random.rand(n_samples, self.n_clusters)
        membership = membership / np.sum(membership, axis=1, keepdims=True)
        return membership
    
    def _calculate_centers(self, X):
        centers = np.zeros((self.n_clusters, X.shape[1]))
        
        for i in range(self.n_clusters):
            u_m = self.membership[:, i] ** self.m
            centers[i] = np.sum(u_m.reshape(-1, 1) * X, axis=0) / np.sum(u_m)
            
        return centers
    
    def _calculate_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - self.centers[i], axis=1)
            
        return distances
    
    def _update_membership(self, distances):
        power = 2.0 / (self.m - 1)
        distances = np.where(distances == 0, 1e-10, distances)
        membership = np.zeros_like(self.membership)
        
        for i in range(self.n_clusters):
            ratio = distances[:, i].reshape(-1, 1) / distances
            membership[:, i] = 1.0 / np.sum(ratio ** power, axis=1)
        
        return membership
    
    def fit(self, X):
        n_samples = X.shape[0]
        
        # Étape 1: Initialisation
        self.membership = self._initialize_membership(n_samples)
        
        print(f"Démarrage de l'algorithme Fuzzy C-Means (K={self.n_clusters}, m={self.m})")
        
        for iteration in range(self.max_iter):
            old_centers = self.centers.copy() if self.centers is not None else None
            self.centers = self._calculate_centers(X)
            distances = self._calculate_distances(X)
            self.membership = self._update_membership(distances)
            if old_centers is not None:
                center_shift = np.linalg.norm(self.centers - old_centers)
                if center_shift < self.error:
                    print(f"✓ Convergence atteinte à l'itération {iteration + 1}")
                    print(f"  Déplacement des centres: {center_shift:.6f}")
                    break
            
            if (iteration + 1) % 10 == 0:
                print(f"  Itération {iteration + 1}/{self.max_iter}")
        else:
            print(f"⚠ Nombre maximum d'itérations atteint ({self.max_iter})")
        
        return self
    
    def predict(self):
        return np.argmax(self.membership, axis=1)
    
    def get_membership_degrees(self):
        return self.membership


def fuzzy_cmeans_grayscale(image_path, n_clusters=2, m=2.0, max_iter=100):
    print("\n" + "="*70)
    print("SEGMENTATION D'IMAGE EN NIVEAUX DE GRIS - FUZZY C-MEANS")
    print("="*70)
    
    # 1. Charger l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    
    print(f"\n📷 Image chargée: {img.shape[0]}x{img.shape[1]} pixels")
    
    h, w = img.shape
    X = img.reshape(-1, 1).astype(np.float64)
    print(f"   Données préparées: {X.shape[0]} points, {X.shape[1]} dimension(s)")
    
    print(f"\n🔄 Application de Fuzzy C-Means...")
    fcm = FuzzyCMeans(n_clusters=n_clusters, max_iter=max_iter, m=m)
    fcm.fit(X)
    labels = fcm.predict()
    membership = fcm.get_membership_degrees()
    
    segmented_img = labels.reshape(h, w)
    clustered_img = np.zeros_like(img, dtype=np.float64)
    for i in range(n_clusters):
        clustered_img[segmented_img == i] = fcm.centers[i, 0]
    
    # Affichage des statistiques détaillées
    print(f"\n" + "="*70)
    print(f"📊 STATISTIQUES DES CLUSTERS (K={n_clusters})")
    print("="*70)
    
    total_pixels = len(labels)
    print(f"\n📷 Image: {h}×{w} pixels = {total_pixels:,} pixels au total\n")
    
    for i in range(n_clusters):
        count = np.sum(labels == i)
        percentage = (count / total_pixels) * 100
        intensity = fcm.centers[i, 0]
        
        print(f"🔹 Cluster {i}:")
        print(f"   • Centre (intensité): {intensity:.2f}")
        print(f"   • Nombre de pixels:   {count:,} pixels")
        print(f"   • Pourcentage:        {percentage:.2f}%")
        print(f"   • Type:               {'Zones sombres' if intensity < 85 else 'Zones moyennes' if intensity < 170 else 'Zones claires'}")
        print()
    
    print("="*70)
    
    # 7. Visualisation
    n_cols = 3 + n_clusters
    fig = plt.figure(figsize=(5*n_cols, 5))
    
    # Image originale
    plt.subplot(1, n_cols, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image Originale', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Image segmentée (labels)
    plt.subplot(1, n_cols, 2)
    plt.imshow(segmented_img, cmap='viridis')
    plt.title(f'Segmentation\n(K={n_clusters} clusters)', fontsize=12, fontweight='bold')
    plt.colorbar(shrink=0.8)
    plt.axis('off')
    
    # Image clusterisée (avec centres)
    plt.subplot(1, n_cols, 3)
    plt.imshow(clustered_img, cmap='gray')
    plt.title('Image Clusterisée\n(valeurs des centres)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Heatmaps des degrés d'appartenance pour chaque cluster
    for i in range(n_clusters):
        plt.subplot(1, n_cols, 4 + i)
        membership_map = membership[:, i].reshape(h, w)
        im = plt.imshow(membership_map, cmap='hot', vmin=0, vmax=1)
        plt.title(f'Heatmap Cluster {i}\n(degré d\'appartenance)', 
                 fontsize=12, fontweight='bold')
        plt.colorbar(im, shrink=0.8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"fuzzy_cmeans_grayscale_{n_clusters}_clusters_results.png", dpi=150, bbox_inches='tight')
    print(f"\n💾 Résultats sauvegardés: fuzzy_cmeans_grayscale_results.png")
    plt.show()
    
    return fcm, segmented_img, clustered_img, membership


def fuzzy_cmeans_color(image_path, n_clusters=3, m=2.0, max_iter=100):
    print("\n" + "="*70)
    print("SEGMENTATION D'IMAGE EN COULEUR (RGB) - FUZZY C-MEANS")
    print("="*70)
    
    # 1. Charger l'image en couleur
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    
    # Convertir BGR (OpenCV) en RGB (matplotlib)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"\n📷 Image chargée: {img_rgb.shape[0]}x{img_rgb.shape[1]} pixels, {img_rgb.shape[2]} canaux")
    
    # 2. Préparer les données (reshape en matrice [N x 3] pour RGB)
    h, w, c = img_rgb.shape
    X = img_rgb.reshape(-1, 3).astype(np.float64)
    print(f"   Données préparées: {X.shape[0]} points, {X.shape[1]} dimensions (RGB)")
    
    # 3. Appliquer Fuzzy C-Means
    print(f"\n🔄 Application de Fuzzy C-Means...")
    fcm = FuzzyCMeans(n_clusters=n_clusters, max_iter=max_iter, m=m)
    fcm.fit(X)
    
    # 4. Obtenir les résultats
    labels = fcm.predict()
    membership = fcm.get_membership_degrees()
    
    # 5. Reconstruire les images
    segmented_img = labels.reshape(h, w)
    
    # Image segmentée avec les couleurs des centres
    clustered_img = np.zeros_like(img_rgb, dtype=np.float64)
    for i in range(n_clusters):
        clustered_img[segmented_img == i] = fcm.centers[i]
    
    clustered_img = np.clip(clustered_img, 0, 255).astype(np.uint8)
    
    # 6. Affichage des statistiques détaillées
    print(f"\n" + "="*70)
    print(f"📊 STATISTIQUES DES CLUSTERS RGB (K={n_clusters})")
    print("="*70)
    
    total_pixels = len(labels)
    print(f"\n📷 Image: {h}×{w} pixels = {total_pixels:,} pixels au total\n")
    
    for i in range(n_clusters):
        count = np.sum(labels == i)
        percentage = (count / total_pixels) * 100
        r, g, b = fcm.centers[i]
        
        # Déterminer la couleur dominante
        color_name = ""
        if r > g and r > b:
            color_name = "Rouge dominant"
        elif g > r and g > b:
            color_name = "Vert dominant"
        elif b > r and b > g:
            color_name = "Bleu dominant"
        elif r > 200 and g > 200 and b > 200:
            color_name = "Blanc/Très clair"
        elif r < 50 and g < 50 and b < 50:
            color_name = "Noir/Très sombre"
        else:
            color_name = "Couleur mixte"
        
        print(f"🔹 Cluster {i}:")
        print(f"   • Centre RGB:         ({r:.0f}, {g:.0f}, {b:.0f})")
        print(f"   • Couleur:            {color_name}")
        print(f"   • Nombre de pixels:   {count:,} pixels")
        print(f"   • Pourcentage:        {percentage:.2f}%")
        print()
    
    print("="*70)
    
    # 7. Visualisation
    n_cols = 3 + n_clusters
    fig = plt.figure(figsize=(5*n_cols, 5))
    
    # Image originale
    plt.subplot(1, n_cols, 1)
    plt.imshow(img_rgb)
    plt.title('Image Originale', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Image segmentée (labels)
    plt.subplot(1, n_cols, 2)
    plt.imshow(segmented_img, cmap='viridis')
    plt.title(f'Segmentation\n(K={n_clusters} clusters)', fontsize=12, fontweight='bold')
    plt.colorbar(shrink=0.8)
    plt.axis('off')
    
    # Image clusterisée (avec couleurs des centres)
    plt.subplot(1, n_cols, 3)
    plt.imshow(clustered_img)
    plt.title('Image Clusterisée\n(couleurs des centres)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Heatmaps des degrés d'appartenance pour chaque cluster
    for i in range(n_clusters):
        plt.subplot(1, n_cols, 4 + i)
        membership_map = membership[:, i].reshape(h, w)
        im = plt.imshow(membership_map, cmap='hot', vmin=0, vmax=1)
        plt.title(f'Heatmap Cluster {i}\n(degré d\'appartenance)', 
                 fontsize=12, fontweight='bold')
        plt.colorbar(im, shrink=0.8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"fuzzy_cmeans_color_{n_clusters}_clusters_results.png", dpi=150, bbox_inches='tight')
    print(f"\n💾 Résultats sauvegardés: fuzzy_cmeans_color_results.png")
    plt.show()
    
    return fcm, segmented_img, clustered_img, membership


def compare_different_k(image_path, k_values=[2, 3, 4, 5], is_grayscale=True):
    """
    Comparer les résultats pour différentes valeurs de K
    """
    print("\n" + "="*70)
    print(f"COMPARAISON POUR DIFFÉRENTES VALEURS DE K: {k_values}")
    print("="*70)
    
    # Charger l'image
    if is_grayscale:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        cmap_original = 'gray'
    else:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cmap_original = None
    
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    
    # Préparer les données
    if is_grayscale:
        h, w = img.shape
        X = img.reshape(-1, 1).astype(np.float64)
    else:
        h, w, c = img.shape
        X = img.reshape(-1, 3).astype(np.float64)
    
    # Visualisation
    fig = plt.figure(figsize=(5*len(k_values), 10))
    
    for idx, k in enumerate(k_values):
        print(f"\n" + "─"*70)
        print(f"🔄 Test avec K={k}")
        print("─"*70)
        
        # Appliquer FCM
        fcm = FuzzyCMeans(n_clusters=k, max_iter=100, m=2.0)
        fcm.fit(X)
        labels = fcm.predict()
        
        # Statistiques pour ce K
        total_pixels = len(labels)
        print(f"\n📊 Statistiques pour K={k}:")
        for i in range(k):
            count = np.sum(labels == i)
            percentage = (count / total_pixels) * 100
            if is_grayscale:
                intensity = fcm.centers[i, 0]
                print(f"   Cluster {i}: intensité={intensity:.2f}, pixels={count:,} ({percentage:.1f}%)")
            else:
                r, g, b = fcm.centers[i]
                print(f"   Cluster {i}: RGB=({r:.0f},{g:.0f},{b:.0f}), pixels={count:,} ({percentage:.1f}%)")
        
        # Reconstruire l'image segmentée
        segmented_img = labels.reshape(h, w)
        
        # Affichage
        plt.subplot(2, len(k_values), idx + 1)
        plt.imshow(segmented_img, cmap='viridis')
        plt.title(f'K = {k}', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Image clusterisée
        if is_grayscale:
            clustered_img = np.zeros_like(img, dtype=np.float64)
            for i in range(k):
                clustered_img[segmented_img == i] = fcm.centers[i, 0]
            plt.subplot(2, len(k_values), len(k_values) + idx + 1)
            plt.imshow(clustered_img, cmap='gray')
        else:
            clustered_img = np.zeros_like(img, dtype=np.float64)
            for i in range(k):
                clustered_img[segmented_img == i] = fcm.centers[i]
            clustered_img = np.clip(clustered_img, 0, 255).astype(np.uint8)
            plt.subplot(2, len(k_values), len(k_values) + idx + 1)
            plt.imshow(clustered_img)
        
        plt.title(f'Clusterisée K={k}', fontsize=14, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    mode = "grayscale" if is_grayscale else "color"
    plt.savefig(f'fuzzy_cmeans_comparison_{mode}.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Comparaison sauvegardée: fuzzy_cmeans_comparison_{mode}.png")
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  TP - IMPLÉMENTATION FUZZY C-MEANS FROM SCRATCH")
    print("="*70)
    
    # Chemins des images
    img_gray = "milky-way-nvg.jpg"
    img_color = "milky-way.jpg"
    
    print("\n" + "🔹"*35)
    print("TEST 1: Image en niveaux de gris (K=2)")
    print("🔹"*35)
    
    try:
        fcm_gray, segmented_gray, clustered_gray, membership_gray = \
            fuzzy_cmeans_grayscale(img_gray, n_clusters=2, m=2.0, max_iter=100)
        print("\n✅ Test 1 réussi!")
    except Exception as e:
        print(f"\n❌ Erreur Test 1: {e}")
        
    print("\n" + "🔹"*35)
    print("TEST 2: Image en niveaux de gris (K=3)")
    print("🔹"*35)
    
    try:
        fcm_gray3, segmented_gray3, clustered_gray3, membership_gray3 = \
            fuzzy_cmeans_grayscale(img_gray, n_clusters=3, m=2.0, max_iter=100)
        print("\n✅ Test 2 réussi!")
    except Exception as e:
        print(f"\n❌ Erreur Test 2: {e}")
    
    print("\n" + "🔹"*35)
    print("TEST 3: Image en couleur RGB (K=3)")
    print("🔹"*35)
    
    try:
        fcm_color, segmented_color, clustered_color, membership_color = \
            fuzzy_cmeans_color(img_color, n_clusters=3, m=2.0, max_iter=100)
        print("\n✅ Test 3 réussi!")
    except Exception as e:
        print(f"\n❌ Erreur Test 3: {e}")
    
    print("\n" + "🔹"*35)
    print("TEST 4: Comparaison K=[2,3,4,5] - Niveaux de gris")
    print("🔹"*35)
    
    try:
        compare_different_k(img_gray, k_values=[2, 3, 4, 5], is_grayscale=True)
        print("\n✅ Test 4 réussi!")
    except Exception as e:
        print(f"\n❌ Erreur Test 4: {e}")
    
    print("\n" + "🔹"*35)
    print("TEST 5: Comparaison K=[2,3,4] - Couleur")
    print("🔹"*35)
    
    try:
        compare_different_k(img_color, k_values=[2, 3, 4], is_grayscale=False)
        print("\n✅ Test 5 réussi!")
    except Exception as e:
        print(f"\n❌ Erreur Test 5: {e}")
    
    print("\n" + "="*70)
    print("  ✅ TOUS LES TESTS TERMINÉS!")
