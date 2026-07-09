import torch
from net.statistical_memory import DWTExtractor, GMMStatisticalMemory

def test_smh():
    print("Testing DWTExtractor...")
    extractor = DWTExtractor()
    dummy_img = torch.randn(2, 3, 256, 256)
    feat = extractor(dummy_img)
    print(f"Extractor output shape: {feat.shape} (Expected: [2, 256] or [2, 512])")
    
    feature_dim = feat.shape[1]
    
    print("\nTesting GMMStatisticalMemory...")
    gmm = GMMStatisticalMemory(feature_dim=feature_dim, num_tasks=2, n_components=3)
    
    print("Fitting Task 0...")
    task0_features = torch.randn(10, feature_dim) + 5.0
    gmm.fit(0, task0_features)
    
    print("Fitting Task 1...")
    task1_features = torch.randn(10, feature_dim) - 5.0
    gmm.fit(1, task1_features)
    
    print("Predicting probabilities for Task 0 features...")
    probs0 = gmm.predict_task_probs(task0_features[:2])
    print(f"Probs for Task 0 features: \n{probs0}")
    
    print("Predicting probabilities for Task 1 features...")
    probs1 = gmm.predict_task_probs(task1_features[:2])
    print(f"Probs for Task 1 features: \n{probs1}")

if __name__ == '__main__':
    test_smh()
