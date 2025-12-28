import os
import numpy as np
import faiss # 這是 Facebook 的搜尋神器

class ImageSearcher:
    def __init__(self, feature_path, path_file_path):
        """
        初始化搜尋引擎
        feature_path: .npy 特徵檔路徑
        path_file_path: .npy 圖片路徑檔
        """
        # 1. 載入資料
        if not os.path.exists(feature_path) or not os.path.exists(path_file_path):
            raise FileNotFoundError("找不到特徵檔案，請先執行 build_features.py")
            
        print("正在載入特徵資料庫...")
        self.features = np.load(feature_path).astype('float32') # FAISS 規定要 float32
        self.image_paths = np.load(path_file_path)
        
        # 取得向量維度 (應該是 2048)
        self.d = self.features.shape[1]
        
        # 2. 建立 FAISS 索引
        # 我們使用 "Inner Product" (IP)，因為向量已經正規化過，
        # 所以內積 (Inner Product) 等同於餘弦相似度 (Cosine Similarity)
        # 這是最適合比對圖像風格的方法
        print("正在建立 FAISS 索引...")
        self.index = faiss.IndexFlatIP(self.d)
        self.index.add(self.features) # 把 8 萬筆資料塞進去
        print(f"索引建立完成！目前資料庫共有 {self.index.ntotal} 張圖片。")

    def search(self, query_vector, k=5):
        """
        輸入: query_vector (1, 2048) 的特徵向量
        輸出: (distances, paths) 
        """
        # 確保格式正確
        query_vector = query_vector.astype('float32')
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # 執行搜尋
        # D: 距離 (相似度分數), I: 索引 (第幾張圖)
        D, I = self.index.search(query_vector, k)
        
        # 整理結果
        results = []
        # I[0] 是因為我們只查 1 張圖 (batch=1)
        for score, idx in zip(D[0], I[0]):
            img_path = self.image_paths[idx]
            results.append((img_path, score))
            
        return results

# --- 測試區 ---
if __name__ == "__main__":
    # 測試路徑 (根據你 build_features.py 的設定)
    feat_path = 'data/processed/wikiart_features.npy'
    path_path = 'data/processed/wikiart_paths.npy'
    
    try:
        searcher = ImageSearcher(feat_path, path_path)
        
        # 隨便造一個假向量來測試
        fake_vec = np.random.rand(2048).astype('float32')
        fake_vec = fake_vec / np.linalg.norm(fake_vec) # 正規化
        
        results = searcher.search(fake_vec, k=3)
        print("\n測試搜尋結果:")
        for path, score in results:
            print(f"相似度: {score:.4f} | 路徑: {path}")
            
    except Exception as e:
        print(f"測試失敗: {e}")