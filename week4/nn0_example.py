"""
nn0_example.py — 神經網路學習範例

演示使用 nn0.py 中的 Value、Adam 優化器進行神經網路訓練。
包含：
  1. MLP 分類器 - 簡單的多層感知機
  2. XOR 問題 - 經典非線性問題
  3. 數字分類 - 簡單的多類分類
"""

from nn0 import Value, Adam, softmax
import random
import math


# ============================================================================
# 1. MLP 多層感知機模型
# ============================================================================

class MLP:
    """多層感知機 (Multi-Layer Perceptron)"""
    
    def __init__(self, layer_sizes):
        """
        初始化 MLP
        
        Args:
            layer_sizes: 列表 [input_size, hidden1, hidden2, ..., output_size]
            例如 [2, 16, 16, 1] 表示 2 輸入層、兩個隱藏層各 16 個神經元、1 輸出層
        """
        self.layers = []
        
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            
            # 隨機初始化權重和偏置
            w = [[Value(random.gauss(0, 0.1)) for _ in range(in_size)] 
                 for _ in range(out_size)]
            b = [Value(0) for _ in range(out_size)]
            
            self.layers.append({'w': w, 'b': b})
    
    def __call__(self, x):
        """
        前向傳播
        
        Args:
            x: 輸入值列表
            
        Returns:
            輸出值列表
        """
        for layer in self.layers[:-1]:
            # 線性變換
            z = [sum(w_row[i] * x[i] for i in range(len(x))) + b 
                 for w_row, b in zip(layer['w'], layer['b'])]
            # ReLU 激活函數
            x = [Value(0) if z_i.data <= 0 else z_i for z_i in z]
        
        # 輸出層（不使用激活函數，交給 softmax 或 sigmoid 處理）
        last_layer = self.layers[-1]
        output = [sum(w_row[i] * x[i] for i in range(len(x))) + b 
                  for w_row, b in zip(last_layer['w'], last_layer['b'])]
        return output
    
    def parameters(self):
        """取得所有參數（用於優化器）"""
        params = []
        for layer in self.layers:
            params.extend(layer['w'])
            params.extend(layer['b'])
        # 扁平化
        params_flat = []
        for p in params:
            if isinstance(p, list):
                params_flat.extend(p)
            else:
                params_flat.append(p)
        return params_flat
    
    def zero_grad(self):
        """清除所有梯度"""
        for p in self.parameters():
            p.grad = 0


# ============================================================================
# 2. XOR 問題 — 經典非線性分類問題
# ============================================================================

def train_xor():
    """
    訓練 MLP 解決 XOR 問題
    
    XOR 是非線性可分的問題，需要至少一個隱藏層才能解決。
    """
    print("=" * 60)
    print("範例 1: XOR 問題")
    print("=" * 60)
    
    # XOR 訓練數據
    X_train = [
        [0, 0],  # XOR(0,0) = 0
        [0, 1],  # XOR(0,1) = 1
        [1, 0],  # XOR(1,0) = 1
        [1, 1],  # XOR(1,1) = 0
    ]
    y_train = [0, 1, 1, 0]
    
    # 建立模型：2 輸入 → 16 隱藏 → 1 輸出
    model = MLP([2, 16, 1])
    
    # Adam 優化器
    optimizer = Adam(model.parameters(), lr=0.1)
    
    # 訓練迴圈
    num_epochs = 500
    print(f"訓練 {num_epochs} 個 epoch...\n")
    
    for epoch in range(num_epochs):
        model.zero_grad()
        
        total_loss = 0
        for x, y in zip(X_train, y_train):
            # 準備輸入
            x_val = [Value(xi) for xi in x]
            
            # 前向傳播
            logits = model(x_val)
            
            # Sigmoid 損失（二分類）
            pred = logits[0]
            loss = (pred - Value(y))**2  # MSE 損失
            loss.backward()
            total_loss += loss.data
        
        # Adam 優化執行一步
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(X_train)
            print(f"Epoch {epoch + 1:3d}, Loss: {avg_loss:.6f}")
    
    # 測試
    print("\n預測結果：")
    for x, y_true in zip(X_train, y_train):
        x_val = [Value(xi) for xi in x]
        pred = model(x_val)[0].data
        print(f"XOR({x[0]}, {x[1]}) = {pred:.4f}  (真值: {y_true})")
    
    print()


# ============================================================================
# 3. 多類分類 — 簡單的 3 類分類問題
# ============================================================================

def train_multiclass():
    """訓練 MLP 解決 3 類分類問題"""
    print("=" * 60)
    print("範例 2: 多類分類 (3 類)")
    print("=" * 60)
    
    # 生成簡單的訓練數據：3 個類別，每個類別 10 個樣本
    random.seed(42)
    X_train = []
    y_train = []
    
    # 類別 0: 中心在 (0, 0)
    for _ in range(10):
        x = random.gauss(0, 0.5)
        y = random.gauss(0, 0.5)
        X_train.append([x, y])
        y_train.append(0)
    
    # 類別 1: 中心在 (3, 0)
    for _ in range(10):
        x = random.gauss(3, 0.5)
        y = random.gauss(0, 0.5)
        X_train.append([x, y])
        y_train.append(1)
    
    # 類別 2: 中心在 (1.5, 3)
    for _ in range(10):
        x = random.gauss(1.5, 0.5)
        y = random.gauss(3, 0.5)
        X_train.append([x, y])
        y_train.append(2)
    
    # 建立模型：2 輸入 → 8 隱藏 → 3 輸出
    model = MLP([2, 8, 3])
    
    # Adam 優化器
    optimizer = Adam(model.parameters(), lr=0.1)
    
    # 訓練迴圈
    num_epochs = 300
    print(f"訓練 {num_epochs} 個 epoch...\n")
    
    for epoch in range(num_epochs):
        model.zero_grad()
        
        total_loss = 0
        for x, y_true in zip(X_train, y_train):
            # 準備輸入
            x_val = [Value(xi) for xi in x]
            
            # 前向傳播
            logits = model(x_val)
            
            # Softmax + 交叉熵損失
            probs = softmax(logits)
            
            # 交叉熵損失：-log(p_true_class)
            loss = -probs[y_true].log()
            loss.backward()
            total_loss += loss.data
        
        # Adam 優化執行一步
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(X_train)
            print(f"Epoch {epoch + 1:3d}, Loss: {avg_loss:.6f}")
    
    # 測試準確度
    print("\n測試準確度：")
    correct = 0
    for x, y_true in zip(X_train, y_train):
        x_val = [Value(xi) for xi in x]
        logits = model(x_val)
        probs = softmax(logits)
        pred = max(range(3), key=lambda i: probs[i].data)
        correct += (pred == y_true)
    
    accuracy = correct / len(X_train) * 100
    print(f"準確度: {accuracy:.1f}%")
    
    print()


# ============================================================================
# 4. 函數擬合 — 學習 sin(x)
# ============================================================================

def train_function_fit():
    """訓練 MLP 擬合 sin(x) 函數"""
    print("=" * 60)
    print("範例 3: 函數擬合 - sin(x)")
    print("=" * 60)
    
    # 生成訓練數據：y = sin(x)
    X_train = []
    y_train = []
    
    random.seed(42)
    for _ in range(50):
        x = random.uniform(-math.pi, math.pi)
        y = math.sin(x)
        X_train.append([x])
        y_train.append([y])
    
    # 建立模型：1 輸入 → 16 隱藏 → 1 輸出
    model = MLP([1, 16, 1])
    
    # Adam 優化器
    optimizer = Adam(model.parameters(), lr=0.05)
    
    # 訓練迴圈
    num_epochs = 500
    print(f"訓練 {num_epochs} 個 epoch...\n")
    
    for epoch in range(num_epochs):
        model.zero_grad()
        
        total_loss = 0
        for x, y_true in zip(X_train, y_train):
            # 準備輸入
            x_val = [Value(xi) for xi in x]
            
            # 前向傳播
            pred = model(x_val)[0]
            
            # MSE 損失
            loss = (pred - Value(y_true[0]))**2
            loss.backward()
            total_loss += loss.data
        
        # Adam 優化執行一步
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(X_train)
            print(f"Epoch {epoch + 1:3d}, Loss: {avg_loss:.6f}")
    
    # 測試
    print("\n預測結果 (sample)：")
    test_x_vals = [-math.pi, -math.pi/2, 0, math.pi/2, math.pi]
    for x_val in test_x_vals:
        x = [Value(x_val)]
        pred = model(x)[0].data
        true = math.sin(x_val)
        error = abs(pred - true)
        print(f"x = {x_val:6.3f}, 預測 = {pred:7.4f}, 真值 = {true:7.4f}, 誤差 = {error:.4f}")
    
    print()


# ============================================================================
# 5. 梯度檢查 — 驗證反向傳播實現
# ============================================================================

def gradient_check():
    """
    梯度檢查：用數值梯度驗證反向傳播的正確性
    
    這是一個重要的調試技巧，確保自動微分實現正確。
    """
    print("=" * 60)
    print("範例 4: 梯度檢查 (Gradient Checking)")
    print("=" * 60)
    
    # 建立簡單的計算圖
    x = Value(3.0)
    y = Value(2.0)
    
    # 計算 z = x^3 + 2xy + y^2
    z = x**3 + Value(2) * x * y + y**2
    
    # 反向傳播
    z.backward()
    
    # 解析梯度
    dz_dx_analytical = z.grad if hasattr(z, 'data') else 0
    dz_dy_analytical = y.grad if hasattr(y, 'data') else 0
    
    # 數值梯度（用有限差分計算）
    eps = 1e-5
    
    # ∂z/∂x 的數值梯度
    x_plus = Value(3.0 + eps)
    y_temp = Value(2.0)
    z_plus = x_plus**3 + Value(2) * x_plus * y_temp + y_temp**2
    
    x_minus = Value(3.0 - eps)
    z_minus = x_minus**3 + Value(2) * x_minus * y_temp + y_temp**2
    
    dz_dx_numerical = (z_plus.data - z_minus.data) / (2 * eps)
    
    # ∂z/∂y 的數值梯度
    x_temp = Value(3.0)
    y_plus = Value(2.0 + eps)
    z_plus = x_temp**3 + Value(2) * x_temp * y_plus + y_plus**2
    
    y_minus = Value(2.0 - eps)
    z_minus = x_temp**3 + Value(2) * x_temp * y_minus + y_minus**2
    
    dz_dy_numerical = (z_plus.data - z_minus.data) / (2 * eps)
    
    print(f"函數：z = x³ + 2xy + y²")
    print(f"在 x = 3.0, y = 2.0 處\n")
    print(f"∂z/∂x:")
    print(f"  反向傳播 (自動微分): {x.grad:.6f}")
    print(f"  數值梯度:          {dz_dx_numerical:.6f}")
    print(f"  誤差:              {abs(x.grad - dz_dx_numerical):.2e}")
    print()
    print(f"∂z/∂y:")
    print(f"  反向傳播 (自動微分): {y.grad:.6f}")
    print(f"  數值梯度:          {dz_dy_numerical:.6f}")
    print(f"  誤差:              {abs(y.grad - dz_dy_numerical):.2e}")
    print()


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  nn0.py 神經網路學習範例".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # 執行所有範例
    train_xor()
    train_multiclass()
    train_function_fit()
    gradient_check()
    
    print("=" * 60)
    print("所有範例執行完成！")
    print("=" * 60)
