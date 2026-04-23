"""
非 Transformer 的語言模型 - 純 NumPy 實現
不依賴 TensorFlow，支援 Windows ARM
包含：N-gram 模型 + 神經網路模型
"""

import numpy as np
import os
from collections import defaultdict

# 設定隨機種子
np.random.seed(42)


class SimpleNeuralNetwork:
    """簡單的前向神經網路 - 純 NumPy 實現"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 初始化權重
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        """ReLU 激活函數"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU 導數"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax 激活函數"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """前向傳播"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        """反向傳播"""
        m = X.shape[0]
        
        # 計算梯度
        dz2 = output.copy()
        dz2[np.arange(m), y] -= 1
        dz2 /= m
        
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # 更新權重
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X, y, epochs=100):
        """訓練模型"""
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            # 計算交叉熵損失
            loss = -np.mean(np.log(output[np.arange(len(y)), y] + 1e-10))
            losses.append(loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """預測"""
        output = self.forward(X)
        return output


class CharLevelLanguageModel:
    """字符級別語言模型"""
    
    def __init__(self, seq_length=20):
        self.seq_length = seq_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.model = None
        
    def load_data(self, filename):
        """讀取文本文件"""
        with open(filename, 'r', encoding='utf-8') as f:
            self.text = f.read()
        return self.text
    
    def build_vocab(self):
        """構建字符詞彙表"""
        chars = sorted(list(set(self.text)))
        self.char_to_idx = {char: i for i, char in enumerate(chars)}
        self.idx_to_char = {i: char for i, char in enumerate(chars)}
        self.vocab_size = len(chars)
        print(f"字彙表大小: {self.vocab_size}")
        print(f"字彙: {chars}")
        return self.char_to_idx, self.idx_to_char
    
    def encode_text(self):
        """將文本編碼為數字"""
        self.encoded = np.array([self.char_to_idx[char] for char in self.text])
        return self.encoded
    
    def generate_sequences(self):
        """生成訓練序列"""
        X, y = [], []
        for i in range(len(self.encoded) - self.seq_length):
            X.append(self.encoded[i:i + self.seq_length])
            y.append(self.encoded[i + self.seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # 轉換 X 為 one-hot 編碼
        X_onehot = np.zeros((X.shape[0], X.shape[1] * self.vocab_size))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_onehot[i, j * self.vocab_size + X[i, j]] = 1
        
        print(f"訓練樣本數: {len(X)}")
        print(f"輸入形狀: {X_onehot.shape}")
        print(f"輸出形狀: {y.shape}")
        
        return X_onehot, y
    
    def build_model(self):
        """構建神經網路模型"""
        input_size = self.seq_length * self.vocab_size
        hidden_size = 256
        output_size = self.vocab_size
        
        self.model = SimpleNeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.01)
        print(f"模型構建完成")
        print(f"  輸入層: {input_size}")
        print(f"  隱藏層: {hidden_size}")
        print(f"  輸出層: {output_size}")
    
    def train(self, epochs=100):
        """訓練模型"""
        print("\n開始訓練...")
        X, y = self.generate_sequences()
        self.model.train(X, y, epochs=epochs)
    
    def generate_text(self, seed_text, num_chars=10, temperature=0.7):
        """生成文本"""
        generated = seed_text
        
        for _ in range(num_chars):
            # 準備輸入
            seed = seed_text[-self.seq_length:]
            
            # 補足長度不足的種子
            if len(seed) < self.seq_length:
                seed = ' ' * (self.seq_length - len(seed)) + seed
            
            # 編碼為 one-hot
            X_input = np.zeros((1, self.seq_length * self.vocab_size))
            for i, char in enumerate(seed):
                if char in self.char_to_idx:
                    X_input[0, i * self.vocab_size + self.char_to_idx[char]] = 1
            
            # 預測
            probabilities = self.model.predict(X_input)[0]
            
            # 使用溫度採樣
            probabilities = np.log(probabilities + 1e-10) / temperature
            probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
            
            # 選擇下一個字符
            predicted_idx = np.random.choice(len(probabilities), p=probabilities)
            predicted_char = self.idx_to_char[predicted_idx]
            
            generated += predicted_char
            seed_text = seed_text[1:] + predicted_char
        
        return generated


class NGramLanguageModel:
    """N-gram 語言模型 - 基於統計的方法"""
    
    def __init__(self, n=3):
        self.n = n
        self.vocab = set()
        self.ngrams = defaultdict(lambda: defaultdict(int))
    
    def load_data(self, filename):
        """讀取文本"""
        with open(filename, 'r', encoding='utf-8') as f:
            self.text = f.read()
    
    def train(self):
        """訓練 N-gram 模型"""
        print(f"訓練 {self.n}-gram 模型...")
        
        # 構建字符詞彙表
        self.vocab = set(self.text)
        print(f"詞彙表大小: {len(self.vocab)}")
        
        # 構建 N-gram
        for i in range(len(self.text) - self.n):
            context = self.text[i:i + self.n - 1]
            next_char = self.text[i + self.n - 1]
            self.ngrams[context][next_char] += 1
        
        print(f"N-gram 數量: {len(self.ngrams)}")
    
    def generate_text(self, seed_text, num_chars=10):
        """生成文本"""
        generated = seed_text
        
        for _ in range(num_chars):
            # 獲得上下文
            context = generated[-(self.n - 1):] if len(generated) >= self.n - 1 else generated
            
            # 如果上下文不夠長，添加空格
            if len(context) < self.n - 1:
                context = ' ' * (self.n - 1 - len(context)) + context
            
            # 獲得候選字符
            if context in self.ngrams:
                candidates = self.ngrams[context]
                # 根據頻率採樣
                total = sum(candidates.values())
                next_char = max(candidates, key=candidates.get)
            else:
                # 如果上下文不存在，隨機選擇
                next_char = np.random.choice(list(self.vocab))
            
            generated += next_char
        
        return generated


def interactive_session(ngram_model, nn_model=None):
    """交互式對話會話"""
    print("\n" + "=" * 60)
    print("進入互動模式 (輸入 'quit' 離開)")
    print("=" * 60)
    
    while True:
        print("\n" + "-" * 60)
        
        # 選擇模型
        print("選擇模型:")
        print("  1. N-gram 模型 (快速)")
        print("  2. 神經網路模型 (緩慢但效果更好)")
        print("  3. 返回主菜單")
        
        choice = input("請選擇 (1/2/3): ").strip()
        
        if choice == '3':
            break
        
        if choice not in ['1', '2']:
            print("❌ 無效選擇，請重試")
            continue
        
        if choice == '2' and nn_model is None:
            print("❌ 神經網路模型未訓練。請先訓練模型。")
            continue
        
        # 輸入種子文本
        seed_text = input("\n輸入種子文本 (或按 Enter 使用預設): ").strip()
        
        if not seed_text:
            seed_text = "小貓"
        
        if seed_text.lower() == 'quit':
            break
        
        # 輸入生成長度
        try:
            num_chars = input("生成字符數 (預設 100): ").strip()
            num_chars = int(num_chars) if num_chars else 100
            num_chars = max(10, min(500, num_chars))  # 限制在 10-500
        except ValueError:
            num_chars = 100
        
        # 生成文本
        print("\n⏳ 生成中...")
        if choice == '1':
            generated = ngram_model.generate_text(seed_text, num_chars=num_chars)
        else:
            generated = nn_model.generate_text(seed_text, num_chars=num_chars)
        
        print(f"\n✅ 生成完成")
        print("-" * 60)
        print(f"【種子文本】: {seed_text}")
        print(f"【生成文本】:")
        print(generated)
        print("-" * 60)
        
        # 選擇是否保存
        save = input("是否保存結果到文件? (y/n): ").strip().lower()
        if save == 'y':
            save_result(seed_text, generated)
    
    print("\n感謝使用！")


def save_result(seed_text, generated_text):
    """保存生成結果"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"種子文本: {seed_text}\n")
        f.write(f"生成文本:\n{generated_text}\n")
    
    print(f"✅ 已保存到 {filename}")


def show_menu():
    """顯示主菜單"""
    print("\n" + "=" * 60)
    print("非 Transformer 語言模型 - 主菜單")
    print("=" * 60)
    print("1. N-gram 模型試驗")
    print("2. 神經網路模型訓練 & 試驗")
    print("3. 互動模式")
    print("4. 退出")
    return input("請選擇 (1/2/3/4): ").strip()


def main():
    # 文件路徑
    data_file = os.path.join(os.path.dirname(__file__), 'tw.txt')
    
    print("=" * 60)
    print("非 Transformer 語言模型 - 純 NumPy 實現 (Windows ARM 相容)")
    print("=" * 60)
    
    # 初始化 N-gram 模型
    print("\n⏳ 初始化 N-gram 模型...")
    ngram_model = NGramLanguageModel(n=4)
    ngram_model.load_data(data_file)
    ngram_model.train()
    print("✅ N-gram 模型準備就緒\n")
    
    # 初始化神經網路模型（可選）
    nn_model = None
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            # N-gram 試驗
            print("\n【N-gram 模型試驗】")
            print("-" * 60)
            test_seeds = ["小貓", "我喜歡", "天上", "小狗", "今天"]
            
            for seed in test_seeds:
                generated = ngram_model.generate_text(seed, num_chars=10)
                print(f"\n種子: {seed}")
                print(f"生成: {generated}")
        
        elif choice == '2':
            # 神經網路訓練
            print("\n【神經網路模型訓練】")
            print("-" * 60)
            print("⏳ 讀取數據...")
            nn_model = CharLevelLanguageModel(seq_length=10)
            text = nn_model.load_data(data_file)
            print(f"✅ 加載數據: {len(text)} 字符")
            
            print("\n⏳ 構建詞彙表...")
            nn_model.build_vocab()
            nn_model.encode_text()
            
            print("\n⏳ 構建模型...")
            nn_model.build_model()
            
            print("\n⏳ 訓練模型 (這可能需要數分鐘)...")
            nn_model.train(epochs=50)
            
            # 測試生成
            print("\n【生成測試】")
            test_seeds = ["小貓", "我喜歡", "天上"]
            for seed in test_seeds:
                generated = nn_model.generate_text(seed, num_chars=10)
                print(f"\n種子: {seed}")
                print(f"生成: {generated}")
                print("-" * 60)
        
        elif choice == '3':
            # 互動模式
            interactive_session(ngram_model, nn_model)
        
        elif choice == '4':
            # 退出
            print("\n👋 再見！")
            break
        
        else:
            print("❌ 無效選擇，請重試")


if __name__ == "__main__":
    main()
