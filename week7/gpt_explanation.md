# microgpt 程式說明

這份說明檔針對 `gpt.py` 的實作進行解析。這個程式是一個最簡化版本的 GPT 模型，完全用純 Python 實作，沒有額外函式庫依賴。

## 1. 程式目的

`gpt.py` 的目的是實現一個最簡單的 GPT 文字生成模型，它會：

- 下載並讀取名字資料集
- 將資料集中的字元轉成 token
- 實作簡單的自動微分（autograd）引擎
- 建立 Transformer/Attention 架構
- 用 Adam 訓練模型來預測下個字元
- 生成新的名字

## 2. 程式架構總覽

程式可以分成 8 個主要部分：

1. **資料集處理（DATASET）**
2. **Tokenizer（TOKENIZER）**
3. **自動微分（AUTOGRAD）**
4. **超參數與模型參數（HYPERPARAMETERS & PARAMETERS）**
5. **輔助函式（HELPER FUNCTIONS）**
6. **GPT 架構（ARCHITECTURE）**
7. **訓練迴圈（TRAINING LOOP）**
8. **推理／生成（INFERENCE）**

---

## 3. 程式細節說明

### 3.1 資料集處理

```python
if not os.path.exists('input.txt'):
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
```

- 若當前目錄沒有 `input.txt`，程式會從網路下載名字資料集。
- `docs` 儲存每一行名字，並做亂數排列。

### 3.2 Tokenizer

```python
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
```

- `uchars` 取得資料集中所有唯一字元。
- `BOS` 是特殊起始／結束符號。
- 詞彙大小 `vocab_size` 等於字元數加上 `BOS`。

### 3.3 自動微分（Autograd）

程式使用 `Value` 類別來追蹤每個 scalar 的運算與梯度。

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
```

- 每個 `Value` 包含數值 `data`、梯度 `grad`、父節點 `_children` 和對父節點的本地梯度 `_local_grads`。
- 常見運算（加、乘、冪、log、exp、ReLU）都被覆寫。
- `backward()` 會建立拓撲排序，並從 loss 反向傳播梯度。

### 3.4 超參數與模型參數

```python
n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head
```

- `n_embd`：嵌入維度
- `n_head`：注意力頭數
- `n_layer`：Transformer 層數
- `block_size`：最大序列長度
- `head_dim`：每個 attention head 的維度

模型參數使用 `Value` 隨機初始化：

```python
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}
```

還包含每層的 `attn_wq`, `attn_wk`, `attn_wv`, `attn_wo`, `mlp_fc1`, `mlp_fc2`。

### 3.5 輔助函式

- `linear(x, w)`：矩陣向量相乘
- `softmax(logits)`：將 logits 轉為機率分布
- `rmsnorm(x)`：簡化版 RMSNorm 標準化

### 3.6 GPT 架構

`gpt(token_id, pos_id, keys, values)` 是模型前向傳播函式。

#### 3.6.1 Embedding

```python
x = [t + p for t, p in zip(tok_emb, pos_emb)]
x = rmsnorm(x)
```

- 將 token 嵌入與位置嵌入相加，再做 RMSNorm。

#### 3.6.2 Multi-head Attention

- 計算 `q`, `k`, `v` 3 個向量
- 把 `k`, `v` 加入 KV cache
- 每個 head 計算自己注意力分數
- softmax 得到權重，然後加權求和
- 最後將所有 head 輸出串接並投影回去

#### 3.6.3 MLP Block

- 接著做一個簡單的前饋網路：
  - 線性層 -> ReLU -> 線性層
  - 再加上殘差連接

#### 3.6.4 最後輸出

```python
logits = linear(x, state_dict['lm_head'])
```

輸出維度為 `vocab_size`，代表下個字元的分數。

### 3.7 訓練迴圈

訓練循環如下：

1. 取一個名字
2. 加上 `BOS` 開頭與結尾
3. 以最大 `block_size` 進行前向傳播
4. 計算每個位置的交叉熵損失
5. `loss.backward()` 反向傳播
6. 使用 Adam 更新參數

Adam 參數：

```python
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
```

每步更新後清空梯度。

### 3.8 推理（生成）

推理時固定模型權重，從 `BOS` 開始生成：

- 先前向傳播得到 logits
- 用溫度 `temperature = 0.5` 調整 logits
- 依照機率採樣下一個 token
- 若抽到 `BOS` 則停止

最終印出 20 個新名字。

---

## 4. 執行方式

在該目錄下執行：

```bash
python gpt.py
```

它會先訓練 1000 步，然後生成 20 個新的名字。

## 5. 你可以如何改進

- 增加 `num_steps` 讓模型訓練更久
- 調整 `n_embd`, `n_head`, `n_layer` 讓模型變大
- 將資料集改成其他文本，像是城市名、英文單字、簡短詩句
- 試試不同 `temperature` 值，控制生成多樣性
