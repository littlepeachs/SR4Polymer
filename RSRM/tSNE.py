from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
font = {'family': 'Arial',
        'weight': 'bold',
        'size': 18
        }
plt.rc('font', **font)
# 加载鸢尾花数据集
# 读取xlsx文件
import pandas as pd
df = pd.read_excel('aging_mirco_macro.xlsx')
X= df.iloc[:,2:11].values
y = df.iloc[:,11].values
target_names = ["low_hardness","middle_hardness","high_hardness"]
# import pdb; pdb.set_trace()
# 对数据进行PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 对数据进行t-SNE降维
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_tsne = tsne.fit_transform(X)

# 定义颜色板和标记符号
palette = sns.color_palette("bright", 3)
markers = ['o', 's', '^']

# 创建一个包含两个子图的图形,并设置背景风格
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
sns.set_style("whitegrid")

# 绘制PCA降维后的数据点
for i, target_name in enumerate(target_names):
    ax1.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=palette[i], label=target_name, marker=markers[i], s=60, alpha=0.8)
ax1.set_xlabel('PC1', fontsize=12)
ax1.set_ylabel('PC2', fontsize=12)
ax1.set_title('PCA of Aging Dataset', fontsize=14)
ax1.legend(fontsize=12)
# plt.tight_layout()
# plt.show()
# plt.savefig('PCA.png')
# 绘制t-SNE降维后的数据点
for i, target_name in enumerate(target_names):
    ax2.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], color=palette[i], label=target_name, marker=markers[i], s=60, alpha=0.8)
ax2.set_xlabel('t-SNE dim 1', fontsize=12)
ax2.set_ylabel('t-SNE dim 2', fontsize=12)
ax2.set_title('t-SNE of Aging Dataset', fontsize=14)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.show()
plt.savefig('tSNE.png',dpi=600)