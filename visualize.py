import torch
import torchvision
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from nyu_uw_dataset import NYUUWDataset, UIEBDataset
from tqdm import tqdm
import random
from networks import UNetEncoder
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from ggplot import ggplot, aes, geom_point, ggtitle


# test_dataset = UIEBDataset('/media/pritish/New Volume/datasets/raw-890/raw-890/', 
#     '/media/pritish/New Volume/datasets/raw-890/raw-890/',
#     size=890,
#     test_start=0,
#     mode='test')

test_dataset = NYUUWDataset('/media/pritish/New Volume/datasets/nyu_underwater_v2/data', 
	'/media/pritish/New Volume/datasets/nyu_underwater_v2/label',
	size=360,
	test_start=33008,
	mode='test')

batch_size = 1
dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
outs = []
labels = []
names = []
fE = UNetEncoder(3).cuda()
fE.load_state_dict(torch.load('./checkpoints/unet_adv/fE_55.pth'))
fE.eval()

for idx, data in tqdm(enumerate(dataloader)):
	uw_img, cl_img, wt, img_name = data
	uw_img = Variable(uw_img).cuda()
	
	fE_out, enc_outs = fE(uw_img)
	outs.append(fE_out.detach().view(-1))
	labels.append(wt)
	names.append(img_name)

fE_outs = torch.stack(outs).cpu().numpy()
y = torch.stack(labels).numpy()
names = torch.stack(names).numpy()
feat_cols = [ 'feat'+str(i) for i in range(fE_outs.shape[1]) ]

df = pd.DataFrame(fE_outs, columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

df['name'] = names
df['name'] = df['name'].apply(lambda i: str(i))

print ('Size of the dataframe: {}'.format(df.shape))

# pca = PCA(n_components=50)
# pca_result = pca.fit_transform(df[feat_cols].values)

# np.save('pca_result.npy', pca_result)
pca_result = np.load("pca_result.npy")
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]

print ('Size of the dataframe: {}'.format(df.shape))
# print ('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))



# import time
# from sklearn.manifold import TSNE

# time_start = time.time()

# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_pca_results = tsne.fit_transform(pca_result)
# np.save('tsne_pca_results.npy', tsne_pca_results)

# print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
tsne_pca_results = np.load('tsne_pca_results.npy')
print (tsne_pca_results.shape)
df_tsne = None
df_tsne = df.loc[:,:].copy()
df_tsne['x-tsne-pca'] = tsne_pca_results[:,0]
df_tsne['y-tsne-pca'] = tsne_pca_results[:,1]

df = df[df['pca-one']<=100] # (df['pca-one'].mean() + df['pca-one'].std())
df = df[df['pca-two']<=50] # (df['pca-two'].mean() + df['pca-two'].std())
df = df[df['pca-one']>=-100] # (df['pca-one'].mean() - df['pca-one'].std())
df = df[df['pca-two']>=-50] # (df['pca-two'].mean() - df['pca-two'].std())
print ('Size of the dataframe after outlier removal: {}'.format(df.shape))

pca_chart = ggplot( df.loc[:,:], aes(x='pca-one', y='pca-two', color='label') ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("First and Second Principal Components colored by digit")

pca_chart.save('chart_pca_unet_adv_4.png', dpi=1080)

tsne_chart = ggplot( df_tsne, aes(x='x-tsne-pca', y='y-tsne-pca', color='name') ) \
        + geom_point(size=70,alpha=0.1) \
        + ggtitle("tSNE dimensions colored by Digit (PCA)")
tsne_chart.save('chart_tsne_unet_adv_4.png', dpi=1080)