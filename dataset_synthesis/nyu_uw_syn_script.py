"""
	Script to synthesize a dataset of 10 Jerlov water types (clubbed into 
	6 classes - (1, 3), 5, 7, 9, (I, IA, IB), (II, III)) following Anwar et al. (2018). 
	The script augments the dataset by generating 6 images of each class using random parameters, 
	thus for every ground truth image, we have corresponding 36 images of different water types.
"""

import numpy as np
import scipy.io
import h5py, random
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import click, os

@click.command()
@click.argument('nyu_rgbd_path', type=click.Path(exists=True))
@click.argument('dataset_path', type=click.Path(exists=True))
def main(nyu_rgbd_path, dataset_path):
	with h5py.File(nyu_rgbd_path, 'r') as file:
		images = file['images']
		depths = file['depths']
		labels = file['labels']

		N_lambda = {"1": [0.875, 0.885, 0.75],
					"3": [0.8, 0.82, 0.71],
					"5": [0.67, 0.73, 0.67],
					"7": [0.5, 0.61, 0.62],
					"9": [0.29, 0.46, 0.55],
					"I": [0.982, 0.961, 0.805],
					"IA": [0.975, 0.955, 0.804],
					"IB": [0.968, 0.95, 0.83],
					"II": [0.94, 0.925, 0.8],
					"III": [0.89, 0.885, 0.75]
		}

		data_path = os.path.join(dataset_path, 'data')
		label_path = os.path.join(dataset_path, 'label')
		if not os.path.exists(data_path):
			os.mkdir(data_path)
		if not os.path.exists(label_path):
			os.mkdir(label_path)

		rand = {"1": 3,
				"3": 3,
				"5": 6,
				"7": 6,
				"9": 6,
				"I": 2,
				"IA": 2,
				"IB": 2,
				"II": 3,
				"III": 3
		}

		save_type = {"1": 0,
					"3": 0,
					"5": 1,
					"7": 2,
					"9": 3,
					"I": 4,
					"IA": 4,
					"IB": 4,
					"II": 5,
					"III": 5
		}

		water_type_label = {"1": "1, 3",
							"3": "1, 3",
							"5": "5",
							"7": "7",
							"9": "9",
							"I": "I, IA, IB",
							"IA": "I, IA, IB",
							"IB": "I, IA, IB",
							"II": "II, III",
							"III": "II, III"
		}

		for idx, org_img in tqdm(enumerate(images)):
			org_img = org_img.transpose(2,1,0)
			org_img = org_img / 255.0

			save_type_cnts = {0: 0,
					1: 0,
					2: 0,
					3: 0,
					4: 0,
					5: 0,
					6: 0,
			}

			imageio.imwrite(os.path.join(label_path, '{}.png'.format(idx)), (org_img * 255).astype(np.uint8))

			org_depth = depths[idx].transpose(1,0)
			org_depth = (org_depth-org_depth.min())/(org_depth.max()-org_depth.min())

			for water_idx, water_type in enumerate(N_lambda.keys()):
				rand_num = rand[water_type]
				max_depth = np.random.uniform(0.5, 15, size=rand_num) 
				B_rand = 5 - 2 * np.random.uniform(0, 1, size=rand_num)

				for i in range(0, rand_num):
					depth = max_depth[i] * org_depth

					T_x = np.ndarray((480, 640, 3))
					T_x[:,:,0] = N_lambda[water_type][2] * depth
					T_x[:,:,1] = N_lambda[water_type][1] * depth
					T_x[:,:,2] = N_lambda[water_type][0] * depth
					T_x = (T_x-T_x.min())/(T_x.max()-T_x.min())

					B_lambda = np.ndarray((480, 640, 3))
					B_lambda[:,:,0].fill(1.5*N_lambda[water_type][2]**B_rand[i])
					B_lambda[:,:,1].fill(1.5*N_lambda[water_type][1]**B_rand[i])
					B_lambda[:,:,2].fill(1.5*N_lambda[water_type][0]**B_rand[i])

					img = org_img * T_x + B_lambda * (1 - T_x)
					img = (img-img.min())/(img.max()-img.min())

					img_name = '{}_{}_{}.png'.format(idx, save_type[water_type], save_type_cnts[save_type[water_type]])

					imageio.imwrite(os.path.join(data_path, img_name), (img * 255).astype(np.uint8))
					save_type_cnts[save_type[water_type]] += 1

if __name__== "__main__":
	main()