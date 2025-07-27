import torch
import dgl
from bgflow.utils import remove_mean
import numpy as np


class aa2_single_dataset(dgl.data.DGLDataset):
    def __init__(self, dataset,h_initial,kabsch=False):
        self.dataset = torch.from_numpy(dataset)
        self.dataset = self.dataset.float()
        original_shape = dataset.shape
        self.n_particles = dataset.shape[1] // 3
        self.kabsch= kabsch
        self.n_dimensions = 3
        self.dim = self.n_particles * self.n_dimensions
        self.dataset = remove_mean(self.dataset, self.n_particles, self.n_dimensions).reshape(-1, self.dim)
        self.dataset = self.dataset.reshape(original_shape)
        self.n_samples = len(dataset)
        self.h_initial = h_initial.cpu()
        self.nodes=torch.arange(self.n_particles)
        self.edges=torch.cartesian_prod(self.nodes,self.nodes)
        self.edges=self.edges[self.edges[:,0]!=self.edges[:,1]].transpose(0,1)
        self.graphs=[]
        for i in range(self.n_samples):
            g = dgl.graph((self.edges[0].cpu(),self.edges[1].cpu()),num_nodes=self.n_particles)
            g.ndata['h'] = self.h_initial
            self.graphs.append(g)
    def __len__(self):
        return self.n_samples
    def kabsch_torch(self,P, Q):
        """
        Computes the optimal rotation and translation to align two sets of points (P -> Q),
        and their RMSD.
        :param P: A Nx3 matrix of points
        :param Q: A Nx3 matrix of points
        :return: A tuple containing the optimal rotation matrix, the optimal
                translation vector, and the RMSD.
        """
        assert P.shape == Q.shape, "Matrix dimensions must match"

        # Compute centroids
        centroid_P = torch.mean(P, dim=0)
        centroid_Q = torch.mean(Q, dim=0)

        # Center the points
        p = P - centroid_P
        q = Q - centroid_Q

        # Compute the covariance matrix
        H = torch.matmul(p.transpose(0, 1), q)

        # SVD
        U, S, Vt = torch.linalg.svd(H)

        # Validate right-handed coordinate system
        if torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))) < 0.0:
            Vt[:, -1] *= -1.0

        # Optimal rotation
        R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

        # RMSD
        #rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(0, 1)) - q)) / P.shape[0])

        return R


    def __getitem__(self, idx):
        x_true = self.dataset[idx].reshape(self.n_particles, self.n_dimensions)
        x_noise = torch.randn_like(x_true)
        if self.kabsch:
            # Kabsch algorithm to align x_true and x_noise
            R = self.kabsch_torch(x_true, x_noise)
            x_true = torch.matmul(x_true, R.transpose(0, 1))
        self.graphs[idx].ndata["x1"] = x_noise
        self.graphs[idx].ndata["x0"] = x_true
        self.graphs[idx].ndata["h"] = self.h_initial
        return self.graphs[idx]
    
def get_aa2_single_dataloader(dataset, h_initial, batch_size=512, shuffle=True, num_workers=8, kabsch=False):
    dataset = aa2_single_dataset(dataset, h_initial, kabsch=kabsch)
    dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset, dataloader