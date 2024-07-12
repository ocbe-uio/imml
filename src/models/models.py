import numpy as np
import pandas as pd
from lightning import Trainer
from lightning.pytorch.utilities.seed import isolate_rng
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding
from snf import compute
from torch.utils.data import DataLoader
from imvc.decomposition import DeepMFDataset

from src.utils import Utils


class Model:
    def __init__(self, alg_name, alg):
        self.alg_name = alg_name.lower() if alg_name in ["GPCA", "AJIVE", "NMF", "DFMF", "MOFA", "MONET"] else "standard"
        self.method = alg_name.lower() if alg_name in ["SNF", "IntNMF", "COCA", "DeepMF"] else "sklearn_method"
        self.alg_name = eval(f"self.{self.alg_name.lower()}")
        self.method = eval(f"self.{self.method.lower()}")
        self.alg = alg


    def sklearn_method(self, train_Xs, n_clusters, random_state, run_n):
        model, params = self.alg["alg"], self.alg["params"]
        model = self.alg_name(model=model, n_clusters=n_clusters, random_state=random_state, run_n=run_n)
        clusters = model.fit_predict(train_Xs)
        if self.alg_name in ["DAIMC", "PIMVC"]:
            transformed_Xs = model[-1].V_
        elif self.alg_name in ["EEIMVC", "LFIMVC", "MKKMIK", "OSLFIMVC"]:
            transformed_Xs = model[-1].H_
        elif self.alg_name == "IMSR":
            transformed_Xs = model[-1].Z_
        elif self.alg_name == "MSNE":
            transformed_Xs = model[-1].embeddings_
        elif self.alg_name == "OMVC":
            transformed_Xs = model[-1].U_star_loss_
        elif self.alg_name == "SIMCADC":
            transformed_Xs = model[-1].U
        elif self.alg_name in ["MVSC", "MVCRSC"]:
            transformed_Xs = model[-1].embedding_
        else:
            transformed_Xs = model[:-1].transform(train_Xs)
        return clusters, transformed_Xs


    def snf(self, train_Xs, n_clusters, random_state, run_n):
        model = self.alg["alg"]
        train_Xs = model.fit_transform(train_Xs)
        k_snf = np.ceil(len(train_Xs[0]) / 10).astype(int)
        affinities = compute.make_affinity(train_Xs, normalize=False, K=k_snf)
        fused = compute.snf(affinities, K=k_snf)
        sc = SpectralClustering(n_clusters=n_clusters, random_state=random_state + run_n)
        clusters = sc.fit_predict(fused)
        transformed_Xs = spectral_embedding(sc.affinity_matrix_, n_components=n_clusters, eigen_solver=sc.eigen_solver,
                                      random_state=sc.random_state, eigen_tol=sc.eigen_tol, drop_first=False)
        transformed_Xs = pd.DataFrame(transformed_Xs, index=train_Xs[0].index)
        return clusters, transformed_Xs


    def intnmf(self, train_Xs, n_clusters, random_state, run_n):
        from rpy2.robjects.packages import importr
        nmf = importr("IntNMF")
        model = self.alg["alg"]
        train_Xs = model.fit_transform(train_Xs)
        clusters = nmf.nmf_mnnals(dat=Utils.convert_df_to_r_object(train_Xs),
                                  k=n_clusters, seed=int(random_state + run_n))[-1]
        clusters = np.array(clusters) - 1
        return clusters, model


    def coca(self, train_Xs, n_clusters, random_state, run_n):
        from rpy2.robjects.packages import importr
        base, coca = importr("base"), importr("coca")
        model = self.alg["alg"]
        train_Xs = model.fit_transform(train_Xs)
        base.set_seed(int(random_state + run_n))
        clusters = coca.buildMOC(Utils.convert_df_to_r_object(train_Xs), M=len(train_Xs), K=n_clusters)[0]
        clusters = coca.coca(clusters, K=n_clusters)[1]
        clusters = np.array(clusters) - 1
        return clusters, model


    def gpca(self, model, n_clusters, random_state, run_n):
        model[1].set_params(n_components=n_clusters, random_state=random_state + run_n, multiview_output=False)
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model


    def ajive(self, model, n_clusters, random_state, run_n):
        model[-3].set_params(joint_rank=n_clusters, random_state=random_state + run_n)
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model


    def nmf(self, model, n_clusters, random_state, run_n):
        model[1].set_params(n_components=n_clusters, random_state=random_state + run_n, multiview_output=False)
        model[-3].set_params(n_components=n_clusters, random_state=random_state + run_n)
        return model


    def monet(self, model, n_clusters, random_state, run_n):
        model[-1].set_params(random_state=random_state + run_n)
        return model


    def deepmf(self, train_Xs, n_clusters, random_state, run_n):
        pipeline = self.alg["alg"]
        transformed_Xs = pipeline[:3].fit_transform(train_Xs)
        train_data = DeepMFDataset(X=transformed_Xs)
        train_dataloader = DataLoader(dataset=train_data, batch_size=50, shuffle=True)
        trainer = Trainer(max_epochs=10, logger=False, enable_checkpointing=False)
        pipeline[3].set_params(X=transformed_Xs)
        with isolate_rng():
            trainer.fit(pipeline[3], train_dataloader)
        train_dataloader = DataLoader(dataset=train_data, batch_size=50, shuffle=False)
        transformed_Xs = pipeline[3].transform(transformed_Xs)
        pipeline[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        clusters = pipeline[4:].fit_predict(transformed_Xs)
        return clusters, transformed_Xs


    def dfmf(self, model, n_clusters, random_state, run_n):
        model[1].set_params(n_components=n_clusters, random_state=random_state + run_n)
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model


    def mofa(self, model, n_clusters, random_state, run_n):
        model[1].set_params(factors=n_clusters, random_state=random_state + run_n)
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model


    def standard(self, model, n_clusters, random_state, run_n):
        model[-1].set_params(n_clusters=n_clusters, random_state=random_state + run_n)
        return model
