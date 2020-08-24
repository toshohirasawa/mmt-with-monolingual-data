import math
import torch
import numpy as np
from sklearn.decomposition import PCA

import logging
logger = logging.getLogger('nmtpytorch')

def center_emb(x):
    logger.info(f'Apply centering')
    zero_rows = np.all(x == 0, axis=1)
    # transform embedding to make the average to zero vector
    # excluding zero vectors
    x -= x[~zero_rows].mean(axis=0)
    # leave zero vector as it was
    x[zero_rows] = 0
    return x

def norm_emb(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm[x_norm==0] = 1.
    return x / x_norm

def apply_abtt(embs, centering=True, n_pca=3, applys_norm=True):
    pad_emb = embs[1]
    word_embs = embs[1:]

    # Centering
    word_embs = center_emb(word_embs) if centering else word_embs

    # Substract PCA components
    if n_pca > 0:
        logger.info(f'Substracting PCA components (n={n_pca})')
        pca = PCA(n_components=n_pca)
        pca.fit(word_embs)
        # use tensor to calculate vectors for each PCA components
        embs_pca = pca.transform(word_embs).reshape(-1, n_pca, 1) * \
            pca.components_.reshape(1, n_pca, -1)
        # and accumulate them to get final PCA values to substract
        embs_pca = embs_pca.sum(axis=-2)

        word_embs -= embs_pca
    
    word_embs = norm_emb(word_embs) if applys_norm else word_embs

    return np.append([pad_emb], word_embs, axis=0)

def apply_lc(embs, k=10, applys_norm=True):
    assert k > 0, f'`k_nn` not found or should be a positive value'
    logger.info(f'Apply localized centering with {k}-NN')

    special_embs = torch.torch.from_numpy(embs[:1])

    # use GPU to calculate matrix production at scale
    word_embs = torch.torch.from_numpy(embs[1:])
    n_vocab, dim = word_embs.shape

    # use cosine similarity to select k-NN
    word_emb_norms = word_embs.norm(dim=-1, keepdim=True)
    word_sims = word_embs @ word_embs.t() / word_emb_norms / word_emb_norms.t()

    # set self similarity to 0 to avoid argsort from selecting itself
    word_sims[range(n_vocab), range(n_vocab)] = 0
    top_k = word_sims.argsort(descending=True)[:, :k]

    # calculate localized centroid
    word_embs = word_embs - word_embs[top_k].mean(dim=-2)

    embs = torch.cat([special_embs, word_embs]).numpy()
    embs = norm_emb(embs) if applys_norm else embs

    return embs
