from sklearn.mixture import *
from sklearn.cluster import AgglomerativeClustering

from helper import *

def GMMfit(mfcc, nth_component, covariance_type = 'full'):
    gmm = GaussianMixture(
                        n_components = nth_component, 
                        covariance_type = covariance_type, 
                        reg_covar = reg_covar,
                        random_state = seed
                        )
    return gmm.fit(mfcc)

def AHCfit(n_clusters, affinity, linkage, X_normalized):
    Hcluster = AgglomerativeClustering(
                                n_clusters = n_clusters, 
                                affinity = affinity, 
                                linkage = linkage
                                    )
    return Hcluster.fit_predict(X_normalized)

def SCALARfit(GMMfeatures):
    scaler = StandardScaler()
    return scaler.fit(GMMfeatures)

def feature_normalization(GMMfeatures):
    # if not os.path.exists(scalar_weights):
    #     print("Saving Scalar Features")
    #     scaler = SCALARfit(GMMfeatures)
    #     save_weights(scalar_weights, scaler)
    # else:
    #     print("Loading Scalar Features")
    #     scaler = load_weights(scalar_weights)
        
    scaler = SCALARfit(GMMfeatures)
    X_scaled = scaler.transform(GMMfeatures)  
    X_normalized = normalize(X_scaled)
    return X_normalized

def ExtractGMMfeatures(vad, mfcc, nth_component):    
    # if not os.path.exists(gmm_weights):
    #     GMM = GMMfit(mfcc, numMix, 'diag')
    #     save_weights(gmm_weights, GMM)
    #     print("Saving GMM High Level Features")
    # else:
    #     print("Loading GMM High Level Features")
    #     GMM = load_weights(gmm_weights)
    
    GMM = GMMfit(mfcc, nth_component, 'diag')
    
    segLikes = []
    segSize = frameRate*segLen
    len_mfcc = float(mfcc.shape[0]) 
    all_segments = len_mfcc / segSize
    all_segments = int(np.ceil(all_segments))
    for ith_segment in range(all_segments):
        ith_segment_start = ith_segment*segSize
        ith_segment_end = (ith_segment+1)*segSize
        if ith_segment_end > mfcc.shape[0]:
            ith_segment_end = mfcc.shape[0]-1
        if ith_segment_end == ith_segment_start:
            break
        segment = mfcc[ith_segment_start : ith_segment_end, :]
        compLikes = np.sum(GMM.predict_proba(segment),0)
        segLikes.append(compLikes/segment.shape[0])
    return np.asarray(segLikes)

def HierarchicalClustering(n_clusters, affinity, GMMfeaturesNormalized):
    if affinity == 'cosine':
        linkage='complete'
    elif affinity == 'euclidean':
        linkage='ward'
    
    # if not os.path.exists(clustering_weights):
    #     print("Training & Saving AHC Model using GMM extracted features")
    #     Hcluster = AHCfit(n_clusters, affinity, linkage, GMMfeaturesNormalized)
    #     save_weights(clustering_weights, Hcluster)
    # else:
    #     print("Loading Hierarchical CLustering Model ")
    #     Hcluster = load_weights(gmm_weights)
        
    cluster_prediction = AHCfit(n_clusters, affinity, linkage, GMMfeaturesNormalized)
    return cluster_prediction