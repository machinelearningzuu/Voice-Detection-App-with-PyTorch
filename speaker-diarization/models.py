import pandas as pd
from sklearn.mixture import *
from sklearn.cluster import AgglomerativeClustering

from Resemblyzer.resemblyzer import preprocess_wav, VoiceEncoder, sampling_rate
from pathlib import Path

from spectralcluster import *
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

def make_resembyzer_diarization(labelling, min_clusters):
    DiarizationDict = {}
    DiarizationDict['SpeakerLabel'] = []
    DiarizationDict['StartTime'] = []
    DiarizationDict['EndTime'] = []
    DiarizationDict['TimeSeconds'] = []
    
    for label in labelling:
        speaker_label ,start_time ,time, duration = label
        DiarizationDict['SpeakerLabel'].append('Person {}'.format(int(speaker_label) + 1))
        DiarizationDict['StartTime'].append(start_time)
        DiarizationDict['EndTime'].append(time)
        DiarizationDict['TimeSeconds'].append(duration)
    
    speakerdf = pd.DataFrame(DiarizationDict)
    speaker_ids = speakerdf.SpeakerLabel.str[-1].astype(int).values

    order_dict = {}
    speaker_id_new = 1
    for speaker_id in speaker_ids:
        if not (speaker_id in order_dict):
            order_dict[speaker_id] = speaker_id_new
            speaker_id_new += 1 
        if len(order_dict) == min_clusters:
            break

    speaker_ids_updated = ['Person {}'.format(order_dict[speaker_id]) for speaker_id in speaker_ids]
    speakerdf.SpeakerLabel = np.array(speaker_ids_updated)
    
    speakerdf = speakerdf.loc[speakerdf.TimeSeconds >= 2]
    speakerdf.reset_index(drop=True, inplace=True)
    return speakerdf

def ResembyzerClustering(
                    WavFile, 
                    min_clusters,
                    rate = {
                        2 : 1.6,
                        3 : 1.2,
                        4 : 1.0,
                        5 : 1.0
                            }):

    wav_fpath = Path(WavFile)
    wav = preprocess_wav(wav_fpath)
    encoder = VoiceEncoder("cuda")
    try:
        _, cont_embeds, wav_splits = encoder.embed_utterance(
                                                            wav, 
                                                            return_partials=True, 
                                                            rate=rate[min_clusters]
                                                            )
    except:
        print("Define a Rate for Given Minimum Number of Clusters")

    refinement_options = RefinementOptions(
                                gaussian_blur_sigma=1,
                                p_percentile=0.9,
                                thresholding_soft_multiplier=0.01,
                                thresholding_type=ThresholdType.RowMax,
                                refinement_sequence=ICASSP2018_REFINEMENT_SEQUENCE)

    autotune = AutoTune(
                    p_percentile_min=0.60,
                    p_percentile_max=0.95,
                    init_search_step=0.01,
                    search_level=3
                        )


    clusterer = SpectralClusterer(
                        min_clusters=min_clusters,
                        max_clusters=6,
                        autotune=autotune,
                        laplacian_type=None,
                        refinement_options=refinement_options,
                        custom_dist="cosine"
                            )

    labels = clusterer.predict(cont_embeds)
        
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    labelling = []
    start_time = 0

    for i,time in enumerate(times):
        if i>0 and labels[i]!=labels[i-1]:
            speaker_label = str(labels[i-1])
            duration = round(time - start_time, 3)
            temp = [speaker_label ,start_time ,time, duration]
            labelling.append(tuple(temp))
            start_time = time

        if i==len(times)-1:
            speaker_label = str(labels[i])
            duration = round(time - start_time, 3)
            temp = [speaker_label ,start_time ,time, duration]
            labelling.append(tuple(temp))

    speakerdf = make_resembyzer_diarization(labelling, min_clusters)
    return speakerdf