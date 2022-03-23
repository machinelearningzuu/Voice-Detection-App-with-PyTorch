from models import *
from ops import SegmentFrame, speakerdiarisationdf,summary

seed_everything(seed)

def run(wavFile):
    if not Algorithm_flag:
        print("Prediction using GMM")
        y, sr = librosa.load(wavFile)
        duration_speech = librosa.get_duration(y=y, sr=sr)

        vad, mfcc = feature_extraction(wavFile, False)
        GMMfeatures = ExtractGMMfeatures(vad, mfcc, nth_component)
        GMMfeaturesNormalized = feature_normalization(GMMfeatures)
        cluster_prediction = HierarchicalClustering(n_clusters, affinity, GMMfeaturesNormalized)

        frameClust = SegmentFrame(cluster_prediction, segLen, frameRate, mfcc.shape[0])

        pass1hyp = -1*np.ones(len(vad))
        pass1hyp[vad] = frameClust
        speakerdf=speakerdiarisationdf(pass1hyp, frameRate, wavFile, duration_speech)

        speakerdf["TimeSeconds"]=speakerdf.EndTime-speakerdf.StartTime
        speakerdf['SpeakerLabel']=speakerdf['SpeakerLabel'].apply(lambda x: '{} {}'.format(x.split(' ')[0], int(x.split(' ')[1]) + 1))

        speakerdf = speakerdf.loc[speakerdf.TimeSeconds > 0 ]
        speakerdf = speakerdf.sort_values(by=['StartTime'])

    else:
        print("Prediction using Spectral Clustering")
        speakerdf = ResembyzerClustering(wavFile, n_clusters)

    summarydf = summary(speakerdf)

    return speakerdf, summarydf