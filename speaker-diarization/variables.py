segLen = 1
frameRate = 25

n_mfcc = 20
numMix = 128
n_mels = 40
n_clusters = 3
reg_covar = 1e-5
affinity = 'euclidean'
nth_component = 20

sample_rate = 16000
hop_length = int(sample_rate/frameRate)

seed = 1234

gmm_weights = "weights/gmm.sav"
scalar_weights = "weights/scalar.sav"
clustering_weights = "weights/clustering.sav"

save_file_name = "result generation/recordings/{}"

template_path = 'result generation/pdf generation/template.docx'
output_dir = "result generation/pdf generation/{}/"
student = 'Isuru'

Algorithm_flag = 1
Algorithm = 'Spectral Clustering' if Algorithm_flag else 'GMM'