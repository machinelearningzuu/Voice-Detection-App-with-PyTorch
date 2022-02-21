segLen = 3
n_mfcc = 20
numMix = 50
n_mels = 40
frameRate = 50
n_clusters = 4
reg_covar = 1e-5
affinity = 'euclidean'
nth_component = 20

sample_rate = 16000
hop_length = int(sample_rate/frameRate)

wavFile="data/3 person/3Person_without_noise.wav"
gtFile = "data/3 person/ground_truth.csv" 

seed = 1234

gmm_weights = "weights/gmm.sav"
scalar_weights = "weights/scalar.sav"
clustering_weights = "weights/clustering.sav"

save_file_name = "result generation/recordings/{}"

template_path = 'result generation/pdf generation/template.docx'
output_dir = "result generation/pdf generation/{}/"
student = 'Isuru'