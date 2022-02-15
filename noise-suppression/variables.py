weight_path = 'files/cp-100.pkl'

clean_dir = 'files/store/clean'
noisy_dir = 'files/store/noisy'
enhanced_dir = 'files/store/enhanced'


window_size = 2 ** 14  # about 1 second of samples
stride = 0.5
negative_slope = 0.03
emph_coeff=0.95

epoches = 15
batch_size = 64
sample_rate = 16000

wave_file = 'sachi_airconditioner6.wav' # Change this with tour file