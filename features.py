##################################################################################################
import numpy             as np
import scipy             as sci
import scipy.fftpack     as fftpack
from   scipy.fftpack     import dct
from   scipy.io.wavfile  import read
import soundfile as sf
##################################################################################################
def load_file(path):
    sig, sample_rate = sf.read(path)
    
    if(sig.ndim > 1):
        sig = stereo_to_mono(sig)
	
    return(sig, sample_rate)

def stereo_to_mono(audio_signal):
    mono = np.amax(audio_signal,axis=1)
    return mono

def pre_emphasis(signal, pre_enfasis_coeff = 0.97):
    emphasized_signal = np.append(signal[0], signal[1:] - pre_enfasis_coeff * signal[:-1])
    return emphasized_signal

def time_to_samples(n_segs, sample_rate):
    n_samples_total = n_segs*sample_rate
    n_samples = 1
    while n_samples < n_samples_total:
        n_samples *= 2
    return n_samples

def framing(signal, frame_len = 512, frame_hop = 128):
    signal_length = len(signal)
    num_frames    = int(np.ceil(float(np.abs(signal_length - frame_len)) / frame_hop))
    pad_signal_length = num_frames * frame_hop + frame_len
        
    z = np.zeros((pad_signal_length - signal_length), dtype=np.float64)
    pad_signal = np.append(signal, z)
        
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(np.arange(0, (num_frames) * frame_hop, frame_hop), (frame_len, 1)).T
    frames  = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frames.shape[1])
    return frames

def registers_needed(total_time = 0.5, frame_size_time = 0.025, frame_hop_time = 0.01):
    numero_registros = 1
    acumulado = frame_size_time
    while acumulado < total_time:
        acumulado += frame_hop_time
        numero_registros += 1
    return(numero_registros)
##################################################################################################
def power_spectrum(frames, nfft):
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))               # Magnitude of the FFT
    pow_frames = ((1.0 / nfft) * ((mag_frames) ** 2))
    return(pow_frames)

def calculate_nfft(window_length):
    nfft = 1
    while nfft < window_length:
        nfft *= 2
    return nfft

def mel_filter_bank(sample_rate, nfilt, nfft):
    low_freq_mel  = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))    # Convert Hz to Mel
    mel_points    = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points     = (700 * (10**(mel_points / 2595) - 1))                 # Convert Mel to Hz
    bin           = np.floor((nfft + 1) * hz_points / sample_rate)
    
    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])                                   # left
        f_m       = int(bin[m])                                       # center
        f_m_plus  = int(bin[m + 1])                                   # right
        
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    base   = 2.0 / (bin[2:nfilt + 2] - bin[:nfilt])
    fbank *= base[:, np.newaxis]
    return(fbank)

def mel_filter_bank_spectrum(pow_frames, fbank):
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)                                     # dB
    return(filter_banks)

def discrete_cosine_transform(filter_banks, num_ceps=12):
	dc_transform = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2 - num_ceps+1
	return(dc_transform)

def liftering(mfccs, cep_lifter):
    (nframes, ncoeff) = mfccs.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfccs *= lift
    return(mfccs)

def delta(mfcc_frames):
    deltas     = (mfcc_frames[0:len(mfcc_frames)-2,:] - mfcc_frames[2:,:])/2
    new_frames = mfcc_frames[1:len(mfcc_frames) - 1,:]
    coeffs     = np.concatenate((new_frames, deltas),axis=1)
    return(coeffs)

def d_delta(mfcc_frames):
    deltas     = (mfcc_frames[0:len(mfcc_frames)-2,:] - mfcc_frames[2:,:])/2
    d_deltas   = (deltas[0:len(deltas)-2,:] - deltas[2:,:])/2
    new_frames = mfcc_frames[2:len(mfcc_frames) - 2,:]
    new_deltas = deltas[1:len(deltas) - 1,:]
    coeffs     = np.concatenate((new_frames,new_deltas),axis=1)
    coeffs     = np.concatenate((coeffs,d_deltas),axis=1)
    return(coeffs)
###############################################################################################################################################
#MFCB y MFCC
def mfb(signal_path, pre_emph_coeff = 0.97, frame_size_time = 0.025, frame_hop_time = 0.01, nfilt = 40):
    signal, sample_rate = load_file(signal_path)
    pre_emph            = pre_emphasis(signal, pre_emph_coeff)
    frame_size_samples  = time_to_samples(frame_size_time, sample_rate)
    frame_hop_samples   = time_to_samples(frame_hop_time,  sample_rate)
    frames              = framing(pre_emph, frame_size_samples, frame_hop_samples)
    nfft                = calculate_nfft(frames.shape[1])
    pow_spect           = power_spectrum(frames, nfft)
    mfb                 = mel_filter_bank(sample_rate, nfilt, nfft)
    mfb_spectrum        = mel_filter_bank_spectrum(pow_spect, mfb)
    return(mfb_spectrum)

def mfcc(signal_path, pre_emph_coeff = 0.97, frame_size_time = 0.025, frame_hop_time = 0.01, nfilt = 40, num_ceps = 12, cep_lifter = 22, deltas = 0, mfb = ''):
	if(mfb == ''):
		signal, sample_rate = load_file(signal_path)
		pre_emph            = pre_emphasis(signal, pre_emph_coeff)
		frame_size          = time_to_samples(frame_size_time, sample_rate)
		frame_hop           = time_to_samples(frame_hop_time,  sample_rate)
		frames              = framing(pre_emph, frame_size, frame_hop)
		nfft                = calculate_nfft(frames.shape[1])
		pow_spect           = power_spectrum(frames, nfft)
		mfb                 = mel_filter_bank(sample_rate, nfilt, nfft)
		mfb_spectrum        = mel_filter_bank_spectrum(pow_spect, mfb)
	else:
		mfb_spectrum        = mfb

	mfcc = discrete_cosine_transform(mfb_spectrum, num_ceps)
    
	if(cep_lifter > 0):
		mfcc  = liftering(mfcc, cep_lifter)
    
	if(deltas == 0):
		coeffs  = mfcc
	elif(deltas == 1):
		coeffs  = delta(mfcc)
	elif(deltas == 2):
		coeffs  = d_delta(mfcc)
	else:
		raise ValueError('Error en el valor de Deltas, no soporta valores mayores a 2.')
	return(coeffs)
#######################################################################################################
def lpc_to_cepstral(coeffs,n=13):
    c      = np.zeros([len(coeffs),n])
    c[:,0] = np.log(coeffs.shape[1])
    c[:,1] = coeffs[:,0]
    
    for i in range(2,coeffs.shape[1]):
        c[:,i] += coeffs[:,i - 1]
        for j in range(1,i):
            c[:,i] += (j/i)*c[:,j]*coeffs[:,i - 1 - j]
    for i in range(coeffs.shape[1], n):
        for j in range(i-coeffs.shape[1],i):
            c[:,i] = (j/i)*c[:,j]*coeffs[:,i - 1 - j]
    return(c)

def lpc_coeffs(frames, orden_p):
	lpc = np.zeros([frames.shape[0], orden_p])
	G   = np.zeros(frames.shape[0])
	for i in range(lpc.shape[0]):
		#print('fit vector ' + str(i) + ' ...')
		lpc[i], G[i] = lpc_fit(frames[i], orden_p)
	return(lpc, G)

def lpc_fit(serie, p=10):
    ac  = autocorr(serie, p+1)
    R   = sci.linalg.toeplitz(ac[:p])
    r   = ac[1:p+1]
    phi = sci.linalg.inv(R).dot(r)
    G   = gain(ac,phi)
    return(phi, G)

def gain(autocorr, coeffs):
    G = np.sqrt(autocorr[0] + np.dot(autocorr[1:],coeffs))
    return(G)

def autocorr(serie, lag=10):
	c    = np.correlate(serie, serie, 'full')
	mid  = len(c)//2
	acov = c[mid:mid+lag]
	acor = acov/acov[0]
	return(acor)
#######################################################################################################
#LPC
def lpc(signal_path, pre_emph_coeff = 0.97, frame_size_time = 0.025, frame_hop_time = 0.01, orden_p = 10, gain = 0):
	signal, sample_rate = load_file(signal_path)
	pre_emph            = pre_emphasis(signal, pre_emph_coeff)
	frame_size          = time_to_samples(frame_size_time, sample_rate)
	frame_hop           = time_to_samples(frame_hop_time,  sample_rate)
	frames              = framing(pre_emph, frame_size, frame_hop)
	lpc, G              = lpc_coeffs(frames, orden_p)
	if(gain == 1):
		G = G[:, np.newaxis]
		lpc = np.concatenate((lpc,G),axis=1)
	elif(gain > 1):
		raise ValueError('Error en el valor de gain, solo toma valores 0 o 1')
	return(lpc)

