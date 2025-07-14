# Feature Extraction code for the base SALSA and SALSA-Lite features are adapted from the official repo- https://github.com/thomeou/SALSA
import librosa
import numpy as np

# ============================
# Feature Extraction Constants
# ============================
FS = 24000
N_FFT = 512
HOP_LEN = 300

# DOA Frequency Bounds
FMIN_DOA = 50
FMAX_DOA = 4000
D_MAX = 0.08    # maximum mic spacing (m)
C = 343.0       # speed of sound (m/s)
F_ALIAS = C / (2 * D_MAX)
FMAX_DOA = min(FMAX_DOA, FS // 2, F_ALIAS)

# STFT Bins
N_BINS = N_FFT // 2 + 1
LOWER_BIN = max(1, int(FMIN_DOA * N_FFT / FS))
UPPER_BIN = int(FMAX_DOA * N_FFT / FS)
CUTOFF_BIN = int(9000 * N_FFT / FS)

# SALSA-Lite parameters
DELTA = 2 * np.pi * FS / (N_FFT * C)
FREQ_VECTOR = np.arange(N_FFT // 2 + 1)
FREQ_VECTOR[0] = 1
FREQ_VECTOR = FREQ_VECTOR[:, None, None]

# SALSA-based Window for 200 bins
W = np.zeros((200, N_BINS), dtype=np.float32)
# First 192 is one-to-one mapping
for i in np.arange(192):
    W[i, i + 1] = 1.0
# Last 8 is averaged
for i in np.arange(192, 200):
    start = 193 + (i - 192) * 8
    length = 8 if i < 199 else 7
    W[i, start:start + length] = 1.0 / 8
N_SALSA_BINS = W.shape[0]

# ======================
# Utility Functions
# ======================
def stft_multi_channel(audio: np.ndarray) -> np.ndarray:
    """
    Compute STFT for each channel in shape (freq, time, channels).
    """
    n_ch, n_samples = audio.shape
    n_frames = int(n_samples / HOP_LEN)
    specs = []
    for ch in range(n_ch):
        S = librosa.stft(y=np.asfortranarray(audio[ch]), n_fft=N_FFT, hop_length=HOP_LEN,
                         win_length=N_FFT, window='hann', pad_mode='reflect')[:, :n_frames]
        specs.append(S)
    return np.stack(specs, axis=-1)  # (freq, time, channels)


def compute_msc_recursive(X, lambda_=0.8):
    """
    Compute the Mean Magnitude-Squared Coherence (MMSC) using recursive time averaging.

    The MMSC provides a smoothed estimate of the MSC over time, leveraging a forgetting factor
    to give more weight to recent frames.

    Parameters:
        X (ndarray): Multichannel time-frequency representation of shape (M, T, F).
        lambda_ (float, optional): Forgetting factor for recursive averaging, 0 < lambda_ < 1.

    Returns:
        gamma_avg (ndarray): Mean magnitude-squared coherence averaged over all channel pairs
                             for each TF bin, shape (T, F).
    """
    if X.ndim != 3:
        raise ValueError(f"Input X must be a 3D array of shape (M, T, F), but got shape {X.shape}.")
    if not (0 < lambda_ < 1):
        raise ValueError(f"Parameter lambda_ must be in the interval (0, 1), but got lambda_={lambda_}.")

    M, T, F = X.shape
    num_pairs = M * (M - 1) // 2
    gamma_avg = np.zeros((T, F), dtype=np.float32)

    # Generate indices for all unique microphone pairs (i < j)
    i_indices, j_indices = np.triu_indices(M, k=1)

    # Initialize recursive variables for auto-power and cross-power spectral densities
    S_ii = np.zeros((M, F), dtype=np.complex64)             # Auto-power spectral densities
    S_ij = np.zeros((num_pairs, F), dtype=np.complex64)     # Cross-power spectral densities

    # Iterate over each time frame to compute recursive averages
    for t in range(T):
        # Extract the current time frame across all channels and frequencies
        X_t = X[:, t, :]            # Shape: (M, F)
        X_t_conj = np.conj(X_t)     # Shape: (M, F)

        # Update auto-power spectral densities with recursive averaging
        S_ii = lambda_ * S_ii + (1 - lambda_) * (X_t * X_t_conj)

        # Update cross-power spectral densities for each microphone pair with recursive averaging
        Xi = X_t[i_indices, :]              # Shape: (num_pairs, F)
        Xj_conj = X_t_conj[j_indices, :]    # Shape: (num_pairs, F)
        S_ij = lambda_ * S_ij + (1 - lambda_) * (Xi * Xj_conj)

        # Compute the numerator and denominator for MSC
        numerator = np.abs(S_ij) ** 2
        denominator = S_ii[i_indices, :] * S_ii[j_indices, :]

        # Compute MSC for each microphone pair and frequency bin
        gamma_ij = numerator / denominator

        # Average MSC across all microphone pairs for the current time frame
        gamma_avg[t, :] = np.mean(gamma_ij.real, axis=0)

    return gamma_avg


# =================================
# SALSA variants feature extraction
# =================================
def get_salsa_dlite(wav_path: str, alpha_snr: float = 2.0, lambda_mmsc: float = 0.8, 
                    n_hopfreq: int = 3, n_hopframes: int = 3) -> np.ndarray:
    """
    Extracts the SALSA-DLite feature set from an audio file for the MIC format. This feature set includes:
    - SALSA-Lite features: Log-linear Spectrograms and Normalized Interaural Phase Differences (NIPDs)
    - Coherence and Direct-Path Dominance: Calculated as gamma_hat * rho_hat
    
    Args:
        wav_path (str): Filepath of the `.wav` audio file.
        alpha_snr (float): Alpha SNR to determine the Magitude Test.
        lambda_mmsc (float): Recursive threshold value to calculate the Mean Magnitude Squared Coherence.
        n_hopfreq (int): T_f for frequency averaging for the covariance matrix.
        n_hopframes (int): T_w for the time averaging for the covariance matrix.

    Returns:
        np.ndarray: (n_feature_channels, time_bins, freq_bins)
    """

    # Load audio data
    audio_data, _ = librosa.load(wav_path, sr=FS, mono=False, dtype=np.float32)

    # Compute STFT for all channels
    stfts = stft_multi_channel(audio=audio_data)

    # Extract SALSA-Lite features
    salsalite_feat = _get_salsalite(audio_path=wav_path) 

    # Select frequency bins and transpose for further processing
    X = stfts[1:CUTOFF_BIN, :, :]  # (Freq, Time, Channels)
    n_bins, n_frames, n_chans = X.shape
    Y = X.T  # (Channels, Time, Freq)

    # Compute average MSC (Magnitude Squared Coherence)
    gamma_avg = compute_msc_recursive(Y, lambda_=lambda_mmsc)  # (Time, Freq)

    # Noise floor tracking parameters
    n_sig_frames = 3
    indicator_countdown = np.full((n_bins,), n_sig_frames, dtype=int)
    alpha = 0.02
    slow_scale = 0.1
    floor_up = 1 + alpha
    floor_up_slow = 1 + slow_scale * alpha
    floor_down = 1 - alpha

    # Pad X for autocorrelation computation
    X = np.pad(X, ((n_hopfreq, n_hopfreq), (n_hopframes, n_hopframes), (0, 0)), 'wrap')

    # Initialize signal magnitude spectrogram
    signal_magspec = np.zeros((n_bins, n_frames))
    n_autocorr_frames = 3

    # Compute autocorrelation-based signal magnitude
    for iframe in np.arange(n_autocorr_frames):
        signal_magspec = signal_magspec + np.abs(X[n_hopfreq:n_bins + n_hopfreq, n_hopframes - iframe:n_hopframes - iframe + n_frames, 0]) ** 2
    signal_magspec = np.sqrt(signal_magspec / n_autocorr_frames)

    # Initialize noise floor based on initial frames
    noise_floor = 0.5 * np.mean(signal_magspec[:, :5], axis=1)  # Shape: (n_bins,)

    # Initialize CDPD spectrogram
    cdpd_spec = np.zeros((n_bins, n_frames))
    rho_default = 1.0 / n_chans

    # =========================================================================
    for iframe in np.arange(n_hopframes, n_frames + n_hopframes):
        # get current frame tracking singal
        xfmag = signal_magspec[:, iframe - n_hopframes]
        # ---------------------------------------------------------------------
        # bg noise tracking: implement direct up/down noise floor tracker
        above_noise_idx = xfmag > noise_floor
        # ------------------------------------
        # if signal above noise floor
        indicator_countdown[above_noise_idx] = indicator_countdown[above_noise_idx] - 1
        negative_indicator_idx = indicator_countdown < 0
        # update noise slow for bin above noise and negative indicator
        an_ni_idx = np.logical_and(above_noise_idx, negative_indicator_idx)
        noise_floor[an_ni_idx] = floor_up_slow * noise_floor[an_ni_idx]
        # update noise for bin above noise and positive indicator
        an_pi_idx = np.logical_and(above_noise_idx, np.logical_not(negative_indicator_idx))
        noise_floor[an_pi_idx] = floor_up * noise_floor[an_pi_idx]
        # reset indicator counter for bin below noise floor
        indicator_countdown[np.logical_not(above_noise_idx)] = n_sig_frames
        # reduce noise floor for bin below noise floor
        noise_floor[np.logical_not(above_noise_idx)] = floor_down * noise_floor[np.logical_not(above_noise_idx)]
        # make sure noise floor does not go to 0
        noise_floor[noise_floor < 1e-6] = 1e-6
        # --------------------------------------
        # select TF bins above noise level
        indicator_sig = xfmag > (alpha_snr * noise_floor)
        # ---------------------------------------------------------------------
        # valid bin after onset and noise background tracking
        valid_bin = indicator_sig

        # ---------------------------------------------------------------------
        for ibin in np.arange(n_bins):
            if valid_bin[ibin]: # Pass magnitude test

                # Compute covariance matrix for the current bin and its neighbors
                padded_bin = ibin + n_hopfreq
                X2 = X[padded_bin - n_hopfreq : padded_bin + n_hopfreq + 1, iframe - n_hopframes:iframe + n_hopframes + 1, :] # Shape: (2*n_hopfreq + 1, 2*n_hopframes + 1, n_chans)
                X2 = X2.reshape(-1, n_chans) # Shape: ((2*n_hopfreq + 1)*(2*n_hopframes + 1), n_chans)

                # Compute normalized covariance matrix
                Rxx1 = np.dot(X2.T, X2.conj()) / float((2 * n_hopframes + 1) * (2 * n_hopfreq + 1))

                # Perform Singular Value Decomposition (SVD) 
                s = np.linalg.svd(Rxx1, compute_uv=False)   # s: n_chans

                # Calculate the normalized singular value ratio
                rho_tf = s[0] / (np.sum(s))

                # Retrieve the average MSC value
                gamma_tf = gamma_avg[iframe - n_hopframes, ibin]

                # Compute DPD for the current bin and frame
                cdpd_tf = rho_tf * gamma_tf
                cdpd_spec[ibin, iframe - n_hopframes] = cdpd_tf
            else:
                # Default DPD value when the bin does not pass the magnitude test
                rho_tf = rho_default
                gamma_tf = gamma_avg[iframe - n_hopframes, ibin]
                cdpd_tf = rho_tf * gamma_tf
                cdpd_spec[ibin, iframe - n_hopframes] = cdpd_tf

    # Concatenation of features
    cdpd_spec = np.expand_dims(np.sqrt(cdpd_spec), axis=0).transpose((0, 2, 1)) # Channels, Time, Frequency
    full_spec = np.concatenate((salsalite_feat[:4], cdpd_spec, salsalite_feat[4:]), axis=0) # Channels, Time, Frequency

    # Check for NaN values in the feature set
    if np.isnan(full_spec).any():
        raise RuntimeError(f"SALSA-DLite produced NaNs for {wav_path}")

    return full_spec


def _get_salsa(audio_path, add_cdpd: bool = False, alpha_snr: float = 2.0, beta_coherence: float = 3.0, 
               lambda_mmsc: float = 0.8, n_hopfreq: int = 3, n_hopframes: int = 3) -> np.ndarray:
    """
    Extracts full-scale SALSA features and optionally CDPD from an audio file.

    Parameters:
        audio_path (str): Filepath of the `.wav` audio file.
        add_cdpd (bool): Whether to include the CDPD feature.
        alpha_snr (float): Alpha SNR to determine the Magitude Test.
        beta_coherence (float): Beta_DRR that is used for SALSA EPV computation.
        lambda_mmsc (float): Recursive threshold value to calculate the Mean Magnitude Squared Coherence.
        n_hopfreq (int): T_f for frequency averaging for the covariance matrix.
        n_hopframes (int): T_w for the time averaging for the covariance matrix.

    Returns:
        np.ndarray: (n_feature_channels, time_bins, freq_bins)
    """

    # Load audio data
    audio_data, _ = librosa.load(audio_path, sr=FS, mono=False, dtype=np.float32)

    # Getting log-linear spectrograms
    stfts = stft_multi_channel(audio=audio_data)
    power = np.abs(stfts) ** 2
    power = np.tensordot(W, power, axes=(1,0))
    log_specs = librosa.power_to_db(power, ref=1.0, amin=1e-10, top_db=None)
    log_features = log_specs.transpose(2, 1, 0)  # Shape: (Channels, Time, Freq)

    # Prepare for SALSA and CDPD computation
    X = stfts[LOWER_BIN:(N_SALSA_BINS + LOWER_BIN), :, :]
    n_bins, n_frames, n_chans = X.shape
    Y = X.T     # (ch, time, freq)
    gamma_avg = compute_msc_recursive(Y, lambda_=lambda_mmsc)

    # noise floor tracking params
    n_sig_frames = 3
    indicator_countdown = np.full((n_bins,), n_sig_frames, dtype=int)
    alpha = 0.02
    slow_scale = 0.1
    floor_up = 1 + alpha
    floor_up_slow = 1 + slow_scale * alpha
    floor_down = 1 - alpha
    c = 343  # Speed of sound in m/s
    delta = 2 * np.pi * FS / (N_FFT * c)

    # --------- Padding for Autocorrelation and Covariance Computations ---------
    X = np.pad(X, ((n_hopfreq, n_hopfreq), (n_hopframes, n_hopframes), (0, 0)), 'wrap')

    # --------- Signal Magnitude Spectrogram Computation ---------
    signal_magspec = np.zeros((n_bins, n_frames))
    n_autocorr_frames = 3
    for iframe in np.arange(n_autocorr_frames):
        signal_magspec = signal_magspec + np.abs(X[n_hopfreq:n_bins + n_hopfreq, n_hopframes - iframe:n_hopframes - iframe + n_frames, 0]) ** 2
    signal_magspec = np.sqrt(signal_magspec / n_autocorr_frames)

    # --------- Initial Noise Floor Estimation ---------
    noise_floor = 0.5 * np.mean(signal_magspec[:, :5], axis=1)  # Shape: (n_bins,)

    # --------- Initialize Output Matrices ---------
    normalized_eigenvector_mat = np.zeros((n_chans - 1, n_bins, n_frames))  # normalized eigenvector of ss tf bin
    cdpd_spec = np.zeros((n_frames, n_bins)) # DRR Spectrogram
    rho_default = 1.0 / n_chans

    # =========================================================================
    for iframe in np.arange(n_hopframes, n_frames + n_hopframes):
        # get current frame tracking singal
        xfmag = signal_magspec[:, iframe - n_hopframes]
        # ---------------------------------------------------------------------
        # bg noise tracking: implement direct up/down noise floor tracker
        above_noise_idx = xfmag > noise_floor
        # ------------------------------------
        # if signal above noise floor
        indicator_countdown[above_noise_idx] = indicator_countdown[above_noise_idx] - 1
        negative_indicator_idx = indicator_countdown < 0
        # update noise slow for bin above noise and negative indicator
        an_ni_idx = np.logical_and(above_noise_idx, negative_indicator_idx)
        noise_floor[an_ni_idx] = floor_up_slow * noise_floor[an_ni_idx]
        # update noise for bin above noise and positive indicator
        an_pi_idx = np.logical_and(above_noise_idx, np.logical_not(negative_indicator_idx))
        noise_floor[an_pi_idx] = floor_up * noise_floor[an_pi_idx]
        # reset indicator counter for bin below noise floor
        indicator_countdown[np.logical_not(above_noise_idx)] = n_sig_frames
        # reduce noise floor for bin below noise floor
        noise_floor[np.logical_not(above_noise_idx)] = floor_down * noise_floor[np.logical_not(above_noise_idx)]
        # make sure noise floor does not go to 0
        noise_floor[noise_floor < 1e-6] = 1e-6
        # --------------------------------------
        # select TF bins above noise level
        indicator_sig = xfmag > (alpha_snr * noise_floor)
        # ---------------------------------------------------------------------
        # valid bin after onset and noise background tracking
        valid_bin = indicator_sig
        # ---------------------------------------------------------------------
        # --------- Coherence Testing and Feature Extraction ---------
        for ibin in np.arange(n_bins):
            if valid_bin[ibin]:

                # Compute covariance matrix for the current bin and its neighbors
                padded_bin = ibin + n_hopfreq # i.e. where our bin is in the padded matrix
                X2 = X[padded_bin - n_hopfreq : padded_bin + n_hopfreq + 1, iframe - n_hopframes:iframe + n_hopframes + 1, :]  # Shape: (2*n_hopfreq + 1, 2*n_hopframes + 1, n_chans)
                X2 = X2.reshape(-1, n_chans)  # Shape: ((2*n_hopfreq + 1)*(2*n_hopframes + 1), n_chans)
                # Compute normalized covariance matrix
                Rxx1 = np.dot(X2.T, X2.conj())
                Rxx1 = Rxx1 / float((2 * n_hopframes + 1) * (2 * n_hopfreq + 1))

                # Perform Singular Value Decomposition (SVD)
                # u: n_chans x n_chans, s: n_chans, columns of u is the singular vectors
                u, s, v = np.linalg.svd(Rxx1)

                # --------- CDPD Feature Extraction ---------
                if add_cdpd:
                    rho_tf = s[0] / np.sum(s)  # Ratio of the largest singular value to the sum of all singular values
                    gamma_tf = gamma_avg[iframe - n_hopframes, ibin]  # Coherence value
                    cdpd_spec[iframe - n_hopframes, ibin] = rho_tf * gamma_tf  # DRR

                # --------- SALSA Feature Extraction ---------
                # coherence test
                if s[0] > s[1] * beta_coherence:
                    indicator_rank1 = True
                else:
                    indicator_rank1 = False

                # Update valid bin based on coherence test
                valid_bin[ibin] = valid_bin[ibin] and indicator_rank1

                # compute doa spectrum
                if valid_bin[ibin]:
                    # normalize largest eigenvector
                    normed_eigenvector = np.angle(u[1:, 0] * np.conj(u[0, 0]))  # get the phase difference
                    # normalized for the frequency and delta
                    normed_eigenvector = normed_eigenvector / (delta * (ibin + LOWER_BIN))
                    # save output
                    normalized_eigenvector_mat[:, ibin, iframe - n_hopframes] = normed_eigenvector
            else:
                if add_cdpd:
                    rho_tf = rho_default # Assign default DRR value for invalid bins
                    gamma_tf = gamma_avg[iframe - n_hopframes, ibin]
                    cdpd_spec[iframe - n_hopframes, ibin] = rho_tf * gamma_tf

    # --------- Construct Full Eigenvector Matrix ---------
    full_eigenvector_mat = np.zeros((n_chans - 1, n_frames, N_SALSA_BINS))
    full_eigenvector_mat[:, :, :(UPPER_BIN - LOWER_BIN)] = np.transpose(normalized_eigenvector_mat[:, :(UPPER_BIN - LOWER_BIN), :], (0, 2, 1))

    if add_cdpd:
        cdpd_final = np.zeros((n_frames, N_SALSA_BINS))
        cdpd_final[:, :N_SALSA_BINS] = cdpd_spec
        cdpd_final[:, 192:] = 0 # Zero off above spatial aliasing frequency
        full_cdpd_final = np.expand_dims(np.sqrt(cdpd_final), axis=0)
        salsa_audio_feature = np.concatenate((log_features, full_cdpd_final, full_eigenvector_mat), axis = 0)
    else:
        salsa_audio_feature = np.concatenate((log_features, full_eigenvector_mat), axis=0)

    return salsa_audio_feature



def _get_salsalite(audio_path):
    # Load audio data
    audio_data , _ = librosa.load(audio_path, sr=FS, mono=False, dtype=np.float32)

    stfts = stft_multi_channel(audio_data)
    power = np.abs(stfts) ** 2
    log_specs = librosa.power_to_db(power, ref=1.0, amin=1e-10)
    log_specs = log_specs.transpose(2, 1, 0)    # (ch, time, freq)

    # Compute spatial feature (NIPD)
    phase_vector = np.angle(stfts[:, :, 1:] * np.conj(stfts[:, :, 0, None]))
    phase_vector = phase_vector / (DELTA * FREQ_VECTOR)
    phase_vector = np.transpose(phase_vector, (2, 1, 0))  # (n_mics, n_frames, n_bins)

    # Crop frequency
    log_specs = log_specs[:, :, LOWER_BIN:CUTOFF_BIN]
    phase_vector = phase_vector[:, :, LOWER_BIN:CUTOFF_BIN]
    phase_vector[:, :, UPPER_BIN:] = 0

    return np.concatenate((log_specs, phase_vector), axis=0) # Stack features



if __name__ == "__main__":
    print(f"Lower Bin: {LOWER_BIN}, Upper Bin: {UPPER_BIN}, Cutoff Bin: {UPPER_BIN}, SALSA-bins: {N_SALSA_BINS}")
    sample_audio_fp = "test_track.wav"
    
    salsalite_feat = _get_salsalite(sample_audio_fp)
    salsadlite_feat = get_salsa_dlite(sample_audio_fp)
    salsa_feat = _get_salsa(sample_audio_fp)
    salsad_feat = _get_salsa(sample_audio_fp)

    print(f"SALSA-Lite: {salsalite_feat.shape}, SALSA-DLite: {salsadlite_feat.shape}, SALSA: {salsa_feat.shape}, SALSA-D: {salsad_feat.shape}")
    
    print(np.__version__, librosa.__version__)