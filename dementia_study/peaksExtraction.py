import numpy as np
import numpy as np
from scipy.signal import find_peaks, peak_prominences


def grand_peak_params(amplitude_array,latency_array,latency_range,variable_name,N100=False,N200=False,N400=False,P300=False):
    """
    Returns the peak amplitude and latency from an amplitude array and a latency array
    """
    if N100 or N200 or N400:
        peak_amplitude = amplitude_array[(latency_array >= latency_range[0]) & (latency_array <= latency_range[1])].min()
        peak_latency = latency_array[(latency_array >= latency_range[0]) & (latency_array <= latency_range[1])][np.argmin(amplitude_array[(latency_array >= latency_range[0]) & (latency_array <= latency_range[1])])]
        peak_amplitude,peak_latency = round(peak_amplitude,3),round(peak_latency,3)
        if peak_amplitude < 0:
            peak__amplitude = peak_amplitude * -1
            peak__latency = peak_latency
            print(f'{variable_name} peak amplitude: {peak__amplitude} | {variable_name} peak latency: {peak__latency}')
            #return peak__amplitude,peak__latency
        if peak_amplitude > 0:
            peak__amplitude = peak_amplitude * -1
            peak__latency = peak_latency
            print(f'{variable_name} peak amplitude: {peak__amplitude} | {variable_name} peak latency: {peak__latency}')
            #return peak__amplitude,peak__latency
    if P300:
        peak_amplitude = amplitude_array[(latency_array >= latency_range[0]) & (latency_array <= latency_range[1])].max()
        peak_latency = latency_array[(latency_array >= latency_range[0]) & (latency_array <= latency_range[1])][np.argmax(amplitude_array[(latency_array >= latency_range[0]) & (latency_array <= latency_range[1])])]
        peak__amplitude,peak__latency = round(peak_amplitude,3),round(peak_latency,3)
        print(f'{variable_name} peak amplitude: {peak__amplitude} | {variable_name} peak latency: {peak__latency}')
    return peak__amplitude,peak__latency


def mean_param_extractor(amplitude_array,latency_array,latency_range,variable_name,single_waveform=False,multiple_waveforms=False,N100=False,N200=False,N400=False,P300=False):
    """
    Returns the mean amplitude and latency from an amplitude array and a latency array
    """

    def mean_params(amplitude_array,latency_array,latency_range,variable_name,N100=False,N200=False,N400=False,P300=False):
        if N100 or N200 or N400:
            mean_amplitude = amplitude_array[(latency_array >= latency_range[0]) & (latency_array <= latency_range[1])]
            mean_amplitude = mean_amplitude * -1
            mean_amplitude = round(np.nanmean(mean_amplitude),3)
            print(f'{variable_name} mean amplitude: {mean_amplitude}')
        if P300:
            mean_amplitude = amplitude_array[(latency_array >= latency_range[0]) & (latency_array <= latency_range[1])]
            mean_amplitude = np.nanmean(mean_amplitude)
            mean_amplitude= round(mean_amplitude,3)
            print(f'{variable_name} mean amplitude: {mean_amplitude}')
        return mean_amplitude

    if single_waveform:
        mean__amplitude = mean_params(amplitude_array,latency_array,latency_range,variable_name,N100,N200,N400,P300)
    if multiple_waveforms:
        mean__amplitude = []
        for i in range(len(amplitude_array)):
            mean__amplitude.append(mean_params(amplitude_array[i],latency_array,latency_range,variable_name,N100,N200,N400,P300))
        mean__amplitude = np.array(mean__amplitude)
        print(f'{variable_name} average of mean amplitudes: {round(np.nanmean(mean__amplitude),3)}')
    return mean__amplitude

def range_param_extractor(amplitude_array,latency_array,latency_range,single_waveform=False,multiple_waveforms=False,N100=False,N200=False,N400=False,P300=False):
    """
    Returns the range of amplitudes and latency from an amplitude array and a latency array
    """

    def range_params(amplitude_array,latency_array,latency_range,N100=False,N200=False,N400=False,P300=False):
        if N100 or N200 or N400:
           range_amplitude = amplitude_array[(latency_array >= latency_range[0]) & (latency_array <= latency_range[1])]
           range_amplitude = range_amplitude * -1

        if P300:
            range_amplitude = amplitude_array[(latency_array >= latency_range[0]) & (latency_array <= latency_range[1])]
        return range_amplitude

    if single_waveform:
        range__amplitude = range_params(amplitude_array,latency_array,latency_range,N100,N200,N400,P300)
    if multiple_waveforms:
        range__amplitude = []
        for i in range(len(amplitude_array)):
            range__amplitude.append(range_params(amplitude_array[i],latency_array,latency_range,N100,N200,N400,P300))
        range__amplitude = np.array(range__amplitude)
        range__amplitude[np.isnan(range__amplitude)] = 0
    return range__amplitude


def BAM(data,time,latency_range,component_direction):
    """
    BAM: Baseline Adjusted Measures
    Once the targeted components were confirmed using mean
    amplitude analysis, adjusted baseline amplitude and peak latency
    were measured for all 3 components in both modalities. Adjusted
    baseline amplitude measures were calculated at any electrodes from peak
    amplitudes relative to the two adjacent peaks of opposite polarity
    (D’Arcy et al., 2011; Ghosh-Hajra et al., 2016a). 

    data: ERP waveform data
    time: ERP waveform time or latency
    component_direction: 'Negative' [N100, N400] or 'Positive' [P300]


    """

    t_start, t_end = latency_range[0], latency_range[1]
    def get_peaks_within_time_range(data, time, t_start, t_end, component,p):
        t_indices = np.where((time >= t_start) & (time <= t_end))[0]

        if component == 'Negative':
            neg_peaks, _ = find_peaks(-data[t_indices], prominence=p)
            prominences = peak_prominences(-data[t_indices], neg_peaks)[0]
            top_two_neg_peaks = neg_peaks[np.argsort(prominences)[-2:]]
            peak_times = time[t_indices][top_two_neg_peaks]
            peak_values = data[t_indices][top_two_neg_peaks]

        if component == 'Positive':
            pos_peaks, _ = find_peaks(data[t_indices], prominence=p)
            prominences = peak_prominences(data[t_indices], pos_peaks)[0]
            top_two_pos_peaks = pos_peaks[np.argsort(prominences)[-2:]]
            peak_times = time[t_indices][top_two_pos_peaks]
            peak_values = data[t_indices][top_two_pos_peaks]

        return peak_times, peak_values

    y, t = data, time

    if component_direction == 'Positive':
        neg_peak_times, neg_peak_values = get_peaks_within_time_range(data, time, t_start, t_end, 'Negative',0.1)
        if neg_peak_times.size != 2:
            p = 0.1
            while neg_peak_times.size != 2:
                neg_peak_times, neg_peak_values = get_peaks_within_time_range(data, time, t_start, t_end, 'Negative',p)
                p -= 0.01
                if p < 0:
                    print('change range')
                    break
        print("Negative peak times (ms):", neg_peak_times)
        print("Negative peak values (uV):", neg_peak_values)

        point1 = (neg_peak_times[0], neg_peak_values[0])
        point2 = (neg_peak_times[1], neg_peak_values[1])
        midpoint = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)
        print("Midpoint (ms,uV):", midpoint)

        if point1[0] < point2[0]:
            y_prime = y[(t >= point1[0]) & (t <= point2[0])]
            t_prime = t[(t >= point1[0]) & (t <= point2[0])]
        else:
            y_prime = y[(t >= point2[0]) & (t <= point1[0])]
            t_prime = t[(t >= point2[0]) & (t <= point1[0])]

        slope_original = (point2[1] - point1[1]) / (point2[0] - point1[0])
        slope_perpendicular = -1 / slope_original
        if midpoint[0] in t_prime:
            t_prime[np.where(t_prime == midpoint[0])] = np.nan
        else:
            pass
        slope_midpoint = (midpoint[1]-y_prime) / (midpoint[0]-t_prime)
        slope_desired = slope_midpoint[np.argmin(np.abs(slope_midpoint - slope_perpendicular))]
        idx_slope_desired = np.argmin(np.abs(slope_midpoint - slope_desired))
        y_adjusted, t_adjusted = y_prime[idx_slope_desired], t_prime[idx_slope_desired]
        y_adjusted, t_adjusted = round(y_adjusted, 3), round(t_adjusted, 3)
        print("Adjusted Amplitude (uV):", y_adjusted)
        print("Adjusted Latency (ms):", t_adjusted)
        print('\n')
        return y_adjusted, t_adjusted

        
    if component_direction == 'Negative':
        pos_peak_times, pos_peak_values = get_peaks_within_time_range(data, time, t_start, t_end, 'Positive',0.1)
        if pos_peak_times.size != 2:
            p = 0.1
            while pos_peak_times.size != 2:
                pos_peak_times, pos_peak_values = get_peaks_within_time_range(data, time, t_start, t_end, 'Positive',p)
                p -= 0.01
                if p < 0:
                    print('change range')
                    break
        print("Positive peak times (ms):", pos_peak_times)
        print("Positive peak values (uV):", pos_peak_values)

        point1 = (pos_peak_times[0], pos_peak_values[0])
        point2 = (pos_peak_times[1], pos_peak_values[1])
        midpoint = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)
        print("Midpoint (ms,uV):", midpoint)

        if point1[0] < point2[0]:
            y_prime = y[(t >= point1[0]) & (t <= point2[0])]
            t_prime = t[(t >= point1[0]) & (t <= point2[0])]
        else:
            y_prime = y[(t >= point2[0]) & (t <= point1[0])]
            t_prime = t[(t >= point2[0]) & (t <= point1[0])]

        slope_original = (point2[1] - point1[1]) / (point2[0] - point1[0])
        slope_perpendicular = -1 / slope_original
        if midpoint[0] in t_prime:
            t_prime[np.where(t_prime == midpoint[0])] = np.nan
        else:
            pass
        slope_midpoint = (midpoint[1]-y_prime) / (midpoint[0]-t_prime)
        slope_desired = slope_midpoint[np.argmin(np.abs(slope_midpoint - slope_perpendicular))]
        idx_slope_desired = np.argmin(np.abs(slope_midpoint - slope_desired))
        y_adjusted, t_adjusted = y_prime[idx_slope_desired], t_prime[idx_slope_desired]
        y_adjusted = -y_adjusted
        y_adjusted, t_adjusted = round(y_adjusted, 3), round(t_adjusted, 3)
        print("Adjusted Amplitude (uV):", y_adjusted)
        print("Adjusted Latency (ms):", t_adjusted)
        print('\n')
        return y_adjusted, t_adjusted