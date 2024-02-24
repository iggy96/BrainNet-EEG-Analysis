import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import pandas as pd
from matplotlib.spines import Spine
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def bvs_individual_grand(measureArray,ref_data,comparative_data,color_code,labels):

    """
    Input: measure_array (numpy array: 2 x 8)
          col_names: Group,N100_amp,N100_lat,P300_amp,P300_lat,N400_amp,N400_lat,Channel
          row: contains the individual measures above for just one subject
          color_code: ccolor string e.g., 'b' for blue
          labels: list of strings e.g., ['A','B'] for the measured_data and comparative_data
    Input: ref_data (pandas dataframe: size of normative database)
            col_names: any dataframe but must contain N100_amp,N100_lat,P300_amp,P300_lat,N400_amp & N400_lat column names


    """

    def remout(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] > (Q1 - 1.5 * IQR)) & (df[col] < (Q3 + 1.5 * IQR))]
        np = df[col].values
        return np

    ref_N100_amp = remout(ref_data, 'N100_amp')
    ref_N100_lat = remout(ref_data, 'N100_lat')
    ref_P300_amp = remout(ref_data, 'P300_amp')
    ref_P300_lat = remout(ref_data, 'P300_lat')
    ref_N400_amp = remout(ref_data, 'N400_amp')
    ref_N400_lat = remout(ref_data, 'N400_lat')

    def input(measure_array, ref_N100_amp, ref_N100_lat, ref_P300_amp, ref_P300_lat, ref_N400_amp, ref_N400_lat):
        
        measure_list = measure_array.tolist()
        
        def scorer(mean, max, min, amplitude=False, latency=False):
            if amplitude:
                if mean > max:
                    score = 100
                elif mean < min:
                    score = 0
                else:
                    best = max
                    score = (round((1 - abs((mean-best)/(max-min))).item(),2))*100
            elif latency:
                if mean > max:
                    score = 0
                elif mean < min:
                    score = 100
                else:
                    best = min
                    score = (round((1 - abs((mean-best)/(max-min))).item(),2))*100
            return score

        amp = float([row[measure_list[0].index('P300_amp')] for row in measure_list[1:]][0])
        norm_max_amp = round(np.max(ref_P300_amp),2)
        norm_min_amp = round(np.min(ref_P300_amp),2)
        p3_amp_score = scorer(amp,norm_max_amp,norm_min_amp,amplitude=True)

        lat = float([row[measure_list[0].index('P300_lat')] for row in measure_list[1:]][0])
        norm_max_lat = round(np.max(ref_P300_lat),2)
        norm_min_lat = round(np.min(ref_P300_lat),2)
        p3_lat_score = scorer(lat,norm_max_lat,norm_min_lat,latency=True)

        amp = float([row[measure_list[0].index('N400_amp')] for row in measure_list[1:]][0])
        norm_max_amp = round(np.max(ref_N400_amp),2)
        norm_min_amp = round(np.min(ref_N400_amp),2)
        n4_amp_score = scorer(amp,norm_max_amp,norm_min_amp,amplitude=True)

        lat = float([row[measure_list[0].index('N400_lat')] for row in measure_list[1:]][0])
        norm_max_lat = round(np.max(ref_N400_lat),2)
        norm_min_lat = round(np.min(ref_N400_lat),2)
        n4_lat_score = scorer(lat,norm_max_lat,norm_min_lat,latency=True)

        amp = float([row[measure_list[0].index('N100_amp')] for row in measure_list[1:]][0])
        norm_max_amp = round(np.max(ref_N100_amp),2)
        norm_min_amp = round(np.min(ref_N100_amp),2)
        n1_amp_score = scorer(amp,norm_max_amp,norm_min_amp,amplitude=True)

        lat = float([row[measure_list[0].index('N100_lat')] for row in measure_list[1:]][0])
        norm_max_lat = round(np.max(ref_N100_lat),2)
        norm_min_lat = round(np.min(ref_N100_lat),2)
        n1_lat_score = scorer(lat,norm_max_lat,norm_min_lat,latency=True)

        return [n1_amp_score,n4_lat_score,n4_amp_score,p3_lat_score,p3_amp_score,n1_lat_score]

    
    comp_data = comparative_data
    comp_data = comp_data[['N100_amp', 'N100_lat', 'P300_amp', 'P300_lat', 'N400_amp', 'N400_lat']]
    comp_data_means = comp_data.mean()
    comp_data_means = pd.concat([comp_data_means], axis=1).T
    comp_data_means = pd.concat([comp_data_means]*3, ignore_index=True)
    comp_data_means = comp_data_means.reset_index().T.reset_index().T
    comp_data_means = comp_data_means.to_numpy()
    comparative_data  = comp_data_means[:,1:]
    fz_measure,cz_measure,pz_measure = np.vstack((measureArray[0,:],measureArray[1,:])),np.vstack((measureArray[0,:],measureArray[2,:])),np.vstack((measureArray[0,:],measureArray[3,:]))
    comp_fz_measure,comp_cz_measure,comp_pz_measure = np.vstack((comparative_data[0,:],comparative_data[1,:])),np.vstack((comparative_data[0,:],comparative_data[2,:])),np.vstack((comparative_data[0,:],comparative_data[3,:]))
    fz = input(fz_measure,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)  
    cz = input(cz_measure,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)
    pz = input(pz_measure,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)
    comp_fz = input(comp_fz_measure,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)
    comp_cz = input(comp_cz_measure,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)
    comp_pz = input(comp_pz_measure,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)

    def radar_factory(num_vars, frame='circle'):
        """
        Create a radar chart with `num_vars` axes.

        This function creates a RadarAxes projection and registers it.

        Parameters
        ----------
        num_vars : int
            Number of variables for radar chart.
        frame : {'circle', 'polygon'}
            Shape of frame surrounding axes.

        """
        # calculate evenly-spaced axis angles
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

        class RadarTransform(PolarAxes.PolarTransform):

            def transform_path_non_affine(self, path):
                # Paths with non-unit interpolation steps correspond to gridlines,
                # in which case we force interpolation (to defeat PolarTransform's
                # autoconversion to circular arcs).
                if path._interpolation_steps > 1:
                    path = path.interpolated(num_vars)
                return Path(self.transform(path.vertices), path.codes)

        class RadarAxes(PolarAxes):

            name = 'radar'
            PolarTransform = RadarTransform

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # rotate plot such that the first axis is at the top
                self.set_theta_zero_location('N')

            def fill(self, *args, closed=True, **kwargs):
                """Override fill so that line is closed by default"""
                return super().fill(closed=closed, *args, **kwargs)

            def plot(self, *args, **kwargs):
                """Override plot so that line is closed by default"""
                lines = super().plot(*args, **kwargs)
                for line in lines:
                    self._close_line(line)

            def _close_line(self, line):
                x, y = line.get_data()
                # FIXME: markers at x[0], y[0] get doubled-up
                if x[0] != x[-1]:
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    line.set_data(x, y)

            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)

            def _gen_axes_patch(self):
                # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
                # in axes coordinates.
                if frame == 'circle':
                    return Circle((0.5, 0.5), 0.5)
                elif frame == 'polygon':
                    return RegularPolygon((0.5, 0.5), num_vars,
                                        radius=.5, edgecolor="k")
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

            def _gen_axes_spines(self):
                if frame == 'circle':
                    return super()._gen_axes_spines()
                elif frame == 'polygon':
                    # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                    spine = Spine(axes=self,
                                spine_type='circle',
                                path=Path.unit_regular_polygon(num_vars))
                    # unit_regular_polygon gives a polygon of radius 1 centered at
                    # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                    # 0.5) in axes coordinates.
                    spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                        + self.transAxes)
                    return {'polar': spine}
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

        register_projection(RadarAxes)
        return theta

    data = [
        ['Auditory \n Sensation \n (Amplitude)','Cognitive \n Processing \n (Latency)',
            'Cognitive \n Processing \n (Amplitude)','Basic \n Attention \n (Latency)',
            'Basic \n Attention \n (Amplitude)','Auditory \n Sensation \n (Latency)'],
        ('Fz', [fz,comp_fz]),
        ('Cz', [cz,comp_cz]),
        ('Pz', [pz,comp_pz])
    ]

    N = 6
    theta = radar_factory(N, frame='polygon')
    spoke_labels = data.pop(0)
    fig, axs = plt.subplots(figsize=(15,25), nrows=1, ncols=3,subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=2, hspace=0.20, top=0.85, bottom=0.05)
    fig.set_dpi(1000)
    colors = [color_code, "violet"]

    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        ax.set_rgrids([0,20,40,60,80,100])
        ax.set_rlabel_position(40)  # adjust rgrids (scale values) angle
        ax.tick_params(pad=22)
        ax.xaxis.grid(False)
        ax.set_title(title, weight='bold', size='large', y=1.55,
                        horizontalalignment='center', verticalalignment='top',pad=50)
        
        for d, color in zip(case_data, colors):
            ax.plot(theta, d,linewidth=4, color=color, alpha=0.8)  # alpha value controls the transparency
            ax.fill(theta, d, facecolor=color, alpha=0, label='_nolegend_')
            ax.set_ylim(0,100)
        ax.set_varlabels(spoke_labels)

# might need to add code for adjust scale based on values of comparison data
    # add legend relative to top-left of each subplot
    labels = (labels, 'Healthy Individuals')
    legend = axs[0].legend(labels, loc=(0.9, 1.3),
                            labelspacing=0.1, fontsize='medium')    
    legend = axs[1].legend(labels, loc=(0.9, 1.3),
                            labelspacing=0.1, fontsize='medium')
    legend = axs[2].legend(labels, loc=(0.9, 1.3),
                            labelspacing=0.1, fontsize='medium')

    plt.show()

def customplot_bvs(comp_data,grand_peaks,ref_data,labels):

    # grand_adj_peaks: numpy array with headers on the first rows
    # ref_data: pandas dataframe with headers on the first rows
    # comp_data: pandas dataframe with headers on the first rows

    def remout(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] > (Q1 - 1.5 * IQR)) & (df[col] < (Q3 + 1.5 * IQR))]
        np = df[col].values
        return np

    ref_N100_amp = remout(ref_data, 'N100_amp')
    ref_N100_lat = remout(ref_data, 'N100_lat')
    ref_P300_amp = remout(ref_data, 'P300_amp')
    ref_P300_lat = remout(ref_data, 'P300_lat')
    ref_N400_amp = remout(ref_data, 'N400_amp')
    ref_N400_lat = remout(ref_data, 'N400_lat')

    # min and max values for ref_N100_amp, ref_N100_lat, ref_P300_amp, ref_P300_lat, ref_N400_amp, ref_N400_lat
    min_N100_amp, max_N100_amp = np.min(ref_N100_amp), np.max(ref_N100_amp)
    min_N100_lat, max_N100_lat = np.min(ref_N100_lat), np.max(ref_N100_lat)
    min_P300_amp, max_P300_amp = np.min(ref_P300_amp), np.max(ref_P300_amp)
    min_P300_lat, max_P300_lat = np.min(ref_P300_lat), np.max(ref_P300_lat)
    min_N400_amp, max_N400_amp = np.min(ref_N400_amp), np.max(ref_N400_amp)
    min_N400_lat, max_N400_lat = np.min(ref_N400_lat), np.max(ref_N400_lat)
    print('range of N100_amp: ', min_N100_amp,'-', max_N100_amp)
    print('range of N100_lat: ', min_N100_lat,'-', max_N100_lat)
    print('range of P300_amp: ', min_P300_amp,'-', max_P300_amp)
    print('range of P300_lat: ', min_P300_lat,'-', max_P300_lat)
    print('range of N400_amp: ', min_N400_amp,'-', max_N400_amp)
    print('range of N400_lat: ', min_N400_lat,'-', max_N400_lat)

    def input(measure_array, ref_N100_amp, ref_N100_lat, ref_P300_amp, ref_P300_lat, ref_N400_amp, ref_N400_lat):
        
        
        def param(measure_array, ref_N100_amp, ref_N100_lat, ref_P300_amp, ref_P300_lat, ref_N400_amp, ref_N400_lat):
            
            if len(measure_array.T)==8:
                measure_array = np.vstack(([['Group', 'N100_amp' ,'N100_lat' ,'P300_amp' ,'P300_lat' ,'N400_amp' ,'N400_lat','Channel']], measure_array))
            elif len(measure_array.T)==6:
                measure_array =  np.vstack(([['N100_amp' ,'N100_lat' ,'P300_amp' ,'P300_lat' ,'N400_amp' ,'N400_lat']], measure_array))
            
            measure_list = measure_array.tolist()
            
            def scorer(mean, max, min, amplitude=False, latency=False):
                if amplitude:
                    if mean > max:
                        score = 100
                        
                    elif mean < min:
                        score = 0
                    else:
                        best = max
                        score = (round((1 - abs((mean-best)/(max-min))).item(),2))*100
                elif latency:
                    if mean > max:
                        score = 0
                    elif mean < min:
                        score = 100
                    else:
                        best = min
                        score = (round((1 - abs((mean-best)/(max-min))).item(),2))*100
                return score

            amp = float([row[measure_list[0].index('P300_amp')] for row in measure_list[1:]][0])
            norm_max_amp = round(np.max(ref_P300_amp),2)
            norm_min_amp = round(np.min(ref_P300_amp),2)
            p3_amp_score = scorer(amp,norm_max_amp,norm_min_amp,amplitude=True)

            lat = float([row[measure_list[0].index('P300_lat')] for row in measure_list[1:]][0])
            norm_max_lat = round(np.max(ref_P300_lat),2)
            norm_min_lat = round(np.min(ref_P300_lat),2)
            p3_lat_score = scorer(lat,norm_max_lat,norm_min_lat,latency=True)

            amp = float([row[measure_list[0].index('N400_amp')] for row in measure_list[1:]][0])
            norm_max_amp = round(np.max(ref_N400_amp),2)
            norm_min_amp = round(np.min(ref_N400_amp),2)
            n4_amp_score = scorer(amp,norm_max_amp,norm_min_amp,amplitude=True)

            lat = float([row[measure_list[0].index('N400_lat')] for row in measure_list[1:]][0])
            norm_max_lat = round(np.max(ref_N400_lat),2)
            norm_min_lat = round(np.min(ref_N400_lat),2)
            n4_lat_score = scorer(lat,norm_max_lat,norm_min_lat,latency=True)

            amp = float([row[measure_list[0].index('N100_amp')] for row in measure_list[1:]][0])
            norm_max_amp = round(np.max(ref_N100_amp),2)
            norm_min_amp = round(np.min(ref_N100_amp),2)
            n1_amp_score = scorer(amp,norm_max_amp,norm_min_amp,amplitude=True)

            lat = float([row[measure_list[0].index('N100_lat')] for row in measure_list[1:]][0])
            norm_max_lat = round(np.max(ref_N100_lat),2)
            norm_min_lat = round(np.min(ref_N100_lat),2)
            n1_lat_score = scorer(lat,norm_max_lat,norm_min_lat,latency=True)
            return [n1_amp_score,n4_lat_score,n4_amp_score,p3_lat_score,p3_amp_score,n1_lat_score]

        output = []
        measure_array = measure_array[1:,:]
        for i in range(len(measure_array)):
            output.append(param(measure_array[i],ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat))
        output = np.array(output)
        if len(output.shape)==3 and output.shape[0] == 1:
            output = output.reshape(output.shape[1],output.shape[2])
            output = output[:,1:-1]
            output = output[1:,:]
            output = output
        if output.shape[0] > 1:
            output_mean = np.mean(output, axis=0)
            output_std = np.std(output, axis=0)
            output = np.vstack((output_mean,output_std))
            output = output
        return output
        
    measuresBNDCz = np.vstack((grand_peaks[0,:],grand_peaks[1,:]))  
    measuresBNDFz = np.vstack((grand_peaks[0,:],grand_peaks[2,:]))
    measuresBNDPz = np.vstack((grand_peaks[0,:],grand_peaks[3,:]))
    measuresBMMDCz = np.vstack((grand_peaks[0,:],grand_peaks[4,:]))
    measuresBMMDFz = np.vstack((grand_peaks[0,:],grand_peaks[5,:]))
    measuresBMMDPz = np.vstack((grand_peaks[0,:],grand_peaks[6,:]))
    measuresBSDCz = np.vstack((grand_peaks[0,:],grand_peaks[7,:]))
    measuresBSDFz = np.vstack((grand_peaks[0,:],grand_peaks[8,:]))
    measuresBSDPz = np.vstack((grand_peaks[0,:],grand_peaks[9,:]))


    comp_data = comp_data[['N100_amp', 'N100_lat', 'P300_amp', 'P300_lat', 'N400_amp', 'N400_lat']]
    comp_test = comp_data.to_numpy()
    comp_test = np.vstack((['N100_amp', 'N100_lat', 'P300_amp', 'P300_lat', 'N400_amp', 'N400_lat'],comp_test))

    radarBNDFz = input(measuresBNDFz,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)
    radarBMMDFz = input(measuresBMMDFz,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)
    radarBSDFz = input(measuresBSDFz,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)
    radarBNDCz = input(measuresBNDCz,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)
    radarBMMDCz = input(measuresBMMDCz,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)
    radarBSDCz = input(measuresBSDCz,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)
    radarBNDPz = input(measuresBNDPz,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)
    radarBMMDPz = input(measuresBMMDPz,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)
    radarBSDPz = input(measuresBSDPz,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)
    comp = input(comp_test,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)[0]
    comp_std = input(comp_test,ref_N100_amp,ref_N100_lat,ref_P300_amp,ref_P300_lat,ref_N400_amp,ref_N400_lat)[1]

    print('radarBNDPz',radarBNDPz)
    print('radarBMMDPz',radarBMMDPz)
    print('radarBSDPz',radarBSDPz)

    def radar_factory(num_vars, frame='circle'):
        """
        Create a radar chart with `num_vars` axes.

        This function creates a RadarAxes projection and registers it.

        Parameters
        ----------
        num_vars : int
            Number of variables for radar chart.
        frame : {'circle', 'polygon'}
            Shape of frame surrounding axes.

        """
        # calculate evenly-spaced axis angles
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

        class RadarTransform(PolarAxes.PolarTransform):

            def transform_path_non_affine(self, path):
                # Paths with non-unit interpolation steps correspond to gridlines,
                # in which case we force interpolation (to defeat PolarTransform's
                # autoconversion to circular arcs).
                if path._interpolation_steps > 1:
                    path = path.interpolated(num_vars)
                return Path(self.transform(path.vertices), path.codes)

        class RadarAxes(PolarAxes):

            name = 'radar'
            PolarTransform = RadarTransform

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # rotate plot such that the first axis is at the top
                self.set_theta_zero_location('N')

            def fill(self, *args, closed=True, **kwargs):
                """Override fill so that line is closed by default"""
                return super().fill(closed=closed, *args, **kwargs)

            def plot(self, *args, **kwargs):
                """Override plot so that line is closed by default"""
                lines = super().plot(*args, **kwargs)
                for line in lines:
                    self._close_line(line)

            def _close_line(self, line):
                x, y = line.get_data()
                # FIXME: markers at x[0], y[0] get doubled-up
                if x[0] != x[-1]:
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    line.set_data(x, y)

            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)

            def _gen_axes_patch(self):
                # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
                # in axes coordinates.
                if frame == 'circle':
                    return Circle((0.5, 0.5), 0.5)
                elif frame == 'polygon':
                    return RegularPolygon((0.5, 0.5), num_vars,
                                        radius=.5, edgecolor="k")
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

            def _gen_axes_spines(self):
                if frame == 'circle':
                    return super()._gen_axes_spines()
                elif frame == 'polygon':
                    # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                    spine = Spine(axes=self,
                                spine_type='circle',
                                path=Path.unit_regular_polygon(num_vars))
                    # unit_regular_polygon gives a polygon of radius 1 centered at
                    # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                    # 0.5) in axes coordinates.
                    spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                        + self.transAxes)
                    return {'polar': spine}
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

        register_projection(RadarAxes)
        return theta

# N1 Amp score from Fz, N4 Lat score from Pz, N4 Amp score from Pz, P3 Lat score from Cz, P3 Amp score from Cz, N1 Lat score from Fz
    eBND = [radarBNDFz[0][0],radarBNDPz[0][1],radarBNDPz[0][2],radarBNDCz[0][3],radarBNDCz[0][4],radarBNDFz[0][5]]
    eBMM = [radarBMMDFz[0][0],radarBMMDPz[0][1],radarBMMDPz[0][2],radarBMMDCz[0][3],radarBMMDCz[0][4],radarBMMDFz[0][5]]
    eBSD = [radarBSDFz[0][0],radarBSDPz[0][1],radarBSDPz[0][2],radarBSDCz[0][3],radarBSDCz[0][4],radarBSDFz[0][5]]

    data = [
        ['Auditory \n Sensation \n (Amplitude)','Cognitive \n Processing \n (Latency)',
            'Cognitive \n Processing \n (Amplitude)','Basic \n Attention \n (Latency)',
            'Basic \n Attention \n (Amplitude)','Auditory \n Sensation \n (Latency)'],
        ('', [eBND,comp]),
        ('', [eBMM,comp]),
        ('', [eBSD,comp])
    ]

    N = 6
    theta = radar_factory(N, frame='polygon')
    spoke_labels = data.pop(0)
    colors = ['g', 'b', 'r','magenta']
    std_colors = ['lightcyan','powderblue'] # https://matplotlib.org/stable/gallery/color/named_colors.html
    fig, axs = plt.subplots(figsize=(13,13),subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=1.6, hspace=0.06, top=0.85, bottom=0.05)
    fig.set_dpi(600) # adjust to 1600 for manuscript
    axs.plot(theta, eBND,linewidth=2, color=colors[0], alpha=0.8, label=labels[0])  # alpha value controls the transparency
    axs.fill(theta, eBND, facecolor=colors[0], alpha=0)
    axs.plot(theta, comp,linewidth=2, color=colors[3], alpha=0.8,label=labels[3])  # alpha value controls the transparency
    axs.fill(theta, comp, facecolor=colors[3], alpha=0)
    theta_ = np.concatenate((theta,[theta[0]])) # added zero to initialise closing the circle
    std_neg_1,std_pos_1 = comp-1*comp_std,comp+1*comp_std
    std_neg_2,std_pos_2 = comp-2*comp_std,comp+2*comp_std
    print('comp mean: ',comp)
    print('comp std: ',comp_std)
    print('negative 1 std from mean: ',std_neg_1)
    print('positive 1 std from mean: ',std_pos_1)
    print('negative 2 std from mean: ',std_neg_2)
    print('positive 2 std from mean: ',std_pos_2)
    std_neg_1, std_pos_1 = np.concatenate((std_neg_1,[std_neg_1[0]])),np.concatenate((std_pos_1,[std_pos_1[0]]))
    std_neg_2, std_pos_2 = np.concatenate((std_neg_2,[std_neg_2[0]])),np.concatenate((std_pos_2,[std_pos_2[0]]))
    comp_ = np.concatenate((comp,[comp[0]]))
    axs.fill_between(theta_, comp_,std_pos_1,facecolor=std_colors[0],alpha=0.5)
    axs.fill_between(theta_, std_pos_1,std_pos_2,facecolor=std_colors[1], alpha=0.5)
    axs.fill_between(theta_, comp_,std_neg_1,facecolor=std_colors[0],alpha=0.5)
    axs.fill_between(theta_, std_neg_1,std_neg_2,facecolor=std_colors[1], alpha=0.5)
    axs.set_ylim(0,100)
    axs.set_varlabels(spoke_labels)
    axs.set_rgrids([0, 20, 40, 60, 80, 100])
    axs.tick_params(axis='y', which='both', labelsize='xx-large', grid_color='grey', grid_alpha=0.2, pad=10) # adjust rgrids (scale values) fontsize
    axs.tick_params(axis='x', labelsize='xx-large', pad=40) # adjust rgrids (labels) fontsize
    # change color of angular axis
    for spine in axs.spines.values():
        spine.set_edgecolor('grey')
        spine.set_linewidth(0.5)
        spine.set_alpha(0.2)
    axs.set_rlabel_position(40)  # adjust rgrids (scale values) angle
    axs.xaxis.grid(False)
    axs.set_title('', weight='bold', size='large', y=1.55,horizontalalignment='center', verticalalignment='top',pad=50)
    legend_elements_1 = [Line2D([0], [0], color=colors[0], lw=2, label=labels[0]),
                          Line2D([0], [0], color=colors[3], lw=2, label=labels[3]),
                            Patch(facecolor=std_colors[0], edgecolor='grey', label='1 std'),
                            Patch(facecolor=std_colors[1], edgecolor='grey', label='2 std')]
    legend_elements_2 = [Line2D([0], [0], color=colors[1], lw=2, label=labels[1]),
                        Line2D([0], [0], color=colors[3], lw=2, label=labels[3]),
                            Patch(facecolor=std_colors[0], edgecolor='grey', label='1 std'),
                            Patch(facecolor=std_colors[1], edgecolor='grey', label='2 std')]
    legend_elements_3 = [Line2D([0], [0], color=colors[2], lw=2, label=labels[2]),
                        Line2D([0], [0], color=colors[3], lw=2, label=labels[3]),
                            Patch(facecolor=std_colors[0], edgecolor='grey', label='1 std'),
                            Patch(facecolor=std_colors[1], edgecolor='grey', label='2 std')]
    axs.legend(handles=legend_elements_1, loc=(0.9, 1),labelspacing=0.1, fontsize='xx-large')
    plt.show()

    fig, axs = plt.subplots(figsize=(13,13),subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=1.6, hspace=0.06, top=0.85, bottom=0.05)
    fig.set_dpi(600) # adjust to 1600 for manuscript
    axs.plot(theta, eBMM,linewidth=2, color=colors[1], alpha=0.8, label=labels[1])  # alpha value controls the transparency
    axs.fill(theta, eBMM, facecolor=colors[1], alpha=0)
    axs.plot(theta, comp,linewidth=2, color=colors[3], alpha=0.8,label=labels[3])  # alpha value controls the transparency
    axs.fill(theta, comp, facecolor=colors[3], alpha=0)
    axs.fill_between(theta_, comp_,std_pos_1,facecolor=std_colors[0],alpha=0.5)
    axs.fill_between(theta_, std_pos_1,std_pos_2,facecolor=std_colors[1], alpha=0.5)
    axs.fill_between(theta_, comp_,std_neg_1,facecolor=std_colors[0],alpha=0.5)
    axs.fill_between(theta_, std_neg_1,std_neg_2,facecolor=std_colors[1], alpha=0.5)
    axs.set_ylim(0,100)
    axs.set_varlabels(spoke_labels)
    axs.set_rgrids([0, 20, 40, 60, 80, 100])
    axs.tick_params(axis='y', which='both', labelsize='xx-large', grid_color='grey', grid_alpha=0.2, pad=10) # adjust rgrids (scale values) fontsize
    axs.tick_params(axis='x', labelsize='xx-large', pad=40) # adjust rgrids (labels) fontsize
    # change color of angular axis
    for spine in axs.spines.values():
        spine.set_edgecolor('grey')
        spine.set_linewidth(0.5)
        spine.set_alpha(0.2)
    axs.set_rlabel_position(40)  # adjust rgrids (scale values) angle
    axs.xaxis.grid(False)
    axs.set_title('', weight='bold', size='large', y=1.55,horizontalalignment='center', verticalalignment='top',pad=50)
    legend_elements_1 = [Line2D([0], [0], color=colors[0], lw=2, label=labels[0]),
                          Line2D([0], [0], color=colors[3], lw=2, label=labels[3]),
                            Patch(facecolor=std_colors[0], edgecolor='grey', label='1 std'),
                            Patch(facecolor=std_colors[1], edgecolor='grey', label='2 std')]
    legend_elements_2 = [Line2D([0], [0], color=colors[1], lw=2, label=labels[1]),
                        Line2D([0], [0], color=colors[3], lw=2, label=labels[3]),
                            Patch(facecolor=std_colors[0], edgecolor='grey', label='1 std'),
                            Patch(facecolor=std_colors[1], edgecolor='grey', label='2 std')]
    legend_elements_3 = [Line2D([0], [0], color=colors[2], lw=2, label=labels[2]),
                        Line2D([0], [0], color=colors[3], lw=2, label=labels[3]),
                            Patch(facecolor=std_colors[0], edgecolor='grey', label='1 std'),
                            Patch(facecolor=std_colors[1], edgecolor='grey', label='2 std')]
    axs.legend(handles=legend_elements_2, loc=(0.9, 1),labelspacing=0.1, fontsize='xx-large')
    plt.show()

    fig, axs = plt.subplots(figsize=(13,13),subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=1.6, hspace=0.06, top=0.85, bottom=0.05)
    fig.set_dpi(600) # adjust to 1600 for manuscript
    axs.plot(theta, eBSD,linewidth=2, color=colors[2], alpha=0.8, label=labels[2])  # alpha value controls the transparency
    axs.fill(theta, eBSD, facecolor=colors[2], alpha=0)
    axs.plot(theta, comp,linewidth=2, color=colors[3], alpha=0.8,label=labels[3])  # alpha value controls the transparency
    axs.fill(theta, comp, facecolor=colors[3], alpha=0)
    axs.fill_between(theta_, comp_,std_pos_1,facecolor=std_colors[0],alpha=0.5)
    axs.fill_between(theta_, std_pos_1,std_pos_2,facecolor=std_colors[1], alpha=0.5)
    axs.fill_between(theta_, comp_,std_neg_1,facecolor=std_colors[0],alpha=0.5)
    axs.fill_between(theta_, std_neg_1,std_neg_2,facecolor=std_colors[1], alpha=0.5)
    axs.set_ylim(0,100)
    axs.set_varlabels(spoke_labels)
    axs.set_rgrids([0, 20, 40, 60, 80, 100])
    axs.tick_params(axis='y', which='both', labelsize='xx-large', grid_color='grey', grid_alpha=0.2, pad=10) # adjust rgrids (scale values) fontsize
    axs.tick_params(axis='x', labelsize='xx-large', pad=40) # adjust rgrids (labels) fontsize
    # change color of angular axis
    for spine in axs.spines.values():
        spine.set_edgecolor('grey')
        spine.set_linewidth(0.5)
        spine.set_alpha(0.2)
    axs.set_rlabel_position(40)  # adjust rgrids (scale values) angle
    axs.xaxis.grid(False)
    axs.set_title('', weight='bold', size='large', y=1.55,horizontalalignment='center', verticalalignment='top',pad=50)
    legend_elements_1 = [Line2D([0], [0], color=colors[0], lw=2, label=labels[0]),
                          Line2D([0], [0], color=colors[3], lw=2, label=labels[3]),
                            Patch(facecolor=std_colors[0], edgecolor='grey', label='1 std'),
                            Patch(facecolor=std_colors[1], edgecolor='grey', label='2 std')]
    legend_elements_2 = [Line2D([0], [0], color=colors[1], lw=2, label=labels[1]),
                        Line2D([0], [0], color=colors[3], lw=2, label=labels[3]),
                            Patch(facecolor=std_colors[0], edgecolor='grey', label='1 std'),
                            Patch(facecolor=std_colors[1], edgecolor='grey', label='2 std')]
    legend_elements_3 = [Line2D([0], [0], color=colors[2], lw=2, label=labels[2]),
                        Line2D([0], [0], color=colors[3], lw=2, label=labels[3]),
                            Patch(facecolor=std_colors[0], edgecolor='grey', label='1 std'),
                            Patch(facecolor=std_colors[1], edgecolor='grey', label='2 std')]
    axs.legend(handles=legend_elements_3, loc=(0.9, 1),labelspacing=0.1, fontsize='xx-large')
    plt.show()