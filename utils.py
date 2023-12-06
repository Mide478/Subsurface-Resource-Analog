from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import seaborn as sns
from fastcluster import linkage
from geostatspy import geostats, GSLIB
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

pio.renderers.default = "browser"
pio.templates.default = 'simple_white'


def tukey(df, feature):
    """
    This function computes univariate Tukey.

    Arguments
    ---------
    df: DataFrame
    The dataset.

    feature: str
    The predictor feature name.
    """

    f1, f2 = None, None
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    inner_fence = 1.5 * IQR  # 1.5 for labelling outliers
    outer_fence = 3.0 * IQR  # 3.0 for labelling "far out" outliers

    # inner fence lower and upper end
    inner_fence_le = Q1 - inner_fence
    inner_fence_ue = Q3 + inner_fence

    # outer fence lower and upper end
    outer_fence_le = Q1 - outer_fence
    outer_fence_ue = Q3 + outer_fence

    outliers_prob = []
    outliers_poss = []
    for index, x in enumerate(df[feature]):
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers_prob.append(index)
        else:
            f1 = "No Outlier"
    outliers_prob.append(f1)

    for index, x in enumerate(df[feature]):
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers_poss.append(index)
        else:
            f2 = "No Outlier"
    outliers_poss.append(f2)
    return outliers_prob, outliers_poss


def outlier_checker(df, features):
    """
    This function checks for and removes bivariate outliers in a dataset using Tukey function.

    Arguments
    ---------
    df: pandas.DataFrame
    Dataset.

    features: list
    The column names as a list consisting of the predictors of interest.
    """

    outliers = [[] for _ in range(len(features))]
    for index, i in enumerate(features):
        probable_outliers, possible_outliers = tukey(df, i)  # Ensure df is a quantitative dataframe.
        outliers[index].append(probable_outliers)

    AllOutliers = []  # list consisting of outlier indexes for every feature. Note that there might be repeated indexes
    # across feature
    for listoflist in outliers:
        for list_ in listoflist:
            for element in list_:
                AllOutliers.append(element)

    outliers_index = []  # Outlier list for dataset, excluding duplicated outlier index to be removed from df
    [outliers_index.append(i) for i in AllOutliers if i not in outliers_index]

    for k in outliers_index:  # Using the unique outlier indexes, let's drop the corresponding row in the data
        if k in outliers_index is None:
            df = df.drop(outliers_index, 0, inplace=True)

    print('There are no outliers to be dropped in the dataset use data as is.')


def check_symmetric(array, rtol=1e-05, atol=1e-08):
    """
    This function checks if the distance matrix is symmetric, prior to making a sorted dissimilarity matrix
    """
    return np.allclose(array, array.T, rtol=rtol, atol=atol)


def seriation(dendrogram, number_points, cur_index):
    """
    This is a function that creates a sorted 2D matrix as a figure

        input:
            - dendrogram is a hierarchical tree
            - number_points is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    """

    if cur_index < number_points:
        return [cur_index]
    else:
        left = int(dendrogram[cur_index - number_points, 0])
        right = int(dendrogram[cur_index - number_points, 1])
        return seriation(dendrogram, number_points, left) + seriation(dendrogram, number_points, right)


def compute_serial_matrix(dist_mat, method="ward"):
    """
        input:
            - dist_mat is a distance matrix
            -  = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarchical tree
            - res_linkage is the hierarchical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    """

    N = len(dist_mat)
    flat_dist_mat = dist_mat if len(dist_mat.shape) == 2 else squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage


def norm_plotter(array_in, norm_name):
    """
    This function makes and visualizes a sorted dissimilarity matrix that can be used to check for inherent
    data groupings.

    Arguments
    ---------
    array_in: an input array used to compute the serial matrix for sorting the dissimilarity matrices obtained

    norm_name: a string consisting of the name of norm used
    """

    methods = ["ward", "single", "average", "complete"]

    for method in methods:
        print("Linkage for agglomerative hierarchical clustering :\t", method)

        ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(array_in, method)
        plt.imshow(ordered_dist_mat, cmap=plt.get_cmap('plasma'))
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.title(method + ' Sorted ' + norm_name + ' Dissimilarity Matrix', size=12)
        plt.xlabel('Data point index, $\it{i}$', size=12)
        plt.ylabel('Data point index, $\it{j}$', size=12)
        plt.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=2.0, wspace=0.5, hspace=0.3)
        plt.savefig('AHC method:' + method + '.png', dpi=300, bbox_inches='tight')
        plt.show()
    return


def matrix_scatter(dataframe, feat_title, left_adj, bottom_adj, right_adj, top_adj, wspace, hspace, title, palette_,
                   hue_=None):
    """
    This function plots the matrix scatter plot for the given data.

    Arguments
    ---------

    dataframe: dataframe

    feat_title: a list consisting of a string column names for each predictor feature of choice

    left_adj: float values that adjusts the left placement of the scatter plot

    bottom_adj: float values that adjusts the bottom placement of the scatter plot

    right_adj: float values that adjusts the right placement of the scatter plot

    top_adj: float values that adjusts the top placement of the scatter plot

    wspace: float values that adjusts the width placement of the scatter plot

    hspace: float values that adjusts the height placement of the scatter plot

    title: a string consisting of the name of the figure

    palette_: an integer that assigns a dictionary of colors that maps the hue variable consisting of the
    classification label

    hue_: string variable that is used to color matrix scatter plot made
    """

    # Palette assignment
    if palette_ == 1:

        palette_ = {1: 'blue', 2: 'magenta', 3: 'green', 4: 'yellow', 5: 'red', 6: 'cyan', 7: 'brown',
                    8: 'burlywood', 9: 'orange', 10: 'cornflowerblue',
                    11: 'darkorchid', 12: 'palevioletred', 13: 'darkgoldenrod', 14: 'thistle',
                    0: 'black'}  # more colors can be added accordingly for more colors #more colors can be added
    # accordingly for more colors

    elif palette_ == 2:
        palette_ = sns.color_palette("bright",
                                     len(dataframe[hue_].unique()))  # can be changed to any seaborn color scheme
    else:
        palette_ = None

    # Hue assignment
    if hue_ is not None:
        hue_ = hue_
    else:
        hue_ is None

    sns.pairplot(dataframe, vars=feat_title, markers='o', plot_kws={'alpha': 0.5}, hue=hue_, corner=True,
                 palette=palette_)
    plt.subplots_adjust(left=left_adj, bottom=bottom_adj, right=right_adj, top=top_adj, wspace=wspace, hspace=hspace)
    plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    return


def normalizer(dataset, features, keep_only_norm_features=False):
    """
    This function normalizes the dataframe of choice between [0.01, 1].

    Arguments
    ---------
    dataset: DataFrame
        A pandas.DataFrame containing the features of interest
    features: list
        A list consisting of features column names to be normalized
    keep_only_norm_features: bool
        True to discard non-normalized features.
    """
    is_string = isinstance(features, str)
    if is_string:
        features = [features]

    df = dataset.copy()
    x = df.loc[:, features].values
    scaler = MinMaxScaler(feature_range=(0.01, 1))
    xs = scaler.fit_transform(x)

    ns_feats = []
    for i, feature in enumerate(features):
        df['NS_' + feature] = xs[:, i]
        ns_feats.append('NS_' + feature)

    if keep_only_norm_features:
        df = df.loc[:, ns_feats]

    return df


def cova2(x1, y1, x2, y2, nst, pmx, cc, aa, it, anis, rotmat, maxcov):
    """
    This function computes the covariance associated with a variogram model specified by a nugget effect and nested
    variogram structures.

    Arguments
    ---------
    x1: x coordinate of first point

    y1: y coordinate of first point

    x2: x coordinate of second point

    y2: y coordinate of second point

    nst: number of nested structures (maximum of 2) to aid variogram modeling

    pmx:

    cc: a list of multiplicative factor of each nested structure's contribution

    aa: a list of the min and max ranges for the structure in the major direction e.g., [hmaj1, hmaj2]

    it: a list of the variogram modeling type e.g.,[it1, it2]

    anis: a list of anisotropy values from variogram modeling parameter per nested structure

    rotmat: rotation matrices, obtained as output from geostats.setup_rotmat seen in D_spatial function

    maxcov: maximum covariance,  obtained as output from geostats.setup_rotmat seen in D_spatial function
    """

    EPSLON = 0.000001

    # Check for very small distance
    dx = x2 - x1
    dy = y2 - y1

    if (dx * dx + dy * dy) < EPSLON:
        cova2_ = maxcov
        return cova2_

    # Non-zero distance, loop over all the structures
    cova2_ = 0.0
    for js in range(0, nst):
        # Compute the appropriate structural distance
        dx1 = dx * rotmat[0, js] + dy * rotmat[1, js]
        dy1 = (dx * rotmat[2, js] + dy * rotmat[3, js]) / anis[js]
        h = np.sqrt(max((dx1 * dx1 + dy1 * dy1), 0.0))
        # print(h)
        if it[js] == 1:
            # Spherical model
            hr = h / aa[js]
            if hr < 1.0:
                cova2_ = cova2_ + cc[js] * (1.0 - hr * (1.5 - 0.5 * hr * hr))
        elif it[js] == 2:
            # Exponential model
            cova2_ = cova2_ + cc[js] * np.exp(-3.0 * h / aa[js])
        elif it[js] == 3:
            # Gaussian model
            hh = -3.0 * (h * h) / (aa[js] * aa[js])
            cova2_ = cova2_ + cc[js] * np.exp(hh)
        elif it[js] == 4:
            # Power model
            cov1 = pmx - cc[js] * (h ** aa[js])
            cova2_ = cova2_ + cov1
    return cova2_


def euclidean_dist(array, wts=None):
    """
    This function computes both the Euclidean distance and the proposed spatially weighted Euclidean distance.

    Arguments
    ---------
    array: numpy.array
        An array consisting of all data pair locations and possible combinations in the order, Xi, Yi, Xj, Yj

    wts: numpy.array
        An array consisting of the spatial weights for all data pairs (i.e., alphas).
    """
    squared_dif = (array[:, 0] - array[:, 2]) ** 2 + (array[:, 1] - array[:, 3]) ** 2
    if wts is not None:
        squared_dif *= wts
    squared_dif = np.sqrt(squared_dif)
    dij_euclidean = squareform(squared_dif)
    return dij_euclidean


def mahalanobis_dist(dataframe, feats, wts=None):
    """
    This function computes both the Mahalanobis distance and the proposed modified weighted Mahalanobis distance.

    Arguments
    ---------
    dataframe: DataFrame
        A dataframe consisting of all normalized multivariate predictors in the feature space.

    feats: list
        A list consisting of all normalized multivariate predictors column names in the feature space.

    wts: numpy.array
        An array consisting of all the complement of the spatial weights for all data pairs (i.e., betas).
    """

    df_array = dataframe[feats].to_numpy()

    if wts is None:
        dij_mahalanobis = squareform(pdist(X=df_array, metric='mahalanobis'))
    else:
        dij_mahalanobis = squareform(wts) * squareform(pdist(X=df_array, metric='mahalanobis'))
    return dij_mahalanobis


def gcs(dataframe, xcol, ycol, cluster_label):
    """
    This function estimates the group consistency scores for every sample point to its analog(centroid) within its
     respective cluster in the normalized MDS space where GCS is in [0,1]

    Arguments
    ---------

    dataframe: a dataframe consisting of X, Y coordinates for GCS metric computation

    xcol: a string for the column name of the X coordinate in MDS space used to index the dataframe 

    ycol: a string for the column name of the Y coordinate in MDS space used to index the dataframe 

    cluster_label: a string representing DBSCAN labels from the MDS space in the input dataframe (df)
    """

    dataframe = normalizer(dataframe, [xcol, ycol])

    centroid = pd.DataFrame()
    centroid['cent ' + xcol] = dataframe.groupby(by=cluster_label)['NS_' + xcol].apply(
        lambda x: np.mean(x.tolist(), axis=0))
    centroid['cent ' + ycol] = dataframe.groupby(by=cluster_label)['NS_' + ycol].apply(
        lambda x: np.mean(x.tolist(), axis=0))
    centroid = centroid.reset_index()

    label_list = []
    for i in range(0, len(centroid)):
        val = centroid.loc[i, cluster_label]
        label_list.append(val)

    centroid_x = np.array(centroid['cent ' + xcol])
    centroid_y = np.array(centroid['cent ' + ycol])
    bins = np.array(label_list)

    df_ = dataframe.copy(deep=True)
    cent_x = df_.assign(Centroid_X=centroid_x[bins.searchsorted(df_[cluster_label].values)])
    cent_y = df_.assign(Centroid_Y=centroid_y[bins.searchsorted(df_[cluster_label].values)])

    df_gcs = dataframe.copy(deep=True)
    df_gcs['Centroid_X'] = cent_x['Centroid_X']
    df_gcs['Centroid_Y'] = cent_y['Centroid_Y']

    df_gcs['GCS'] = np.sqrt(
        (df_gcs['NS_MDS 1'] - df_gcs['Centroid_X']) ** 2 + (df_gcs['NS_MDS 2'] - df_gcs['Centroid_Y']) ** 2)
    return df_gcs


def pss_all(dataframe, xcol, ycol):
    """
    This function estimates the pairwise-similarity scores for all possible combinatorial of the data in the normalized
     MDS space where PSS is in [0,1]

    Arguments
    ---------

    dataframe: a dataframe consisting of X, Y coordinates for PSS metric computation

    xcol: a string for the column name of the X coordinate in MDS space used to index the dataframe

    ycol: a string for the column name of the Y coordinate in MDS space used to index the dataframe

    """

    dataframe = normalizer(dataframe, [xcol, ycol])

    results = []
    starts = []
    array = dataframe[['NS_' + xcol, 'NS_' + ycol]].to_numpy()
    for i in range(array.shape[0] - 1):
        from_point = array[i].reshape(-1, 2)
        idx_to_compare = np.arange(i + 1, array.shape[0])
        from_point = np.repeat(from_point, repeats=[len(idx_to_compare)], axis=0)
        dif = from_point - array[idx_to_compare]
        results.extend(list(np.sum(dif ** 2, axis=1)))
        starts.extend(list(from_point))
    results = np.array(results)

    plt.imshow(squareform(results), cmap='plasma')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.title('PSS Metric Combinatorial Matrix', size=12)
    plt.xlabel('Data point index, $\it{i}$', size=12)
    plt.ylabel('Data point index, $\it{j}$', size=12)
    plt.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=2.0, wspace=0.5, hspace=0.3)
    plt.savefig('PSS for Entire Combinatorial Space.png', dpi=300, bbox_inches='tight')
    plt.show()
    return starts, results


class Kriging:
    """
    A class to aid in kriging computation.

    Attributes
    ----------
    """

    def __init__(self, dataset, feature, xcoor, ycoor):
        """
        Parameters
        ----------
        dataset: DataFrame
            The dataset that contains the 2D spatial coordinates and its corresponding values.
        feature: str
            The name of the feature to krige.
        xcoor, ycoor: str
            The names of the spatial coordinates.
        """
        self.dataset = dataset
        self.ycoor = ycoor
        self.xcoor = xcoor
        self.feature = feature
        self.half_area_interest = self._half_distance_area()
        self.vario_model = None
        self.xsiz = None
        self.ysiz = None

    def _half_distance_area(self):
        """Find the largest half distance within the area of interest."""
        xmax = self.dataset[self.xcoor].max()
        ymax = self.dataset[self.ycoor].max()
        xmin = self.dataset[self.xcoor].min()
        ymin = self.dataset[self.ycoor].min()

        distx = (xmax - xmin) / 2
        disty = (ymax - ymin) / 2
        if distx > disty:
            distance = distx
        else:
            distance = disty

        return distance

    def experimental_variog(self, lag_distance, xmax=None, azimuths=None, extend_half_dist=1.0, tmin=-9999, tmax=9999,
                            lag_tol_factor=2, bandwidth_factor=999, nlag=None, plot=True, major_dir_index=None,
                            ):
        """
        Plot the experimental and directional variograms.

        Parameters
        ----------
        lag_distance: float
            Lag distance for the experimental variograms. A recommended value is the average of the K nearest neighbors,
            with k=1.
        xmax: float
            The maximum limit for the X-axis plot.
        azimuths: list
            A list of size four with the azimuths to compute. The script automatically computes the complementary
            azimuths.
        extend_half_dist: float
            Distance to compute the experimental variogram with respect to half_distance_area_interest.
        lag_tol_factor: float
            A multiplicator for the lag tolerance, computed as lag_tolerance = lag_distance * lag_tol_factor.
        bandwidth_factor: float
            The horizontal bandwith.
        nlag: int
            Number of lags to compute
        plot: bool
            True to plot.
        major_dir_index: int
            An index from azimuths corresponding to the major direction of spatial continuity.

        Returns
        -------

        """
        if xmax is None:
            half_distance_area_interest = self.half_area_interest
        iazi = None

        if azimuths is None:
            azimuths = [0, 30, 45, 60]
        tempo = [i + 90 if i < 90 else i - 90 for i in azimuths]
        azimuths += tempo  # add 90 degrees to directions you chose

        lag_tol = lag_distance * lag_tol_factor
        bandh = lag_distance * bandwidth_factor

        if nlag is None:
            nlag = int((half_distance_area_interest * extend_half_dist) / lag_distance)
        # Arrays to store the results
        lag = np.zeros((len(azimuths), nlag + 2))
        gamma = np.zeros_like(lag)
        npp = np.zeros_like(lag)

        # Loop over all directions
        for iazi in range(0, len(azimuths)):
            lag[iazi, :], gamma[iazi, :], npp[iazi, :] = geostats.gamv(
                self.dataset, self.xcoor, self.ycoor, self.feature, tmin, tmax, lag_distance,
                lag_tol, nlag, azimuths[iazi], 22.5, bandh, isill=1
            )

        if plot:
            simbolo = ['circle', 'diamond', 'cross', 'triangle-up', 'triangle-down', 'star', 'x', 'square']
            colores = ['blue', 'red', 'black', 'green', 'blue', 'red', 'black', 'green']

            # Create traces
            fig = make_subplots(rows=1, cols=2)
            # First subplot
            for iazi in range(0, 4):
                fig.add_trace(go.Scatter(
                    x=np.round(lag[iazi, :-1], 2),
                    y=np.round(gamma[iazi, :], 2),
                    mode='markers',
                    name='Azimuth:' + str(azimuths[iazi]),
                    hovertemplate='<br><b>Gamma</b>: %{y:.2f}<br>' +
                                  '<b>Lag distance</b>: %{x:.2f}<br>' +
                                  '<b>%{text}</b>',
                    text=[f'Number of pairs {i:.0f}' for i in npp[iazi, :]],
                    marker=dict(size=7, symbol=simbolo[iazi],
                                color=colores[iazi])),
                    row=1, col=1)

            # Second subplot
            for iazi in range(4, 8):
                fig.add_trace(go.Scatter(
                    x=np.round(lag[iazi, :-1], 2),
                    y=np.round(gamma[iazi, :], 2),
                    mode='markers',
                    name='Azimuth' + str(azimuths[iazi - 4]) + '\u00B1 90 :' + str(azimuths[iazi]),
                    hovertemplate='<br><b>Gamma</b>: %{y:.2f}<br>' +
                                  '<b>Lag distance</b>: %{x:.2f}<br>' +
                                  '<b>%{text}</b>',
                    text=[f'Number of pairs {i:.0f}' for i in npp[iazi, :]],
                    marker=dict(size=7, symbol=simbolo[iazi],
                                color=colores[iazi])),
                    row=1, col=2)

            # add the sill to both subplots
            fig.add_trace(go.Scatter(x=[0, 5e9], y=[1.0, 1.0],
                                     line=dict(color='firebrick', width=3, dash='dot'),
                                     name='Sill', text=npp[iazi, :],
                                     showlegend=False),
                          row=1, col=1)

            fig.add_trace(go.Scatter(x=[0, 5e9], y=[1.0, 1.0],
                                     # dash options include 'dash', 'dot', and 'dashdot'
                                     line=dict(color='firebrick', width=3, dash='dot'),
                                     name='Sill', text=npp[iazi, :],
                                     showlegend=False),
                          row=1, col=2)

            # Update xaxis properties
            fig.update_xaxes(title_text="Lag distance <b>h(m)<b>",
                             row=1, range=[0, half_distance_area_interest * extend_half_dist], col=1)
            fig.update_xaxes(title_text="Lag distance <b>h(m)<b>",
                             row=1, range=[0, half_distance_area_interest * extend_half_dist], col=2)

            # Update yaxis properties
            fig.update_yaxes(title_text="<b>\u03B3<b>", row=1, range=[0, np.max(gamma) * 1.02],
                             col=1)
            fig.update_yaxes(title_text="<b>\u03B3<b>", row=1, range=[0, np.max(gamma) * 1.02],
                             col=2)

            # Edit the plot
            fig.update_layout(title='Directional ' + self.feature + ' Variogram',
                              autosize=False,
                              width=950,
                              height=500,
                              template='simple_white', )

            fig.show()

        if major_dir_index is not None:
            minor_dir_index = major_dir_index + 4 if major_dir_index <= 3 else major_dir_index + 4

            major_tensor = np.hstack((
                lag[major_dir_index].reshape(-1, 1),
                gamma[major_dir_index].reshape(-1, 1),
                npp[major_dir_index].reshape(-1, 1)
            ))
            minor_tensor = np.hstack((
                lag[minor_dir_index].reshape(-1, 1),
                gamma[minor_dir_index].reshape(-1, 1),
                npp[minor_dir_index].reshape(-1, 1)
            ))

            experimental_vario = np.zeros((2, major_tensor.shape[0], major_tensor.shape[1]))
            experimental_vario[0] = major_tensor
            experimental_vario[1] = minor_tensor

            return experimental_vario

    def variogram_modeling(self, experimental_vario, nugget, nst, it1, cc1, azi1, hmaj1, hmin1, it2=1, cc2=0,
                           hmaj2=0, hmin2=0, x_max=None, max_model_range=50000, left_adj=0.0, bot_adj=0.0,
                           right_adj=2.2, top_adj=1.6, w_adj=0.2, h_adj=0.3):
        """
        Model the experimental variogram.

        Parameters
        ----------
        experimental_vario: np.array
            The resulting array from experimental_variog when major_dir_index is different from None.
        nugget: float
            The nugget effect.
        nst: int
            Number of structures/models to consider: 1 or 2.
        it1, it2: int
            Variogram model: Spherical 0, Exponential: 1, Gaussian: 2.
        cc1, cc2: float
            The variance contribution from the models. Recall that nugget + cc1 + cc2 = 1.
        azi1, azi2: float
            The major direction of spatial continuity. Both azi1 and azi2 are the same.
        hmaj1, hmaj2: float
            The range of spatial continuity for each variogram model in the major direction of spatial continuity.
        hmin1, hmin2: float
            The range of spatial continuity for each variogram model in the minor direction of spatial continuity.
        x_max: float
            The xlimit for the visualizations purposes.
        max_model_range: int
            For extending the variogram modeling fit (max_model_range * 0.004)

        left_adj, bot_adj, right_adj, top_adj, w_adj, h_adj: float
        Returns
        -------

        """
        if x_max is None:
            x_max = self.half_area_interest

        minor_azi = azi1 + 90 if azi1 < 90 else azi1 - 90

        # make model object
        self.vario_model = GSLIB.make_variogram(nugget, nst, it1, cc1, azi1, hmaj1, hmin1, it2, cc2, azi1, hmaj2, hmin2)
        # variogram model
        azm = azi1  # project the model in the maj azimuth
        indexmaj, hmaj, gammaj, covmaj, romaj = geostats.vmodel(max_model_range, 0.004, azm, self.vario_model)
        azm = azi1 + 90  # project the model in the min azimuth
        indexmin, hmin, gammin, covmin, romin = geostats.vmodel(max_model_range, 0.004, azm, self.vario_model)

        # major
        plt.subplot(1, 2, 1)
        plt.plot(experimental_vario[0, :, 0], experimental_vario[0, :, 1], 'x', color='black',
                 label='Azimuth ' + str(azi1))
        plt.plot([0, x_max], [1.0, 1.0], color='black')
        plt.plot(hmaj, gammaj, color='black')
        plt.xlabel(r'Lag Distance $\bf(h)$, (m)')
        plt.ylabel(r'$\gamma \bf(h)$')
        plt.xlim([0, x_max])
        plt.ylim([0, 1.8])
        plt.legend(loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(experimental_vario[1, :, 0], experimental_vario[1, :, 1], 'x', color='black',
                 label='Azimuth ' + str(minor_azi))
        plt.plot([0, x_max], [1.0, 1.0], color='black')
        plt.plot(hmin, gammin, color='black')
        plt.xlabel(r'Lag Distance $\bf(h)$, (m)')
        plt.ylabel(r'$\gamma \bf(h)$')

        plt.xlim([0, x_max])
        plt.ylim([0, 1.8])
        plt.legend(loc='upper left')

        plt.subplots_adjust(left=left_adj, bottom=bot_adj, right=right_adj, top=top_adj, wspace=w_adj, hspace=h_adj)
        plt.show()

    def compute_kriging(self, ktype, skmean=None, nx=300, ny=300, radius=30, ndmin=0, ndmax=30):
        """
        Compute either simple or ordinary kriging and its corresponding variance map array.

        Parameters
        ----------
        ktype: int
            0 for simple kriging, 1 for ordinary kriging.
        skmean: float
            Stationary mean for simple kriging.
        nx, ny: int
            Number of cells in X and Y.
        radius: float
            The maximum isotropic search radius.
        ndmin, ndmax: int
            The minimum and maximum number of data points to use for kriging a block.

        Returns
        -------
        kmvap, vmap: numpy.array
            The kriging and variance maps.
        """
        skmean = self.dataset[self.feature].mean() if skmean is None else skmean
        # search radius for neighbouring data ..... @ Variogram too: should be always be greater than the major range
        # and the maximium simulated nodes in the previous data
        nxdis = 1
        nydis = 1
        tmin = -9999
        tmax = 9999
        Xrange = self.dataset[self.xcoor].max() - self.dataset[self.xcoor].min()
        Yrange = self.dataset[self.ycoor].max() - self.dataset[self.ycoor].min()
        total_dist = max(Yrange, Xrange)
        self.xsiz = total_dist / nx
        self.ysiz = total_dist / ny
        xmn = self.dataset[self.xcoor].min() - self.xsiz / 2
        ymn = self.dataset[self.ycoor].min() - self.xsiz / 2

        kmap, vmap = geostats.kb2d(self.dataset, self.xcoor, self.ycoor, self.feature, tmin, tmax, nx, xmn, self.xsiz,
                                   ny, ymn, self.ysiz, nxdis, nydis, ndmin, ndmax, radius, ktype, skmean,
                                   self.vario_model)

        return kmap, vmap

    def plot_kriging(self, kmap, vmap, xsize=None, kmap_title="Kriging map", vmap_title="Kriging variance", xlabel=None,
                     ylabel=None, save=False, left_adj=0.0, bot_adj=0.0, right_adj=2.2, top_adj=1.6, w_adj=0.2,
                     h_adj=0.3):
        """
        Plot the kriging and variance maps.`

        Parameters
        ----------
        kmap, vmap: numpy.array
            The kriging and variance maps from compute_kriging.
        xsize: float
            The cell size
        kmap_title, vmap_title: str
            The title for the kriging and variance maps.
        xlabel, ylabel: str
            Title for the x and y labels.
        save: bool
            True to save the plots as a png image.

        left_adj, bot_adj, right_adj, top_adj, w_adj, h_adj: float
        """
        # Extract min and max for location vector and feature
        if xsize is None:
            xsize = self.xsiz
        Xmin = self.dataset[self.xcoor].min()
        Xmax = self.dataset[self.xcoor].max()
        Ymin = self.dataset[self.ycoor].min()
        Ymax = self.dataset[self.ycoor].max()

        if xlabel is None:
            xlabel = self.xcoor
        if ylabel is None:
            ylabel = self.ycoor

        plt.subplot(121)
        GSLIB.locpix_st(
            kmap, Xmin, Xmax, Ymin, Ymax, xsize, self.dataset[self.feature].min() * 0.9,
                                                 self.dataset[self.feature].max() * 1.1, self.dataset, self.xcoor,
            self.ycoor, self.feature, kmap_title,
            xlabel, ylabel, self.feature, plt.cm.plasma
        )

        plt.subplot(122)
        GSLIB.locpix_st(
            vmap, Xmin, Xmax, Ymin, Ymax, xsize, 0, 1, self.dataset, self.xcoor, self.ycoor, self.feature,
            vmap_title, xlabel, ylabel, self.feature + ' $^2$)', plt.cm.plasma
        )

        plt.subplots_adjust(left=left_adj, bottom=bot_adj, right=right_adj, top=top_adj, wspace=w_adj, hspace=h_adj)
        if save:
            plt.savefig('Kriging Estimate and Variance in Euclidean Space.tiff', dpi=300, bbox_inches='tight')
        plt.show()


class Dissimilarity:
    def __init__(self, xcoor, ycoor):
        """
        Parameters
        ----------
        xcoor, ycoor: str
            The names of the spatial coordinates.
        """
        self.xcoor = xcoor
        self.ycoor = ycoor

    def dij_spatial(self, vario_model, normalized_eucl_df, verbose=True):
        """
        This function computes the spatial dissimilarity matrix in Euclidean space

        Arguments
        ---------
        vario_model:dict
            A dictionary with all variogram parameters for the response.
        normalized_eucl_df: DataFrame
            A dataframe that contains both X and Y coordinates in normalized Euclidean space.
        verbose: bool
            True to print additional information.

        Returns
        -------
        covariances, vec_gamma, vec_cov, data_pairs, Dij_spatial, Dij_spatial_wt
        """

        # Load the variogram
        nst = vario_model["nst"]
        cc = np.zeros(nst)
        aa = np.zeros(nst)
        it = np.zeros(nst)
        ang = np.zeros(nst)
        anis = np.zeros(nst)

        c0 = vario_model["nug"]
        cc[0] = vario_model["cc1"]
        it[0] = vario_model["it1"]
        ang[0] = vario_model["azi1"]
        aa[0] = vario_model["hmaj1"]
        anis[0] = vario_model["hmin1"] / vario_model["hmaj1"]
        if nst == 2:
            cc[1] = vario_model["cc2"]
            it[1] = vario_model["it2"]
            ang[1] = vario_model["azi2"]
            aa[1] = vario_model["hmaj2"]
            anis[1] = vario_model["hmin2"] / vario_model["hmaj2"]

        rotmat, maxcov = geostats.setup_rotmat(c0, nst, it, cc, ang, 99999.9)

        X = normalized_eucl_df.loc[:, 'NS_' + self.xcoor].values
        Y = normalized_eucl_df.loc[:, 'NS_' + self.ycoor].values
        i_list = []
        j_list = []
        covariances = []
        alphas = []
        betas = []

        for i in range(0, len(X)):
            for j in range(i + 1, len(Y)):
                cov = cova2(X[i], Y[i], X[j], Y[j], nst, 9999.9, cc, aa, it, anis, rotmat, maxcov)
                i_list.append((X[i], Y[i]))
                j_list.append((X[j], Y[j]))
                alpha = 1 - cov
                beta = 1 - alpha
                covariances.append(cov)
                alphas.append(alpha)
                betas.append(beta)

        mat_i = np.asarray(i_list)
        mat_j = np.asarray(j_list)
        data_pairs = np.column_stack((mat_i, mat_j))
        vec_gamma = np.asarray(alphas)
        vec_cov = np.asarray(betas)

        # non-weighted euclidean distance
        Dij_spatial = euclidean_dist(data_pairs)

        # weighted euclidean distance
        Dij_spatial_wt = euclidean_dist(data_pairs, vec_gamma)

        if verbose:
            print('The minimum and maximum variogram covariances are ', round(min(covariances), 5), 'and',
                  round(max(covariances), 5), 'respectively.\n')
            print('The minimum and maximum variogram spatial weights are ', round(min(alphas), 5), 'and',
                  round(max(alphas), 5), 'respectively.')

        return covariances, vec_gamma, vec_cov, data_pairs, Dij_spatial, Dij_spatial_wt

    def dij_multivariate(self, normalized_eucl_df, features, betas=None):
        """
        This function computes the multivariate dissimilarity matrix in feature space

        Arguments
        ---------
        normalized_eucl_df: pandas.DataFrame
            A dataframe of both X and Y coordinates in normalized Euclidean space.

        features: pandas.DataFrame
            A dataframe consisting ONLY of all normalized multivariate predictors in the feature space.

        betas: np.array
            A vector array of weights used to modify the mahalanobis distance calculation obtained from the
            Euclidean space.

        Returns
        -------
        df, df_multi, Dij_multiv, Dij_multiv_wt
        """

        normalized_eucl_df_copy = normalized_eucl_df.copy()

        X = normalized_eucl_df_copy.loc[:, 'NS_' + self.xcoor].values
        Y = normalized_eucl_df_copy.loc[:, 'NS_' + self.ycoor].values
        features_title = features.columns
        df = pd.DataFrame()

        listi = []
        listj = []
        listq = []

        for feat_i in range(features.shape[1]):  # loop through the feature
            for samples in range(0, len(X)):  # samples
                for q in range(samples + 1, len(features[features_title[feat_i]])):
                    listi.append(X[samples])
                    listj.append(Y[samples])
                    listq.append(features[features_title[feat_i]][q])

        mati = np.asarray(listi)
        matq = np.asarray(listq)
        pair_feat = np.column_stack((mati, matq))
        multi = np.asarray(np.split(pair_feat, len(features_title), axis=0))

        for k in range(features.shape[1]):
            df['i'] = multi[0, :][:, 0]
            df[features_title[k]] = multi[k, :][:, 1]
        df_multi = df.iloc[:, 1:].copy(deep=True)  # multiariate features

        # non-weighted mahalanobis
        Dij_multiv = mahalanobis_dist(features, features_title)

        # weighted mahalanobis
        if betas is not None:
            Dij_multiv_wt = mahalanobis_dist(features, features_title, betas)
            return df, df_multi, Dij_multiv, Dij_multiv_wt
        else:
            return df, df_multi, Dij_multiv

    @staticmethod
    def dij_spatialmultivariate(Dij_spatial_wt, Dij_multiv_wt):
        """
        This function computes the dissimilarity matrix consisting of both spatial and multivariate contributions.

        Arguments
        ---------
        Dij_spatial_wt: numpy.array
            A squareform ndarray consisting of the dissimilarity calculated from the spatially weighted Euclidean
            distance function dij_spatial.

        Dij_multiv_wt: numpy.array
            A squareform ndarray consisting of the dissimilarity calculated from the spatially weighted Mahalanobis
            distance function dij_multivariate.

        Returns
        -------
        Dij_SM
        """

        Dij_SM = Dij_spatial_wt + Dij_multiv_wt
        return Dij_SM


class SubRoutine:
    def __init__(self, dataset, xcoor, ycoor, normalize=False, keep_only_norm_features=False):
        """
        Parameters
        ----------
        dataset: DataFrame
            The dataframe consisting of the columns to be clustered both in a space of choice and normalized version of
            same space.
        xcoor, ycoor: str
            The names of the spatial coordinates.
        """
        self.normalize = False
        if normalize:
            self.coords = normalizer(dataset, [xcoor, ycoor], keep_only_norm_features=keep_only_norm_features)
        else:
            self.coords = dataset.loc[:, [xcoor, ycoor]].copy()
        self.xcoor = xcoor
        self.ycoor = ycoor

    def nearest_neighbor(self, space_index, elbow_title_coord=None, annot_coords_x=None, annot_coords_y=None,
                         verbose=True, save=False):
        """
        This function evaluates the intersample distance in Euclidean space using Nearest Neighbor (NN) graphically,
        which corresponds to the eps parameter value in DBSCAN.

        Parameters
        ----------
        space_index: int
            An integer that assigns the name of the current space. Euclidean: 1; Feature: 2; MDS: other

        elbow_title_coord: list
            A list consisting of x,y text coordinate placement for eps graphical identification.

        annot_coords_x: list
            A list of length 2 consisting of x coordinate placement for eps graphical identification.

        annot_coords_y: list
            A list of length 2 consisting of y coordinate placement for eps graphical identification.

        verbose: bool
            True to print additional information.

        save: bool
            True to save the resulting image in tiff format.

        Returns
        -------
        lag_dist: numpy.array
            
        """
        X = self.coords.values
        if space_index == 1:
            space_index = 'Euclidean'

        elif space_index == 2:
            space_index = 'Feature'

        else:
            space_index = 'MDS'

        # Input preparation
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]

        # get nominal distance between samples such that it can be used as lag_distance input
        lag_dist = np.mean(distances)

        # Make figure
        plt.plot(distances, c='black')
        plt.xlabel('Sorted Ascending, Data Index')
        plt.ylabel('Intersample Distance in ' + space_index + ' Space')
        plt.title('Nearest Neighbour ')
        if annot_coords_x is not None or annot_coords_y is not None:
            plt.plot(annot_coords_x, annot_coords_y, color='black', linestyle='--')
            plt.text(elbow_title_coord[0], elbow_title_coord[1], r'Elbow location with optimal epsilon', size=10)
        if save:
            plt.savefig('Eps determination using NN.tiff', dpi=300, bbox_inches='tight')
        plt.show()

        if verbose:
            print(f'The intersample distance is {lag_dist:.5f}')

        return lag_dist

    def dbscan_plotter(self, eps, min_samp, cluster_label, xlabel, ylabel, space_index, palette_, workflow, test,
                       xlabel_units, ylabel_units, save=False):
        """
        Plot the final clusters assigned by DBSCAN using tuned parameters with grouping proportion.

        Parameters
        ---------
        eps: float
            Tuned epsilon parameter from DBSCAN clustering.

        min_samp: int
            Tuned minimum samples parameter form DBSCAN clustering.

        cluster_label: str
            A string representing DBSCAN labels in the dataframe inputted when created.

        xlabel, ylabel: str
            Coordinates labels consisting  representing the X and Y coordinates in space.

        palette_: int
            Either 1 or 2 that assigns a dictionary of colors that maps the hue variable consisting of the DBSCAN label.

        space_index: int
            It is an integer that assigns the name of the space worked in
            T = ['Euclidean Space' , 'Feature Space' ,'MDS Space'].

        workflow: str
            A string for type of workflow to run, close-ology, multivariate or proposed method.

        test: bool
            A True/False boolean to indicate workflow use (i.e., main workflow or validation workflow for proposed
             method).

        xlabel_units, ylabel_units: str

        save: bool

        Returns
        -------
        df_cluster, df_outlier, merged_df, df_cluster[label].unique()
        """
        # Input preparation
        # todo self.normalize if True call NS_ + xcoor, else call xcoor
        model = self.coords.loc[:, [xlabel, ylabel]].copy().values
        dbscan = DBSCAN(eps=eps, min_samples=min_samp).fit(model)
        self.coords[cluster_label] = dbscan.labels_ + 1
        df_cluster = self.coords.loc[(self.coords[cluster_label] != 0)]
        df_outlier = self.coords.loc[(self.coords[cluster_label] == 0)]
        merged_df = pd.concat([df_cluster, df_outlier])

        # Make Plotter
        fig, axs = plt.subplots(nrows=1, ncols=2)

        # Plot for group proportions of clusters in data
        N, bins, patches = axs[1].hist(self.coords[cluster_label], alpha=0.5, edgecolor="black",
                                       bins=np.arange(-0.5, len(self.coords[cluster_label].unique()), 1),
                                       range=[0.5, len(self.coords[cluster_label].unique()) + 0.5], density=True)

        # Palette assignment
        if palette_ == 1:
            palette_ = {1: 'blue', 2: 'magenta', 3: 'green', 4: 'yellow', 5: 'red', 6: 'cyan', 7: 'brown',
                        8: 'burlywood', 9: 'orange', 10: 'cornflowerblue',
                        11: 'darkorchid', 12: 'palevioletred', 13: 'darkgoldenrod', 14: 'thistle',
                        0: 'black'}  # more colors can be added accordingly for more colors #more colors can be 
            # added accordingly for more colors

            for i in range(0, len(self.coords[cluster_label].unique())):
                patches[i].set_facecolor(str(palette_[i]))

        elif palette_ == 2:
            color_labels = self.coords['DB-label'].unique()
            rgb = sns.color_palette("bright",
                                    len(color_labels))  # this can be changed to "husl", "tab10",colorblind",
            # "deep","Set1" or any other color scheme that can be found online in seaborn
            palette_ = dict(zip(color_labels, rgb))

            for i in palette_:
                patches[i].set_facecolor(palette_[i])

        axs[1].set_title('Group Proportions', size=16)
        axs[1].set_xlabel('Cluster Labels', size=16)
        axs[1].set_ylabel("Proportion", size=16)
        axs[1] = plt.gca()
        axs[1].set_xticks(np.arange(0, len(self.coords[cluster_label].unique()), 1))
        axs[1].set_yticks(np.arange(0, 1.2, 0.2))
        axs[1].set_yticks(np.arange(0, 1.2, 0.05), minor=True)
        axs[1].tick_params(axis='both', which='major', labelsize=14)
        axs[1].grid(which='both', alpha=0.01)

        T = ['Euclidean Space', 'Feature Space', 'MDS Space']
        P = [xlabel_units, xlabel_units, 'MDS 1']
        Q = [ylabel_units, ylabel_units, 'MDS 2']

        if space_index == 1:
            space_index = space_index - 1

        elif space_index == 2:
            space_index = space_index - 1

        else:
            space_index = 2

        if test is True:
            array = ['10001A', '10002A', '10003A', '10004A']  # array of test samples for validation
            test_samp = merged_df.loc[merged_df['API'].isin(array)]

            if workflow == 'closeology':
                tester = pd.DataFrame([1, 4, 0, 2], columns=[cluster_label])
                annot = ['c', 'i', 'o', 's']

            else:
                tester = pd.DataFrame([1, 2, 4, 0], columns=[cluster_label])
                annot = ['c', 's', 'i', 'o']

            axs[0].scatter(test_samp[xlabel], test_samp[ylabel], marker='*', s=800,
                           c=test_samp[cluster_label].map(palette_),
                           alpha=1.0,  # START & PICK cuz of size edit
                           linewidths=1.0, edgecolor='black')
            axs[0].scatter(df_cluster[xlabel], df_cluster[ylabel], s=100, c=df_cluster[cluster_label].map(palette_),
                           alpha=0.8,
                           linewidths=1.5, edgecolor='black')
            axs[0].scatter(df_outlier[xlabel], df_outlier[ylabel], c='black', s=80, marker='x')

            xs = test_samp[xlabel].tolist()
            ys = test_samp[ylabel].tolist()

            for k, txt in enumerate(annot):
                axs[0].annotate(txt, (xs[k], ys[k]), fontsize=12, ha='center', va='center', c='red',
                                weight='bold')  # START & STOP

        else:
            axs[0].scatter(df_cluster[xlabel], df_cluster[ylabel], s=100, c=df_cluster[cluster_label].map(palette_),
                           alpha=0.8,
                           linewidths=1.5, edgecolor='black')
            axs[0].scatter(df_outlier[xlabel], df_outlier[ylabel], c='black', s=80, marker='x')

        axs[0].set_title('DBSCAN in ' + T[space_index], size=16)
        axs[0].set_xlabel(P[space_index], size=16)
        axs[0].set_ylabel(Q[space_index], size=16)
        axs[0].tick_params(axis='both', which='major', labelsize=14)
        axs[0].grid(which='both', color='black', alpha=0.01)

        # For legend assignment
        elements = []

        # Hardcode outlier labeled cluster into legend
        outlier_ = [Line2D([0], [0], marker='X', color='w', label='cluster 0',
                           markerfacecolor='black',
                           markersize=12)]
        elements.append(outlier_)

        for i in range(1, (len(df_cluster[cluster_label].unique()) + 1)):
            element = [Line2D([0], [0], marker='o', color='w', label='cluster ' + str(i),
                              markerfacecolor=palette_[i], markeredgecolor='black', markeredgewidth=1.0, markersize=12)]
            elements.append(element)

        legend_elements = list(chain(*elements))  # unlist the legend handles

        # Aesthetics for plot
        plt.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.50, wspace=0.2, hspace=0.2)
        axs[0].legend(bbox_to_anchor=(-0.5, 0.5), handles=legend_elements, loc='center', fontsize=17)
        if save:
            plt.savefig('DBSCAN final figure with cluster proportions.tif', dpi=300, bbox_inches='tight')
        plt.show()

        return df_cluster, df_outlier, merged_df, df_cluster[cluster_label].unique()

    def dbscan_tuner(self, min_samp, start, stop, step, x, y, x_ext, y_ext, save=False):
        """
        Tune the DBSCAN algorithm and output already tuned parameters epsilon and min. samples with its associated
        silhouette scores.

        Arguments
        ---------
        min_samp: int
            Arbitrary minimum number of samples required to make a cluster from DBSCAN, as low as reasonably possible as
            it will be tuned eventually.

        start, stop, step: float
            Values consisting of the start, stop, and step values used to make the range of epsilon values to try to
             find its tuned variant.

        x, y: float
            Start values required to make a rectangle patch.

        x_ext, y_ext: float
            Delta values x,y need to be extended by to make a rectangle patch.
            
        save: bool
            True to save image in tiff format.

        Returns
        ---------
        eps, max(silhouette_list), min_samples, min_samples2, max(silhouette_avg_n_cluster)
        """
        X = self.coords[['NS_' + self.xcoor, 'NS_' + self.ycoor]].values
        # Input preparation
        # To determine the hyperparameter epsilon in DBSCAN, we use nearest neighbors to obtain "eps" range.
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]

        # We obtain the range where epsilon lies from the elbow method, and use that to get the tuned epsilon and tune
        # the value for min_samples using silhouette scores.

        # Epsilon tuning
        silhouette_list = []
        eps_list = []
        n_clusters = []
        for i in np.arange(start, stop + step, step):
            db = DBSCAN(eps=i, min_samples=min_samp).fit(X)
            labels = db.labels_ + 1
            silhouette_avg_ = silhouette_score(X, labels)
            eps_list.append(i)
            silhouette_list.append(silhouette_avg_)
            n_clusters.append(len(np.unique(labels)))

        maxi = silhouette_list.index(max(silhouette_list))
        min_samples = n_clusters[maxi]
        eps = eps_list[maxi]

        # Plotter for epsilon determination
        fig, axs = plt.subplots(nrows=1, ncols=2)

        axs[0].plot(distances, c='black')
        axs[0].add_patch(
            Rectangle((x, y), x_ext, y_ext, alpha=1, ls='--', edgecolor='red', facecolor='none'))  # fill=None))
        axs[0].set_xlabel('Sorted Ascending, Data Index', size=16)
        axs[0].set_ylabel('Intersample Distance', size=16)
        axs[0].set_title("Nearest Neighbor Method", size=16)  # Epsilon Determination via Elbow Method
        axs[0].tick_params(axis='both', which='major', labelsize=14)

        axs[1].plot(eps_list, silhouette_list, c='blue', linestyle='--', label='Epsilon')
        axs[1].scatter(eps, max(silhouette_list), marker='s', s=60, c='blue', label='Eps')
        axs[1].set_xlabel('Epsilon', color='black', size=16)
        axs[1].set_ylabel('Silhouette Score', size=16)
        axs[1].set_title("Silhouette Method", size=16)
        axs[1].tick_params(axis='both', which='major', labelsize=14)

        # For min. samples tuning
        range_n_cluster = range(2, min_samp + 1)
        silhouette_avg_n_cluster = []

        for n_cluster in range_n_cluster:
            # Initialize the clusterer with the Eps value gotten from DBBSCAN_tune_plotter function
            clusterer = DBSCAN(eps=eps, min_samples=n_cluster).fit(X)
            cluster_labels = clusterer.labels_ + 1

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters
            silhouette_avg2 = silhouette_score(X, cluster_labels)
            silhouette_avg_n_cluster.append(silhouette_avg2)

            # Compute the silhouette scores for each sample
            # sample_silhouette_values2 = silhouette_samples(X, cluster_labels)

        maxi2 = silhouette_avg_n_cluster.index(max(silhouette_avg_n_cluster))
        min_samples2 = range_n_cluster[maxi2]

        # Aesthetics for plot
        lines, labels = axs[1].get_legend_handles_labels()
        axs[1].legend(lines, labels, fontsize=14, loc='best')

        plt.subplots_adjust(left=0.0, bottom=0.0, right=3.0, top=1.50, wspace=0.35, hspace=0.2)
        if save:
            plt.savefig('Epsilon determination.tiff', dpi=300, bbox_inches='tight')
        plt.show()

        return eps, max(silhouette_list), min_samples, min_samples2, max(silhouette_avg_n_cluster)

    def dbscan_sensitivity(self, df, eps_mat, min_sample_mat, cluster_label, space_index, xlabel_units, ylabel_units,
                           verbose=True, save=False):
        """
        This function performs a graphical sensitivity check for tuning the appropriate min samples to use in DBSCAN.

        Arguments
        ---------
        df: the dataframe consisting of the columns to be clustered both in a space of choice and normalized version of
         same space i.e., [X1, X2, X1 normalized, X2 normalized] in this order

        eps_mat: a list of DBSCAN eps parameter to be checked.

        min_sample_mat: a list of minimum samples DBSCAN parameter to be checked, should be the same length as eps_mat,
        which is preferably 3. If you increase this value adjust the figure visualization parameters to see all.

        cluster_label: a string name representing DBSCAN labels in the dataframe inputted when created

        space_index: is an integer that assigns the name of the space worked in T

        verbose: bool

        xlabel_units, ylabel_units: str

        save: bool

        True to print information.
        """
        cmap = plt.cm.plasma
        cluster_list = []  # list that stores the unique clusters for each DBSCAN realization made for sensitivity

        T = ['Euclidean Space', 'Feature Space', 'MDS Space']
        P = [xlabel_units, xlabel_units, 'MDS 1']
        Q = [ylabel_units, ylabel_units, 'MDS 2']

        if space_index == 1:
            space_index = space_index - 1

        elif space_index == 2:
            space_index = space_index - 1

        else:
            space_index = 2

        index = 1
        for eps in eps_mat:
            for min_sample in min_sample_mat:
                dbscan = DBSCAN(eps=eps, min_samples=min_sample).fit(df.iloc[:, [2, 3]].values)
                df[cluster_label] = dbscan.labels_ + 1

                plt.subplot(len(eps_mat), len(eps_mat), index)
                df_in = df.loc[(df[cluster_label] != 0)]
                plt.scatter(df_in.iloc[:, 0], df_in.iloc[:, 1], c=df_in[cluster_label], alpha=0.5, edgecolor='k',
                            cmap=cmap)
                df_outlier = df.loc[(df[cluster_label] == 0)]
                number_clusters = len(df_in[cluster_label].unique())
                cluster_list.append(number_clusters)
                plt.scatter(df_outlier.iloc[:, 0], df_outlier.iloc[:, 1], c='black', s=50, marker='x',
                            )
                plt.title('DBSCAN in ' + T[space_index] + ',\n eps = ' + str(eps) + ', min sample = ' + str(min_sample),
                          size=16)
                plt.xlabel(P[space_index], size=16)
                plt.ylabel(Q[space_index], size=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                index = index + 1
        plt.subplots_adjust(left=0.0, bottom=0.0, right=3.0, top=3.50, wspace=0.3, hspace=0.4)
        if save:
            plt.savefig('DBSCAN min_sample sensitivity.tiff', dpi=300, bbox_inches='tight')
        plt.show()

        if verbose:
            print('Number of unique clusters in each subplot realization', cluster_list)

        return cluster_list


class ProcessPlots:
    def __init__(self, df, kriging_response_euclidean, kriging_response_mds):
        self.krig_response_eucl = kriging_response_euclidean
        self.krig_response_mds = kriging_response_mds
        self.df = df

        self.k1min = kriging_response_euclidean.min()
        self.k2min = kriging_response_mds.min()
        self.k1max = kriging_response_euclidean.max()
        self.k2max = kriging_response_mds.max()

    def analog_maps(self, cluster, outlier, X, Y, T, cluster_label, x_labels, y_labels, cb_title, palette_, workflow,
                    cmap, test, save=False, offset_eucl_x=(1, 1), offset_eucl_y=(1, 1), offset_mds_x=(1, 1),
                    offset_mds_y=(1, 1)):
        """
        This function assigns analogs and plots the mappings for the data in both feature and MDS spaces.
        NOTE: The column name for the DBSCAN label in the dataframes in cluster and outlier variables should be the
        same.

        Arguments
        ----------
        cluster: list
        A list consisting of two dataframes made of only the main clusters found via DBSCAN in feature and MDS space,
        respectively.

        outlier: list
        A list consisting of two dataframes made of only the outliers found from DBSCAN in feature and MDS space.

        X: list
        A list consisting of two items with type string representing the x coordinates in feature and MDS space.

        Y: list
        A list consisting of two items with type string representing the y coordinates in feature and MDS space.

        T: list
        A list comprising two items with type string representing the titles of the subplots made for the feature and
        MDS spaces.

        x_labels: list
        A list comprising two items with type string representing the x-labels of the subplots made in feature, and
        MDS spaces respectively

        y_labels: list
        A list comprising two items with type string representing the y-labels of the subplots made in feature, and
        MDS spaces respectively

        cluster_label: str
        A string consisting representing DBSCAN labels from a dataframe

        cb_title: str
        A string representing the title of the color bar

        palette_: int
        An integer either 1 or 2 that assigns a dictionary of colors that maps the hue variable consisting of the DBSCAN
        label.
-
        workflow: str
        A string for type of workflow to run, closeology, multivariate or proposed method.

        cmap: matplotlib.pyplot.cm
        Atring that assigns a colormap of the values displayed from matplotlib.pyplot.cm.

        test: bool
        A True/False boolean to indicate workflow use i.e., main workflow or validation workflow for the proposed
        method.

        save: bool

        offset_eucl_x, offset_eucl_y: tuple

        offset_mds_x, offset_mds_y: tuple
        """
        im1 = None

        # Create extent for background map and joint color bar for the subplots using X,Y coordinates

        # Feature Space
        df1 = pd.concat([cluster[0], outlier[0]], axis=0)

        xmin = df1[X[0]].min() * offset_eucl_x[0]
        xmax = df1[X[0]].max() * offset_eucl_x[1]
        ymin = df1[Y[0]].min() * offset_eucl_y[0]
        ymax = df1[Y[0]].max() * offset_eucl_y[1]

        # MDS Space
        df2 = pd.concat([cluster[1], outlier[1]], axis=0)

        xmin2 = df2[X[1]].min() * offset_mds_x[0]
        xmax2 = df2[X[1]].max() * offset_mds_x[1]  # increment added to ensure all samples are fully captured
        ymin2 = df2[Y[1]].min() * offset_mds_y[0]
        ymax2 = df2[Y[1]].max() * offset_mds_y[1]

        # Obtain input for plot making
        Xmins = [xmin, xmin2]
        Xmaxs = [xmax, xmax2]
        Ymins = [ymin, ymin2]
        Ymaxs = [ymax, ymax2]
        K = [self.krig_response_eucl, self.krig_response_mds]
        Vmin = [self.k1min, self.k2min]
        Vmax = [self.k1max, self.k2max]

        cen_df = cluster[1].groupby(
            cluster_label).mean()  # dataframe consisting of the centroids of each cluster assigned as analogs in the MDS space
        # should be in the same format as df, and df2 inclusive of column names.

        # Basis for automated subplot
        num_cols = 2
        subplot_nos = 2
        if subplot_nos % num_cols == 0:
            num_rows = subplot_nos // num_cols
        else:
            num_rows = (subplot_nos // num_cols) + 1

        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols)

        # For plot making
        for j in range(0, len(cluster)):

            # Palette assignment
            if palette_ == 1:
                palette_ = {1: 'blue', 2: 'magenta', 3: 'green', 4: 'yellow', 5: 'red', 6: 'cyan', 7: 'brown',
                            8: 'burlywood', 9: 'orange', 10: 'cornflowerblue',
                            11: 'darkorchid', 12: 'palevioletred', 13: 'darkgoldenrod', 14: 'thistle',
                            0: 'black'}  # more colors can be added accordingly for more colors #more colors can be
                # added accordingly for more colors

                palette_2 = list(palette_.values())[:len(cluster[j][cluster_label].unique())]

            elif palette_ == 2:
                color_labels = cluster[j][cluster_label].unique()
                rgb = sns.color_palette("bright",
                                        len(color_labels))  # this can be changed to "husl", "tab10",colorblind","deep",
                # "Set1" or any other color scheme that can be found online in seaborn
                palette_ = dict(zip(color_labels, rgb))
                palette_2 = cen_df.index.map(palette_)

            ax = axs[j]

            if test == True:
                merged_df = pd.concat([cluster[j], outlier[j]])
                array = ['10001A', '10002A', '10003A', '10004A']  # array of test samples for validation
                test_samp = merged_df.loc[merged_df['API'].isin(array)]

                if workflow == 'closeology':
                    tester = pd.DataFrame([1, 4, 0, 2], columns=[cluster_label])
                    annot = ['c', 'i', 'o', 's']

                else:
                    tester = pd.DataFrame([1, 2, 4, 0], columns=[cluster_label])
                    annot = ['c', 's', 'i', 'o']

                im1 = ax.imshow(
                    K[j],
                    vmin=Vmin[j],
                    vmax=Vmax[j],
                    extent=(Xmins[j], Xmaxs[j], Ymins[j], Ymaxs[j]),
                    aspect=1,
                    cmap=cmap,
                    interpolation=None,
                    origin='lower'
                )
                im2 = ax.scatter(
                    cluster[j][X[j]],
                    cluster[j][Y[j]],
                    c=cluster[j][cluster_label].map(palette_),
                    s=60,
                    alpha=1.0,
                    linewidths=1.0,
                    edgecolors="black"
                )
                im3 = ax.scatter(
                    outlier[j][X[j]],
                    outlier[j][Y[j]],
                    c='black',
                    s=60,
                    marker='x',
                    alpha=1.0,
                    linewidths=1.0
                )
                im4 = axs[1].scatter(
                    cen_df[X[1]],
                    cen_df[Y[1]],
                    marker='P',
                    s=150,
                    c=palette_2,
                    linewidths=1.0,
                    edgecolors="black"
                )
                im5 = ax.scatter(
                    test_samp[X[j]],
                    test_samp[Y[j]],
                    c=test_samp[cluster_label].map(palette_),
                    marker='*',
                    s=700,
                    alpha=1.0,
                    linewidths=1.0,
                    edgecolors='black'
                )

                xs = test_samp[X[j]].tolist()
                ys = test_samp[Y[j]].tolist()

                for k, txt in enumerate(annot):
                    ax.annotate(txt, (xs[k], ys[k]), fontsize=12, ha='center', va='center', c='red',
                                weight='bold')  # START & STOP

            else:
                im1 = ax.imshow(
                    K[j],
                    vmin=Vmin[j],
                    vmax=Vmax[j],
                    extent=(Xmins[j], Xmaxs[j], Ymins[j], Ymaxs[j]),
                    aspect=1,
                    cmap=cmap,
                    interpolation=None,
                    origin='lower'
                )
                im2 = ax.scatter(
                    cluster[j][X[j]],
                    cluster[j][Y[j]],
                    c=cluster[j][cluster_label].map(palette_),
                    s=60,
                    alpha=1.0,
                    linewidths=1.0,
                    edgecolors="black"
                )
                im3 = ax.scatter(
                    outlier[j][X[j]],
                    outlier[j][Y[j]],
                    c='black',
                    s=60,
                    marker='x',
                    alpha=1.0,
                    linewidths=1.0
                )
                im4 = axs[1].scatter(
                    cen_df[X[1]],
                    cen_df[Y[1]],
                    marker='P',
                    s=150,
                    c=palette_2,
                    linewidths=1.0,
                    edgecolors="black"
                )

            ax.set_aspect('auto')
            ax.set_title(T[j], size=14)
            ax.set_xlabel(x_labels[j], size=14)

            ax.set_ylabel(y_labels[j], size=14)
            ax.tick_params(axis='both', which='major', labelsize=12)

        # For legend assignment
        elements = []

        # Hardcode outlier labeled cluster into legend
        outlier_ = [Line2D([0], [0], marker='X', color='w', label='cluster 0', markerfacecolor='black', markersize=10)]
        elements.append(outlier_)

        for i in range(1, len(palette_2) + 1):  # for when we have feature space coords too
            element = [Line2D([0], [0], marker='o', color='w', label='cluster ' + str(i), markerfacecolor=palette_[i],
                              markeredgecolor='black', markeredgewidth=1.0, markersize=8)]
            elements.append(element)

        legend_elements = list(chain(*elements))  # unlist the legend handles

        # Aesthetics for plot
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.9, top=1.3, wspace=0.25, hspace=0.3)
        cbar_ax = fig.add_axes([1.97, 0., 0.04, 1.3])  # Left,bottom, width, length
        cbar = fig.colorbar(im1, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(cb_title, rotation=270, labelpad=20, size=14)
        ax.legend(bbox_to_anchor=(-1.6, 0.5), handles=legend_elements, loc='center', fontsize=14)
        if save:
            plt.savefig('Analog Maps.tif', dpi=300, bbox_inches='tight')
        plt.show()

    def trend_map(self, df, Xcoord, Ycoord, K, cb_title, feat_name, units, response, save=False, offset_mds_x=(1, 1),
                  offset_mds_y=(1, 1), shrink_cb=1.0, left_adj=0.0, bot_adj=0.0, right_adj=1.3, top_adj=1,
                  w_adj=0.1, h_adj=0.5
                  ):
        """
        This function visualizes the trend maps of all predictors in tthe MDS space

        Arguments
        ----------
        df: pandas,dataframe
        The dataset of interest.

        Xcoord, Ycoord: str
        Strings for the X and Y coordinate in MDS space.

        K: list
        A list consisting of the array of kriged predictors in MDS space.

        cb_title: str
        Color bar title for the intensity of the response krigged in the background.

        feat_name: list
        All predictors to be visualized in trend map i.e., has been krigged.

        units: list
        A list containing the units of the kriged predictors.

        response: str
        Name for the response title.

        save: bool

        offset_mds_x, offset_mds_y: tuple

        shrink_cb: float
        Shrink parameter of the predictor colorbars.

        left_adj, bot_adj, right_adj, top_adj, w_adj, h_adj: float
            Width and height space among subplots.
        """

        num_cols = 2
        subplot_nos = len(K)
        if subplot_nos % num_cols == 0:
            num_rows = subplot_nos // num_cols
        else:
            num_rows = (subplot_nos // num_cols) + 1
        # Make figure
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 9))
        axs = axs.ravel()
        # Set axes limits in MDS space
        xmin = df[Xcoord].min() * offset_mds_x[0]
        xmax = df[Xcoord].max() * offset_mds_x[1]
        ymin = df[Ycoord].min() * offset_mds_y[0]
        ymax = df[Ycoord].max() * offset_mds_y[1]
        # Set color map
        cmap = 'plasma'
        is_odd = subplot_nos % 2 != 0
        for j in range(0, subplot_nos):
            ax = axs[j]
            # make background figure of kriged response in MDS space
            im = ax.imshow(self.krig_response_mds, vmin=self.k2min, vmax=self.k2max, extent=(xmin, xmax, ymin, ymax),
                           aspect=1,
                           cmap=cmap, interpolation=None, origin='lower')
            # make contours out of kriged predictors
            f1 = ax.contour(K[j], colors='k', vmin=K[j].min(), vmax=K[j].max(), extent=(xmin, xmax, ymin, ymax),
                            levels=np.linspace(K[j].min(), K[j].max(), 8), linewidths=np.linspace(.25, 4, 20))
            # Make a colorbar for the ContourSet returned by the contourf call.
            cbar_f1 = fig.colorbar(f1, ax=ax, shrink=shrink_cb, orientation='vertical', location='right', aspect=10)
            cbar_f1.ax.set_ylabel('Kriged ' + feat_name[j] + units[j], rotation=270, labelpad=30, size=14)
            cbar_f1.ax.tick_params(labelsize=12)
            # Aesthetics for individual subplots
            ax.clabel(f1, fontsize=0, inline=True)
            ax.set_title('Trend Map of ' + feat_name[j] + ' Contoured on \n' + response, size=14)
            ax.set_xlabel('MDS 1', size=14)
            ax.set_ylabel('MDS 2', size=14)
            ax.tick_params(axis='both', which='major', labelsize=12)
            is_last_axes = j == subplot_nos - 1
            if is_last_axes & is_odd:
                fig.delaxes(axs[-1])
        # Aesthetics for plot
        plt.subplots_adjust(left=left_adj, bottom=bot_adj, right=right_adj, top=top_adj,
                            wspace=w_adj, hspace=h_adj)
        cbar_ax = fig.add_axes([1.4, 0.15, 0.03, 0.70])  # Left, Bottom, Width, Length,
        clb = fig.colorbar(im, cax=cbar_ax)
        clb.ax.tick_params(labelsize=12)
        cc = fig.colorbar(im, cax=cbar_ax)
        cc.set_label(cb_title, rotation=270, labelpad=50, size=14)
        if save:
            plt.savefig('Trend Map of Important Predictors in MDS space.tif', dpi=300, bbox_inches='tight')
        plt.show()

    def well_plotter(self, df1, df2, X, Y, subplot_titles, x_labels, y_labels, cb_title, cmap, test, save=False,
                     offset_eucl_x=(1, 1), offset_eucl_y=(1, 1), offset_mds_x=(1, 1), offset_mds_y=(1, 1)):
        """
        This function plots well placements in both feature and MDS spaces.

        Arguments
        ---------
        df1: pandas.DataFrame
        Dataframes with x,y coordinates in feature space.
    
        df2: pandas.DataFrame
        Dataframes with x,y coordinates in MDS space.
    
        X: list
        A list consisting of two items with type string representing the x coordinates in feature and MDS space,
        respectively.

        Y: list
        A list consisting of two items with type string representing the y coordinates in feature and MDS space,
        respectively.

        subplot_titles: list
        A list comprising two items with type string representing the titles of the subplots made for the feature and
        MDS spaces.
    
        x_labels: list
        A list comprising two items with type string representing the x-labels of the subplots made in feature
        and MDS spaces, respectively.
    
        y_labels: list
        A list comprising two items with type string representing the y-labels of the subplots made in feature
        and MDS spaces, respectively.
    
        cb_title: str
        A string representing the title of the color bar.
    
        cmap: str
        String that assigns a colormap of the values displayed from matplotlib.pyplot.cm.
    
        test: bool
        A True/False boolean to indicate workflow use i.e., main workflow or validation workflow for the proposed
        method.

        save: bool
        True to save the image as tif.

        offset_eucl_x, offset_eucl_y: tuple

        offset_mds_x, offset_mds_y: tuple

        """

        # Create extent for background map and joint color bar for the subplots using X,Y coordinates
        # Feature Space
        xmin = df1[X[0]].min() * offset_eucl_x[0]
        xmax = df1[X[0]].max() * offset_eucl_x[1]
        ymin = df1[Y[0]].min() * offset_eucl_y[0]
        ymax = df1[Y[0]].max() * offset_eucl_y[1]

        # MDS Space
        xmin2 = df2[X[1]].min() * offset_mds_x[0]
        xmax2 = df2[X[1]].max() * offset_mds_x[1]  # increment added to ensure all samples are fully captured
        ymin2 = df2[Y[1]].min() * offset_mds_y[0]
        ymax2 = df2[Y[1]].max() * offset_mds_y[1]

        # Obtain input for plot making
        Xmins = [xmin, xmin2]
        Xmaxs = [xmax, xmax2]
        Ymins = [ymin, ymin2]
        Ymaxs = [ymax, ymax2]
        Vmin = [self.k1min, self.k2min]
        Vmax = [self.k1max, self.k2max]
        K = [self.krig_response_eucl, self.krig_response_mds]
        df_list = [df1, df2]
        annot = ['c', 'o', 's', 'i']

        fig, axs = plt.subplots(nrows=1, ncols=2)

        # For plot making
        for j in range(0, len(df_list)):
            ax = axs[j]

            im1 = ax.imshow(K[j], vmin=Vmin[j], vmax=Vmax[j], extent=(Xmins[j], Xmaxs[j], Ymins[j], Ymaxs[j]), aspect=1,
                            cmap=cmap, interpolation=None, origin='lower')

            if test is True:
                im2 = ax.scatter(
                    df_list[j][X[j]][:(len(df_list[j]) - 4)],
                    df_list[j][Y[j]][:(len(df_list[j]) - 4)],
                    c='white',
                    s=60,
                    alpha=1.0,
                    linewidths=1.0,
                    edgecolors="black",
                    label='well'
                )
                ax.scatter(
                    df_list[j][X[j]][(len(df_list[j]) - 4):],
                    df_list[j][Y[j]][(len(df_list[j]) - 4):],
                    marker='*',
                    c='white',
                    s=300,
                    alpha=1.0,
                    linewidths=1.0,  # START & PICK cuz of size edit
                    edgecolors="black", label='test'
                )

                im3 = ax.scatter(
                    df_list[j][X[j]][(len(df_list[j]) - 4):],
                    df_list[j][Y[j]][(len(df_list[j]) - 4):],
                    marker='*',
                    c='white',
                    s=700,
                    alpha=1.0,
                    linewidths=1.0,
                    # START & PICK cuz of size edit
                    edgecolors="black")

                xs = df_list[j][X[j]][(len(df_list[j]) - 4):].tolist()
                ys = df_list[j][Y[j]][(len(df_list[j]) - 4):].tolist()

                for k, txt in enumerate(annot):
                    ax.annotate(txt, (xs[k], ys[k]), fontsize=12, ha='center', va='center', c='black',
                                weight='bold')  # START & STOP

            else:
                im2 = ax.scatter(df_list[j][X[j]], df_list[j][Y[j]], c='white', s=60, alpha=1.0, linewidths=1.0,
                                 edgecolors="black", label='sample')

            ax.legend(fontsize=12)
            ax.set_aspect('auto')
            ax.set_title(subplot_titles[j], size=14)
            ax.set_xlabel(x_labels[j], size=14)
            ax.set_ylabel(y_labels[j], size=14)
            ax.tick_params(axis='both', which='major', labelsize=12)

        # Aesthetics for plot
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.9, top=1.3, wspace=0.25, hspace=0.3)
        cbar_ax = fig.add_axes([1.97, 0., 0.04, 1.3])  # Left,bottom, width, length
        cbar = fig.colorbar(im1, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(cb_title, rotation=270, labelpad=20, size=14)
        if save:
            plt.savefig('Well Placements in Space.tif', dpi=300, bbox_inches='tight')
        plt.show()


# future work
def visualize_model(model,
                    xfeature,
                    x_min,
                    x_max,
                    yfeature,
                    y_min,
                    y_max,
                    response,
                    title,
                    xlab,
                    ylab):
    """
    This function plots the data points and the decision tree prediction.

    Arguments
    ---------

    model: classification model built/fit

    xfeature: series consisting of feature to be on xaxis i.e., MDS 1

    x_min: min value for xfeature

    x_max: max value for xfeature

    yfeature: series consisting of the feature to be on the yaxis i.e., MDS 2

    y_min: min value for yfeature

    y_max: max value for yfeature

    response: series consisting of DBSCAN clustering labels/groupings found

    title: string representing title for visuals made

    xlab: x-axis label

    ylab: y-axis label
    """

    # Input preparation

    z_min = min(response)  # min values for response variable i.e., clusters found excluding outlier groupings
    z_max = max(response)  # max values for response variable i.e., clusters found excluding outlier groupings

    cmap = plt.cm.plasma
    # Resolution of the model visualization
    xplot_step = (x_max - x_min) / 300.0
    yplot_step = (y_max - y_min) / 300.0

    # Set up the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, xplot_step),
                         np.arange(y_min, y_max, yplot_step))

    # Predict with our trained model over the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the predictions
    plt.contourf(xx, yy, Z, cmap=cmap, vmin=z_min, vmax=z_max, levels=100)

    # Add the data values as a colored by response feature scatter plot
    im = plt.scatter(xfeature, yfeature, s=None, c=response, marker=None, cmap=cmap, norm=None, vmin=z_min, vmax=z_max,
                     alpha=0.8, linewidths=0.3, edgecolors="black")

    # Aesthetics for plot
    plt.title(title, size=12)
    plt.xlabel(xlab, size=12)
    plt.ylabel(ylab, size=12)
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar = plt.colorbar(im, orientation='vertical')  # add the color bar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Cluster labels', rotation=270, labelpad=20, size=12)
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.2, wspace=0.2, hspace=0.4)
    plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    return


def visualize_model_prob(model, xfeature, x_min, x_max, yfeature, y_min, y_max, response, title, xlab, ylab):
    """
    This function plots the data points and the prediction probabilities.

    Arguments
    ---------

    model: classification model built/fit

    xfeature: series consisting of feature to be on xaxis i.e., MDS 1

    x_min, x_max: min and max values for xfeature

    yfeature: series consisting of the feature to be on the yaxis i.e., MDS 2

    y_min, y_max: min and max values for yfeature

    response: series consisting of DBSCAN clustering labels/groupings found

    title: string consisting of title for figure generated

    xlab, ylab: x-axis and y-axis labels
    """

    # Input preparation
    cluster_nos = len(response.unique())
    cmap = plt.cm.plasma
    xplot_step = (x_max - x_min) / 300.0
    yplot_step = (y_max - y_min) / 300.0  # resolution of the model visualization
    xx, yy = np.meshgrid(np.arange(x_min, x_max, xplot_step),  # set up the mesh
                         np.arange(y_min, y_max, yplot_step))

    z_min = 0.0
    z_max = 1.0
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    num_cols = 2
    if cluster_nos % num_cols == 0:
        num_rows = cluster_nos // num_cols
    else:
        num_rows = (cluster_nos // num_cols) + 1

    fig, axs = plt.subplots(figsize=(17, 12), nrows=num_rows, ncols=num_cols)

    for i in range(0, cluster_nos):
        Zi = Z[:, i].reshape(xx.shape)
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]

        cs1 = ax.contourf(xx, yy, Zi, cmap=cmap, vmin=z_min, vmax=z_max, levels=np.linspace(z_min, z_max, 100))
        ax.scatter(xfeature, yfeature, s=500, c=response, marker=None, cmap=plt.cm.Greys, norm=None, vmin=z_min,
                   vmax=z_max, alpha=0.8, linewidths=0.3, edgecolors="black")
        ax.set_aspect('auto')
        ax.set_title(title + ' Probability of Label ' + str(i + 1), size=42)
        ax.set_xlabel(xlab, size=42)
        ax.set_ylabel(ylab, size=42)
        ax.tick_params(axis='both', which='major', labelsize=42)
        cbar = fig.colorbar(cs1, ax=ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=42)
        cbar.set_label('Probability', rotation=270, labelpad=60, size=42)

    plt.subplots_adjust(left=0.0, bottom=0.0, right=2.5, top=3.2, wspace=0.2, hspace=0.3)
    plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    return


def make_confusion_matrix(prediction, response,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='RdBu',
                          title=None):
    """
    This function will make a pretty plot of a sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    prediction:    series consisting of the predicted DBSCAN clustering labels/groupings found for the attributed train
     or test set

    response:      series consisting of the actual DBSCAN clustering labels/groupings found for the attributed train or
     test set

    group_names:   List of strings that represent the labels row by row to be  shown in each square.

    categories:    List of strings containing the categories to be displayed on the x-axis,y-axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
    Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'RdBu'
                   See https:///matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    """

    # Make the confusion matrix
    cf = confusion_matrix(response, prediction)

    # Generate text in each square
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # Code to generate summary statistics & text for summary stats
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # If it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # Set figure parameters according to other arguments
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks is False:
        # Do not show categories if xyticks is False
        categories = False

    # Make the heatmap visualization
    plt.figure(figsize=figsize)
    no_cluster = max(response)
    l_ = range(1, no_cluster + 1, 1)
    categories = list(map(str, l_))
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label', size=12)
        plt.xlabel('Predicted label' + stats_text, size=12)
        plt.tick_params(axis='both', which='major', labelsize=12)

    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
        plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
        plt.show()
    return


def silhouette_plotter(X, min_samp_max, eps):
    """
    This is a function that checks if the partition found reflects a clustering structure actually present in the data
    or if we have partitioned the data into artificial groups based on tuning min samples.

    Arguments
    ---------
    X: TODO

    min_samp_max: TODO

    eps: TODO
    """

    range_n_clusters = range(2, min_samp_max + 1)
    silhouette_avg_n_clusters = []

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 you can truncate the limit to lie within [a, b] based on data
        ax1.set_xlim([-1., 1.])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with the Eps value gotten from DBBSCAN_tune_plotter function
        clusterer = DBSCAN(eps=eps, min_samples=n_clusters).fit(X)
        cluster_labels = clusterer.labels_ + 1

        # The silhouette_score gives the average value for all the samples, which gives perspective into the density
        # and separation of the formed clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_avg_n_clusters.append(silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.grid(False)
        ax1.set_title("Silhouette Plot for Various Clusters", size=13)
        ax1.set_xlabel("Silhouette Coefficient", size=13)
        ax1.set_ylabel("Cluster label", size=13)

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.tick_params(axis='both', which='major', labelsize=13)

        # 2nd Plot showing the actual clusters formed
        colors = plt.cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=80, lw=0, alpha=1.0,
                    c=colors, edgecolor='k')

        ax2.grid(False)
        ax2.tick_params(axis='both', which='major', labelsize=13)
        ax2.set_title("The visualization of the clustered data.", size=13)
        ax2.set_xlabel("MDS 1", size=13)
        ax2.set_ylabel("MDS 2", size=13)

        plt.suptitle(("Silhouette analysis for DBSCAN clustering on sample data "
                      "with " + str(n_clusters) + " clusters, inclusive of outlier label 0"),
                     fontsize=14, fontweight='bold')
        plt.savefig(
            'Silhouette analysis for DBSCAN clustering on sample data with' + str(n_clusters) + 'number of clusters')

    plt.show()

    # A plot of the number of clusters vs silhouette score to aid in choice for min samples parameter in DBSCAN
    plt.plot(range_n_clusters, silhouette_avg_n_clusters)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.grid(False)
    plt.xlabel("Number of Clusters", size=13)
    plt.ylabel("Silhouette score", size=13)
    plt.savefig(
        'Silhouette scores for DBSCAN clustering on sample data with ' + str(min_samp_max) + 'number of clusters')
    plt.show()
    return


def block_avg(dataframe):
    """
    This function computes the block average for a dataset with collocated samples.

    Arguments
    ---------
    dataframe: dataframe consisting of features to be blocked averaged on X coordinate i.e., location column
    """

    feat_titles = dataframe.columns.values.tolist()  # list containing features names

    if len(dataframe[feat_titles[0]].unique()) == len(dataframe[feat_titles[1]].unique()):

        data = pd.DataFrame()
        data[feat_titles[0]] = dataframe[feat_titles[0]].unique()

        for i in range(1, len(feat_titles)):
            data[feat_titles[i]] = dataframe.copy(deep=True).groupby(feat_titles[0])[feat_titles[i]].mean().values

    else:
        print("Block average is not needed")

    return data


def dbn_plotter(df1, df2, x_label):
    """
      This function compares the distributions of the features in the original data to the block averaged data.

      Arguments
      ---------
      df1: original dataframe

      df2: block averaged dataframe

      x_label: x-axis label
      """
    feat_titles = df2.columns.tolist()

    # Basis for automated subplot
    num_cols = 2
    subplot_nos = len(feat_titles)
    if subplot_nos % num_cols == 0:
        num_rows = subplot_nos // num_cols
    else:
        num_rows = (subplot_nos // num_cols) + 1

    # Make Plots
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols)

    for i in range(0, len(feat_titles)):
        # Original Data
        kde_obj = stats.gaussian_kde(df1[feat_titles[i]])
        Xmin = df1[feat_titles[i]].min()
        Xmax = df1[feat_titles[i]].max()
        grid = np.linspace(Xmin, Xmax, 5000)
        est_pdf = kde_obj.evaluate(grid)

        # Block Averaged Data
        kde_obj_2 = stats.gaussian_kde(df2[feat_titles[i]])
        # Xmin_2 = df2[feat_titles[i]].min()
        # Xmax_2 = df2[feat_titles[i]].max()
        grid_2 = np.linspace(Xmin, Xmax, 5000)  # Xmin_2, Xmax_2
        est_pdf_2 = kde_obj_2.evaluate(grid_2)

        ax = plt.subplot(num_rows, num_cols, i + 1)

        ax.plot(grid, est_pdf, alpha=1.0, label='Original')
        ax.plot(grid_2, est_pdf_2, alpha=0.6, label='Block averaged')

        # Figure info
        ax.set_aspect('auto')
        ax.set_title('Gaussian Kernel Density Comparison', size=12)
        ax.set_xlabel(x_label + feat_titles[i], size=12)
        ax.set_ylabel('Density', size=12)
        ax.legend(fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=12)

    # Aesthetics
    plt.subplots_adjust(left=0.0, bottom=0.5, right=1.9, top=2.3, wspace=0.3, hspace=0.5)
    plt.savefig('Volume-variance distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    return
