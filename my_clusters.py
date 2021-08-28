import pandas as pd


class Cluster:

    def __init__(self, file_name):
        clusters_df = pd.read_csv(file_name, index_col=0)
        self.r = 0.965 * clusters_df.loc[0, 'R']
        self.cluster_size = clusters_df.shape[0]
        self.clusters_number = clusters_df.shape[1] // 2
        self.clusters_df = clusters_df

        print(f'Received {self.clusters_number} clusters. Each contains {self.cluster_size} points.')

    def get_cluster(self, cluster_number, points_number=None):
        if points_number is None:
            points_number = self.cluster_size

        return self.clusters_df.loc[
               :points_number - 1,
               [f'x_{cluster_number}', f'y_{cluster_number}']
               ] / (2*self.r)
