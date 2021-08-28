import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, sin, cos, pi

from settings import *
from my_clusters import Cluster


class CytoskeletonRestriction:

    def __init__(self,
                 diffusion_rate,  # скорость распространения сигнала
                 melting_activity,  # интенсивность сигнала
                 tonic_signal_level,  # тонический уровень сигнала
                 bcr_number,  # количество рецепторов в модели
                 time_step,  # временной шаг модели
                 epsilon,  # минимальный уровень сигнала (в долях тонического)
                 ):
        """
        Поле значений степени "таяния" цитоскелета.

        Parameters
        ----------
        diffusion_rate : integer or real
            Скорость распространения сигнала таяния от источника.
        melting_activity : integer or real
            Интенсивность испускаемого сигнала таяния (возвращаемая
            степень "таяния" не привысит 1 вне зависимости от
            указанного значения).
        tonic_signal_level : integer or real
            Тонический уровень таяния в отсутствии источников сигнала.
        bcr_number : integer
            Количество потенциальных источников сигнала таяния.
        time_step : integer or real
            Временной шаг моделирования.
        epsilon : real
            Доля величины тонического сигнала, падая ниже которой
            сигналом можно принебречь.
        """

        self.diffusion_rate = diffusion_rate
        self.melting_activity = melting_activity * time_step
        self.tonic_signal_level = tonic_signal_level

        max_sources = 1000 * bcr_number  ######## рассчитать коэффициент!!!!!
        self.indices = np.arange(max_sources)
        self.sources_coords = np.empty((max_sources, 2))
        self.activation_times = np.empty((max_sources, 1))
        self.active_sources = np.zeros(max_sources, dtype=bool)

        # убиваем сигналы, упавшие до epsilon * tonic_signal_level
        self.death_time = np.sqrt(melting_activity * time_step / (epsilon * tonic_signal_level))

    @property
    def is_active(self):
        return self.indices[self.active_sources]

    @property
    def is_not_active(self):
        return self.indices[~self.active_sources]

    def add_melting_sources(self, points, time):
        """
        Вносит источники сигнала.

        Parameters
        ----------
        points : ndarray (n x 2)
            Вектор координат источников.
        time : integer or real
            Момент времени.
        """
        indices = self.is_not_active[:points.shape[0]]
        self.sources_coords[indices] = points
        self.activation_times[indices] = time
        self.active_sources[indices] = True

    def remove_dead_sources(self, time):
        if not self.active_sources.any():
            return
        dead_indices = (time - self.activation_times[self.is_active]) > self.death_time
        self.active_sources[self.is_active[dead_indices.flatten()]] = False

    def get_diffusion_coeffs(self, points, time):
        """
        Возвращает значения "таяния" цитоскелета в данных точках.

        Parameters
        ----------
        points : ndarray (n x 2)
            Вектор координат точек.
        time : integer or real
            Момент времени.

        Returns
        -------
        melting_coeffs : ndarray (n x 1)
            Столбец коэффициентов таяния цитоскелета.
        """
        self.remove_dead_sources(time)

        is_exist = (self.activation_times[self.is_active] < time).flatten()

        sources = self.sources_coords[self.is_active][is_exist]
        distances_2 = CytoskeletonRestriction.square_distance_matrix(points, sources)

        time = np.tile(
            (time - self.activation_times[self.is_active][is_exist]) * self.diffusion_rate,
            points.shape[0]
        )

        melting_signals = self.gauss_diffusion(distances_2, time)
        diffusion_coeffs = self.tonic_signal_level + self.melting_activity * melting_signals.sum(axis=0)

        return np.clip(diffusion_coeffs[:, None], 0, 1)

    @staticmethod
    def gauss_diffusion(square_distance, time):
        time_2 = time ** 2
        return np.exp(-square_distance / (2 * time_2)) / time_2

    @staticmethod
    def square_distance_matrix(points, sources):
        x_p = points[:, 0:1]
        y_p = points[:, 1:2]
        x_s = sources[:, 0:1]
        y_s = sources[:, 1:2]
        return (x_s - x_p.T) ** 2 + (y_s - y_p.T) ** 2


class MembraneDynamics:

    def __init__(self,
                 time_step,
                 surface_area,
                 bcr_number,
                 bcr_radius,
                 # bcr_valence,
                 bcr_rafts_radius,
                 bcr_diffusion_coef,
                 free_rafts_number,
                 raft_diffusion_coef,
                 ):
        """
        Модель клеточной мембраны, содержащая липидные рафты и BCR

        Parameters
        ----------
        surface_area : integer or real
            Площадь квадратной ячейки моделирования.
        bcr_number : integer
            Количество молекул BCR.
        bcr_radius : integer or real
            Радиус молекулы BCR.
        bcr_rafts_radius : integer or real
            Радиус "собственного" липидного рафта молекулы BCR.
        bcr_diffusion_coef : integer or real
            Коэффициент диффузии молекулы BCR в липидном рафте.
        free_rafts_number : integer
            Количество свободных липидных рафтов.
        free_rafts_radius_mean : integer or real
            Средний радиус свободного липидного рафта (радиусы
            свободных рафтов имеют распределение хи-квадрат с 3
            степенями свободы и заданным средним значением).
        raft_diffusion_coef : integer or real
            Коэффициент диффузии липидного рафта единичного радиуса
            в мембране при полностью "растаявшем" цитоскелете.
        """

        # General
        self.side_length = np.sqrt(surface_area)
        self.time = 0
        self.time_step = time_step

        # BCR
        self.bcr_number = bcr_number
        # self.bcr_is_clustered = np.zeros(bcr_number, dtype=bool)
        # self.bcr_is_clustered[:ANTIGEN_NUMBER * ANTIGEN_VALENCY] = True
        self.bcr_relative_coordinates = np.zeros((bcr_number, 2))
        # self.bcr_angles = np.random.rand(bcr_number, 1) * 2*pi
        # self.bcr_radius = np.ones((bcr_number, 1)) * bcr_radius
        # self.bcr_cluster_number = np.arange(bcr_number)
        self.bcr_radius = bcr_radius
        self.bcr_raft_number = np.arange(bcr_number)
        self.bcr_is_activated = np.zeros(bcr_number, dtype=bool)
        self.bcr_diffusion_coef = bcr_diffusion_coef
        # self.bcr_valence = bcr_valence

        # Lipid rafts
        rafts_number = bcr_number + free_rafts_number
        self.lipid_rafts_radii = np.empty((rafts_number, 1))
        self.lipid_rafts_radii[:bcr_number] = bcr_rafts_radius
        self.lipid_rafts_radii[bcr_number:] = (
                FREE_RAFTS_RADIUS_MIN +
                np.random.rand(free_rafts_number, 1) * (FREE_RAFTS_RADIUS_MAX - FREE_RAFTS_RADIUS_MIN)
        )
        # self.lipid_rafts_radii[bcr_number:] = np.random.chisquare(
        #     DEG_FREEDOM,
        #     size=(free_rafts_number, 1),
        # ) * (free_rafts_radius_mean / DEG_FREEDOM)
        self.lipid_rafts_coordinates = np.empty((rafts_number, 2))
        self.init_bcr_rafts(rafts_number)
        self.raft_diffusion_coef = raft_diffusion_coef

        ################################################# ПЕРЕДЕЛАТЬ НОРМАЛЬНО!
        # Cytoskeleton
        self.cytoskeleton = CytoskeletonRestriction(
            diffusion_rate=MELTING_DIFFUSION_RATE,
            melting_activity=MELTING_ACTIVITY,
            tonic_signal_level=TONIC_MELTING_LEVEL,
            bcr_number=bcr_number,
            time_step=time_step,
            epsilon=MELTING_SIGNAL_DEATH_EPSILON,
        )

        ############################
        # print(self.clustered_percent_and_mean_raft_radius())
        ############################

        self.evolve_system()

    def clustered_percent_and_mean_raft_radius(self):
        connect_comps = MembraneDynamics.connectivity_components(
            self.bcr_number,
            MembraneDynamics.calculate_overlap_list(
                self.get_bcrs()[0],
                self.bcr_radius * 1.01,
            )
        )
        nums = np.unique(list(map(len, connect_comps)), return_counts=True)
        return (nums[0] * nums[1]).sum() / self.bcr_number, self.lipid_rafts_radii.mean()

    def init_bcr_rafts(self, rafts_number):
        clusters = Cluster('100_clusters_2500_points.csv')
        for i in range(ANTIGEN_NUMBER):
            self.lipid_rafts_coordinates[i * ANTIGEN_VALENCY: (i + 1) * ANTIGEN_VALENCY] = (
                    np.random.rand(1, 2) * self.side_length +
                    clusters.get_cluster(i % 100, ANTIGEN_VALENCY) * ANTIGEN_SPOT_DISTANCE
            )

        self.lipid_rafts_coordinates[ANTIGEN_NUMBER * ANTIGEN_VALENCY:] = np.random.rand(
            rafts_number - ANTIGEN_NUMBER * ANTIGEN_VALENCY,
            2,
        ) * self.side_length

        (  # fusion
            self.lipid_rafts_coordinates,
            self.lipid_rafts_radii,
        ) = self.fusion_rafts(
            self.lipid_rafts_coordinates,
            self.lipid_rafts_radii,
            internal_elements=(
                self.bcr_raft_number,
                self.bcr_relative_coordinates,
            ),
            connectivity_comps=np.arange(ANTIGEN_NUMBER * ANTIGEN_VALENCY).reshape(ANTIGEN_NUMBER, ANTIGEN_VALENCY),
        )

    # @staticmethod
    # def clusters_to_centers(clusters):
    #     """
    #     Превращает координаты, содержащие кластеры в координаты центров масс
    #     """
    #     edge = ANTIGEN_NUMBER * ANTIGEN_VALENCY
    #     return np.vstack((
    #         clusters[:edge].reshape(ANTIGEN_NUMBER, -1, 2).mean(axis=1),
    #         clusters[edge:]
    #     ))
    #
    # @staticmethod
    # def centers_to_clusters(centers):
    #     """
    #     Превращает значения адресованные центрам масс в значения для элементов кластера
    #     """
    #     return np.vstack((
    #         np.tile(centers[:ANTIGEN_NUMBER], ANTIGEN_VALENCY).reshape(-1,2),
    #         centers[ANTIGEN_NUMBER:]
    #     ))

    @staticmethod
    def clusters_to_centers(clusters, edge=None):
        """
        Превращает координаты, содержащие кластеры в координаты центров масс
        """
        if edge is None:
            edge = ANTIGEN_NUMBER * ANTIGEN_VALENCY

        return np.vstack((
            clusters[:edge].reshape(edge // ANTIGEN_VALENCY, -1, 2).mean(axis=1),
            clusters[edge:]
        ))

    @staticmethod
    def unique_for_clusters(clusters, edge=None):
        """
        """
        if edge is None:
            edge = ANTIGEN_NUMBER * ANTIGEN_VALENCY

        return np.vstack((
            clusters[:edge:ANTIGEN_VALENCY],
            clusters[edge:]
        ))

    @staticmethod
    def centers_to_clusters(centers, edge=None):
        """
        Превращает значения адресованные центрам масс в значения для элементов кластера
        """
        if edge is None:
            edge = ANTIGEN_NUMBER * ANTIGEN_VALENCY

        return np.vstack((
            np.tile(centers[:edge // ANTIGEN_VALENCY], ANTIGEN_VALENCY).reshape(-1, 2),
            centers[edge // ANTIGEN_VALENCY:]
        ))

    def evolve_system(self):
        """
        Выполняет раунд эволюции системы.
        """
        dt = self.time_step
        self.time += dt
        sqrt_half_dt = sqrt(dt / 2)

        # Lipid rafts
        self.diffusion_rafts(
            self.lipid_rafts_coordinates,
            self.lipid_rafts_radii,
            np.sqrt(self.raft_diffusion_coef * self.cytoskeleton.get_diffusion_coeffs(
                self.lipid_rafts_coordinates,
                self.time,
            )) * sqrt_half_dt,
        )
        self.border_conditions_toroidal(
            self.lipid_rafts_coordinates,
            self.side_length,
        )
        (  # fusion
            self.lipid_rafts_coordinates,
            self.lipid_rafts_radii,
        ) = self.fusion_rafts(
            self.lipid_rafts_coordinates,
            self.lipid_rafts_radii,
            internal_elements=(
                self.bcr_raft_number,
                self.bcr_relative_coordinates,
            )
        )

        # BCR
        self.bcr_relative_coordinates = MembraneDynamics.diffusion_bcr(
            self.bcr_relative_coordinates,
            sqrt(self.bcr_diffusion_coef) * sqrt_half_dt,
        )

        # движение по потенциалу (к границе рафта)
        centers_relative_coordinates = MembraneDynamics.clusters_to_centers(
            self.bcr_relative_coordinates
        )
        norm = np.linalg.norm(centers_relative_coordinates, axis=1)[:, None]
        not_central = (norm > 0).flatten()
        potential_movement = POTENTIAL_COEF * (
                MembraneDynamics.unique_for_clusters(
                    self.lipid_rafts_radii[self.bcr_raft_number]
                ) - norm - self.bcr_radius
        )[not_central]
        centers_shift = (
                centers_relative_coordinates[not_central] *
                (potential_movement / norm[not_central])
        )
        self.bcr_relative_coordinates += MembraneDynamics.centers_to_clusters(centers_shift)

        self.border_conditions_hard(
            self.bcr_relative_coordinates,
            self.bcr_radius,
            self.lipid_rafts_radii[self.bcr_raft_number],
        )

        for n_raft in range(self.bcr_raft_number.max() + 1):
            indices = (self.bcr_raft_number == n_raft)

            if indices.sum() > 1:
                # разрешение интерференций
                (
                    dist_matr,
                    self.bcr_relative_coordinates[indices],
                ) = MembraneDynamics.resolve_overlaps(
                    self.bcr_relative_coordinates[indices],
                    self.bcr_radius,
                    (self.bcr_raft_number[:ANTIGEN_NUMBER * ANTIGEN_VALENCY] == n_raft).sum(),
                )

                # активация/деактивация
                self.activation_deactivation(dist_matr, indices)

        # таяние цитоскелета
        self.add_melting_centers()

    def activation_deactivation(self, dist_matr, indices):
        dist_matr[np.diag_indices(dist_matr.shape[0])] = np.inf
        cluster_members = (dist_matr < 2.1 * self.bcr_radius).any(axis=0)

        can_be_activated = np.arange(self.bcr_number)[indices][cluster_members]
        self.bcr_is_activated[can_be_activated] |= np.random.choice(
            [True, False],
            size=can_be_activated.size,
            p=[ACTIVATION_PROB, 1 - ACTIVATION_PROB])

        can_be_deactivated = np.arange(self.bcr_number)[indices][~cluster_members]
        distances_to_edge = (
                self.lipid_rafts_radii[self.bcr_raft_number].flatten() -
                np.linalg.norm(self.bcr_relative_coordinates, axis=1)
        )[can_be_deactivated]
        self.bcr_is_activated[can_be_deactivated] &= ~(
                np.random.rand(can_be_deactivated.size) <
                DEACTIVATION_PROB * np.exp(-INHIBITION_REDUCTION_FACTOR * distances_to_edge)
        )

    def add_melting_centers(self):
        melting_centers = self.get_bcrs()[0][self.bcr_is_activated]
        self.cytoskeleton.add_melting_sources(melting_centers, self.time)

    def get_rafts(self):
        """
        Возвращает координаты и радиусы липидных рафтов.
        """
        return (self.lipid_rafts_coordinates,
                self.lipid_rafts_radii,
                )

    def get_bcrs(self):
        """
        Возвращает координаты (абсолютные) и радиусы молекул BCR.
        """
        bcr_coordinates = self.bcr_relative_coordinates + self.lipid_rafts_coordinates[self.bcr_raft_number]
        return (bcr_coordinates,
                self.bcr_radius,
                self.bcr_is_activated,
                )

    def get_image(self, imsize=(10, 10)):
        """
        Отрисовывает текущее состояние мембраны.
        """
        side_length = np.sqrt(SURFACE_AREA)

        fig, ax = plt.subplots(1, figsize=imsize)
        ax.set_xlim(0, side_length)
        ax.set_ylim(0, side_length)

        # Lipid rafts
        ax.scatter(*self.lipid_rafts_coordinates.T,
                   s=scale_radii_to_points(self.lipid_rafts_radii, fig, ax),
                   c='sandybrown',
                   alpha=0.5,
                   )

        # BCR
        bcr_coordinates = self.bcr_relative_coordinates + self.lipid_rafts_coordinates[self.bcr_raft_number]
        bcr = ax.scatter(*bcr_coordinates.T,
                         s=scale_radii_to_points(self.bcr_radius, fig, ax),
                         c='darkblue',
                         alpha=0.8,
                         )
        bcr.set_array(self.bcr_is_activated)

        N = 200
        grid = np.mgrid[0:side_length:side_length / N, 0:side_length:side_length / N].reshape(2, -1).T
        melting_field = self.cytoskeleton.get_diffusion_coeffs(
            grid,
            self.time,
        )
        ax.imshow(
            melting_field.reshape(N, N)[:, ::-1].T,
            cmap='Blues',
            vmin=TONIC_MELTING_LEVEL,
            vmax=2,
            extent=[0, side_length, 0, side_length],
        )

        plt.show()

    @staticmethod
    def distance_matrix(points, points1=None):
        """
        Вычисляет матрицу расстояний между точками.
        """
        x = points[:, 0:1]
        y = points[:, 1:2]
        if points1 is None:
            return np.sqrt((x - x.T) ** 2 + (y - y.T) ** 2)
        x1 = points1[:, 0:1]
        y1 = points1[:, 1:2]
        return np.sqrt((x - x1.T) ** 2 + (y - y1.T) ** 2)

    @staticmethod
    def combined_params(coordinates, radii):
        """
        Вычисляет обновленные центр и радиус после слияния объектов.
        """
        radii_sq_sum = (radii ** 2).sum()
        new_center = (coordinates * radii ** 2).sum(axis=0) / radii_sq_sum
        return new_center, sqrt(radii_sq_sum)

    @staticmethod
    def diffusion_rafts(centers, radii, diff_time_coef):
        """
        Выполняет раунд диффузии, изменяя координаты липидных рафтов.
        """
        shift = np.random.normal(size=(centers.shape[0], 2)) * diff_time_coef
        centers += shift * (np.log(1 + BCR_RADIUS) / np.log(1 + radii))  # это плохая нормировка, но я исправлю

    @staticmethod
    def diffusion_bcr(centers, diff_time_coef):
        """
        Выполняет раунд диффузии, изменяя координаты липидных рафтов.
        """
        linked_bcr_number = ANTIGEN_NUMBER * (ANTIGEN_VALENCY - 1)
        shift = np.random.normal(size=(centers.shape[0] - linked_bcr_number, 2)) * diff_time_coef
        shift[:ANTIGEN_NUMBER] /= sqrt(ANTIGEN_VALENCY)
        angle = (np.random.normal(size=ANTIGEN_NUMBER) *
                 diff_time_coef * 2 * pi / sqrt(ANTIGEN_VALENCY)) * 20  # Плохо! Рассчитать коэффициент вращения!
        for i in range(ANTIGEN_NUMBER):
            centers[i * ANTIGEN_VALENCY: (i + 1) * ANTIGEN_VALENCY] = MembraneDynamics.move_and_rotation(
                centers[i * ANTIGEN_VALENCY: (i + 1) * ANTIGEN_VALENCY],
                shift[i],
                angle[i],
            )

        centers[ANTIGEN_NUMBER * ANTIGEN_VALENCY:] += shift[ANTIGEN_NUMBER:]
        return centers

    @staticmethod
    def move_and_rotation(points, offset, angle):
        center = points.mean(axis=0)
        cos_a = cos(angle)
        sin_a = sin(angle)
        R = np.array([
            [cos_a, sin_a],
            [-sin_a, cos_a],
        ])
        return (points - center).dot(R) + center + offset

    @staticmethod
    def border_conditions_toroidal(coordinates, side_length):
        """
        Применяет к координатам тороидальные граничные условия.
        """
        coordinates %= side_length

    # @staticmethod
    # def border_conditions_hard(relative_coordinates, self_radii, spot_radii):
    #     """
    #     Применяет жесткие граничные условия к относительным координатам BCR,
    #     не позволяя им выходить за границы липидного рафта.
    #     """
    #     deviation_norm = np.linalg.norm(relative_coordinates, axis=1)
    #     effective_radii = (spot_radii - self_radii).flatten()
    #     aberration = deviation_norm > effective_radii
    #     returning_coefficient = effective_radii[aberration] / deviation_norm[aberration]
    #     relative_coordinates[aberration] *= returning_coefficient[:, None]

    @staticmethod
    def border_conditions_hard(relative_coordinates, self_radii, spot_radii):
        """
        Применяет жесткие граничные условия к относительным координатам BCR,
        не позволяя им выходить за границы липидного рафта.
        """
        edge = ANTIGEN_NUMBER * ANTIGEN_VALENCY
        distance_norm = np.linalg.norm(relative_coordinates, axis=1)
        effective_radii = (spot_radii - self_radii).flatten()
        deviation_norm = distance_norm - effective_radii
        deviation_norm[deviation_norm < 0] = 0
        max_norm = deviation_norm[:edge].reshape(-1, ANTIGEN_VALENCY).argmax(axis=1)
        indices = np.arange(ANTIGEN_NUMBER), max_norm
        return_coefficient = deviation_norm / distance_norm
        return_shift_clusters = (
           -relative_coordinates[:edge].reshape(ANTIGEN_NUMBER, -1, 2)[indices] *
           return_coefficient[:edge].reshape(ANTIGEN_NUMBER, -1, 1)[indices]
        )
        return_shift_single = -relative_coordinates[edge:] * return_coefficient[edge:][:, None]
        relative_coordinates += np.vstack((
            np.tile(return_shift_clusters, ANTIGEN_VALENCY).reshape(-1, 2),
            return_shift_single,
        ))

    @staticmethod
    def fusion_rafts(centers, radii, internal_elements, connectivity_comps=None):
        """
        Выполняет слияния липидных рафтов.
        """

        inter_locations, inter_centers = internal_elements

        if connectivity_comps is None:
            fusion_list = MembraneDynamics.calculate_overlap_list(centers, radii)
            connectivity_comps = MembraneDynamics.connectivity_components(
                centers.shape[0],
                fusion_list,
            )

        disappeared_lines = []  # список поглощенных элементов
        for component in connectivity_comps:

            # рассчет нового центра и радиуса
            new_center, new_radius = MembraneDynamics.combined_params(
                centers[component],
                radii[component],
            )

            # Обновляем данные внутренних элементов
            dominant_class = component[0]
            for i in component:
                updated_items = (inter_locations == i)
                # присваиваем внутренние элементы объединенному рафту
                inter_locations[updated_items] = dominant_class
                # смещаем внутренние элементы к центру оъединенного рафта
                inter_centers[updated_items] += centers[i] - new_center

            # Создаем объединенный рафт
            centers[dominant_class] = new_center
            radii[dominant_class] = new_radius

            # Убираем поглощенные рафты
            for i in component[1:]:
                disappeared_lines.append(i)

        for i in sorted(disappeared_lines, reverse=True):
            inter_locations[inter_locations > i] -= 1

        return (np.delete(arr, disappeared_lines, axis=0) for arr in (centers, radii))

    @staticmethod
    def calculate_overlap_list(centers, radii):
        """
        Возвращает список пар интерферирующих элементов.

        Parameters
        ----------
        centers : ndarray (n x 2)
            Центры элементов.
        radii : ndarray (n x 1)
            Радиусы элементов.

        Returns
        -------
        overlap_list : list <(i, j)>
            Список пар интерферирующих элементов.
        """
        n_clusters = centers.shape[0]
        dist_matr = MembraneDynamics.distance_matrix(centers)
        if isinstance(radii, float):
            overlap_mask = np.ones_like(dist_matr) * 2 * radii
        else:
            overlap_mask = radii + radii.T

        overlap_list = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if dist_matr[i, j] <= overlap_mask[i, j]:
                    overlap_list.append((i, j))

        return overlap_list

    # @staticmethod
    # def resolve_overlaps(centers, radius, edge):
    #     """
    #     Пересчитывает координаты объектов, разрешая интерференции.
    #
    #     Parameters
    #     ----------
    #     centers : ndarray (n x 2)
    #         Центры элементов.
    #     radius : integer or real
    #         Радиус элементов.
    #
    #     Returns
    #     -------
    #     dist_matr : darray (n x n)
    #         Матрица расстояний.
    #     centers : ndarray (n x 2)
    #         Скорректированные центры элементов.
    #     """
    #     n_clusters = centers.shape[0]
    #     dist_matr = MembraneDynamics.distance_matrix(centers)
    #
    #     for i in range(n_clusters):
    #         for j in range(i + 1, n_clusters):
    #             if dist_matr[i, j] <= 2 * radius:
    #                 shift = (centers[j] - centers[i]) * (2 * radius / dist_matr[i, j] - 1) / 2
    #                 centers[i] -= shift
    #                 centers[j] += shift
    #
    #                 new_i = MembraneDynamics.distance_matrix(centers, centers[i:i + 1])
    #                 new_j = MembraneDynamics.distance_matrix(centers, centers[j:j + 1])
    #                 dist_matr[i:i + 1, :] = new_i.T
    #                 dist_matr[:, i:i + 1] = new_i
    #                 dist_matr[j:j + 1, :] = new_j.T
    #                 dist_matr[:, j:j + 1] = new_j
    #
    #     return dist_matr, centers

    @staticmethod
    def resolve_overlaps(centers, radius, edge):
        """
        Пересчитывает координаты объектов, разрешая интерференции.

        Parameters
        ----------
        centers : ndarray (n x 2)
            Центры элементов.
        radius : integer or real
            Радиус элементов.

        Returns
        -------
        dist_matr : darray (n x n)
            Матрица расстояний.
        centers : ndarray (n x 2)
            Скорректированные центры элементов.
        """
        n_clusters = centers.shape[0]
        dist_matr = MembraneDynamics.distance_matrix(centers)

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if dist_matr[i, j] <= 2 * radius:
                    if i < edge and j < edge:
                        i_cluster = ANTIGEN_VALENCY * (i // ANTIGEN_VALENCY)
                        j_cluster = ANTIGEN_VALENCY * (j // ANTIGEN_VALENCY)
                        if i_cluster == j_cluster:
                            shift = (centers[j] - centers[i]) * (2 * radius / dist_matr[i, j] - 1) / 2
                            centers[i] -= shift
                            centers[j] += shift

                            new_i = MembraneDynamics.distance_matrix(centers, centers[i:i + 1])
                            new_j = MembraneDynamics.distance_matrix(centers, centers[j:j + 1])
                            dist_matr[i:i + 1, :] = new_i.T
                            dist_matr[:, i:i + 1] = new_i
                            dist_matr[j:j + 1, :] = new_j.T
                            dist_matr[:, j:j + 1] = new_j

                        shift = (centers[j] - centers[i]) * (2 * radius / dist_matr[i, j] - 1) / 2
                        centers[i_cluster: i_cluster + ANTIGEN_VALENCY] -= shift
                        centers[j_cluster: j_cluster + ANTIGEN_VALENCY] += shift

                        new_i = MembraneDynamics.distance_matrix(centers, centers[i_cluster: i_cluster + ANTIGEN_VALENCY])
                        new_j = MembraneDynamics.distance_matrix(centers, centers[j_cluster: j_cluster + ANTIGEN_VALENCY])
                        dist_matr[i_cluster: i_cluster + ANTIGEN_VALENCY, :] = new_i.T
                        dist_matr[:, i_cluster: i_cluster + ANTIGEN_VALENCY] = new_i
                        dist_matr[j_cluster: j_cluster + ANTIGEN_VALENCY, :] = new_j.T
                        dist_matr[:, j_cluster: j_cluster + ANTIGEN_VALENCY] = new_j
                    elif i < edge:
                        i_cluster = ANTIGEN_VALENCY * (i // ANTIGEN_VALENCY)
                        shift = (centers[j] - centers[i]) * (2 * radius / dist_matr[i, j] - 1) / (ANTIGEN_VALENCY + 1)
                        centers[i_cluster: i_cluster + ANTIGEN_VALENCY] -= shift
                        centers[j] += shift * ANTIGEN_VALENCY

                        new_i = MembraneDynamics.distance_matrix(centers, centers[i_cluster: i_cluster + ANTIGEN_VALENCY])
                        new_j = MembraneDynamics.distance_matrix(centers, centers[j:j + 1])
                        dist_matr[i_cluster: i_cluster + ANTIGEN_VALENCY, :] = new_i.T
                        dist_matr[:, i_cluster: i_cluster + ANTIGEN_VALENCY] = new_i
                        dist_matr[j:j + 1, :] = new_j.T
                        dist_matr[:, j:j + 1] = new_j
                    elif j < edge:
                        i, j = j, i
                        i_cluster = ANTIGEN_VALENCY * (i // ANTIGEN_VALENCY)
                        shift = (centers[j] - centers[i]) * (2 * radius / dist_matr[i, j] - 1) / (ANTIGEN_VALENCY + 1)
                        centers[i_cluster: i_cluster + ANTIGEN_VALENCY] -= shift
                        centers[j] += shift * ANTIGEN_VALENCY

                        new_i = MembraneDynamics.distance_matrix(centers,
                                                                 centers[i_cluster: i_cluster + ANTIGEN_VALENCY])
                        new_j = MembraneDynamics.distance_matrix(centers, centers[j:j + 1])
                        dist_matr[i_cluster: i_cluster + ANTIGEN_VALENCY, :] = new_i.T
                        dist_matr[:, i_cluster: i_cluster + ANTIGEN_VALENCY] = new_i
                        dist_matr[j:j + 1, :] = new_j.T
                        dist_matr[:, j:j + 1] = new_j
                    else:
                        shift = (centers[j] - centers[i]) * (2 * radius / dist_matr[i, j] - 1) / 2
                        centers[i] -= shift
                        centers[j] += shift

                        new_i = MembraneDynamics.distance_matrix(centers, centers[i:i + 1])
                        new_j = MembraneDynamics.distance_matrix(centers, centers[j:j + 1])
                        dist_matr[i:i + 1, :] = new_i.T
                        dist_matr[:, i:i + 1] = new_i
                        dist_matr[j:j + 1, :] = new_j.T
                        dist_matr[:, j:j + 1] = new_j

        return dist_matr, centers

    @staticmethod
    def connectivity_components(n_elements, edges):
        """
        По списку ребер возвращает список компонент связности.

        Parameters
        ----------
        n_elements : integer
            Количество вершин.
        edges : list <(i, j)>
            Список ребер.

        Returns
        -------
        connectivity_comps : list <list>
            Список компонент связности (списков вершин).
        """

        # breadth-first search
        def bfs(i, vertex, adjacency_list, comps):
            if comps[vertex]:
                return False

            comps[vertex] = i
            for v in adjacency_list[vertex]:
                bfs(i, v, adjacency_list, comps)

            return True

        # creating adjacency list (dict) from edges
        adjacency_list = {}
        for a, b in edges:
            adjacency_list[a] = adjacency_list.get(a, []) + [b]
            adjacency_list[b] = adjacency_list.get(b, []) + [a]

        # searching for connectivity components
        elements = np.arange(n_elements)
        comps = np.zeros_like(elements)
        i = 1
        for vertex in adjacency_list.keys():
            if bfs(i, vertex, adjacency_list, comps):
                i += 1

        # extracting components
        connectivity_comps = []
        for k in range(1, i):
            connectivity_comps.append(elements[comps == k])

        return connectivity_comps

    #     def get_potential(self, point):
    #         print(point)

    #         def edge_potential(r, radius, steepness):
    #             return 1 / (1 + np.exp(-steepness * (r - radius))) - 1

    #         def gauss_potential(r, sigma, value=1):
    #             return value * np.exp(-r**2 / (2*sigma**2))

    #         point = np.array(point)[None, :]
    #         potential = 1

    #         # Lipid rafts
    #         dist_matr_lr = MembraneDynamics.distance_matrix(
    #             point,
    #             self.lipid_rafts_coordinates,
    #         )
    #         potential += edge_potential(
    #             dist_matr_lr,
    #             np.tile(self.lipid_rafts_radii, points.shape[0]).T,
    #             LIPID_RAFT_EDGE_STEEPNESS,
    #         ).sum(axis=1)

    #         # BCRs
    #         dist_matr_bcr = MembraneDynamics.distance_matrix(
    #             point,
    #             self.get_bcrs()[0],
    #         )
    #         potential -= edge_potential(
    #             dist_matr_bcr,
    #             self.bcr_radius,
    #             BCR_EDGE_STEEPNESS,
    #         ).sum(axis=1)

    #         return potential[0]

    # def get_potential(self, points):
    #
    #     def edge_potential(r, radius, steepness):
    #         return 1 / (1 + np.exp(-steepness * (r - radius))) - 1
    #
    #     def gauss_potential(r, sigma, value=1):
    #         return value * np.exp(-r ** 2 / (2 * sigma ** 2))
    #
    #     potential = 1
    #
    #     # Lipid rafts
    #     dist_matr_lr = MembraneDynamics.distance_matrix(
    #         points,
    #         self.lipid_rafts_coordinates,
    #     )
    #     potential += edge_potential(
    #         dist_matr_lr,
    #         np.tile(self.lipid_rafts_radii, points.shape[0]).T,
    #         LIPID_RAFT_EDGE_STEEPNESS,
    #     ).sum(axis=1)
    #
    #     # BCRs
    #     dist_matr_bcr = MembraneDynamics.distance_matrix(
    #         points,
    #         self.get_bcrs()[0],
    #     )
    #     potential -= edge_potential(
    #         dist_matr_bcr,
    #         self.bcr_radius,
    #         BCR_EDGE_STEEPNESS,
    #     ).sum(axis=1)
    #
    #     return potential[:, None]


def scale_radii_to_points(radii, fig, ax):
    """
    Возвращает размеры круглых объектов в точках
    для их отрисовки функцией scatter.

    Parameters
    ----------
    radii : ndarray (n x 1)
        Столбец радиусов объектов.
    fig : matplotlib.figure.Figure
        Фигура matplotlib.
    ax : matplotlib.axes._axes.Axes
        Оси matplotlib.

    Returns
    -------
    radii_in_points : ndarray
        Столбец радиусов объектов в точках.
    """
    POINT_SIZE_COEF = 1.243 * 10 ** 4
    fig_scale = fig.get_size_inches()[0]
    x_lim = ax.get_xlim()
    side_lenght = x_lim[1] - x_lim[0]
    return POINT_SIZE_COEF * (radii * fig_scale / side_lenght) ** 2
