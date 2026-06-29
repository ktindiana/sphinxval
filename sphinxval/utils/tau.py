import copy
import numpy as np
from typing import Callable
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
import distinctipy
import pandas as pd
import numpy.typing as npt
# from contingency_space.confusion_matrix import ConfusionMatrix


class ConfusionMatrix:
    """
    Confusion matrix class for multi-class problems.
    """
    def __init__(self, table: dict[str, list[int]]={}):
        """
        The class constructor.

        An example of a confusion matrix for multiple classes ('a', 'b', and 'c')
        is given below::

                          (pred)
                         a   b   c
                       ____________
                    a  | ta  fb  fc
             (real) b  | fa  tb  fc
                    c  | fa  fb  tc

        which should be input as a dictionary as follows::

             {'a': [ta, fb, fc],
              'b': [fa, tb, fc],
              'c': [fa, fb, tc]}



        :param __table: a dictionary with all class names as keys and their corresponding frequencies
        as values.
        """
        
        self.__table = copy.deepcopy(table)
        self.__num_classes = len(self.__table)

        for row in self.__table.values():
            if len(row) != self.__num_classes:
                raise ValueError('Length of each row must be equal to the number of classes.')
        
        self.class_freqs = {}
        for k, v in self.__table.items():
            self.class_freqs.update({k: int(np.sum(np.array(v)))})
        self.dim = len(self.class_freqs.keys())

    def add_class(self, cls: str, values: list[int]) -> None:
        """Adds a row to the matrix. Do not use this function unless you are building
        a matrix from scratch.

        Args:
            cls (str): The name of the class being added.
            values (list[int]): Values of the row.
        """
        self.__table.update({cls: values})
        self.class_freqs.update({cls: int(np.sum(np.array(values)))})
        self.__num_classes += 1
        
    def normalize(self):
        """
        Normalizes all entries of the confusion matrix::
        
                           (pred)                           (pred)
                         a   b   c                        a   b   c
                       ____________                      ____________
                    a  | 30  60  10      =>          a  |0.3 0.6 0.1
             (real) b  | 60  20  20      =>   (real) b  |0.6 0.2 0.2
                    c  | 30  20  50      =>          c  |0.3 0.2 0.5
        """
        
        
        cm_normalized = {}
        for k, freqs in self.__table.items():
            norm_freqs = [e / self.class_freqs[k] if self.class_freqs[k] != 0 else 0 for e in freqs]
            cm_normalized.update({k: norm_freqs})
        self.__init__(cm_normalized)

    def get_total_true(self, per_class: bool = False) -> int | dict[str, int]:
        """ Returns the total number of true classifications in the matrix.
        
        Args:
            per_class (bool): Whether to return the number of true classifications per class. Defaults to False.
        Returns:
            int: sum of the counts along the diagonal of the table.
        """
        a = np.array(list(self.__table.values()))
        
        match per_class:
            case True:
                
                return_dict: dict[str, int] = {}
                
                preds = a.diagonal()
                for cls, val in zip(self.__table.keys(), preds):
                    return_dict.update({cls: val})
                
                return return_dict
            case False:
                return np.sum(a.diagonal())
            case _:
                raise ValueError('per_class must be either True or False.')

    def get_wrong_classifications(self, cls: str = None) -> dict[str, int] | int:
        """
        For each class i, the total amount of false classifications is the sum of the counts in column i, except the one on the diagonal. 
        For binary classification, this will return the number of false positives in the matrix.
        
        In terms of a classification problem, this is the number of times that the model predicted a class, when the real value was a different one. 
        
        For example, given a matrix::
        
                           (pred)
                         a   b   c
                       ____________
                    a  | ta  fba fca
             (real) b  | fab tb  fcb
                    c  | fac fbc tc
                    
        The values would be returned like so::
        
                    {'a': (fab + fac),
                     'b': (fba + fbc),
                     'c': (fca + fcb)}

        Args:
            cls (str, optional): 
                The class for which you wish to find the number of false classifications. If left blank, this function will return a list of false classifications by class.

        Returns:
            list[int] | int: 
                -list[int]: List of the number of false classifications by class.
                -int:       The total number of false classifications for the specified class.
        """
        
        
        matrix = np.array(list(self.__table.values()))
        keys = list(self.__table.keys())
        
        diagonal_mask = np.eye(len(matrix), dtype=bool) #create a mask for the diagonal of the matrix.
        
        if cls is None:
            
            #return the sum of all values in the matrix, except for the diagonal.
            matrix_without_hits = matrix * (1 - diagonal_mask)
            
            summed_list = np.sum(matrix_without_hits, axis=0)
            
            summed_dict = {cls: value for cls, value in zip(self.__table.keys(), summed_list)}
            
            return summed_dict
        else:
            try:
                column_index = keys.index(cls)
            except:
                raise ValueError(f'The class {cls} was not found in this matrix.')
            
            
            #return the sum of all values in the matrix, except for the diagonal.
            return matrix[:, column_index][~diagonal_mask[:, column_index]].sum()
            
    def get_missed_classifications(self, cls: str = None) -> list[int] | int:
        """
        For each class i, the total amount of missed classifications is the sum of the counts in row i, except the one on the diagonal. 
        For binary classification, this will return the number of false negatives in the matrix.
        
        In terms of a classification problem, this is the number of times that the model a different class than the one given. 
        
        For example, given a matrix::
        
                           (pred)
                         a   b   c
                       ____________
                    a  | ta  fba fca
             (real) b  | fab tb  fcb
                    c  | fac fbc tc
                    
        The values would like like so::
        
                    {'a': (fba + fca),
                     'b': (fab + fcb),
                     'c': (fac + fbc)}

        Args:
            cls (str, optional): 
                The class for which you wish to find the number of missed classifications. If left blank, this function will return a list of missed classifications by class. Defaults to None.

        Returns:
            result: list[int] | int: 
                -list[int]: List of the number of missed classifications by class.
                -int:       The total number of missed classifications for the specified class.
        """
        
        matrix = np.array(list(self.__table.values()))
        keys = list(self.__table.keys())
        
        #create a diagonal mask for the matrix
        diagonal_mask = np.eye(len(matrix), dtype=bool)
        
        if cls is None:
            
            #return list of misses per class.
            
            matrix_without_hits = matrix * (1 - diagonal_mask)
            
            return np.sum(matrix_without_hits, axis=1)
        else:
            try:
                row_index = keys.index(cls)
            except:
                raise ValueError(f'The class {cls} was not found in this matrix.')
                
            #return sum of all values within the row, excluding the diagonal
            return matrix[row_index, :][~diagonal_mask[row_index, :]].sum()

    def get_matrix(self):
        return np.array(list(self.__table.values()))

    def vector(self, return_type: tuple | list = tuple, metric: Callable[[], float]=None) -> tuple[float, ...] | list[float]:
        """Returns a tuple representing the position of the confusion matrix within a contingency space. 
        
        Args:
        
            return_type (tuple | list): 
                The type of structure you wish the point to be returned as. Defaults to tuple.
            metric: (Callable[[ConfusionMatrix], float]): 
                A function that takes in a ConfusionMatrix, and returns a float representing an evaluation score for the metric. 

        Returns:
            c (tuple[int, ...] | list[int]): The tuple taking the form (x1, x2, ..., xk), where k is the number of classes.
        """
        
        rates = []
        cm = np.array(list(self.__table.values()))
        
        total_real = np.sum(cm, axis=1) #the total # of instances of each class.
        true_pred = cm.diagonal() #the list of # of times the model classifications each class correctly.
        
        
        for real, pred in zip(total_real, true_pred): #create each coordinate
            rates.append(pred / real)
            
        rates.reverse() # flip to (tnr, tpr)
            
        # if metric is not None:
        #     print(metric)
        #     # if metric.__module__.startswith('sklearn'):
        #     #     true_labels, predicted_labels = self.labels()
        #     #     rates.append(true_labels, predicted_labels)
        #     # else:
        #     rates.append(metric(self))
        metric_val = metric
        rates.append(metric_val)
        
        return return_type(rates)
    
    def num_samples(self, per_class:bool = False):
        """Returns the total number of samples in the matrix.

        Args:
            per_class (bool, optional): Whether or not to return the number of samples per class. Defaults to False.
        """
        
        arr = np.array(list(self.__table.values()))
        
        if per_class == True:
            return np.sum(arr, axis=1)
        return np.sum(np.array(list(self.__table.values())))
    
    def array(self) -> npt.NDArray:
        """Returns the matrix as a numpy array.

        Returns:
            npt.NDArray: A numpy array representation of the ConfusionMatrix.
        """
        return np.array(list(self.__table.values()))
    
    def labels(self) -> tuple[list[int], list[int]]:
        """
        Returns the true and predicted labels in the form of a tuple.
        
        Extracts and returns the true and predicted labels from the confusion matrix.
        The method iterates over the confusion matrix stored in `self.table`, where the keys are the actual labels and the values are lists of counts corresponding to predicted labels. It constructs two lists: one for the true labels and one for the predicted labels, by repeating each label according to its count in the confusion matrix.

        Returns:
            tuple[list[int], list[int]]: A tuple containing two lists: the first list contains the true labels, and the second list contains the predicted labels.
        """
        
        
        true_labels = []
        predicted_labels = []

        for actual_label, counts in self.table.items():
            for predicted_index, count in enumerate(counts):
                predicted_label = list(self.table.keys())[predicted_index]
                true_labels.extend([actual_label] * count)
                predicted_labels.extend([predicted_label] * count)

        return true_labels, predicted_labels

    
    @property
    def matrix(self):
        return self.__table
    @matrix.getter
    def matrix(self):
        return self.__table
    @matrix.setter
    def matrix(self, new_table = dict[str, list[int]]):
        if len(new_table) != len(self.__table):
            raise ValueError("New matrix must be the same size as the old matrix.")
        if set(self.__table.keys()) != set(new_table.keys()):
            raise ValueError("New matrix must contain the same classes as the previous matrix.")
        for row in new_table.values():
            if len(row) != self.__num_classes:
                raise ValueError("Number of elements in each row must match the number of classes in the original matrix.")
            
        self.__table = new_table
    
    @property
    def num_classes(self):
        return self.__num_classes
        

    def __repr__(self) -> str:
        #called when printing the object
        df = pd.DataFrame.from_dict(self.__table, orient='index', columns=self.__table.keys())
        df.index = self.__table.keys()
        return str(df)
    
    def __getitem__(self, index: str, give_index: bool = False):
        
        if index in self.__table:
            return self.__table[index]
        
        if self.num_classes == 2:
            if index.__contains__('t') or index.__contains__('p'):
                for i, cls in enumerate(self.__table.keys()):
                    if cls.__contains__('t') or cls.__contains__('p'):
                        if give_index == True:
                            return (self.__table[cls], i)
                        return self.__table[cls]
            if index.__contains__('f') or index.__contains__('n'):
                for i, cls in enumerate(self.__table.keys()):
                    if cls.__contains__('f') or cls.__contains('n'):
                        if give_index == True:
                            return (self.__table[cls], i)
                        return self.__table[cls]
                    
        raise IndexError(f'Class "{index}" not found within the confusion matrix.')
    
    def __eq__(self, other) -> bool:
        """Compares this CM with another CM. 
        
        Returns whether the frequencies in this matrix match the frequencies of
        another matrix.

        Args:
            other (CM): 
                The other matrix that will be compared with this one.

        Returns:
            bool: 
                Returns True if the frequencies of the given matrices match, and
                False if they do not.
        """
        if other.__class__ is self.__class__:
            for this_freq, that_freq in zip(self.class_freqs, other.class_freqs):
                if this_freq != that_freq:
                    return False
            return True
        else:
            return NotImplemented


class ContingencySpace:
    """ 
    An implementation of the Contingency Space.
    """
    
    def show_history(self):
        """
        Shows the history of the space in confusion matrices. 
        
        Args:
            None.
            
        Prints:
            The matrices.
        """
        
        for index, matrix in self.matrices.items():
            
            print(f'--[{index}]-----------------------------------------')
            print(matrix) #adapt to show 
    
    def add_history(self, values: list | dict):
        """Add a history entry.

        Args:
            values: 
                the rates at which the model performed. Can either be a list with the rates of a single model 
                or a dictionary consisting of multiple models with associated keys.
            
        """
        match values:
            case list():
                #add to the dict, generate a key.
                
                #if the number of classes does not match, throw an error to the user <-- to be implemented
                
                
                index: int = len(self.matrices.keys())
                self.matrices.update({str(index), values})
                return
            case dict():
                #add all rows to the dict
                
                for key, matrix in values.items():
                    self.matrices.update({key: matrix})

                return
            case _:
                print('Something has gone wrong. You must pass a list or dictionary of ConfusionMatrix')
                return
    
    def grab_entry(self, key: int | str) -> ConfusionMatrix | None:
        """
        Grabs a matrix from the space history using a string or integer key.

        Args:
            key (int/str): 
                the index or key for the Confusion Matrix to retrieve.
        
            
        Returns:
            ConfusionMatrix:
                The Confusion Matrix requested. If there is no matrix found with that key, returns None.
        """
        
        matrix: ConfusionMatrix = self.matrices[str(key)]
        
        if not matrix:
            return None
        
        return self.matrices[str(key)]
    
    def learning_path_length_2D(self, points: tuple[str, str]) -> float:
        """Calculate the learning path between the first and last points given. Currently only works for binary classification problems.
        
        Args:
            points (tuple): a tuple consisting of two keys that correspond to CMs within the contingency space.
            
        Returns:
            float: the length of the learning path from the first point to the last.
        """
        
        keys = list(self.matrices.keys())
        
        #get the keys for the user-provided matrices.
        (first, last) = points
        
        #convert the keys to an index representing their location within the space.
        first_matrix_index = keys.index(first)
        last_matrix_index = keys.index(last)
        
        #total distance traveled.
        distance_traveled = 0
        
        previous_key = keys[first_matrix_index]
        
        
        for key in keys[first_matrix_index + 1 : last_matrix_index+1]:
            
            #get the first coordinates
            (previous_x, previous_y) = self.matrices[previous_key].vector()
            #get the coordinates of the previous point.
            (current_x, current_y) = self.matrices[key].vector()
            
            #calculate the distance from the previous point to the current point.
            d = np.sqrt((current_x - previous_x)**2 + (current_y - previous_y)**2)
            
            distance_traveled += d
            previous_key = key
        
        return distance_traveled
    
    def learning_path_length_3D(self, points: tuple[str, str], metric: Callable[[ConfusionMatrix], float]) -> float:
        """Calculate the learning path between the first and last points given, using an accuracy metric to determine a third dimension. Currently only works for binary classification problems. 

        Args:
            points (tuple[str, str]): The points you wish to calculate the learning path between. Defaults to None.
            metric (MetricType, optional): The metric you wish to assess the model with. Defaults to Accuracy.
        
        Returns:
            float : The distance between the first point given and the last point given across the contingency space.
        """
        
        keys = list(self.matrices.keys())
        
        #get the keys for the user-provided matrices.
        (first, last) = points
        
        #convert the keys to an index representing their location within the space.
        first_matrix_index = keys.index(first)
        last_matrix_index = keys.index(last)
        
        #total distance traveled.
        distance_traveled = 0
        
        previous_key = keys[first_matrix_index]
        
        for key in keys[first_matrix_index + 1 : last_matrix_index+1]:
            
            #get the first coordinates
            (previous_x, previous_y) = self.matrices[previous_key].vector(return_type=tuple)
            previous_z = metric(self.matrices[previous_key])
            
            #get the coordinates of the next point.
            (current_x, current_y) = self.matrices[key].vector()
            current_z = metric(self.matrices[key])
            
            #calculate the distance from the previous point to the current point.
            d = np.sqrt((current_x - previous_x)**2 + (current_y - previous_y)**2 + (current_z - previous_z)**2)
        
            distance_traveled += d
            previous_key = key
        
        return distance_traveled
    
    def learning_path(self, points: tuple[str, str], metric: Callable[[ConfusionMatrix], float] = None) -> float:
        """Calculate the learning path between the first and last points given. Currently only works for binary classification problems. 
        If a metric is provided, the function will calculate the learning path in 3D.
        
        Args:
            points (tuple): a tuple consisting of two keys that correspond to CMs within the contingency space.
            metric (Callable): The metric to use for the z-axis. Defaults to None.
        """
    
        match metric:
            case None:
                result = self.learning_path_length_2D(points)
                return result
            
        self.learning_path_length_3D()
        
    def visualize(self, metric: Callable[[ConfusionMatrix], float] | list[Callable[[ConfusionMatrix], float]], labels = None, step_size: int = 30, ax=None, projection: str = '2d', **kwargs):

        """Visualize the contingency space in 2D or 3D.
        
        Args:
            metric (Callable): 
                The metric to be used to determine the z-axis. If a list of metrics is provided, the user will be prompted to select one.
            step_size (int): 
                The granularity of the space. Defaults to 30.
            ax (matplotlib.axes._subplots.AxesSubplot): 
                The axes to plot on. If not provided, a new figure will be created.
            projection (str): 
                The projection to use. Can be either '2d' or '3d'. Defaults to '2d'.
            fig_size (tuple):
                The size of the figure. Defaults to (5, 5).
            point_size (int):
                The size of the points on the plot. Defaults to 10.
            lines (bool):
                Whether to draw lines between the points. Defaults to True.
            title (str):
                The title of the plot. Defaults to None.
        """

        point_size_list = [kwargs.get('point_size') for _ in range(len(self.matrices.keys()))]
        fig_size = kwargs.get('fig_size', (5, 5))
        point_size = kwargs.get('point_size', 100)
        lines = kwargs.get('lines', False)
        try:
            weights = metric.weights
            print(weights)
        except:
            weights = None
        title = kwargs.get('title', None)
        lines = kwargs.get('lines', False)
        # print('in cs', weights)
        import matplotlib.pyplot as plt
        from contingency_space.cm_generator import CMGenerator
        
        point_size_list = [point_size for _ in range(len(self.matrices.keys()))]
        
        # Generate the space we will draw the points on.
        # ----------------------------------------------
        example_matrix: ConfusionMatrix = list(self.matrices.values())[0]
        
        # 0: positives, 1: negatives
        matrix_instances = {cls: sum(row) for (cls, row) in example_matrix.matrix.items()}
        matrix_instances_per_class_list = [x for x in matrix_instances.values()]
        

        generator = CMGenerator(self.num_classes, instances_per_class = matrix_instances)
        generator.generate_cms(granularity=step_size)
        
        base_matrices = generator.all_cms

        total_negatives = matrix_instances['f']
        total_positives = matrix_instances['t']
        if weights is not None:
            base_points = [matrix.vector(metric=metric(cm = matrix, do_normalize = False, weights = weights)) for matrix in base_matrices]
        else:
            base_points = [matrix.vector(metric=metric(cm = matrix, do_normalize = False)) for matrix in base_matrices]

        base_x = [x*matrix_instances_per_class_list[1]/total_negatives for x, y, z in base_points]
        base_y = [y*matrix_instances_per_class_list[0]/total_positives for x, y, z in base_points]
        base_z = [z for x, y, z in base_points]

        
        
        base_x = np.array(base_x[:step_size]) # every nth element
        base_y = np.array(base_y[::step_size])  # first n elements
        base_z = np.array(base_z).reshape((step_size, step_size))
        
        base_x_mesh, base_y_mesh = np.meshgrid(base_x, base_y)
        # print(example_matrix)
        # print("this one here", [matrix.vector(metric=metric(cm = matrix, do_normalize = False)) for matrix in self.matrices.values()])
        # print("PLEASE PLEASE PLEASE PLEASE PLEASE PLEAST", self.matrices.values())
        if weights is not None:
            space_matrix_points = [matrix.vector(metric=metric(cm = matrix, do_normalize = False, weights = weights)) for matrix in self.matrices.values()]
        else:
            space_matrix_points = [matrix.vector(metric=metric(cm = matrix, do_normalize = False)) for matrix in self.matrices.values()]
        
        
        # deconstruct tuple
        model_points_x, model_points_y, model_points_z = map(list, zip(*space_matrix_points))
        
        # rescale values
        model_points_x = [x * matrix_instances_per_class_list[1]/total_negatives for x in model_points_x]
        model_points_y = [y * matrix_instances_per_class_list[0]/total_positives for y in model_points_y]
        # print(model_points_z)
        model_points_z = [np.round(z, 2) for z in model_points_z]

        
        match projection:
            case '2d':
                if ax is not None:
                    cs = ax.contourf(base_x_mesh, base_y_mesh, base_z, 30, vmin=-1, vmax=1, alpha=1, cmap='twilight_shifted')
                    # cs2 = ax.contour(base_x_mesh, base_y_mesh, base_z, 30, vmin=-1, vmax=1, cmap='twilight', edgecolor='black', linewidth=9, linewidths=1.7, linestyles='solid')
                    # cs = ax.contourf(base_x_mesh, base_y_mesh, base_z, 30, alpha=1, cmap='twilight_shifted')
                    cs2 = ax.contour(base_x_mesh, base_y_mesh, base_z, 30, cmap='twilight', edgecolor='black', linewidth=9, linewidths=1.7, linestyles='solid')
                    if title is not None: 
                        ax.set_title(title)
                else:
                    plt.figure(figsize=fig_size)
                    cs = plt.contourf(base_x_mesh, base_y_mesh, base_z, 30, vmin=-1, vmax=1, alpha=1, cmap='twilight_shifted', linestyles=None)
                    cs2 = plt.contour(base_x_mesh, base_y_mesh, base_z, 30, vmin=-1, vmax=1, cmap='twilight', linewidths=0.6, linestyles=None)
                    #for c in cs.collections:
                    #    c.set_edgecolor("face")
                     #   c.set_linewidth(0.000000000001)
                    #for c in cs2.collections:
                    #   c.set_edgecolor("face")
                    #    c.set_linewidth(0.000000000001)
            case '3d':
                if ax is not None:
                    base_x_mesh, base_y_mesh = np.meshgrid(base_x, base_y)
                    
                    surf = ax.plot_surface(base_x_mesh, base_y_mesh, base_z, cmap='twilight_shifted', vmin=-1, vmax=1, alpha=0.8)

                else:
                    fig = plt.figure(figsize=fig_size)
                    ax = fig.add_subplot(111, projection='3d')

                    base_x_mesh, base_y_mesh = np.meshgrid(base_x, base_y)

                    surf = ax.plot_surface(base_x_mesh, base_y_mesh, base_z, cmap='twilight_shifted', vmin=-1, vmax=1, alpha=0.90)
        
        match projection:
            case '2d':
                if ax is not None:
                    if lines:
                        ax.plot(model_points_x, model_points_y, color='white', linewidth=1.5, linestyle='-', alpha=1.0, marker='o', markersize=point_size)
                        ax.plot(model_points_x[0], model_points_y[0], color='blue', alpha=1.0, marker='o', markersize=2*point_size)
                        ax.plot(model_points_x[-1], model_points_y[-1], color='red', alpha=1.0, marker='o', markersize=2*point_size)
                    else:
                        # colors = sns.color_palette('tab20', len(labels))
                        colors = distinctipy.get_colors(len(model_points_x))
                        
                        for pt in range(len(model_points_x)):
                            if 'SWPC' in labels[pt] or labels[pt] == 'ZEUS+iPATH_Flare' or labels[pt] == 'GSU All Clear':
                                ax.scatter(model_points_x[pt], model_points_y[pt], label = labels[pt], color=colors[pt], edgecolors='black', alpha=0.8, s=point_size, marker='*')
                            else:
                                ax.scatter(model_points_x[pt], model_points_y[pt], label = labels[pt], color=colors[pt], edgecolors='black', alpha=0.8, s=point_size)
                        plt.colorbar(cs)
                        ax.legend(fontsize = 'small', bbox_to_anchor=(1.35, 1.00), loc='upper right')
                else:
                    if lines:
                        plt.plot(model_points_x, model_points_y, color='white', linewidth=1.5, linestyle='-', alpha=0.8, marker='o', markersize=point_size)
                    else:
                        colors = sns.color_palette('viridis', len(labels))
                        for pt in range(len(model_points_x)):
                            ax.scatter(model_points_x[pt], model_points_y[pt], label = labels[pt], color=colors[pt], edgecolors='black', s=point_size)
                        ax.legend(fontsize = 'small', bbox_to_anchor=(1.15, 1.00), loc='upper right')
                ax.set_xlabel('True Negative Rate')
                ax.set_ylabel('True Positive Rate')
            case '3d':
                
                
            
                if labels is not None:
                    colors = sns.color_palette('viridis', len(labels))
                    for pt in range(len(model_points_x)):
                        ax.scatter(model_points_x[pt], model_points_y[pt], model_points_z[pt], label = labels[pt], color = colors[pt], s=75)
                    ax.legend(fontsize = 'small', bbox_to_anchor=(1.3, 1.0), loc='upper right')
                else:
                    ax.scatter(model_points_x, model_points_y, model_points_z, s=100)   
                ax.contour3D(base_x_mesh, base_y_mesh, base_z, levels=20, colors = 'black')
                # ax.plot(model_points_x, model_points_y, model_points_z, color='black', linewidth=1.5, linestyle='-', alpha=1)
                ax.set_xlabel('True Negative Rate')
                ax.set_ylabel('True Positive Rate')
        if ax is None:
            plt.show()
        
        
    def __init__(self, matrices: dict[str, ConfusionMatrix] | list[ConfusionMatrix] = None):
        """
        
        The constructor for the contingency space.
        
        If initialized with values, they should be passed in as a dictionary of keys and confusion matrices, or as a list
        of confusion matrices.
        
        
        Args:
            matrices: A pre-defined set of models and their values to be plotted on the contingency space. If one is not provided, an empty dictionary will be generated.
        
        """
        
        #If the user has passed in matrices, copy them to the object. Otherwise, initialize an empty dictionary.
        
        if not matrices:
            self.matrices = {}
            self.num_classes = 2
        else:
            self.matrices = {}
            self.num_classes = 2
            match matrices:
                case list():
                    #generate keys for each ConfusionMatrix
                    for index, cm in enumerate(matrices):
                        self.matrices.update({str(index): cm})
                        if self.num_classes != cm.num_classes:
                            if self.num_classes == 0:
                                self.num_classes = cm.num_classes
                            else:
                                raise ValueError('Number of classes must remain the same over every matrix.')
                case dict():
                    self.matrices = copy.deepcopy(matrices)
                    # add num classes










class Tau:
    """
    This is the new metric I am now investigating. It forms two axes by stacking the values tp
    and fn (from the confusion matrix) as the y-axis and tn and fp as the x-axis. It normalizes
    each axis with respect to p and n, respectively. The point located at (x=tn, y=tp) represents
    the model under study in the space of all possible performances. Moreover, the model at (1, 1)
    of this 2D space represents the Perfect model, while the one at (0.5, 0.5) represents the
    Random-guess model's performance.

    This space provides a visualization for each model with respect to the Perfect model and also
    with respect to the Random-guess mode. The distance between any given model and the
    Random-guess model indicates how well the model performs. Similarly, its performance can be
    measured with respect to the Perfect model. Our investigation shows that comparing a model with
    respect to the Perfect model is more meaningful.

    Non-normalized version of tau can be formulated as follows::

            tua = sqrt((fp/n)^2 + (fn/p)^2)
    in other words::

            tau = sqrt( FPR^2 + FNR ^ 2)
    where FPR and FNR are False-Positive Rate and False-Negative Rate, respectively.
    """
    def __init__(self, cm: ConfusionMatrix, do_normalize: bool = True):
        """

        :param cm:
        :param do_normalize: if `True`, normalize TP and FN with respect to P, and TN and FP with
        respect to N.
        """
        self.cm: ConfusionMatrix = cm
        self.model_point, self.perfect_point, self.random_point = self.__measure(do_normalize)
        self.dist_from_random = self.__get_dist_from_random()
        self.dist_from_perfect = self.__get_dist_from_perfect()
        self.value = self.__get_tau(do_normalize)
        self.tn = cm['f'][1]
        self.tp = cm['t'][0]
        self.fp = cm['f'][0]
        self.fn = cm['t'][1]
        manual_calc = 1 - (np.sqrt((self.fp/(self.tn + self.fp))**2 + (self.fn/(self.tp + self.fn))**2)/np.sqrt(2))
        # manual_calc = (np.sqrt((self.fp/(self.fn + self.tp))**2 + (self.fn/(self.tp + self.fn))**2)/np.sqrt(2))
        self.value = manual_calc
        return self.value
        
    def __call__(self, cm: ConfusionMatrix, do_normalize: bool = True):
        """

        :param cm:
        :param do_normalize: if `True`, normalize TP and FN with respect to P, and TN and FP with
        respect to N.
        """
        self.cm: ConfusionMatrix = cm
        self.tn = cm['f'][1]
        self.tp = cm['t'][0]
        self.fp = cm['f'][0]
        self.fn = cm['t'][1]

        self.model_point, self.perfect_point, self.random_point = self.__measure(do_normalize)
        self.dist_from_random = self.__get_dist_from_random()
        self.dist_from_perfect = self.__get_dist_from_perfect()
        self.value = self.__get_tau(do_normalize)
        manual_calc =1 - (np.sqrt((self.fp/(self.tn + self.fp))**2 + (self.fn/(self.tp + self.fn))**2)/np.sqrt(2))
        # manual_calc = (np.sqrt((self.fp/(self.fn + self.tp))**2 + (self.fn/(self.tp + self.fn))**2)/np.sqrt(2))

        return manual_calc

    def __measure(self, do_normalize: bool = True):
        """

        :param do_normalize: see the docstring of the class constructor.

        :return:
        """
        cm: CM = copy.deepcopy(self.cm)
        self.tn = cm['f'][1]
        self.tp = cm['t'][0]
        self.fp = cm['f'][0]
        self.fn = cm['t'][1]
        if do_normalize:
            cm.normalize()
            perfect_point = np.array([1, 1])
        else:
            perfect_point = np.array([self.tn + self.fp, self.tp + self.fn])
        # Position of performance for "the model"
        model_point = np.array([self.tn, self.tp])
        # Position of performance for "Random-Guess Model"
        random_point = np.array([(self.tn + self.fp) / 2, (self.tp + self.fn) / 2])
        return model_point, perfect_point, random_point

    def __get_dist_from_random(self):
        """
        :return: the euclidean distance from the model's point to random-guess's point.
        """
        return np.linalg.norm(self.model_point - self.random_point)

    def __get_dist_from_perfect(self):
        """
        :return: the euclidean distance from the model's point to the perfect model's point (i.e.
        Origin = (0, 0)).
        """

        return np.linalg.norm(self.model_point - self.perfect_point)

    def __get_tau(self, do_normalize: bool = True):
        """
        normalizes `dist_from_perfect` so that it ranges from 0 to 1.
        If the CM is not normalized, it simply returns `dist_from_perfect`
        as is.
        :return: normalized `dist_from_perfect`.
        """
        if do_normalize:
            dist_upper_bound = np.sqrt(2)
            tau = 1 - (self.dist_from_perfect / dist_upper_bound)
        else:
            tau = self.dist_from_perfect


        # tau_surface= []

        return tau
    
    def surface(self):
        return


    def value(self):
        manual_calc =1 - (np.sqrt((self.fp/(self.tn + self.fp))**2 + (self.fn/(self.tp + self.fn))**2)/np.sqrt(2))
        # manual_calc = (np.sqrt((self.fp/(self.fn + self.tp))**2 + (self.fn/(self.tp + self.fn))**2)/np.sqrt(2))

        return manual_calc

