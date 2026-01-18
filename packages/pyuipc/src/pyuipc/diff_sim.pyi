import numpy
import numpy.typing
import scipy.sparse
import typing

class ParameterCollection:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def broadcast(self) -> None:
        """broadcast(self: pyuipc.diff_sim.ParameterCollection) -> None

        Broadcast parameter values across all instances.
        """
    def resize(self, N: typing.SupportsInt, default_value: typing.SupportsFloat = ...) -> None:
        """resize(self: pyuipc.diff_sim.ParameterCollection, N: typing.SupportsInt, default_value: typing.SupportsFloat = 0.0) -> None

        Resize the parameter collection.
        Args:
            N: New size.
            default_value: Default value for new parameters (default: 0.0).
        """
    def view(self) -> numpy.typing.NDArray[numpy.float64]:
        """view(self: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

        Get a view of the parameters as a numpy array.
        Returns:
            numpy.ndarray: Array view of parameters.
        """

class SparseCOOView:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def col_indices(self) -> numpy.typing.NDArray[numpy.int32]:
        """col_indices(self: pyuipc.diff_sim.SparseCOOView) -> numpy.typing.NDArray[numpy.int32]

        Get the column indices.
        Returns:
            numpy.ndarray: Array of column indices.
        """
    def row_indices(self) -> numpy.typing.NDArray[numpy.int32]:
        """row_indices(self: pyuipc.diff_sim.SparseCOOView) -> numpy.typing.NDArray[numpy.int32]

        Get the row indices.
        Returns:
            numpy.ndarray: Array of row indices.
        """
    def shape(self) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], '[2, 1]']:
        '''shape(self: pyuipc.diff_sim.SparseCOOView) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[2, 1]"]

        Get the matrix shape.
        Returns:
            tuple: (rows, cols) shape of the matrix.
        '''
    def to_dense(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], '[m, n]']:
        '''to_dense(self: pyuipc.diff_sim.SparseCOOView) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]

        Convert to dense matrix representation.
        Returns:
            numpy.ndarray: Dense matrix.
        '''
    def to_sparse(self) -> scipy.sparse.csc_matrix[numpy.float64]:
        """to_sparse(self: pyuipc.diff_sim.SparseCOOView) -> scipy.sparse.csc_matrix[numpy.float64]

        Convert to sparse matrix representation (scipy.sparse).
        Returns:
            scipy.sparse matrix: Sparse matrix object.
        """
    def values(self) -> numpy.typing.NDArray[numpy.float64]:
        """values(self: pyuipc.diff_sim.SparseCOOView) -> numpy.typing.NDArray[numpy.float64]

        Get the values.
        Returns:
            numpy.ndarray: Array of matrix values.
        """
