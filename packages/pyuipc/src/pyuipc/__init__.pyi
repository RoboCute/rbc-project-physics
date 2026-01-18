import collections.abc
import numpy
import numpy.typing
import typing
from . import backend as backend, builtin as builtin, constitution as constitution, core as core, diff_sim as diff_sim, geometry as geometry, unit as unit, usd as usd
from pyuipc.core import Animation as Animation, Engine as Engine, Scene as Scene, SceneIO as SceneIO, World as World
from typing import Any, ClassVar, overload

__version__: str

class AngleAxis:
    @overload
    def __init__(self, angle: typing.SupportsFloat, axis: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: pyuipc.AngleAxis, angle: typing.SupportsFloat, axis: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None

        Create an angle-axis from an angle and axis vector.
        Args:
            angle: Rotation angle in radians.
            axis: 3D axis vector (will be normalized).

        2. __init__(self: pyuipc.AngleAxis, quaternion: pyuipc.Quaternion) -> None

        Create an angle-axis from a quaternion.
        Args:
            quaternion: Quaternion to convert.
        """
    @overload
    def __init__(self, quaternion: Quaternion) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: pyuipc.AngleAxis, angle: typing.SupportsFloat, axis: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None

        Create an angle-axis from an angle and axis vector.
        Args:
            angle: Rotation angle in radians.
            axis: 3D axis vector (will be normalized).

        2. __init__(self: pyuipc.AngleAxis, quaternion: pyuipc.Quaternion) -> None

        Create an angle-axis from a quaternion.
        Args:
            quaternion: Quaternion to convert.
        """
    @staticmethod
    def Identity() -> AngleAxis:
        """Identity() -> pyuipc.AngleAxis

        Create an identity angle-axis (no rotation).
        Returns:
            AngleAxis: Identity angle-axis.
        """
    def angle(self) -> float:
        """angle(self: pyuipc.AngleAxis) -> float

        Get the rotation angle.
        Returns:
            float: Rotation angle in radians.
        """
    def axis(self) -> numpy.typing.NDArray[numpy.float64]:
        """axis(self: pyuipc.AngleAxis) -> numpy.typing.NDArray[numpy.float64]

        Get the rotation axis.
        Returns:
            numpy.ndarray: 3D axis vector.
        """
    @overload
    def __mul__(self, other: AngleAxis) -> Quaternion:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: pyuipc.AngleAxis, other: pyuipc.AngleAxis) -> pyuipc.Quaternion

        Multiply two angle-axis rotations (returns quaternion).
        Args:
            other: Another angle-axis.
        Returns:
            Quaternion: Result of rotation composition.

        2. __mul__(self: pyuipc.AngleAxis, quaternion: pyuipc.Quaternion) -> pyuipc.Quaternion

        Multiply angle-axis by quaternion.
        Args:
            quaternion: Quaternion to multiply with.
        Returns:
            Quaternion: Result of multiplication.
        """
    @overload
    def __mul__(self, quaternion: Quaternion) -> Quaternion:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: pyuipc.AngleAxis, other: pyuipc.AngleAxis) -> pyuipc.Quaternion

        Multiply two angle-axis rotations (returns quaternion).
        Args:
            other: Another angle-axis.
        Returns:
            Quaternion: Result of rotation composition.

        2. __mul__(self: pyuipc.AngleAxis, quaternion: pyuipc.Quaternion) -> pyuipc.Quaternion

        Multiply angle-axis by quaternion.
        Args:
            quaternion: Quaternion to multiply with.
        Returns:
            Quaternion: Result of multiplication.
        """

class Float:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def One() -> float:
        """One() -> float

        Get one value.
        Returns:
            Scalar: One value.
        """
    @staticmethod
    def Value(value: typing.SupportsFloat) -> float:
        """Value(value: typing.SupportsFloat) -> float

        Create a scalar value.
        Args:
            value: Scalar value.
        Returns:
            Scalar: The value.
        """
    @staticmethod
    def Zero() -> float:
        """Zero() -> float

        Get zero value.
        Returns:
            Scalar: Zero value.
        """
    @staticmethod
    def size_bytes() -> int:
        """size_bytes() -> int

        Get the size in bytes.
        Returns:
            int: Size in bytes.
        """

class I32:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def One() -> int:
        """One() -> int

        Get one value.
        Returns:
            Scalar: One value.
        """
    @staticmethod
    def Value(value: typing.SupportsInt) -> int:
        """Value(value: typing.SupportsInt) -> int

        Create a scalar value.
        Args:
            value: Scalar value.
        Returns:
            Scalar: The value.
        """
    @staticmethod
    def Zero() -> int:
        """Zero() -> int

        Get zero value.
        Returns:
            Scalar: Zero value.
        """
    @staticmethod
    def size_bytes() -> int:
        """size_bytes() -> int

        Get the size in bytes.
        Returns:
            int: Size in bytes.
        """

class I64:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def One() -> int:
        """One() -> int

        Get one value.
        Returns:
            Scalar: One value.
        """
    @staticmethod
    def Value(value: typing.SupportsInt) -> int:
        """Value(value: typing.SupportsInt) -> int

        Create a scalar value.
        Args:
            value: Scalar value.
        Returns:
            Scalar: The value.
        """
    @staticmethod
    def Zero() -> int:
        """Zero() -> int

        Get zero value.
        Returns:
            Scalar: Zero value.
        """
    @staticmethod
    def size_bytes() -> int:
        """size_bytes() -> int

        Get the size in bytes.
        Returns:
            int: Size in bytes.
        """

class IndexT:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def One() -> int:
        """One() -> int

        Get one value.
        Returns:
            Scalar: One value.
        """
    @staticmethod
    def Value(value: typing.SupportsInt) -> int:
        """Value(value: typing.SupportsInt) -> int

        Create a scalar value.
        Args:
            value: Scalar value.
        Returns:
            Scalar: The value.
        """
    @staticmethod
    def Zero() -> int:
        """Zero() -> int

        Get zero value.
        Returns:
            Scalar: Zero value.
        """
    @staticmethod
    def size_bytes() -> int:
        """size_bytes() -> int

        Get the size in bytes.
        Returns:
            int: Size in bytes.
        """

class Logger:
    class Level:
        __members__: ClassVar[dict] = ...  # read-only
        Critical: ClassVar[Logger.Level] = ...
        Debug: ClassVar[Logger.Level] = ...
        Error: ClassVar[Logger.Level] = ...
        Info: ClassVar[Logger.Level] = ...
        Trace: ClassVar[Logger.Level] = ...
        Warn: ClassVar[Logger.Level] = ...
        __entries: ClassVar[dict] = ...
        def __init__(self, value: typing.SupportsInt) -> None:
            """__init__(self: pyuipc.Logger.Level, value: typing.SupportsInt) -> None"""
        def __eq__(self, other: object) -> bool:
            """__eq__(self: object, other: object, /) -> bool"""
        def __hash__(self) -> int:
            """__hash__(self: object, /) -> int"""
        def __index__(self) -> int:
            """__index__(self: pyuipc.Logger.Level, /) -> int"""
        def __int__(self) -> int:
            """__int__(self: pyuipc.Logger.Level, /) -> int"""
        def __ne__(self, other: object) -> bool:
            """__ne__(self: object, other: object, /) -> bool"""
        @property
        def name(self): ...
        @property
        def value(self) -> int: ...
    Critical: ClassVar[Logger.Level] = ...
    Debug: ClassVar[Logger.Level] = ...
    Error: ClassVar[Logger.Level] = ...
    Info: ClassVar[Logger.Level] = ...
    Trace: ClassVar[Logger.Level] = ...
    Warn: ClassVar[Logger.Level] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def critical(msg: str) -> None:
        """critical(msg: str) -> None

        Log a critical message.
        Args:
            msg: Message to log.
        """
    @staticmethod
    def debug(msg: str) -> None:
        """debug(msg: str) -> None

        Log a debug message.
        Args:
            msg: Message to log.
        """
    @staticmethod
    def error(msg: str) -> None:
        """error(msg: str) -> None

        Log an error message.
        Args:
            msg: Message to log.
        """
    @staticmethod
    def info(msg: str) -> None:
        """info(msg: str) -> None

        Log an info message.
        Args:
            msg: Message to log.
        """
    @staticmethod
    def log(level: Logger.Level, msg: str) -> None:
        """log(level: pyuipc.Logger.Level, msg: str) -> None

        Log a message at the specified level.
        Args:
            level: Logging level.
            msg: Message to log.
        """
    @staticmethod
    def set_level(level: Logger.Level) -> None:
        """set_level(level: pyuipc.Logger.Level) -> None

        Set the logging level.
        Args:
            level: Logging level (Trace, Debug, Info, Warn, Error, Critical).
        """
    @staticmethod
    def set_pattern(pattern: str) -> None:
        """set_pattern(pattern: str) -> None

        Set the logging pattern format.
        Args:
            pattern: Format pattern string for log messages.
        """
    @staticmethod
    def warn(msg: str) -> None:
        """warn(msg: str) -> None

        Log a warning message.
        Args:
            msg: Message to log.
        """

class Matrix12x12:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def Identity() -> numpy.typing.NDArray[numpy.float64]:
        """Identity() -> numpy.typing.NDArray[numpy.float64]

        Create an identity matrix (only for square matrices).
        Returns:
            numpy.ndarray: Identity matrix.
        """
    @staticmethod
    def Ones() -> numpy.typing.NDArray[numpy.float64]:
        """Ones() -> numpy.typing.NDArray[numpy.float64]

        Create a matrix of ones.
        Returns:
            numpy.ndarray: Matrix filled with ones.
        """
    @staticmethod
    def Random() -> numpy.typing.NDArray[numpy.float64]:
        """Random() -> numpy.typing.NDArray[numpy.float64]

        Create a random matrix.
        Returns:
            numpy.ndarray: Random matrix.
        """
    @staticmethod
    def Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Create a matrix from a numpy array.
        Args:
            value: Numpy array to convert.
        Returns:
            numpy.ndarray: Matrix.
        """
    @staticmethod
    def Zero() -> numpy.typing.NDArray[numpy.float64]:
        """Zero() -> numpy.typing.NDArray[numpy.float64]

        Create a zero matrix.
        Returns:
            numpy.ndarray: Zero matrix.
        """
    @staticmethod
    def shape() -> tuple[int, int]:
        """shape() -> tuple[int, int]

        Get the matrix shape.
        Returns:
            tuple: (rows, cols) shape tuple.
        """

class Matrix2x2:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def Identity() -> numpy.typing.NDArray[numpy.float64]:
        """Identity() -> numpy.typing.NDArray[numpy.float64]

        Create an identity matrix (only for square matrices).
        Returns:
            numpy.ndarray: Identity matrix.
        """
    @staticmethod
    def Ones() -> numpy.typing.NDArray[numpy.float64]:
        """Ones() -> numpy.typing.NDArray[numpy.float64]

        Create a matrix of ones.
        Returns:
            numpy.ndarray: Matrix filled with ones.
        """
    @staticmethod
    def Random() -> numpy.typing.NDArray[numpy.float64]:
        """Random() -> numpy.typing.NDArray[numpy.float64]

        Create a random matrix.
        Returns:
            numpy.ndarray: Random matrix.
        """
    @staticmethod
    def Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Create a matrix from a numpy array.
        Args:
            value: Numpy array to convert.
        Returns:
            numpy.ndarray: Matrix.
        """
    @staticmethod
    def Zero() -> numpy.typing.NDArray[numpy.float64]:
        """Zero() -> numpy.typing.NDArray[numpy.float64]

        Create a zero matrix.
        Returns:
            numpy.ndarray: Zero matrix.
        """
    @staticmethod
    def shape() -> tuple[int, int]:
        """shape() -> tuple[int, int]

        Get the matrix shape.
        Returns:
            tuple: (rows, cols) shape tuple.
        """

class Matrix3x3:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def Identity() -> numpy.typing.NDArray[numpy.float64]:
        """Identity() -> numpy.typing.NDArray[numpy.float64]

        Create an identity matrix (only for square matrices).
        Returns:
            numpy.ndarray: Identity matrix.
        """
    @staticmethod
    def Ones() -> numpy.typing.NDArray[numpy.float64]:
        """Ones() -> numpy.typing.NDArray[numpy.float64]

        Create a matrix of ones.
        Returns:
            numpy.ndarray: Matrix filled with ones.
        """
    @staticmethod
    def Random() -> numpy.typing.NDArray[numpy.float64]:
        """Random() -> numpy.typing.NDArray[numpy.float64]

        Create a random matrix.
        Returns:
            numpy.ndarray: Random matrix.
        """
    @staticmethod
    def Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Create a matrix from a numpy array.
        Args:
            value: Numpy array to convert.
        Returns:
            numpy.ndarray: Matrix.
        """
    @staticmethod
    def Zero() -> numpy.typing.NDArray[numpy.float64]:
        """Zero() -> numpy.typing.NDArray[numpy.float64]

        Create a zero matrix.
        Returns:
            numpy.ndarray: Zero matrix.
        """
    @staticmethod
    def shape() -> tuple[int, int]:
        """shape() -> tuple[int, int]

        Get the matrix shape.
        Returns:
            tuple: (rows, cols) shape tuple.
        """

class Matrix4x4:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def Identity() -> numpy.typing.NDArray[numpy.float64]:
        """Identity() -> numpy.typing.NDArray[numpy.float64]

        Create an identity matrix (only for square matrices).
        Returns:
            numpy.ndarray: Identity matrix.
        """
    @staticmethod
    def Ones() -> numpy.typing.NDArray[numpy.float64]:
        """Ones() -> numpy.typing.NDArray[numpy.float64]

        Create a matrix of ones.
        Returns:
            numpy.ndarray: Matrix filled with ones.
        """
    @staticmethod
    def Random() -> numpy.typing.NDArray[numpy.float64]:
        """Random() -> numpy.typing.NDArray[numpy.float64]

        Create a random matrix.
        Returns:
            numpy.ndarray: Random matrix.
        """
    @staticmethod
    def Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Create a matrix from a numpy array.
        Args:
            value: Numpy array to convert.
        Returns:
            numpy.ndarray: Matrix.
        """
    @staticmethod
    def Zero() -> numpy.typing.NDArray[numpy.float64]:
        """Zero() -> numpy.typing.NDArray[numpy.float64]

        Create a zero matrix.
        Returns:
            numpy.ndarray: Zero matrix.
        """
    @staticmethod
    def shape() -> tuple[int, int]:
        """shape() -> tuple[int, int]

        Get the matrix shape.
        Returns:
            tuple: (rows, cols) shape tuple.
        """

class Matrix6x6:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def Identity() -> numpy.typing.NDArray[numpy.float64]:
        """Identity() -> numpy.typing.NDArray[numpy.float64]

        Create an identity matrix (only for square matrices).
        Returns:
            numpy.ndarray: Identity matrix.
        """
    @staticmethod
    def Ones() -> numpy.typing.NDArray[numpy.float64]:
        """Ones() -> numpy.typing.NDArray[numpy.float64]

        Create a matrix of ones.
        Returns:
            numpy.ndarray: Matrix filled with ones.
        """
    @staticmethod
    def Random() -> numpy.typing.NDArray[numpy.float64]:
        """Random() -> numpy.typing.NDArray[numpy.float64]

        Create a random matrix.
        Returns:
            numpy.ndarray: Random matrix.
        """
    @staticmethod
    def Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Create a matrix from a numpy array.
        Args:
            value: Numpy array to convert.
        Returns:
            numpy.ndarray: Matrix.
        """
    @staticmethod
    def Zero() -> numpy.typing.NDArray[numpy.float64]:
        """Zero() -> numpy.typing.NDArray[numpy.float64]

        Create a zero matrix.
        Returns:
            numpy.ndarray: Zero matrix.
        """
    @staticmethod
    def shape() -> tuple[int, int]:
        """shape() -> tuple[int, int]

        Get the matrix shape.
        Returns:
            tuple: (rows, cols) shape tuple.
        """

class Matrix9x9:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def Identity() -> numpy.typing.NDArray[numpy.float64]:
        """Identity() -> numpy.typing.NDArray[numpy.float64]

        Create an identity matrix (only for square matrices).
        Returns:
            numpy.ndarray: Identity matrix.
        """
    @staticmethod
    def Ones() -> numpy.typing.NDArray[numpy.float64]:
        """Ones() -> numpy.typing.NDArray[numpy.float64]

        Create a matrix of ones.
        Returns:
            numpy.ndarray: Matrix filled with ones.
        """
    @staticmethod
    def Random() -> numpy.typing.NDArray[numpy.float64]:
        """Random() -> numpy.typing.NDArray[numpy.float64]

        Create a random matrix.
        Returns:
            numpy.ndarray: Random matrix.
        """
    @staticmethod
    def Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Create a matrix from a numpy array.
        Args:
            value: Numpy array to convert.
        Returns:
            numpy.ndarray: Matrix.
        """
    @staticmethod
    def Zero() -> numpy.typing.NDArray[numpy.float64]:
        """Zero() -> numpy.typing.NDArray[numpy.float64]

        Create a zero matrix.
        Returns:
            numpy.ndarray: Zero matrix.
        """
    @staticmethod
    def shape() -> tuple[int, int]:
        """shape() -> tuple[int, int]

        Get the matrix shape.
        Returns:
            tuple: (rows, cols) shape tuple.
        """

class Quaternion:
    @overload
    def __init__(self, wxyz: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: pyuipc.Quaternion, wxyz: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None

        Create a quaternion from w, x, y, z components.
        Args:
            wxyz: Array of 4 floats [w, x, y, z] representing the quaternion.

        2. __init__(self: pyuipc.Quaternion, angle_axis: pyuipc.AngleAxis) -> None

        Create a quaternion from an AngleAxis.
        Args:
            angle_axis: AngleAxis object to convert.
        """
    @overload
    def __init__(self, angle_axis: AngleAxis) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: pyuipc.Quaternion, wxyz: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None

        Create a quaternion from w, x, y, z components.
        Args:
            wxyz: Array of 4 floats [w, x, y, z] representing the quaternion.

        2. __init__(self: pyuipc.Quaternion, angle_axis: pyuipc.AngleAxis) -> None

        Create a quaternion from an AngleAxis.
        Args:
            angle_axis: AngleAxis object to convert.
        """
    @staticmethod
    def Identity() -> Quaternion:
        """Identity() -> pyuipc.Quaternion

        Create an identity quaternion (no rotation).
        Returns:
            Quaternion: Identity quaternion.
        """
    def conjugate(self) -> Quaternion:
        """conjugate(self: pyuipc.Quaternion) -> pyuipc.Quaternion

        Get the conjugate quaternion.
        Returns:
            Quaternion: Conjugate quaternion.
        """
    def inverse(self) -> Quaternion:
        """inverse(self: pyuipc.Quaternion) -> pyuipc.Quaternion

        Get the inverse quaternion.
        Returns:
            Quaternion: Inverse quaternion.
        """
    @overload
    def norm(self) -> float:
        """norm(self: pyuipc.Quaternion) -> float

        Get the norm (magnitude) of the quaternion.
        Returns:
            float: Norm of the quaternion.
        """
    @overload
    def norm(self, magnitude) -> Any:
        """norm(self: pyuipc.Quaternion) -> float

        Get the norm (magnitude) of the quaternion.
        Returns:
            float: Norm of the quaternion.
        """
    def normalized(self) -> Quaternion:
        """normalized(self: pyuipc.Quaternion) -> pyuipc.Quaternion

        Get a normalized copy of the quaternion.
        Returns:
            Quaternion: Normalized quaternion.
        """
    @overload
    def __mul__(self, other: Quaternion) -> Quaternion:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: pyuipc.Quaternion, other: pyuipc.Quaternion) -> pyuipc.Quaternion

        Multiply two quaternions (composition of rotations).
        Args:
            other: Another quaternion.
        Returns:
            Quaternion: Result of quaternion multiplication.

        2. __mul__(self: pyuipc.Quaternion, angle_axis: pyuipc.AngleAxis) -> pyuipc.Quaternion

        Multiply quaternion by angle-axis.
        Args:
            angle_axis: AngleAxis to multiply with.
        Returns:
            Quaternion: Result of multiplication.
        """
    @overload
    def __mul__(self, angle_axis: AngleAxis) -> Quaternion:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: pyuipc.Quaternion, other: pyuipc.Quaternion) -> pyuipc.Quaternion

        Multiply two quaternions (composition of rotations).
        Args:
            other: Another quaternion.
        Returns:
            Quaternion: Result of quaternion multiplication.

        2. __mul__(self: pyuipc.Quaternion, angle_axis: pyuipc.AngleAxis) -> pyuipc.Quaternion

        Multiply quaternion by angle-axis.
        Args:
            angle_axis: AngleAxis to multiply with.
        Returns:
            Quaternion: Result of multiplication.
        """

class ResidentThread:
    def __init__(self) -> None:
        """__init__(self: pyuipc.ResidentThread) -> None

        Create a new resident thread.
        """
    def is_ready(self) -> bool:
        """is_ready(self: pyuipc.ResidentThread) -> bool

        Check if the resident thread is ready to accept new tasks.
        Returns:
            bool: True if the thread is ready, False otherwise.
        """
    def post(self, func: collections.abc.Callable) -> bool:
        """post(self: pyuipc.ResidentThread, func: collections.abc.Callable) -> bool

        Post a callable to be executed in the resident thread.
        Args:
            func: Python callable object (function) with no arguments to execute.
        Returns:
            bool: True if the function was successfully posted.
        """

class SizeT:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def One() -> int:
        """One() -> int

        Get one value.
        Returns:
            Scalar: One value.
        """
    @staticmethod
    def Value(value: typing.SupportsInt) -> int:
        """Value(value: typing.SupportsInt) -> int

        Create a scalar value.
        Args:
            value: Scalar value.
        Returns:
            Scalar: The value.
        """
    @staticmethod
    def Zero() -> int:
        """Zero() -> int

        Get zero value.
        Returns:
            Scalar: Zero value.
        """
    @staticmethod
    def size_bytes() -> int:
        """size_bytes() -> int

        Get the size in bytes.
        Returns:
            int: Size in bytes.
        """

class Timer:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def disable_all() -> None:
        """disable_all() -> None

        Disable all timers.
        """
    @staticmethod
    def enable_all() -> None:
        """enable_all() -> None

        Enable all timers for performance measurement.
        """
    @staticmethod
    def report() -> None:
        """report() -> None

        Print timing report to the console.
        """
    @staticmethod
    def report_as_json() -> json:
        """report_as_json() -> json

        Get timing report as a JSON object.
        Returns:
            dict: Timing report as a dictionary.
        """

class Transform:
    def __init__(self, matrix: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """__init__(self: pyuipc.Transform, matrix: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None

        Create a transform from a 4x4 matrix.
        Args:
            matrix: 4x4 transformation matrix.
        """
    @staticmethod
    def Identity() -> Transform:
        """Identity() -> pyuipc.Transform

        Create an identity transform (no translation, no rotation, unit scale).
        Returns:
            Transform: Identity transformation matrix.
        """
    def apply_to(self, vectors: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """apply_to(self: pyuipc.Transform, vectors: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Apply transform to a vector or array of vectors (in-place for arrays).
        Args:
            vectors: Single 3D vector or array of 3D vectors.
        Returns:
            numpy.ndarray: Transformed vector(s).
        """
    def inverse(self, arg0) -> Transform:
        """inverse(self: pyuipc.Transform, arg0: Eigen::TransformTraits) -> pyuipc.Transform

        Get the inverse transform.
        Returns:
            Transform: Inverse transformation matrix.
        """
    def matrix(self) -> numpy.typing.NDArray[numpy.float64]:
        """matrix(self: pyuipc.Transform) -> numpy.typing.NDArray[numpy.float64]

        Get the 4x4 transformation matrix.
        Returns:
            numpy.ndarray: 4x4 transformation matrix.
        """
    @overload
    def prerotate(self, angle_axis: AngleAxis) -> Transform:
        """prerotate(*args, **kwargs)
        Overloaded function.

        1. prerotate(self: pyuipc.Transform, angle_axis: pyuipc.AngleAxis) -> pyuipc.Transform

        Apply rotation to the transform (pre-multiply).
        Args:
            angle_axis: AngleAxis rotation.
        Returns:
            Transform: Reference to self for chaining.

        2. prerotate(self: pyuipc.Transform, quaternion: pyuipc.Quaternion) -> pyuipc.Transform

        Apply rotation to the transform (pre-multiply).
        Args:
            quaternion: Quaternion rotation.
        Returns:
            Transform: Reference to self for chaining.
        """
    @overload
    def prerotate(self, quaternion: Quaternion) -> Transform:
        """prerotate(*args, **kwargs)
        Overloaded function.

        1. prerotate(self: pyuipc.Transform, angle_axis: pyuipc.AngleAxis) -> pyuipc.Transform

        Apply rotation to the transform (pre-multiply).
        Args:
            angle_axis: AngleAxis rotation.
        Returns:
            Transform: Reference to self for chaining.

        2. prerotate(self: pyuipc.Transform, quaternion: pyuipc.Quaternion) -> pyuipc.Transform

        Apply rotation to the transform (pre-multiply).
        Args:
            quaternion: Quaternion rotation.
        Returns:
            Transform: Reference to self for chaining.
        """
    def prescale(self, scale: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> Transform:
        """prescale(self: pyuipc.Transform, scale: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> pyuipc.Transform

        Apply scaling to the transform (pre-multiply).
        Args:
            scale: Either a scalar (uniform scale) or 3D vector (non-uniform scale).
        Returns:
            Transform: Reference to self for chaining.
        """
    def pretranslate(self, translation: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> Transform:
        """pretranslate(self: pyuipc.Transform, translation: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> pyuipc.Transform

        Apply translation to the transform (pre-multiply).
        Args:
            translation: 3D translation vector.
        Returns:
            Transform: Reference to self for chaining.
        """
    @overload
    def rotate(self, angle_axis: AngleAxis) -> Transform:
        """rotate(*args, **kwargs)
        Overloaded function.

        1. rotate(self: pyuipc.Transform, angle_axis: pyuipc.AngleAxis) -> pyuipc.Transform

        Apply rotation to the transform (post-multiply).
        Args:
            angle_axis: AngleAxis rotation.
        Returns:
            Transform: Reference to self for chaining.

        2. rotate(self: pyuipc.Transform, quaternion: pyuipc.Quaternion) -> pyuipc.Transform

        Apply rotation to the transform (post-multiply).
        Args:
            quaternion: Quaternion rotation.
        Returns:
            Transform: Reference to self for chaining.
        """
    @overload
    def rotate(self, quaternion: Quaternion) -> Transform:
        """rotate(*args, **kwargs)
        Overloaded function.

        1. rotate(self: pyuipc.Transform, angle_axis: pyuipc.AngleAxis) -> pyuipc.Transform

        Apply rotation to the transform (post-multiply).
        Args:
            angle_axis: AngleAxis rotation.
        Returns:
            Transform: Reference to self for chaining.

        2. rotate(self: pyuipc.Transform, quaternion: pyuipc.Quaternion) -> pyuipc.Transform

        Apply rotation to the transform (post-multiply).
        Args:
            quaternion: Quaternion rotation.
        Returns:
            Transform: Reference to self for chaining.
        """
    def scale(self, scale: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> Transform:
        """scale(self: pyuipc.Transform, scale: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> pyuipc.Transform

        Apply scaling to the transform (post-multiply).
        Args:
            scale: Either a scalar (uniform scale) or 3D vector (non-uniform scale).
        Returns:
            Transform: Reference to self for chaining.
        """
    def translate(self, translation: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> Transform:
        """translate(self: pyuipc.Transform, translation: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> pyuipc.Transform

        Apply translation to the transform (post-multiply).
        Args:
            translation: 3D translation vector.
        Returns:
            Transform: Reference to self for chaining.
        """
    def translation(self) -> numpy.typing.NDArray[numpy.float64]:
        """translation(self: pyuipc.Transform) -> numpy.typing.NDArray[numpy.float64]

        Get the translation component of the transform.
        Returns:
            numpy.ndarray: 3D translation vector.
        """
    @overload
    def __mul__(self, other: Transform) -> Transform:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: pyuipc.Transform, other: pyuipc.Transform) -> pyuipc.Transform

        Compose two transforms (matrix multiplication).
        Args:
            other: Another transform.
        Returns:
            Transform: Result of transform composition.

        2. __mul__(self: pyuipc.Transform, vector: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Apply transform to a 3D vector.
        Args:
            vector: 3D vector to transform.
        Returns:
            numpy.ndarray: Transformed 3D vector.
        """
    @overload
    def __mul__(self, vector: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """__mul__(*args, **kwargs)
        Overloaded function.

        1. __mul__(self: pyuipc.Transform, other: pyuipc.Transform) -> pyuipc.Transform

        Compose two transforms (matrix multiplication).
        Args:
            other: Another transform.
        Returns:
            Transform: Result of transform composition.

        2. __mul__(self: pyuipc.Transform, vector: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Apply transform to a 3D vector.
        Args:
            vector: 3D vector to transform.
        Returns:
            numpy.ndarray: Transformed 3D vector.
        """

class U32:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def One() -> int:
        """One() -> int

        Get one value.
        Returns:
            Scalar: One value.
        """
    @staticmethod
    def Value(value: typing.SupportsInt) -> int:
        """Value(value: typing.SupportsInt) -> int

        Create a scalar value.
        Args:
            value: Scalar value.
        Returns:
            Scalar: The value.
        """
    @staticmethod
    def Zero() -> int:
        """Zero() -> int

        Get zero value.
        Returns:
            Scalar: Zero value.
        """
    @staticmethod
    def size_bytes() -> int:
        """size_bytes() -> int

        Get the size in bytes.
        Returns:
            int: Size in bytes.
        """

class U64:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def One() -> int:
        """One() -> int

        Get one value.
        Returns:
            Scalar: One value.
        """
    @staticmethod
    def Value(value: typing.SupportsInt) -> int:
        """Value(value: typing.SupportsInt) -> int

        Create a scalar value.
        Args:
            value: Scalar value.
        Returns:
            Scalar: The value.
        """
    @staticmethod
    def Zero() -> int:
        """Zero() -> int

        Get zero value.
        Returns:
            Scalar: Zero value.
        """
    @staticmethod
    def size_bytes() -> int:
        """size_bytes() -> int

        Get the size in bytes.
        Returns:
            int: Size in bytes.
        """

class Vector12:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def Identity() -> numpy.typing.NDArray[numpy.float64]:
        """Identity() -> numpy.typing.NDArray[numpy.float64]

        Create an identity matrix (only for square matrices).
        Returns:
            numpy.ndarray: Identity matrix.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @staticmethod
    def Ones() -> numpy.typing.NDArray[numpy.float64]:
        """Ones() -> numpy.typing.NDArray[numpy.float64]

        Create a matrix of ones.
        Returns:
            numpy.ndarray: Matrix filled with ones.
        """
    @staticmethod
    def Random() -> numpy.typing.NDArray[numpy.float64]:
        """Random() -> numpy.typing.NDArray[numpy.float64]

        Create a random matrix.
        Returns:
            numpy.ndarray: Random matrix.
        """
    @staticmethod
    def Unit(i: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """Unit(i: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector with 1 at index i.
        Args:
            i: Index of the unit element (0-based).
        Returns:
            numpy.ndarray: Unit vector.
        """
    @staticmethod
    def UnitW() -> numpy.typing.NDArray[numpy.float64]:
        """UnitW() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in W direction.
        Returns:
            numpy.ndarray: Unit W vector.
        """
    @staticmethod
    def UnitX() -> numpy.typing.NDArray[numpy.float64]:
        """UnitX() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in X direction.
        Returns:
            numpy.ndarray: Unit X vector.
        """
    @staticmethod
    def UnitY() -> numpy.typing.NDArray[numpy.float64]:
        """UnitY() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in Y direction.
        Returns:
            numpy.ndarray: Unit Y vector.
        """
    @staticmethod
    def UnitZ() -> numpy.typing.NDArray[numpy.float64]:
        """UnitZ() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in Z direction.
        Returns:
            numpy.ndarray: Unit Z vector.
        """
    @staticmethod
    def Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Create a matrix from a numpy array.
        Args:
            value: Numpy array to convert.
        Returns:
            numpy.ndarray: Matrix.
        """
    @staticmethod
    def Zero() -> numpy.typing.NDArray[numpy.float64]:
        """Zero() -> numpy.typing.NDArray[numpy.float64]

        Create a zero matrix.
        Returns:
            numpy.ndarray: Zero matrix.
        """
    @staticmethod
    def shape() -> tuple[int, int]:
        """shape() -> tuple[int, int]

        Get the matrix shape.
        Returns:
            tuple: (rows, cols) shape tuple.
        """

class Vector2:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def Identity() -> numpy.typing.NDArray[numpy.float64]:
        """Identity() -> numpy.typing.NDArray[numpy.float64]

        Create an identity matrix (only for square matrices).
        Returns:
            numpy.ndarray: Identity matrix.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @staticmethod
    def Ones() -> numpy.typing.NDArray[numpy.float64]:
        """Ones() -> numpy.typing.NDArray[numpy.float64]

        Create a matrix of ones.
        Returns:
            numpy.ndarray: Matrix filled with ones.
        """
    @staticmethod
    def Random() -> numpy.typing.NDArray[numpy.float64]:
        """Random() -> numpy.typing.NDArray[numpy.float64]

        Create a random matrix.
        Returns:
            numpy.ndarray: Random matrix.
        """
    @staticmethod
    def Unit(i: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """Unit(i: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector with 1 at index i.
        Args:
            i: Index of the unit element (0-based).
        Returns:
            numpy.ndarray: Unit vector.
        """
    @staticmethod
    def UnitX() -> numpy.typing.NDArray[numpy.float64]:
        """UnitX() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in X direction.
        Returns:
            numpy.ndarray: Unit X vector.
        """
    @staticmethod
    def UnitY() -> numpy.typing.NDArray[numpy.float64]:
        """UnitY() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in Y direction.
        Returns:
            numpy.ndarray: Unit Y vector.
        """
    @staticmethod
    def Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Create a matrix from a numpy array.
        Args:
            value: Numpy array to convert.
        Returns:
            numpy.ndarray: Matrix.
        """
    @staticmethod
    def Zero() -> numpy.typing.NDArray[numpy.float64]:
        """Zero() -> numpy.typing.NDArray[numpy.float64]

        Create a zero matrix.
        Returns:
            numpy.ndarray: Zero matrix.
        """
    @staticmethod
    def shape() -> tuple[int, int]:
        """shape() -> tuple[int, int]

        Get the matrix shape.
        Returns:
            tuple: (rows, cols) shape tuple.
        """

class Vector3:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def Identity() -> numpy.typing.NDArray[numpy.float64]:
        """Identity() -> numpy.typing.NDArray[numpy.float64]

        Create an identity matrix (only for square matrices).
        Returns:
            numpy.ndarray: Identity matrix.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @staticmethod
    def Ones() -> numpy.typing.NDArray[numpy.float64]:
        """Ones() -> numpy.typing.NDArray[numpy.float64]

        Create a matrix of ones.
        Returns:
            numpy.ndarray: Matrix filled with ones.
        """
    @staticmethod
    def Random() -> numpy.typing.NDArray[numpy.float64]:
        """Random() -> numpy.typing.NDArray[numpy.float64]

        Create a random matrix.
        Returns:
            numpy.ndarray: Random matrix.
        """
    @staticmethod
    def Unit(i: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """Unit(i: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector with 1 at index i.
        Args:
            i: Index of the unit element (0-based).
        Returns:
            numpy.ndarray: Unit vector.
        """
    @staticmethod
    def UnitX() -> numpy.typing.NDArray[numpy.float64]:
        """UnitX() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in X direction.
        Returns:
            numpy.ndarray: Unit X vector.
        """
    @staticmethod
    def UnitY() -> numpy.typing.NDArray[numpy.float64]:
        """UnitY() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in Y direction.
        Returns:
            numpy.ndarray: Unit Y vector.
        """
    @staticmethod
    def UnitZ() -> numpy.typing.NDArray[numpy.float64]:
        """UnitZ() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in Z direction.
        Returns:
            numpy.ndarray: Unit Z vector.
        """
    @staticmethod
    def Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Create a matrix from a numpy array.
        Args:
            value: Numpy array to convert.
        Returns:
            numpy.ndarray: Matrix.
        """
    @staticmethod
    def Zero() -> numpy.typing.NDArray[numpy.float64]:
        """Zero() -> numpy.typing.NDArray[numpy.float64]

        Create a zero matrix.
        Returns:
            numpy.ndarray: Zero matrix.
        """
    @staticmethod
    def shape() -> tuple[int, int]:
        """shape() -> tuple[int, int]

        Get the matrix shape.
        Returns:
            tuple: (rows, cols) shape tuple.
        """

class Vector4:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def Identity() -> numpy.typing.NDArray[numpy.float64]:
        """Identity() -> numpy.typing.NDArray[numpy.float64]

        Create an identity matrix (only for square matrices).
        Returns:
            numpy.ndarray: Identity matrix.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @staticmethod
    def Ones() -> numpy.typing.NDArray[numpy.float64]:
        """Ones() -> numpy.typing.NDArray[numpy.float64]

        Create a matrix of ones.
        Returns:
            numpy.ndarray: Matrix filled with ones.
        """
    @staticmethod
    def Random() -> numpy.typing.NDArray[numpy.float64]:
        """Random() -> numpy.typing.NDArray[numpy.float64]

        Create a random matrix.
        Returns:
            numpy.ndarray: Random matrix.
        """
    @staticmethod
    def Unit(i: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """Unit(i: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector with 1 at index i.
        Args:
            i: Index of the unit element (0-based).
        Returns:
            numpy.ndarray: Unit vector.
        """
    @staticmethod
    def UnitW() -> numpy.typing.NDArray[numpy.float64]:
        """UnitW() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in W direction.
        Returns:
            numpy.ndarray: Unit W vector.
        """
    @staticmethod
    def UnitX() -> numpy.typing.NDArray[numpy.float64]:
        """UnitX() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in X direction.
        Returns:
            numpy.ndarray: Unit X vector.
        """
    @staticmethod
    def UnitY() -> numpy.typing.NDArray[numpy.float64]:
        """UnitY() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in Y direction.
        Returns:
            numpy.ndarray: Unit Y vector.
        """
    @staticmethod
    def UnitZ() -> numpy.typing.NDArray[numpy.float64]:
        """UnitZ() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in Z direction.
        Returns:
            numpy.ndarray: Unit Z vector.
        """
    @staticmethod
    def Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Create a matrix from a numpy array.
        Args:
            value: Numpy array to convert.
        Returns:
            numpy.ndarray: Matrix.
        """
    @staticmethod
    def Zero() -> numpy.typing.NDArray[numpy.float64]:
        """Zero() -> numpy.typing.NDArray[numpy.float64]

        Create a zero matrix.
        Returns:
            numpy.ndarray: Zero matrix.
        """
    @staticmethod
    def shape() -> tuple[int, int]:
        """shape() -> tuple[int, int]

        Get the matrix shape.
        Returns:
            tuple: (rows, cols) shape tuple.
        """

class Vector6:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def Identity() -> numpy.typing.NDArray[numpy.float64]:
        """Identity() -> numpy.typing.NDArray[numpy.float64]

        Create an identity matrix (only for square matrices).
        Returns:
            numpy.ndarray: Identity matrix.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @staticmethod
    def Ones() -> numpy.typing.NDArray[numpy.float64]:
        """Ones() -> numpy.typing.NDArray[numpy.float64]

        Create a matrix of ones.
        Returns:
            numpy.ndarray: Matrix filled with ones.
        """
    @staticmethod
    def Random() -> numpy.typing.NDArray[numpy.float64]:
        """Random() -> numpy.typing.NDArray[numpy.float64]

        Create a random matrix.
        Returns:
            numpy.ndarray: Random matrix.
        """
    @staticmethod
    def Unit(i: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """Unit(i: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector with 1 at index i.
        Args:
            i: Index of the unit element (0-based).
        Returns:
            numpy.ndarray: Unit vector.
        """
    @staticmethod
    def UnitW() -> numpy.typing.NDArray[numpy.float64]:
        """UnitW() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in W direction.
        Returns:
            numpy.ndarray: Unit W vector.
        """
    @staticmethod
    def UnitX() -> numpy.typing.NDArray[numpy.float64]:
        """UnitX() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in X direction.
        Returns:
            numpy.ndarray: Unit X vector.
        """
    @staticmethod
    def UnitY() -> numpy.typing.NDArray[numpy.float64]:
        """UnitY() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in Y direction.
        Returns:
            numpy.ndarray: Unit Y vector.
        """
    @staticmethod
    def UnitZ() -> numpy.typing.NDArray[numpy.float64]:
        """UnitZ() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in Z direction.
        Returns:
            numpy.ndarray: Unit Z vector.
        """
    @staticmethod
    def Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Create a matrix from a numpy array.
        Args:
            value: Numpy array to convert.
        Returns:
            numpy.ndarray: Matrix.
        """
    @staticmethod
    def Zero() -> numpy.typing.NDArray[numpy.float64]:
        """Zero() -> numpy.typing.NDArray[numpy.float64]

        Create a zero matrix.
        Returns:
            numpy.ndarray: Zero matrix.
        """
    @staticmethod
    def shape() -> tuple[int, int]:
        """shape() -> tuple[int, int]

        Get the matrix shape.
        Returns:
            tuple: (rows, cols) shape tuple.
        """

class Vector9:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def Identity() -> numpy.typing.NDArray[numpy.float64]:
        """Identity() -> numpy.typing.NDArray[numpy.float64]

        Create an identity matrix (only for square matrices).
        Returns:
            numpy.ndarray: Identity matrix.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @overload
    @staticmethod
    def LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
        """LinSpaced(*args, **kwargs)
        Overloaded function.

        1. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, n: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector.
        Args:
            start: Start value.
            end: End value.
            n: Number of points.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        2. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector (size M).
        Args:
            start: Start value.
            end: End value.
        Returns:
            numpy.ndarray: Linearly spaced vector.

        3. LinSpaced(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]

        Create a linearly spaced vector with step size.
        Args:
            start: Start value.
            end: End value.
            step: Step size.
        Returns:
            numpy.ndarray: Linearly spaced vector.
        """
    @staticmethod
    def Ones() -> numpy.typing.NDArray[numpy.float64]:
        """Ones() -> numpy.typing.NDArray[numpy.float64]

        Create a matrix of ones.
        Returns:
            numpy.ndarray: Matrix filled with ones.
        """
    @staticmethod
    def Random() -> numpy.typing.NDArray[numpy.float64]:
        """Random() -> numpy.typing.NDArray[numpy.float64]

        Create a random matrix.
        Returns:
            numpy.ndarray: Random matrix.
        """
    @staticmethod
    def Unit(i: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
        """Unit(i: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector with 1 at index i.
        Args:
            i: Index of the unit element (0-based).
        Returns:
            numpy.ndarray: Unit vector.
        """
    @staticmethod
    def UnitW() -> numpy.typing.NDArray[numpy.float64]:
        """UnitW() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in W direction.
        Returns:
            numpy.ndarray: Unit W vector.
        """
    @staticmethod
    def UnitX() -> numpy.typing.NDArray[numpy.float64]:
        """UnitX() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in X direction.
        Returns:
            numpy.ndarray: Unit X vector.
        """
    @staticmethod
    def UnitY() -> numpy.typing.NDArray[numpy.float64]:
        """UnitY() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in Y direction.
        Returns:
            numpy.ndarray: Unit Y vector.
        """
    @staticmethod
    def UnitZ() -> numpy.typing.NDArray[numpy.float64]:
        """UnitZ() -> numpy.typing.NDArray[numpy.float64]

        Create a unit vector in Z direction.
        Returns:
            numpy.ndarray: Unit Z vector.
        """
    @staticmethod
    def Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        """Values(value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

        Create a matrix from a numpy array.
        Args:
            value: Numpy array to convert.
        Returns:
            numpy.ndarray: Matrix.
        """
    @staticmethod
    def Zero() -> numpy.typing.NDArray[numpy.float64]:
        """Zero() -> numpy.typing.NDArray[numpy.float64]

        Create a zero matrix.
        Returns:
            numpy.ndarray: Zero matrix.
        """
    @staticmethod
    def shape() -> tuple[int, int]:
        """shape() -> tuple[int, int]

        Get the matrix shape.
        Returns:
            tuple: (rows, cols) shape tuple.
        """

def config() -> json:
    """config() -> json

    Get the current configuration of libuipc.
    Returns:
        dict: Current configuration dictionary.
    """
def default_config() -> json:
    """default_config() -> json

    Get the default configuration for libuipc.
    Returns:
        dict: Default configuration dictionary.
    """
def init(dict: dict) -> None:
    """init(dict: dict) -> None

    Initialize the libuipc library with the given configuration.
    Args:
        dict: Configuration dictionary containing library settings.
    """
@overload
def view(slot: geometry.AttributeSlotString) -> geometry.StringSpan:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(slot: geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
@overload
def view(pc: diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]:
    """view(*args, **kwargs)
    Overloaded function.

    1. view(slot: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.StringSpan

    Get a view of a string attribute slot.
    Args:
        slot: AttributeSlotString to view.
    Returns:
        StringSpan: View of string attribute data.

    2. view(slot: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    3. view(slot: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    4. view(slot: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    5. view(slot: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    6. view(slot: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    7. view(slot: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    8. view(slot: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    9. view(slot: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    10. view(slot: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    11. view(slot: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    12. view(slot: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    13. view(slot: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    14. view(slot: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    15. view(slot: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    16. view(slot: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    17. view(slot: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    18. view(slot: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    19. view(slot: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    20. view(slot: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    21. view(slot: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

    Get a numpy array view of an attribute slot.
    Args:
        slot: AttributeSlot to view.
    Returns:
        numpy.ndarray: Array view of attribute data.

    22. view(pc: pyuipc.diff_sim.ParameterCollection) -> numpy.typing.NDArray[numpy.float64]

    Get a view of parameter collection as a numpy array.
    Args:
        pc: ParameterCollection to view.
    Returns:
        numpy.ndarray: Array view of parameters.
    """
