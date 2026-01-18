import collections.abc
import numpy
import numpy.typing
import pyuipc
import typing
from typing import overload

class AttributeCollection:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def attribute_count(self) -> int:
        """attribute_count(self: pyuipc.geometry.AttributeCollection) -> int

        Get the number of attributes.
        Returns:
            int: Number of attributes.
        """
    def clear(self) -> None:
        """clear(self: pyuipc.geometry.AttributeCollection) -> None

        Clear all attributes.
        """
    @overload
    def create(self, name: str, value: int) -> IAttributeSlot:
        """create(*args, **kwargs)
        Overloaded function.

        1. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create an I32 (32-bit integer) attribute.
        Args:
            name: Attribute name.
            value: I32 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        2. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create an I64 (64-bit integer) attribute.
        Args:
            name: Attribute name.
            value: I64 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        3. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create a U64 (64-bit unsigned integer) attribute.
        Args:
            name: Attribute name.
            value: U64 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        4. create(self: pyuipc.geometry.AttributeCollection, name: str, value: float) -> pyuipc.geometry.IAttributeSlot

        Create a Float (floating-point) attribute.
        Args:
            name: Attribute name.
            value: Float scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        5. create(self: pyuipc.geometry.AttributeCollection, name: str, value: numpy.typing.NDArray[numpy.float64]) -> pyuipc.geometry.IAttributeSlot

        Create an attribute from a numpy array (auto-detects type: scalar, vector, or matrix).
        Args:
            name: Attribute name.
            value: Numpy array (can be scalar, vector of size 2/3/4/6/9/12, or matrix of size 2x2/3x3/4x4/6x6/9x9/12x12).
        Returns:
            AttributeSlot: Created attribute slot.

        6. create(self: pyuipc.geometry.AttributeCollection, name: str, value: numpy.typing.NDArray[numpy.int32]) -> pyuipc.geometry.IAttributeSlot

        Create an integer attribute from a numpy array (auto-detects type: scalar or integer vector).
        Args:
            name: Attribute name.
            value: Numpy array (can be scalar or integer vector of size 2/3/4).
        Returns:
            AttributeSlot: Created attribute slot.

        7. create(self: pyuipc.geometry.AttributeCollection, name: str, value: str) -> pyuipc.geometry.AttributeSlotString

        Create a string attribute.
        Args:
            name: Attribute name.
            value: String value.
        Returns:
            AttributeSlot: Created attribute slot.
        """
    @overload
    def create(self, name: str, value: int) -> IAttributeSlot:
        """create(*args, **kwargs)
        Overloaded function.

        1. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create an I32 (32-bit integer) attribute.
        Args:
            name: Attribute name.
            value: I32 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        2. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create an I64 (64-bit integer) attribute.
        Args:
            name: Attribute name.
            value: I64 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        3. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create a U64 (64-bit unsigned integer) attribute.
        Args:
            name: Attribute name.
            value: U64 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        4. create(self: pyuipc.geometry.AttributeCollection, name: str, value: float) -> pyuipc.geometry.IAttributeSlot

        Create a Float (floating-point) attribute.
        Args:
            name: Attribute name.
            value: Float scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        5. create(self: pyuipc.geometry.AttributeCollection, name: str, value: numpy.typing.NDArray[numpy.float64]) -> pyuipc.geometry.IAttributeSlot

        Create an attribute from a numpy array (auto-detects type: scalar, vector, or matrix).
        Args:
            name: Attribute name.
            value: Numpy array (can be scalar, vector of size 2/3/4/6/9/12, or matrix of size 2x2/3x3/4x4/6x6/9x9/12x12).
        Returns:
            AttributeSlot: Created attribute slot.

        6. create(self: pyuipc.geometry.AttributeCollection, name: str, value: numpy.typing.NDArray[numpy.int32]) -> pyuipc.geometry.IAttributeSlot

        Create an integer attribute from a numpy array (auto-detects type: scalar or integer vector).
        Args:
            name: Attribute name.
            value: Numpy array (can be scalar or integer vector of size 2/3/4).
        Returns:
            AttributeSlot: Created attribute slot.

        7. create(self: pyuipc.geometry.AttributeCollection, name: str, value: str) -> pyuipc.geometry.AttributeSlotString

        Create a string attribute.
        Args:
            name: Attribute name.
            value: String value.
        Returns:
            AttributeSlot: Created attribute slot.
        """
    @overload
    def create(self, name: str, value: int) -> IAttributeSlot:
        """create(*args, **kwargs)
        Overloaded function.

        1. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create an I32 (32-bit integer) attribute.
        Args:
            name: Attribute name.
            value: I32 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        2. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create an I64 (64-bit integer) attribute.
        Args:
            name: Attribute name.
            value: I64 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        3. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create a U64 (64-bit unsigned integer) attribute.
        Args:
            name: Attribute name.
            value: U64 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        4. create(self: pyuipc.geometry.AttributeCollection, name: str, value: float) -> pyuipc.geometry.IAttributeSlot

        Create a Float (floating-point) attribute.
        Args:
            name: Attribute name.
            value: Float scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        5. create(self: pyuipc.geometry.AttributeCollection, name: str, value: numpy.typing.NDArray[numpy.float64]) -> pyuipc.geometry.IAttributeSlot

        Create an attribute from a numpy array (auto-detects type: scalar, vector, or matrix).
        Args:
            name: Attribute name.
            value: Numpy array (can be scalar, vector of size 2/3/4/6/9/12, or matrix of size 2x2/3x3/4x4/6x6/9x9/12x12).
        Returns:
            AttributeSlot: Created attribute slot.

        6. create(self: pyuipc.geometry.AttributeCollection, name: str, value: numpy.typing.NDArray[numpy.int32]) -> pyuipc.geometry.IAttributeSlot

        Create an integer attribute from a numpy array (auto-detects type: scalar or integer vector).
        Args:
            name: Attribute name.
            value: Numpy array (can be scalar or integer vector of size 2/3/4).
        Returns:
            AttributeSlot: Created attribute slot.

        7. create(self: pyuipc.geometry.AttributeCollection, name: str, value: str) -> pyuipc.geometry.AttributeSlotString

        Create a string attribute.
        Args:
            name: Attribute name.
            value: String value.
        Returns:
            AttributeSlot: Created attribute slot.
        """
    @overload
    def create(self, name: str, value: float) -> IAttributeSlot:
        """create(*args, **kwargs)
        Overloaded function.

        1. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create an I32 (32-bit integer) attribute.
        Args:
            name: Attribute name.
            value: I32 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        2. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create an I64 (64-bit integer) attribute.
        Args:
            name: Attribute name.
            value: I64 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        3. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create a U64 (64-bit unsigned integer) attribute.
        Args:
            name: Attribute name.
            value: U64 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        4. create(self: pyuipc.geometry.AttributeCollection, name: str, value: float) -> pyuipc.geometry.IAttributeSlot

        Create a Float (floating-point) attribute.
        Args:
            name: Attribute name.
            value: Float scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        5. create(self: pyuipc.geometry.AttributeCollection, name: str, value: numpy.typing.NDArray[numpy.float64]) -> pyuipc.geometry.IAttributeSlot

        Create an attribute from a numpy array (auto-detects type: scalar, vector, or matrix).
        Args:
            name: Attribute name.
            value: Numpy array (can be scalar, vector of size 2/3/4/6/9/12, or matrix of size 2x2/3x3/4x4/6x6/9x9/12x12).
        Returns:
            AttributeSlot: Created attribute slot.

        6. create(self: pyuipc.geometry.AttributeCollection, name: str, value: numpy.typing.NDArray[numpy.int32]) -> pyuipc.geometry.IAttributeSlot

        Create an integer attribute from a numpy array (auto-detects type: scalar or integer vector).
        Args:
            name: Attribute name.
            value: Numpy array (can be scalar or integer vector of size 2/3/4).
        Returns:
            AttributeSlot: Created attribute slot.

        7. create(self: pyuipc.geometry.AttributeCollection, name: str, value: str) -> pyuipc.geometry.AttributeSlotString

        Create a string attribute.
        Args:
            name: Attribute name.
            value: String value.
        Returns:
            AttributeSlot: Created attribute slot.
        """
    @overload
    def create(self, name: str, value: numpy.typing.NDArray[numpy.float64]) -> IAttributeSlot:
        """create(*args, **kwargs)
        Overloaded function.

        1. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create an I32 (32-bit integer) attribute.
        Args:
            name: Attribute name.
            value: I32 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        2. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create an I64 (64-bit integer) attribute.
        Args:
            name: Attribute name.
            value: I64 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        3. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create a U64 (64-bit unsigned integer) attribute.
        Args:
            name: Attribute name.
            value: U64 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        4. create(self: pyuipc.geometry.AttributeCollection, name: str, value: float) -> pyuipc.geometry.IAttributeSlot

        Create a Float (floating-point) attribute.
        Args:
            name: Attribute name.
            value: Float scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        5. create(self: pyuipc.geometry.AttributeCollection, name: str, value: numpy.typing.NDArray[numpy.float64]) -> pyuipc.geometry.IAttributeSlot

        Create an attribute from a numpy array (auto-detects type: scalar, vector, or matrix).
        Args:
            name: Attribute name.
            value: Numpy array (can be scalar, vector of size 2/3/4/6/9/12, or matrix of size 2x2/3x3/4x4/6x6/9x9/12x12).
        Returns:
            AttributeSlot: Created attribute slot.

        6. create(self: pyuipc.geometry.AttributeCollection, name: str, value: numpy.typing.NDArray[numpy.int32]) -> pyuipc.geometry.IAttributeSlot

        Create an integer attribute from a numpy array (auto-detects type: scalar or integer vector).
        Args:
            name: Attribute name.
            value: Numpy array (can be scalar or integer vector of size 2/3/4).
        Returns:
            AttributeSlot: Created attribute slot.

        7. create(self: pyuipc.geometry.AttributeCollection, name: str, value: str) -> pyuipc.geometry.AttributeSlotString

        Create a string attribute.
        Args:
            name: Attribute name.
            value: String value.
        Returns:
            AttributeSlot: Created attribute slot.
        """
    @overload
    def create(self, name: str, value: numpy.typing.NDArray[numpy.int32]) -> IAttributeSlot:
        """create(*args, **kwargs)
        Overloaded function.

        1. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create an I32 (32-bit integer) attribute.
        Args:
            name: Attribute name.
            value: I32 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        2. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create an I64 (64-bit integer) attribute.
        Args:
            name: Attribute name.
            value: I64 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        3. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create a U64 (64-bit unsigned integer) attribute.
        Args:
            name: Attribute name.
            value: U64 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        4. create(self: pyuipc.geometry.AttributeCollection, name: str, value: float) -> pyuipc.geometry.IAttributeSlot

        Create a Float (floating-point) attribute.
        Args:
            name: Attribute name.
            value: Float scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        5. create(self: pyuipc.geometry.AttributeCollection, name: str, value: numpy.typing.NDArray[numpy.float64]) -> pyuipc.geometry.IAttributeSlot

        Create an attribute from a numpy array (auto-detects type: scalar, vector, or matrix).
        Args:
            name: Attribute name.
            value: Numpy array (can be scalar, vector of size 2/3/4/6/9/12, or matrix of size 2x2/3x3/4x4/6x6/9x9/12x12).
        Returns:
            AttributeSlot: Created attribute slot.

        6. create(self: pyuipc.geometry.AttributeCollection, name: str, value: numpy.typing.NDArray[numpy.int32]) -> pyuipc.geometry.IAttributeSlot

        Create an integer attribute from a numpy array (auto-detects type: scalar or integer vector).
        Args:
            name: Attribute name.
            value: Numpy array (can be scalar or integer vector of size 2/3/4).
        Returns:
            AttributeSlot: Created attribute slot.

        7. create(self: pyuipc.geometry.AttributeCollection, name: str, value: str) -> pyuipc.geometry.AttributeSlotString

        Create a string attribute.
        Args:
            name: Attribute name.
            value: String value.
        Returns:
            AttributeSlot: Created attribute slot.
        """
    @overload
    def create(self, name: str, value: str) -> AttributeSlotString:
        """create(*args, **kwargs)
        Overloaded function.

        1. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create an I32 (32-bit integer) attribute.
        Args:
            name: Attribute name.
            value: I32 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        2. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create an I64 (64-bit integer) attribute.
        Args:
            name: Attribute name.
            value: I64 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        3. create(self: pyuipc.geometry.AttributeCollection, name: str, value: int) -> pyuipc.geometry.IAttributeSlot

        Create a U64 (64-bit unsigned integer) attribute.
        Args:
            name: Attribute name.
            value: U64 scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        4. create(self: pyuipc.geometry.AttributeCollection, name: str, value: float) -> pyuipc.geometry.IAttributeSlot

        Create a Float (floating-point) attribute.
        Args:
            name: Attribute name.
            value: Float scalar value.
        Returns:
            AttributeSlot: Created attribute slot.

        5. create(self: pyuipc.geometry.AttributeCollection, name: str, value: numpy.typing.NDArray[numpy.float64]) -> pyuipc.geometry.IAttributeSlot

        Create an attribute from a numpy array (auto-detects type: scalar, vector, or matrix).
        Args:
            name: Attribute name.
            value: Numpy array (can be scalar, vector of size 2/3/4/6/9/12, or matrix of size 2x2/3x3/4x4/6x6/9x9/12x12).
        Returns:
            AttributeSlot: Created attribute slot.

        6. create(self: pyuipc.geometry.AttributeCollection, name: str, value: numpy.typing.NDArray[numpy.int32]) -> pyuipc.geometry.IAttributeSlot

        Create an integer attribute from a numpy array (auto-detects type: scalar or integer vector).
        Args:
            name: Attribute name.
            value: Numpy array (can be scalar or integer vector of size 2/3/4).
        Returns:
            AttributeSlot: Created attribute slot.

        7. create(self: pyuipc.geometry.AttributeCollection, name: str, value: str) -> pyuipc.geometry.AttributeSlotString

        Create a string attribute.
        Args:
            name: Attribute name.
            value: String value.
        Returns:
            AttributeSlot: Created attribute slot.
        """
    def destroy(self, name: str) -> None:
        """destroy(self: pyuipc.geometry.AttributeCollection, name: str) -> None

        Destroy an attribute by name.
        Args:
            name: Attribute name to destroy.
        """
    def find(self, name: str) -> IAttributeSlot:
        """find(self: pyuipc.geometry.AttributeCollection, name: str) -> pyuipc.geometry.IAttributeSlot

        Find an attribute by name.
        Args:
            name: Attribute name.
        Returns:
            AttributeSlot or None: Attribute slot if found, None otherwise.
        """
    def reorder(self, indices: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64]) -> None:
        """reorder(self: pyuipc.geometry.AttributeCollection, indices: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64]) -> None

        Reorder attributes according to the given indices.
        Args:
            indices: Array of new indices for reordering.
        """
    def reserve(self, size: typing.SupportsInt) -> None:
        """reserve(self: pyuipc.geometry.AttributeCollection, size: typing.SupportsInt) -> None

        Reserve capacity for attributes.
        Args:
            size: Capacity to reserve.
        """
    def resize(self, size: typing.SupportsInt) -> None:
        """resize(self: pyuipc.geometry.AttributeCollection, size: typing.SupportsInt) -> None

        Resize all attributes to the specified size.
        Args:
            size: New size for all attributes.
        """
    def share(self, name: str, slot: IAttributeSlot) -> None:
        """share(self: pyuipc.geometry.AttributeCollection, name: str, slot: pyuipc.geometry.IAttributeSlot) -> None

        Share an existing attribute slot with a new name.
        Args:
            name: New name for the shared attribute.
            slot: Attribute slot to share.
        """
    def size(self) -> int:
        """size(self: pyuipc.geometry.AttributeCollection) -> int

        Get the size of attributes.
        Returns:
            int: Size of attributes.
        """

class AttributeIO:
    def __init__(self, file: str) -> None:
        """__init__(self: pyuipc.geometry.AttributeIO, file: str) -> None

        Create an AttributeIO instance.
        Args:
            file: File path to read from.
        """
    def read(self, name: str, slot: IAttributeSlot) -> None:
        """read(self: pyuipc.geometry.AttributeIO, name: str, slot: pyuipc.geometry.IAttributeSlot) -> None

        Read an attribute from the file.
        Args:
            name: Attribute name to read.
            slot: AttributeSlot to store the read data.
        """

class AttributeSlotFloat(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.float64]:
        """view(self: pyuipc.geometry.AttributeSlotFloat) -> numpy.typing.NDArray[numpy.float64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotI32(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.int32]:
        """view(self: pyuipc.geometry.AttributeSlotI32) -> numpy.typing.NDArray[numpy.int32]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotI64(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.int64]:
        """view(self: pyuipc.geometry.AttributeSlotI64) -> numpy.typing.NDArray[numpy.int64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotMatrix12x12(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.float64]:
        """view(self: pyuipc.geometry.AttributeSlotMatrix12x12) -> numpy.typing.NDArray[numpy.float64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotMatrix2x2(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.float64]:
        """view(self: pyuipc.geometry.AttributeSlotMatrix2x2) -> numpy.typing.NDArray[numpy.float64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotMatrix3x3(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.float64]:
        """view(self: pyuipc.geometry.AttributeSlotMatrix3x3) -> numpy.typing.NDArray[numpy.float64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotMatrix4x4(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.float64]:
        """view(self: pyuipc.geometry.AttributeSlotMatrix4x4) -> numpy.typing.NDArray[numpy.float64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotMatrix6x6(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.float64]:
        """view(self: pyuipc.geometry.AttributeSlotMatrix6x6) -> numpy.typing.NDArray[numpy.float64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotMatrix9x9(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.float64]:
        """view(self: pyuipc.geometry.AttributeSlotMatrix9x9) -> numpy.typing.NDArray[numpy.float64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotString(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> CStringSpan:
        """view(self: pyuipc.geometry.AttributeSlotString) -> pyuipc.geometry.CStringSpan

        Get a view of the string attribute data.
        Returns:
            StringSpan: View of string attribute data.
        """

class AttributeSlotU32(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.uint32]:
        """view(self: pyuipc.geometry.AttributeSlotU32) -> numpy.typing.NDArray[numpy.uint32]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotU64(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.uint64]:
        """view(self: pyuipc.geometry.AttributeSlotU64) -> numpy.typing.NDArray[numpy.uint64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotVector12(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.float64]:
        """view(self: pyuipc.geometry.AttributeSlotVector12) -> numpy.typing.NDArray[numpy.float64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotVector2(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.float64]:
        """view(self: pyuipc.geometry.AttributeSlotVector2) -> numpy.typing.NDArray[numpy.float64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotVector2i(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.int32]:
        """view(self: pyuipc.geometry.AttributeSlotVector2i) -> numpy.typing.NDArray[numpy.int32]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotVector3(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.float64]:
        """view(self: pyuipc.geometry.AttributeSlotVector3) -> numpy.typing.NDArray[numpy.float64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotVector3i(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.int32]:
        """view(self: pyuipc.geometry.AttributeSlotVector3i) -> numpy.typing.NDArray[numpy.int32]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotVector4(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.float64]:
        """view(self: pyuipc.geometry.AttributeSlotVector4) -> numpy.typing.NDArray[numpy.float64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotVector4i(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.int32]:
        """view(self: pyuipc.geometry.AttributeSlotVector4i) -> numpy.typing.NDArray[numpy.int32]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotVector6(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.float64]:
        """view(self: pyuipc.geometry.AttributeSlotVector6) -> numpy.typing.NDArray[numpy.float64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class AttributeSlotVector9(IAttributeSlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def view(self) -> numpy.typing.NDArray[numpy.float64]:
        """view(self: pyuipc.geometry.AttributeSlotVector9) -> numpy.typing.NDArray[numpy.float64]

        Get a numpy array view of the attribute data.
        Returns:
            numpy.ndarray: Array view of attribute data.
        """

class CStringSpan:
    def __init__(self) -> None:
        """__init__(self: pyuipc.geometry.CStringSpan) -> None"""
    def __getitem__(self, arg0: typing.SupportsInt) -> str:
        """__getitem__(self: pyuipc.geometry.CStringSpan, arg0: typing.SupportsInt) -> str"""
    def __iter__(self) -> collections.abc.Iterator[str]:
        """__iter__(self: pyuipc.geometry.CStringSpan) -> collections.abc.Iterator[str]"""
    def __len__(self) -> int:
        """__len__(self: pyuipc.geometry.CStringSpan) -> int"""

class Geometry(IGeometry):
    class InstanceAttributes:
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def clear(self) -> None:
            """clear(self: pyuipc.geometry.Geometry.InstanceAttributes) -> None

            Clear all instance attributes.
            """
        def create(self, name: str, object: object) -> IAttributeSlot:
            """create(self: pyuipc.geometry.Geometry.InstanceAttributes, name: str, object: object) -> pyuipc.geometry.IAttributeSlot

            Create a new instance attribute from a Python object.
            Args:
                name: Attribute name.
                object: Python object to create attribute from.
            Returns:
                AttributeSlot: Created attribute slot.
            """
        def destroy(self, name: str) -> None:
            """destroy(self: pyuipc.geometry.Geometry.InstanceAttributes, name: str) -> None

            Destroy an instance attribute by name.
            Args:
                name: Attribute name to destroy.
            """
        def find(self, name: str) -> IAttributeSlot:
            """find(self: pyuipc.geometry.Geometry.InstanceAttributes, name: str) -> pyuipc.geometry.IAttributeSlot

            Find an instance attribute by name.
            Args:
                name: Attribute name.
            Returns:
                AttributeSlot or None: Attribute slot if found, None otherwise.
            """
        def reserve(self, size: typing.SupportsInt) -> None:
            """reserve(self: pyuipc.geometry.Geometry.InstanceAttributes, size: typing.SupportsInt) -> None

            Reserve capacity for instance attributes.
            Args:
                size: Capacity to reserve.
            """
        def resize(self, size: typing.SupportsInt) -> None:
            """resize(self: pyuipc.geometry.Geometry.InstanceAttributes, size: typing.SupportsInt) -> None

            Resize the instance attributes to the specified size.
            Args:
                size: New size for instance attributes.
            """
        def share(self, name: str, attribute: IAttributeSlot) -> None:
            """share(self: pyuipc.geometry.Geometry.InstanceAttributes, name: str, attribute: pyuipc.geometry.IAttributeSlot) -> None

            Share an existing attribute slot with a new name.
            Args:
                name: New name for the shared attribute.
                attribute: Attribute slot to share.
            """
        def size(self) -> int:
            """size(self: pyuipc.geometry.Geometry.InstanceAttributes) -> int

            Get the number of instances.
            Returns:
                int: Number of instances.
            """
        def to_json(self) -> json:
            """to_json(self: pyuipc.geometry.Geometry.InstanceAttributes) -> json

            Convert instance attributes to JSON.
            Returns:
                dict: JSON representation of instance attributes.
            """

    class MetaAttributes:
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def create(self, name: str, object: object) -> IAttributeSlot:
            """create(self: pyuipc.geometry.Geometry.MetaAttributes, name: str, object: object) -> pyuipc.geometry.IAttributeSlot

            Create a new meta attribute from a Python object.
            Args:
                name: Attribute name.
                object: Python object to create attribute from.
            Returns:
                AttributeSlot: Created attribute slot.
            """
        def destroy(self, name: str) -> None:
            """destroy(self: pyuipc.geometry.Geometry.MetaAttributes, name: str) -> None

            Destroy a meta attribute by name.
            Args:
                name: Attribute name to destroy.
            """
        def find(self, name: str) -> IAttributeSlot:
            """find(self: pyuipc.geometry.Geometry.MetaAttributes, name: str) -> pyuipc.geometry.IAttributeSlot

            Find a meta attribute by name.
            Args:
                name: Attribute name.
            Returns:
                AttributeSlot or None: Attribute slot if found, None otherwise.
            """
        def share(self, name: str, attribute: IAttributeSlot) -> None:
            """share(self: pyuipc.geometry.Geometry.MetaAttributes, name: str, attribute: pyuipc.geometry.IAttributeSlot) -> None

            Share an existing attribute slot with a new name.
            Args:
                name: New name for the shared attribute.
                attribute: Attribute slot to share.
            """
        def to_json(self) -> json:
            """to_json(self: pyuipc.geometry.Geometry.MetaAttributes) -> json

            Convert meta attributes to JSON.
            Returns:
                dict: JSON representation of meta attributes.
            """
    def __init__(self) -> None:
        """__init__(self: pyuipc.geometry.Geometry) -> None

        Create an empty geometry.
        """
    def instances(self) -> Geometry.InstanceAttributes:
        """instances(self: pyuipc.geometry.Geometry) -> pyuipc.geometry.Geometry.InstanceAttributes

        Get the instance attributes.
        Returns:
            InstanceAttributes: Instance attributes of the geometry.
        """
    def meta(self) -> Geometry.MetaAttributes:
        """meta(self: pyuipc.geometry.Geometry) -> pyuipc.geometry.Geometry.MetaAttributes

        Get the meta attributes.
        Returns:
            MetaAttributes: Meta attributes of the geometry.
        """
    def __getitem__(self, name: str) -> AttributeCollection:
        """__getitem__(self: pyuipc.geometry.Geometry, name: str) -> pyuipc.geometry.AttributeCollection

        Get an attribute collection by name.
        Returns:
            AttributeCollection: Attribute collection with the given name, if not found, create a new one.
        """

class GeometryAtlas:
    def __init__(self) -> None:
        """__init__(self: pyuipc.geometry.GeometryAtlas) -> None

        Create an empty geometry atlas.
        """
    def attribute_collection_count(self) -> int:
        """attribute_collection_count(self: pyuipc.geometry.GeometryAtlas) -> int

        Get the number of attribute collections.
        Returns:
            int: Number of attribute collections.
        """
    def attribute_collection_names(self) -> list:
        """attribute_collection_names(self: pyuipc.geometry.GeometryAtlas) -> list

        Get the names of all attribute collections.
        Returns:
            list: List of attribute collection names.
        """
    @overload
    def create(self, geo: Geometry) -> int:
        """create(*args, **kwargs)
        Overloaded function.

        1. create(self: pyuipc.geometry.GeometryAtlas, geo: pyuipc.geometry.Geometry) -> int

        Create a geometry in the atlas.
        Args:
            geo: Geometry to add.
        Returns:
            GeometrySlot: Slot containing the created geometry.

        2. create(self: pyuipc.geometry.GeometryAtlas, name: str, ac: pyuipc.geometry.AttributeCollection) -> None

        Create a geometry from an attribute collection.
        Args:
            name: Name for the geometry.
            ac: AttributeCollection to create geometry from.
        Returns:
            GeometrySlot: Slot containing the created geometry.
        """
    @overload
    def create(self, name: str, ac: AttributeCollection) -> None:
        """create(*args, **kwargs)
        Overloaded function.

        1. create(self: pyuipc.geometry.GeometryAtlas, geo: pyuipc.geometry.Geometry) -> int

        Create a geometry in the atlas.
        Args:
            geo: Geometry to add.
        Returns:
            GeometrySlot: Slot containing the created geometry.

        2. create(self: pyuipc.geometry.GeometryAtlas, name: str, ac: pyuipc.geometry.AttributeCollection) -> None

        Create a geometry from an attribute collection.
        Args:
            name: Name for the geometry.
            ac: AttributeCollection to create geometry from.
        Returns:
            GeometrySlot: Slot containing the created geometry.
        """
    def from_json(self, json: json) -> None:
        """from_json(self: pyuipc.geometry.GeometryAtlas, json: json) -> None

        Load atlas from JSON representation.
        Args:
            json: JSON dictionary to load from.
        """
    def geometry_count(self) -> int:
        """geometry_count(self: pyuipc.geometry.GeometryAtlas) -> int

        Get the number of geometries in the atlas.
        Returns:
            int: Number of geometries.
        """
    def to_json(self) -> json:
        """to_json(self: pyuipc.geometry.GeometryAtlas) -> json

        Convert atlas to JSON representation.
        Returns:
            dict: JSON representation of the atlas.
        """

class GeometrySlot:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def geometry(self) -> Geometry:
        """geometry(self: pyuipc.geometry.GeometrySlot) -> pyuipc.geometry.Geometry

        Get the geometry in this slot.
        Returns:
            Geometry: Reference to the geometry.
        """
    def id(self) -> int:
        """id(self: pyuipc.geometry.GeometrySlot) -> int

        Get the geometry slot ID.
        Returns:
            int: Geometry slot ID.
        """

class IAttributeSlot:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def allow_destroy(self) -> bool:
        """allow_destroy(self: pyuipc.geometry.IAttributeSlot) -> bool

        Check if the attribute can be destroyed.
        Returns:
            bool: True if destroyable, False otherwise.
        """
    def is_shared(self) -> bool:
        """is_shared(self: pyuipc.geometry.IAttributeSlot) -> bool

        Check if the attribute is shared.
        Returns:
            bool: True if shared, False otherwise.
        """
    def name(self) -> str:
        """name(self: pyuipc.geometry.IAttributeSlot) -> str

        Get the attribute name.
        Returns:
            str: Attribute name.
        """
    def size(self) -> int:
        """size(self: pyuipc.geometry.IAttributeSlot) -> int

        Get the size of the attribute.
        Returns:
            int: Number of elements.
        """
    def type_name(self) -> str:
        """type_name(self: pyuipc.geometry.IAttributeSlot) -> str

        Get the attribute type name.
        Returns:
            str: Type name.
        """
    def view(self) -> numpy.ndarray:
        """view(self: pyuipc.geometry.IAttributeSlot) -> numpy.ndarray

        Get a view of the attribute data (virtual method, returns None for base class).
        Returns:
            numpy.ndarray or None: Array view if available, None otherwise.
        """

class IGeometry:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def clone(self) -> IGeometry:
        """clone(self: pyuipc.geometry.IGeometry) -> pyuipc.geometry.IGeometry

        Create a deep copy of the geometry.
        Returns:
            IGeometry: Cloned geometry.
        """
    def to_json(self) -> json:
        """to_json(self: pyuipc.geometry.IGeometry) -> json

        Convert geometry to JSON representation.
        Returns:
            dict: JSON representation of the geometry.
        """
    def type(self) -> str:
        """type(self: pyuipc.geometry.IGeometry) -> str

        Get the geometry type name.
        Returns:
            str: Geometry type name.
        """

class ImplicitGeometry(Geometry):
    def __init__(self) -> None:
        """__init__(self: pyuipc.geometry.ImplicitGeometry) -> None

        Create an empty implicit geometry.
        """

class ImplicitGeometrySlot(GeometrySlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def geometry(self) -> ImplicitGeometry:
        """geometry(self: pyuipc.geometry.ImplicitGeometrySlot) -> pyuipc.geometry.ImplicitGeometry

        Get the implicit geometry in this slot.
        Returns:
            ImplicitGeometry: Reference to the implicit geometry.
        """

class SimplicialComplex(Geometry):
    class EdgeAttributes:
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def clear(self) -> None:
            """clear(self: pyuipc.geometry.SimplicialComplex.EdgeAttributes) -> None

            Clear all attributes.
            """
        def create(self, name: str, object: object) -> IAttributeSlot:
            """create(self: pyuipc.geometry.SimplicialComplex.EdgeAttributes, name: str, object: object) -> pyuipc.geometry.IAttributeSlot

            Create a new attribute from a Python object.
            Args:
                name: Attribute name.
                object: Python object to create attribute from.
            Returns:
                AttributeSlot: Created attribute slot.
            """
        def destroy(self, name: str) -> None:
            """destroy(self: pyuipc.geometry.SimplicialComplex.EdgeAttributes, name: str) -> None

            Destroy an attribute by name.
            Args:
                name: Attribute name to destroy.
            """
        def find(self, name: str) -> IAttributeSlot:
            """find(self: pyuipc.geometry.SimplicialComplex.EdgeAttributes, name: str) -> pyuipc.geometry.IAttributeSlot

            Find an attribute by name.
            Args:
                name: Attribute name.
            Returns:
                AttributeSlot or None: Attribute slot if found, None otherwise.
            """
        def reserve(self, size: typing.SupportsInt) -> None:
            """reserve(self: pyuipc.geometry.SimplicialComplex.EdgeAttributes, size: typing.SupportsInt) -> None

            Reserve capacity for attributes.
            Args:
                size: Capacity to reserve.
            """
        def resize(self, size: typing.SupportsInt) -> None:
            """resize(self: pyuipc.geometry.SimplicialComplex.EdgeAttributes, size: typing.SupportsInt) -> None

            Resize the attributes to the specified size.
            Args:
                size: New size.
            """
        def share(self, name: str, attribute: IAttributeSlot) -> None:
            """share(self: pyuipc.geometry.SimplicialComplex.EdgeAttributes, name: str, attribute: pyuipc.geometry.IAttributeSlot) -> None

            Share an existing attribute slot with a new name.
            Args:
                name: New name for the shared attribute.
                attribute: Attribute slot to share.
            """
        def size(self) -> int:
            """size(self: pyuipc.geometry.SimplicialComplex.EdgeAttributes) -> int

            Get the number of simplices.
            Returns:
                int: Number of simplices.
            """
        def to_json(self) -> json:
            """to_json(self: pyuipc.geometry.SimplicialComplex.EdgeAttributes) -> json

            Convert attributes to JSON representation.
            Returns:
                dict: JSON representation of the attributes.
            """
        def topo(self) -> AttributeSlotVector2i:
            """topo(self: pyuipc.geometry.SimplicialComplex.EdgeAttributes) -> pyuipc.geometry.AttributeSlotVector2i

            Get the topology attribute slot.
            Returns:
                AttributeSlot: Reference to topology attribute slot.
            """

    class TetrahedronAttributes:
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def clear(self) -> None:
            """clear(self: pyuipc.geometry.SimplicialComplex.TetrahedronAttributes) -> None

            Clear all attributes.
            """
        def create(self, name: str, object: object) -> IAttributeSlot:
            """create(self: pyuipc.geometry.SimplicialComplex.TetrahedronAttributes, name: str, object: object) -> pyuipc.geometry.IAttributeSlot

            Create a new attribute from a Python object.
            Args:
                name: Attribute name.
                object: Python object to create attribute from.
            Returns:
                AttributeSlot: Created attribute slot.
            """
        def destroy(self, name: str) -> None:
            """destroy(self: pyuipc.geometry.SimplicialComplex.TetrahedronAttributes, name: str) -> None

            Destroy an attribute by name.
            Args:
                name: Attribute name to destroy.
            """
        def find(self, name: str) -> IAttributeSlot:
            """find(self: pyuipc.geometry.SimplicialComplex.TetrahedronAttributes, name: str) -> pyuipc.geometry.IAttributeSlot

            Find an attribute by name.
            Args:
                name: Attribute name.
            Returns:
                AttributeSlot or None: Attribute slot if found, None otherwise.
            """
        def reserve(self, size: typing.SupportsInt) -> None:
            """reserve(self: pyuipc.geometry.SimplicialComplex.TetrahedronAttributes, size: typing.SupportsInt) -> None

            Reserve capacity for attributes.
            Args:
                size: Capacity to reserve.
            """
        def resize(self, size: typing.SupportsInt) -> None:
            """resize(self: pyuipc.geometry.SimplicialComplex.TetrahedronAttributes, size: typing.SupportsInt) -> None

            Resize the attributes to the specified size.
            Args:
                size: New size.
            """
        def share(self, name: str, attribute: IAttributeSlot) -> None:
            """share(self: pyuipc.geometry.SimplicialComplex.TetrahedronAttributes, name: str, attribute: pyuipc.geometry.IAttributeSlot) -> None

            Share an existing attribute slot with a new name.
            Args:
                name: New name for the shared attribute.
                attribute: Attribute slot to share.
            """
        def size(self) -> int:
            """size(self: pyuipc.geometry.SimplicialComplex.TetrahedronAttributes) -> int

            Get the number of simplices.
            Returns:
                int: Number of simplices.
            """
        def to_json(self) -> json:
            """to_json(self: pyuipc.geometry.SimplicialComplex.TetrahedronAttributes) -> json

            Convert attributes to JSON representation.
            Returns:
                dict: JSON representation of the attributes.
            """
        def topo(self) -> AttributeSlotVector4i:
            """topo(self: pyuipc.geometry.SimplicialComplex.TetrahedronAttributes) -> pyuipc.geometry.AttributeSlotVector4i

            Get the topology attribute slot.
            Returns:
                AttributeSlot: Reference to topology attribute slot.
            """

    class TriangleAttributes:
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def clear(self) -> None:
            """clear(self: pyuipc.geometry.SimplicialComplex.TriangleAttributes) -> None

            Clear all attributes.
            """
        def create(self, name: str, object: object) -> IAttributeSlot:
            """create(self: pyuipc.geometry.SimplicialComplex.TriangleAttributes, name: str, object: object) -> pyuipc.geometry.IAttributeSlot

            Create a new attribute from a Python object.
            Args:
                name: Attribute name.
                object: Python object to create attribute from.
            Returns:
                AttributeSlot: Created attribute slot.
            """
        def destroy(self, name: str) -> None:
            """destroy(self: pyuipc.geometry.SimplicialComplex.TriangleAttributes, name: str) -> None

            Destroy an attribute by name.
            Args:
                name: Attribute name to destroy.
            """
        def find(self, name: str) -> IAttributeSlot:
            """find(self: pyuipc.geometry.SimplicialComplex.TriangleAttributes, name: str) -> pyuipc.geometry.IAttributeSlot

            Find an attribute by name.
            Args:
                name: Attribute name.
            Returns:
                AttributeSlot or None: Attribute slot if found, None otherwise.
            """
        def reserve(self, size: typing.SupportsInt) -> None:
            """reserve(self: pyuipc.geometry.SimplicialComplex.TriangleAttributes, size: typing.SupportsInt) -> None

            Reserve capacity for attributes.
            Args:
                size: Capacity to reserve.
            """
        def resize(self, size: typing.SupportsInt) -> None:
            """resize(self: pyuipc.geometry.SimplicialComplex.TriangleAttributes, size: typing.SupportsInt) -> None

            Resize the attributes to the specified size.
            Args:
                size: New size.
            """
        def share(self, name: str, attribute: IAttributeSlot) -> None:
            """share(self: pyuipc.geometry.SimplicialComplex.TriangleAttributes, name: str, attribute: pyuipc.geometry.IAttributeSlot) -> None

            Share an existing attribute slot with a new name.
            Args:
                name: New name for the shared attribute.
                attribute: Attribute slot to share.
            """
        def size(self) -> int:
            """size(self: pyuipc.geometry.SimplicialComplex.TriangleAttributes) -> int

            Get the number of simplices.
            Returns:
                int: Number of simplices.
            """
        def to_json(self) -> json:
            """to_json(self: pyuipc.geometry.SimplicialComplex.TriangleAttributes) -> json

            Convert attributes to JSON representation.
            Returns:
                dict: JSON representation of the attributes.
            """
        def topo(self) -> AttributeSlotVector3i:
            """topo(self: pyuipc.geometry.SimplicialComplex.TriangleAttributes) -> pyuipc.geometry.AttributeSlotVector3i

            Get the topology attribute slot.
            Returns:
                AttributeSlot: Reference to topology attribute slot.
            """

    class VertexAttributes:
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def clear(self) -> None:
            """clear(self: pyuipc.geometry.SimplicialComplex.VertexAttributes) -> None

            Clear all attributes.
            """
        def create(self, name: str, object: object) -> IAttributeSlot:
            """create(self: pyuipc.geometry.SimplicialComplex.VertexAttributes, name: str, object: object) -> pyuipc.geometry.IAttributeSlot

            Create a new attribute from a Python object.
            Args:
                name: Attribute name.
                object: Python object to create attribute from.
            Returns:
                AttributeSlot: Created attribute slot.
            """
        def destroy(self, name: str) -> None:
            """destroy(self: pyuipc.geometry.SimplicialComplex.VertexAttributes, name: str) -> None

            Destroy an attribute by name.
            Args:
                name: Attribute name to destroy.
            """
        def find(self, name: str) -> IAttributeSlot:
            """find(self: pyuipc.geometry.SimplicialComplex.VertexAttributes, name: str) -> pyuipc.geometry.IAttributeSlot

            Find an attribute by name.
            Args:
                name: Attribute name.
            Returns:
                AttributeSlot or None: Attribute slot if found, None otherwise.
            """
        def reserve(self, size: typing.SupportsInt) -> None:
            """reserve(self: pyuipc.geometry.SimplicialComplex.VertexAttributes, size: typing.SupportsInt) -> None

            Reserve capacity for attributes.
            Args:
                size: Capacity to reserve.
            """
        def resize(self, size: typing.SupportsInt) -> None:
            """resize(self: pyuipc.geometry.SimplicialComplex.VertexAttributes, size: typing.SupportsInt) -> None

            Resize the attributes to the specified size.
            Args:
                size: New size.
            """
        def share(self, name: str, attribute: IAttributeSlot) -> None:
            """share(self: pyuipc.geometry.SimplicialComplex.VertexAttributes, name: str, attribute: pyuipc.geometry.IAttributeSlot) -> None

            Share an existing attribute slot with a new name.
            Args:
                name: New name for the shared attribute.
                attribute: Attribute slot to share.
            """
        def size(self) -> int:
            """size(self: pyuipc.geometry.SimplicialComplex.VertexAttributes) -> int

            Get the number of simplices.
            Returns:
                int: Number of simplices.
            """
        def to_json(self) -> json:
            """to_json(self: pyuipc.geometry.SimplicialComplex.VertexAttributes) -> json

            Convert attributes to JSON representation.
            Returns:
                dict: JSON representation of the attributes.
            """
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def copy(self) -> SimplicialComplex:
        """copy(self: pyuipc.geometry.SimplicialComplex) -> pyuipc.geometry.SimplicialComplex

        Create a copy of the simplicial complex.
        Returns:
            SimplicialComplex: Copy of the simplicial complex.
        """
    def dim(self) -> int:
        """dim(self: pyuipc.geometry.SimplicialComplex) -> int

        Get the dimension of the simplicial complex.
        Returns:
            int: Dimension (0=points, 1=edges, 2=triangles, 3=tetrahedra).
        """
    def edges(self) -> SimplicialComplex.EdgeAttributes:
        """edges(self: pyuipc.geometry.SimplicialComplex) -> pyuipc.geometry.SimplicialComplex.EdgeAttributes

        Get the edge attributes.
        Returns:
            EdgeAttributes: Edge attributes collection.
        """
    def positions(self) -> AttributeSlotVector3:
        """positions(self: pyuipc.geometry.SimplicialComplex) -> pyuipc.geometry.AttributeSlotVector3

        Get the position attribute slot (3D vertex positions).
        Returns:
            AttributeSlot: Reference to position attribute slot.
        """
    def tetrahedra(self) -> SimplicialComplex.TetrahedronAttributes:
        """tetrahedra(self: pyuipc.geometry.SimplicialComplex) -> pyuipc.geometry.SimplicialComplex.TetrahedronAttributes

        Get the tetrahedron attributes.
        Returns:
            TetrahedronAttributes: Tetrahedron attributes collection.
        """
    def transforms(self) -> AttributeSlotMatrix4x4:
        """transforms(self: pyuipc.geometry.SimplicialComplex) -> pyuipc.geometry.AttributeSlotMatrix4x4

        Get the transform attribute slot (4x4 transformation matrices).
        Returns:
            AttributeSlot: Reference to transform attribute slot.
        """
    def triangles(self) -> SimplicialComplex.TriangleAttributes:
        """triangles(self: pyuipc.geometry.SimplicialComplex) -> pyuipc.geometry.SimplicialComplex.TriangleAttributes

        Get the triangle attributes.
        Returns:
            TriangleAttributes: Triangle attributes collection.
        """
    def vertices(self) -> SimplicialComplex.VertexAttributes:
        """vertices(self: pyuipc.geometry.SimplicialComplex) -> pyuipc.geometry.SimplicialComplex.VertexAttributes

        Get the vertex attributes.
        Returns:
            VertexAttributes: Vertex attributes collection.
        """

class SimplicialComplexIO:
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: pyuipc.geometry.SimplicialComplexIO) -> None

        Create a SimplicialComplexIO instance without pre-transform.

        2. __init__(self: pyuipc.geometry.SimplicialComplexIO, pre_transform: pyuipc.Transform) -> None

        Create a SimplicialComplexIO instance with a pre-transform.
        Args:
            pre_transform: Transform to apply before reading/writing.

        3. __init__(self: pyuipc.geometry.SimplicialComplexIO, pre_transform: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None

        Create a SimplicialComplexIO instance with a pre-transform matrix.
        Args:
            pre_transform: 4x4 transformation matrix.
        """
    @overload
    def __init__(self, pre_transform: pyuipc.Transform) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: pyuipc.geometry.SimplicialComplexIO) -> None

        Create a SimplicialComplexIO instance without pre-transform.

        2. __init__(self: pyuipc.geometry.SimplicialComplexIO, pre_transform: pyuipc.Transform) -> None

        Create a SimplicialComplexIO instance with a pre-transform.
        Args:
            pre_transform: Transform to apply before reading/writing.

        3. __init__(self: pyuipc.geometry.SimplicialComplexIO, pre_transform: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None

        Create a SimplicialComplexIO instance with a pre-transform matrix.
        Args:
            pre_transform: 4x4 transformation matrix.
        """
    @overload
    def __init__(self, pre_transform: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: pyuipc.geometry.SimplicialComplexIO) -> None

        Create a SimplicialComplexIO instance without pre-transform.

        2. __init__(self: pyuipc.geometry.SimplicialComplexIO, pre_transform: pyuipc.Transform) -> None

        Create a SimplicialComplexIO instance with a pre-transform.
        Args:
            pre_transform: Transform to apply before reading/writing.

        3. __init__(self: pyuipc.geometry.SimplicialComplexIO, pre_transform: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None

        Create a SimplicialComplexIO instance with a pre-transform matrix.
        Args:
            pre_transform: 4x4 transformation matrix.
        """
    def read(self, filename: str) -> SimplicialComplex:
        """read(self: pyuipc.geometry.SimplicialComplexIO, filename: str) -> pyuipc.geometry.SimplicialComplex

        Read a simplicial complex from a file.
        Args:
            filename: Input file path.
        Returns:
            SimplicialComplex: Loaded simplicial complex.
        """
    def write(self, sc: str, filename: SimplicialComplex) -> None:
        """write(self: pyuipc.geometry.SimplicialComplexIO, sc: str, filename: pyuipc.geometry.SimplicialComplex) -> None

        Write a simplicial complex to a file.
        Args:
            sc: SimplicialComplex to write.
            filename: Output file path.
        """

class SimplicialComplexSlot(GeometrySlot):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def geometry(self) -> SimplicialComplex:
        """geometry(self: pyuipc.geometry.SimplicialComplexSlot) -> pyuipc.geometry.SimplicialComplex

        Get the simplicial complex in this slot.
        Returns:
            SimplicialComplex: Reference to the simplicial complex.
        """

class SpreadSheetIO:
    def __init__(self, output_folder: str = ...) -> None:
        """__init__(self: pyuipc.geometry.SpreadSheetIO, output_folder: str = './') -> None

        Create a SpreadSheetIO instance.
        Args:
            output_folder: Output folder path (default: './').
        """
    @overload
    def write_csv(self, geo_name: str, simplicial_complex: SimplicialComplex) -> None:
        """write_csv(*args, **kwargs)
        Overloaded function.

        1. write_csv(self: pyuipc.geometry.SpreadSheetIO, geo_name: str, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> None

        Write a simplicial complex to CSV format with a geometry name.
        Args:
            geo_name: Name for the geometry.
            simplicial_complex: SimplicialComplex to write.

        2. write_csv(self: pyuipc.geometry.SpreadSheetIO, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> None

        Write a simplicial complex to CSV format.
        Args:
            simplicial_complex: SimplicialComplex to write.
        """
    @overload
    def write_csv(self, simplicial_complex: SimplicialComplex) -> None:
        """write_csv(*args, **kwargs)
        Overloaded function.

        1. write_csv(self: pyuipc.geometry.SpreadSheetIO, geo_name: str, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> None

        Write a simplicial complex to CSV format with a geometry name.
        Args:
            geo_name: Name for the geometry.
            simplicial_complex: SimplicialComplex to write.

        2. write_csv(self: pyuipc.geometry.SpreadSheetIO, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> None

        Write a simplicial complex to CSV format.
        Args:
            simplicial_complex: SimplicialComplex to write.
        """
    @overload
    def write_json(self, geo_name: str, simplicial_complex: SimplicialComplex) -> None:
        """write_json(*args, **kwargs)
        Overloaded function.

        1. write_json(self: pyuipc.geometry.SpreadSheetIO, geo_name: str, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> None

        Write a simplicial complex to JSON format with a geometry name.
        Args:
            geo_name: Name for the geometry.
            simplicial_complex: SimplicialComplex to write.

        2. write_json(self: pyuipc.geometry.SpreadSheetIO, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> None

        Write a simplicial complex to JSON format.
        Args:
            simplicial_complex: SimplicialComplex to write.
        """
    @overload
    def write_json(self, simplicial_complex: SimplicialComplex) -> None:
        """write_json(*args, **kwargs)
        Overloaded function.

        1. write_json(self: pyuipc.geometry.SpreadSheetIO, geo_name: str, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> None

        Write a simplicial complex to JSON format with a geometry name.
        Args:
            geo_name: Name for the geometry.
            simplicial_complex: SimplicialComplex to write.

        2. write_json(self: pyuipc.geometry.SpreadSheetIO, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> None

        Write a simplicial complex to JSON format.
        Args:
            simplicial_complex: SimplicialComplex to write.
        """

class StringSpan:
    def __init__(self) -> None:
        """__init__(self: pyuipc.geometry.StringSpan) -> None"""
    def __getitem__(self, arg0: typing.SupportsInt) -> str:
        """__getitem__(self: pyuipc.geometry.StringSpan, arg0: typing.SupportsInt) -> str"""
    def __iter__(self) -> collections.abc.Iterator[str]:
        """__iter__(self: pyuipc.geometry.StringSpan) -> collections.abc.Iterator[str]"""
    def __len__(self) -> int:
        """__len__(self: pyuipc.geometry.StringSpan) -> int"""
    def __setitem__(self, arg0: typing.SupportsInt, arg1: str) -> None:
        """__setitem__(self: pyuipc.geometry.StringSpan, arg0: typing.SupportsInt, arg1: str) -> None"""

class UrdfController:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def apply_to(self, attr: str) -> None:
        """apply_to(self: pyuipc.geometry.UrdfController, attr: str) -> None

        Apply controller state to attributes.
        Args:
            attr: Attribute collection to apply to.
        """
    def links(self) -> list:
        """links(self: pyuipc.geometry.UrdfController) -> list

        Get the list of link names.
        Returns:
            list: List of link names.
        """
    def move_root(self, xyz: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], rpy: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """move_root(self: pyuipc.geometry.UrdfController, xyz: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], rpy: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None

        Move the root link of the robot.
        Args:
            xyz: Translation vector (3D).
            rpy: Rotation vector (roll, pitch, yaw in radians).
        """
    def revolute_joints(self) -> ImplicitGeometrySlot:
        """revolute_joints(self: pyuipc.geometry.UrdfController) -> pyuipc.geometry.ImplicitGeometrySlot

        Get the list of revolute joint names.
        Returns:
            list: List of revolute joint names.
        """
    def rotate_to(self, joint_name: str, angle: typing.SupportsFloat) -> None:
        """rotate_to(self: pyuipc.geometry.UrdfController, joint_name: str, angle: typing.SupportsFloat) -> None

        Set the rotation angle of a revolute joint.
        Args:
            joint_name: Name of the joint.
            angle: Rotation angle in radians.
        """
    def sync_visual_mesh(self) -> None:
        """sync_visual_mesh(self: pyuipc.geometry.UrdfController) -> None

        Synchronize visual mesh with current joint positions.
        """

class UrdfIO:
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.geometry.UrdfIO, config: json = {'load_visual_mesh': True}) -> None

        Create a UrdfIO instance.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default UrdfIO configuration.
        Returns:
            dict: Default configuration dictionary.
        """
    def read(self, object, urdf_path: str) -> UrdfController:
        """read(self: pyuipc.geometry.UrdfIO, object: uipc::core::Object, urdf_path: str) -> pyuipc.geometry.UrdfController

        Read a URDF file and populate an object.
        Args:
            object: Object to populate with URDF data.
            urdf_path: Path to the URDF file.
        Returns:
            UrdfController: Controller for the loaded URDF model.
        """

def apply_region(sc: SimplicialComplex) -> list:
    """apply_region(sc: pyuipc.geometry.SimplicialComplex) -> list

    Split a simplicial complex by regions.
    Args:
        sc: SimplicialComplex with region labels.
    Returns:
        list: List of SimplicialComplex objects, one per region.
    """
def apply_transform(sc: SimplicialComplex) -> list:
    """apply_transform(sc: pyuipc.geometry.SimplicialComplex) -> list

    Apply transforms to a simplicial complex, splitting into multiple complexes.
    Args:
        sc: SimplicialComplex with transforms.
    Returns:
        list: List of transformed SimplicialComplex objects.
    """
def compute_mesh_d_hat(sc: SimplicialComplex, max_d_hat: typing.SupportsFloat) -> AttributeSlotFloat:
    """compute_mesh_d_hat(sc: pyuipc.geometry.SimplicialComplex, max_d_hat: typing.SupportsFloat) -> pyuipc.geometry.AttributeSlotFloat

    Compute mesh d_hat (characteristic length) parameter.
    Args:
        sc: SimplicialComplex to compute for.
        max_d_hat: Maximum d_hat value.
    Returns:
        float: Computed d_hat value.
    """
def constitution_type(geo: Geometry) -> str:
    """constitution_type(geo: pyuipc.geometry.Geometry) -> str

    Get the constitution type for a geometry.
    Args:
        geo: Geometry to check.
    Returns:
        str: Constitution type name.
    """
@overload
def extract_surface(sc: SimplicialComplex) -> SimplicialComplex:
    """extract_surface(*args, **kwargs)
    Overloaded function.

    1. extract_surface(sc: pyuipc.geometry.SimplicialComplex) -> pyuipc.geometry.SimplicialComplex

    Extract surface from a simplicial complex.
    Args:
        sc: SimplicialComplex to extract surface from.
    Returns:
        SimplicialComplex: Surface mesh.

    2. extract_surface(sc: list) -> pyuipc.geometry.SimplicialComplex

    Extract surface from multiple simplicial complexes.
    Args:
        sc: List of SimplicialComplex objects.
    Returns:
        SimplicialComplex: Combined surface mesh.
    """
@overload
def extract_surface(sc: list) -> SimplicialComplex:
    """extract_surface(*args, **kwargs)
    Overloaded function.

    1. extract_surface(sc: pyuipc.geometry.SimplicialComplex) -> pyuipc.geometry.SimplicialComplex

    Extract surface from a simplicial complex.
    Args:
        sc: SimplicialComplex to extract surface from.
    Returns:
        SimplicialComplex: Surface mesh.

    2. extract_surface(sc: list) -> pyuipc.geometry.SimplicialComplex

    Extract surface from multiple simplicial complexes.
    Args:
        sc: List of SimplicialComplex objects.
    Returns:
        SimplicialComplex: Combined surface mesh.
    """
def facet_closure(sc: SimplicialComplex) -> SimplicialComplex:
    """facet_closure(sc: pyuipc.geometry.SimplicialComplex) -> pyuipc.geometry.SimplicialComplex

    Compute the facet closure of a simplicial complex.
    Args:
        sc: SimplicialComplex to compute closure for.
    Returns:
        SimplicialComplex: Facet closure.
    """
def flip_inward_triangles(sc: SimplicialComplex) -> SimplicialComplex:
    """flip_inward_triangles(sc: pyuipc.geometry.SimplicialComplex) -> pyuipc.geometry.SimplicialComplex

    Flip inward-facing triangles to face outward.
    Args:
        sc: SimplicialComplex to modify.
    """
def ground(height: typing.SupportsFloat = ..., N: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] = ...) -> ImplicitGeometry:
    """ground(height: typing.SupportsFloat = 0.0, N: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] = array([[0.], [1.], [0.]])) -> pyuipc.geometry.ImplicitGeometry

    Create a ground plane implicit geometry.
    Args:
        height: Height of the ground plane (default: 0.0).
        N: Normal vector of the ground plane (default: Y-axis).
    Returns:
        ImplicitGeometry: Ground plane.
    """
def halfplane(P: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] = ..., N: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] = ...) -> ImplicitGeometry:
    """halfplane(P: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] = array([[0.], [0.], [0.]]), N: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] = array([[0.], [1.], [0.]])) -> pyuipc.geometry.ImplicitGeometry

    Create a half-plane implicit geometry.
    Args:
        P: Point on the plane (default: origin).
        N: Normal vector of the plane (default: Y-axis).
    Returns:
        ImplicitGeometry: Half-plane.
    """
def is_trimesh_closed(sc: SimplicialComplex) -> bool:
    """is_trimesh_closed(sc: pyuipc.geometry.SimplicialComplex) -> bool

    Check if a triangular mesh is closed (watertight).
    Args:
        sc: SimplicialComplex to check.
    Returns:
        bool: True if closed, False otherwise.
    """
def label_connected_vertices(sc: SimplicialComplex) -> AttributeSlotI32:
    """label_connected_vertices(sc: pyuipc.geometry.SimplicialComplex) -> pyuipc.geometry.AttributeSlotI32

    Label connected components of vertices.
    Args:
        sc: SimplicialComplex to label.
    """
def label_region(sc: SimplicialComplex) -> None:
    """label_region(sc: pyuipc.geometry.SimplicialComplex) -> None

    Label regions in a simplicial complex.
    Args:
        sc: SimplicialComplex to label.
    """
def label_surface(sc: SimplicialComplex) -> None:
    """label_surface(sc: pyuipc.geometry.SimplicialComplex) -> None

    Label surface elements in a simplicial complex.
    Args:
        sc: SimplicialComplex to label.
    """
def label_triangle_orient(sc: SimplicialComplex) -> AttributeSlotI32:
    """label_triangle_orient(sc: pyuipc.geometry.SimplicialComplex) -> pyuipc.geometry.AttributeSlotI32

    Label triangle orientations in a simplicial complex.
    Args:
        sc: SimplicialComplex to label.
    """
def linemesh(Vs: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], Es: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> SimplicialComplex:
    """linemesh(Vs: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], Es: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> pyuipc.geometry.SimplicialComplex

    Create a line mesh from vertices and edges.
    Args:
        Vs: Array of vertex positions (Nx3).
        Es: Array of edge indices (Mx2).
    Returns:
        SimplicialComplex: Line mesh.
    """
def merge(sc_list: list) -> SimplicialComplex:
    """merge(sc_list: list) -> pyuipc.geometry.SimplicialComplex

    Merge multiple simplicial complexes into one.
    Args:
        sc_list: List of SimplicialComplex objects to merge.
    Returns:
        SimplicialComplex: Merged simplicial complex.
    """
@overload
def optimal_transform(src: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], dst: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """optimal_transform(*args, **kwargs)
    Overloaded function.

    1. optimal_transform(src: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], dst: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

    Compute optimal transform between two point sets.
    Args:
        src: Source points (Nx3 array).
        dst: Destination points (Nx3 array).
    Returns:
        numpy.ndarray: 4x4 transformation matrix.

    2. optimal_transform(src: pyuipc.geometry.SimplicialComplex, dst: pyuipc.geometry.SimplicialComplex) -> numpy.typing.NDArray[numpy.float64]

    Compute optimal transform between two simplicial complexes.
    Args:
        src: Source SimplicialComplex.
        dst: Destination SimplicialComplex.
    Returns:
        numpy.ndarray: 4x4 transformation matrix.
    """
@overload
def optimal_transform(src: SimplicialComplex, dst: SimplicialComplex) -> numpy.typing.NDArray[numpy.float64]:
    """optimal_transform(*args, **kwargs)
    Overloaded function.

    1. optimal_transform(src: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], dst: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]

    Compute optimal transform between two point sets.
    Args:
        src: Source points (Nx3 array).
        dst: Destination points (Nx3 array).
    Returns:
        numpy.ndarray: 4x4 transformation matrix.

    2. optimal_transform(src: pyuipc.geometry.SimplicialComplex, dst: pyuipc.geometry.SimplicialComplex) -> numpy.typing.NDArray[numpy.float64]

    Compute optimal transform between two simplicial complexes.
    Args:
        src: Source SimplicialComplex.
        dst: Destination SimplicialComplex.
    Returns:
        numpy.ndarray: 4x4 transformation matrix.
    """
def pointcloud(Vs: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> SimplicialComplex:
    """pointcloud(Vs: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> pyuipc.geometry.SimplicialComplex

    Create a point cloud from vertices.
    Args:
        Vs: Array of vertex positions (Nx3).
    Returns:
        SimplicialComplex: Point cloud.
    """
def points_from_volume(sc: SimplicialComplex, resolution: typing.SupportsFloat = ...) -> SimplicialComplex:
    """points_from_volume(sc: pyuipc.geometry.SimplicialComplex, resolution: typing.SupportsFloat = 0.01) -> pyuipc.geometry.SimplicialComplex

    Generate points from a volume mesh.
    Args:
        sc: SimplicialComplex (volume mesh).
        resolution: Point sampling resolution (default: 0.01).
    Returns:
        SimplicialComplex: Point cloud.
    """
def tetmesh(Vs: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], Ts: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> SimplicialComplex:
    """tetmesh(Vs: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], Ts: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> pyuipc.geometry.SimplicialComplex

    Create a tetrahedral mesh from vertices and tetrahedra.
    Args:
        Vs: Array of vertex positions (Nx3).
        Ts: Array of tetrahedron indices (Mx4).
    Returns:
        SimplicialComplex: Tetrahedral mesh.
    """
def trimesh(Vs: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], Fs: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> SimplicialComplex:
    """trimesh(Vs: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], Fs: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> pyuipc.geometry.SimplicialComplex

    Create a triangular mesh from vertices and faces.
    Args:
        Vs: Array of vertex positions (Nx3).
        Fs: Array of face indices (Mx3 for triangles or Mx4 for quads).
    Returns:
        SimplicialComplex: Triangular mesh.
    """
