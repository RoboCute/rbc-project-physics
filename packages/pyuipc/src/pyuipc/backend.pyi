import collections.abc
import pyuipc.core
import pyuipc.diff_sim
import pyuipc.geometry
import typing
from typing import overload

class Buffer:
    def __init__(self, resize_func: collections.abc.Callable, get_buffer_view_func: collections.abc.Callable) -> None:
        """__init__(self: pyuipc.backend.Buffer, resize_func: collections.abc.Callable, get_buffer_view_func: collections.abc.Callable) -> None

        Constructs a Buffer object with provided resize and get_buffer_view functions.
        Args:
            resize_func f:(int)->None: Function to resize the buffer.
            get_buffer_view_func f:()->BufferView: Function to retrieve the buffer view.
        """
    def resize(self, size: typing.SupportsInt) -> None:
        """resize(self: pyuipc.backend.Buffer, size: typing.SupportsInt) -> None

        Resize the buffer.
        Args:
            size: New size in elements.
        """
    def view(self) -> BufferView:
        """view(self: pyuipc.backend.Buffer) -> pyuipc.backend.BufferView

        Get a view of the buffer.
        Returns:
            BufferView: View of the buffer.
        """

class BufferView:
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: pyuipc.backend.BufferView) -> None

        Create an empty buffer view.

        2. __init__(self: pyuipc.backend.BufferView, handle: typing.SupportsInt, element_offset: typing.SupportsInt, element_count: typing.SupportsInt, element_size: typing.SupportsInt, element_stride: typing.SupportsInt, backend_name: str) -> None

        Create a buffer view with specified parameters.
        Args:
            handle: Backend handle to the buffer.
            element_offset: Offset in elements from the start of the buffer.
            element_count: Number of elements in the view.
            element_size: Size of each element in bytes.
            element_stride: Stride between elements in bytes.
            backend_name: Name of the backend.
        """
    @overload
    def __init__(self, handle: typing.SupportsInt, element_offset: typing.SupportsInt, element_count: typing.SupportsInt, element_size: typing.SupportsInt, element_stride: typing.SupportsInt, backend_name: str) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: pyuipc.backend.BufferView) -> None

        Create an empty buffer view.

        2. __init__(self: pyuipc.backend.BufferView, handle: typing.SupportsInt, element_offset: typing.SupportsInt, element_count: typing.SupportsInt, element_size: typing.SupportsInt, element_stride: typing.SupportsInt, backend_name: str) -> None

        Create a buffer view with specified parameters.
        Args:
            handle: Backend handle to the buffer.
            element_offset: Offset in elements from the start of the buffer.
            element_count: Number of elements in the view.
            element_size: Size of each element in bytes.
            element_stride: Stride between elements in bytes.
            backend_name: Name of the backend.
        """
    def backend(self) -> str:
        """backend(self: pyuipc.backend.BufferView) -> str

        Get the backend name.
        Returns:
            str: Backend name.
        """
    def element_size(self) -> int:
        """element_size(self: pyuipc.backend.BufferView) -> int

        Get the element size in bytes.
        Returns:
            int: Element size in bytes.
        """
    def element_stride(self) -> int:
        """element_stride(self: pyuipc.backend.BufferView) -> int

        Get the element stride in bytes.
        Returns:
            int: Element stride in bytes.
        """
    def handle(self) -> int:
        """handle(self: pyuipc.backend.BufferView) -> int

        Get the backend handle.
        Returns:
            int: Backend handle value.
        """
    def offset(self) -> int:
        """offset(self: pyuipc.backend.BufferView) -> int

        Get the element offset.
        Returns:
            int: Element offset.
        """
    def size(self) -> int:
        """size(self: pyuipc.backend.BufferView) -> int

        Get the number of elements.
        Returns:
            int: Number of elements.
        """
    def size_in_bytes(self) -> int:
        """size_in_bytes(self: pyuipc.backend.BufferView) -> int

        Get the total size in bytes.
        Returns:
            int: Total size in bytes.
        """
    def subview(self, offset: typing.SupportsInt, element_count: typing.SupportsInt) -> BufferView:
        """subview(self: pyuipc.backend.BufferView, offset: typing.SupportsInt, element_count: typing.SupportsInt) -> pyuipc.backend.BufferView

        Create a subview of this buffer view.
        Args:
            offset: Element offset from the start of this view.
            element_count: Number of elements in the subview.
        Returns:
            BufferView: New buffer view representing the subview.
        """
    def __bool__(self) -> bool:
        """__bool__(self: pyuipc.backend.BufferView) -> bool

        Check if the buffer view is valid.
        Returns:
            bool: True if valid, False otherwise.
        """

class DiffSimVisitor:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def parameters(self) -> pyuipc.diff_sim.ParameterCollection:
        """parameters(self: pyuipc.backend.DiffSimVisitor) -> pyuipc.diff_sim.ParameterCollection

        Get the parameter collection.
        Returns:
            ParameterCollection: Reference to parameter collection.
        """

class SceneVisitor:
    class GeometrySlotSpan:
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def __getitem__(self, index: typing.SupportsInt) -> pyuipc.geometry.GeometrySlot:
            """__getitem__(self: pyuipc.backend.SceneVisitor.GeometrySlotSpan, index: typing.SupportsInt) -> pyuipc.geometry.GeometrySlot

            Get an element by index.
            Args:
                index: Element index.
            Returns:
                Element value.
            """
        def __iter__(self) -> collections.abc.Iterator[pyuipc.geometry.GeometrySlot]:
            """__iter__(self: pyuipc.backend.SceneVisitor.GeometrySlotSpan) -> collections.abc.Iterator[pyuipc.geometry.GeometrySlot]

            Create an iterator over the span.
            Returns:
                iterator: Iterator over span elements.
            """
        def __len__(self) -> int:
            """__len__(self: pyuipc.backend.SceneVisitor.GeometrySlotSpan) -> int

            Get the size of the span.
            Returns:
                int: Number of elements.
            """
        def __setitem__(self, index: typing.SupportsInt, value: pyuipc.geometry.GeometrySlot) -> None:
            """__setitem__(self: pyuipc.backend.SceneVisitor.GeometrySlotSpan, index: typing.SupportsInt, value: pyuipc.geometry.GeometrySlot) -> None

            Set an element by index.
            Args:
                index: Element index.
                value: Value to set.
            """
    def __init__(self, scene: pyuipc.core.Scene) -> None:
        """__init__(self: pyuipc.backend.SceneVisitor, scene: pyuipc.core.Scene) -> None

        Create a SceneVisitor for a scene.
        Args:
            scene: Scene to visit.
        """
    def begin_pending(self) -> None:
        """begin_pending(self: pyuipc.backend.SceneVisitor) -> None

        Begin processing pending geometries.
        """
    def config(self) -> pyuipc.geometry.AttributeCollection:
        """config(self: pyuipc.backend.SceneVisitor) -> pyuipc.geometry.AttributeCollection

        Get the scene configuration.
        Returns:
            dict: Configuration dictionary.
        """
    def constitution_tabular(self) -> pyuipc.core.ConstitutionTabular:
        """constitution_tabular(self: pyuipc.backend.SceneVisitor) -> pyuipc.core.ConstitutionTabular

        Get the constitution tabular.
        Returns:
            ConstitutionTabular: Reference to constitution tabular.
        """
    def contact_tabular(self) -> pyuipc.core.ContactTabular:
        """contact_tabular(self: pyuipc.backend.SceneVisitor) -> pyuipc.core.ContactTabular

        Get the contact tabular.
        Returns:
            ContactTabular: Reference to contact tabular.
        """
    def diff_sim(self) -> DiffSimVisitor:
        """diff_sim(self: pyuipc.backend.SceneVisitor) -> pyuipc.backend.DiffSimVisitor

        Get the differential simulator visitor.
        Returns:
            DiffSimVisitor: Reference to diff sim visitor.
        """
    def geometries(self) -> SceneVisitor.GeometrySlotSpan:
        """geometries(self: pyuipc.backend.SceneVisitor) -> pyuipc.backend.SceneVisitor.GeometrySlotSpan

        Get all geometries.
        Returns:
            GeometrySlotSpan: Span of geometry slots.
        """
    def get(self) -> pyuipc.core.Scene:
        """get(self: pyuipc.backend.SceneVisitor) -> pyuipc.core.Scene

        Get the scene.
        Returns:
            Scene: Scene object.
        """
    def pending_geometries(self) -> SceneVisitor.GeometrySlotSpan:
        """pending_geometries(self: pyuipc.backend.SceneVisitor) -> pyuipc.backend.SceneVisitor.GeometrySlotSpan

        Get pending geometries.
        Returns:
            GeometrySlotSpan: Span of pending geometry slots.
        """
    def pending_rest_geometries(self) -> SceneVisitor.GeometrySlotSpan:
        """pending_rest_geometries(self: pyuipc.backend.SceneVisitor) -> pyuipc.backend.SceneVisitor.GeometrySlotSpan

        Get pending rest geometries.
        Returns:
            GeometrySlotSpan: Span of pending rest geometry slots.
        """
    def rest_geometries(self) -> SceneVisitor.GeometrySlotSpan:
        """rest_geometries(self: pyuipc.backend.SceneVisitor) -> pyuipc.backend.SceneVisitor.GeometrySlotSpan

        Get rest geometries.
        Returns:
            GeometrySlotSpan: Span of rest geometry slots.
        """
    def solve_pending(self) -> None:
        """solve_pending(self: pyuipc.backend.SceneVisitor) -> None

        Solve pending geometries.
        """

class WorldVisitor:
    def __init__(self, world: pyuipc.core.World) -> None:
        """__init__(self: pyuipc.backend.WorldVisitor, world: pyuipc.core.World) -> None

        Create a WorldVisitor for a world.
        Args:
            world: World to visit.
        """
    def animator(self, *args, **kwargs):
        """animator(self: pyuipc.backend.WorldVisitor) -> uipc::backend::AnimatorVisitor

        Get the animator.
        Returns:
            Animator: Reference to animator.
        """
    def get(self) -> pyuipc.core.World:
        """get(self: pyuipc.backend.WorldVisitor) -> pyuipc.core.World

        Get the world.
        Returns:
            World: World object.
        """
    def scene(self) -> SceneVisitor:
        """scene(self: pyuipc.backend.WorldVisitor) -> pyuipc.backend.SceneVisitor

        Get the scene visitor.
        Returns:
            SceneVisitor: Scene visitor.
        """
