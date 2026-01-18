import collections.abc
import numpy
import numpy.typing
import pyuipc.constitution
import pyuipc.diff_sim
import pyuipc.geometry
import typing
from typing import Any, ClassVar, overload

Error: SanityCheckResult
Success: SanityCheckResult
Warning: SanityCheckResult

class AffineBodyStateAccessorFeature(Feature):
    FeatureName: ClassVar[str] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def body_count(self) -> int:
        """body_count(self: pyuipc.core.AffineBodyStateAccessorFeature) -> int

        Get the number of affine bodies.
        Returns:
            int: Number of affine bodies.
        """
    def copy_from(self, state_geo: pyuipc.geometry.SimplicialComplex) -> None:
        """copy_from(self: pyuipc.core.AffineBodyStateAccessorFeature, state_geo: pyuipc.geometry.SimplicialComplex) -> None

        Copy state from geometry.
        Args:
            state_geo: Geometry containing state data to copy from.
        """
    def copy_to(self, state_geo: pyuipc.geometry.SimplicialComplex) -> None:
        """copy_to(self: pyuipc.core.AffineBodyStateAccessorFeature, state_geo: pyuipc.geometry.SimplicialComplex) -> None

        Copy state to geometry.
        Args:
            state_geo: Geometry to copy state data to.
        """
    def create_geometry(self, body_offset: typing.SupportsInt = ..., body_count: typing.SupportsInt = ...) -> pyuipc.geometry.SimplicialComplex:
        """create_geometry(self: pyuipc.core.AffineBodyStateAccessorFeature, body_offset: typing.SupportsInt = 0, body_count: typing.SupportsInt = 18446744073709551615) -> pyuipc.geometry.SimplicialComplex

        Create geometry from state data.
        Args:
            body_offset: Starting body index (default: 0).
            body_count: Number of bodies to include (default: all).
        Returns:
            Geometry: Geometry created from state data.
        """

class Animation:
    class UpdateHint:
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""

    class UpdateInfo:
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def dt(self) -> float:
            """dt(self: pyuipc.core.Animation.UpdateInfo) -> float

            Get the time step.
            Returns:
                float: Time step (delta time).
            """
        def frame(self) -> int:
            """frame(self: pyuipc.core.Animation.UpdateInfo) -> int

            Get the current frame number.
            Returns:
                int: Current frame number.
            """
        def geo_slots(self) -> list:
            """geo_slots(self: pyuipc.core.Animation.UpdateInfo) -> list

            Get the geometry slots.
            Returns:
                list: List of geometry slots.
            """
        def hint(self) -> Animation.UpdateHint:
            """hint(self: pyuipc.core.Animation.UpdateInfo) -> pyuipc.core.Animation.UpdateHint

            Get the update hint.
            Returns:
                UpdateHint: Update hint value.
            """
        def object(self) -> Object:
            """object(self: pyuipc.core.Animation.UpdateInfo) -> pyuipc.core.Object

            Get the object being animated.
            Returns:
                Object: Reference to the object.
            """
        def rest_geo_slots(self) -> list:
            """rest_geo_slots(self: pyuipc.core.Animation.UpdateInfo) -> list

            Get the rest geometry slots.
            Returns:
                list: List of rest geometry slots.
            """
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class Animator:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def erase(self, object: typing.SupportsInt) -> None:
        """erase(self: pyuipc.core.Animator, object: typing.SupportsInt) -> None

        Remove animation callback for an object.
        Args:
            object: Object to remove animation from.
        """
    def insert(self, object: Object, callable: collections.abc.Callable) -> None:
        """insert(self: pyuipc.core.Animator, object: pyuipc.core.Object, callable: collections.abc.Callable) -> None

        Insert an animation callback for an object.
        Args:
            object: Object to animate.
            callable: Python function that takes an UpdateInfo argument and updates the object.
        """
    @overload
    def substep(self, substep: typing.SupportsInt) -> None:
        """substep(*args, **kwargs)
        Overloaded function.

        1. substep(self: pyuipc.core.Animator, substep: typing.SupportsInt) -> None

        Set the substep count.
        Args:
            substep: Number of substeps per frame.

        2. substep(self: pyuipc.core.Animator) -> int

        Get the substep count.
        Returns:
            int: Number of substeps per frame.
        """
    @overload
    def substep(self) -> int:
        """substep(*args, **kwargs)
        Overloaded function.

        1. substep(self: pyuipc.core.Animator, substep: typing.SupportsInt) -> None

        Set the substep count.
        Args:
            substep: Number of substeps per frame.

        2. substep(self: pyuipc.core.Animator) -> int

        Get the substep count.
        Returns:
            int: Number of substeps per frame.
        """

class ConstitutionTabular:
    def __init__(self) -> None:
        """__init__(self: pyuipc.core.ConstitutionTabular) -> None

        Create an empty constitution tabular.
        """
    def insert(self, constitution: pyuipc.constitution.IConstitution) -> None:
        """insert(self: pyuipc.core.ConstitutionTabular, constitution: pyuipc.constitution.IConstitution) -> None

        Insert a constitution into the tabular.
        Args:
            constitution: Constitution to insert.
        """
    def types(self) -> set[str]:
        """types(self: pyuipc.core.ConstitutionTabular) -> set[str]

        Get the types of all constitutions.
        Returns:
            list: List of constitution type names.
        """
    def uids(self) -> numpy.typing.NDArray[numpy.uint64]:
        """uids(self: pyuipc.core.ConstitutionTabular) -> numpy.typing.NDArray[numpy.uint64]

        Get the UIDs of all constitutions.
        Returns:
            numpy.ndarray: Array of constitution UIDs.
        """

class ContactElement:
    def __init__(self, id: typing.SupportsInt, name: str) -> None:
        """__init__(self: pyuipc.core.ContactElement, id: typing.SupportsInt, name: str) -> None

        Create a contact element.
        Args:
            id: Element ID.
            name: Element name.
        """
    def apply_to(self, object: pyuipc.geometry.Geometry) -> pyuipc.geometry.AttributeSlotI32:
        """apply_to(self: pyuipc.core.ContactElement, object: pyuipc.geometry.Geometry) -> pyuipc.geometry.AttributeSlotI32

        Apply this contact element to an object.
        Args:
            object: Object to apply to.
        """
    def id(self) -> int:
        """id(self: pyuipc.core.ContactElement) -> int

        Get the element ID.
        Returns:
            int: Element ID.
        """
    def name(self) -> str:
        """name(self: pyuipc.core.ContactElement) -> str

        Get the element name.
        Returns:
            str: Element name.
        """

class ContactModel:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def config(self) -> json:
        """config(self: pyuipc.core.ContactModel) -> json

        Get the configuration dictionary.
        Returns:
            dict: Configuration dictionary.
        """
    def friction_rate(self) -> float:
        """friction_rate(self: pyuipc.core.ContactModel) -> float

        Get the friction rate.
        Returns:
            float: Friction rate value.
        """
    def is_enabled(self) -> bool:
        """is_enabled(self: pyuipc.core.ContactModel) -> bool

        Check if the contact model is enabled.
        Returns:
            bool: True if enabled, False otherwise.
        """
    def resistance(self) -> float:
        """resistance(self: pyuipc.core.ContactModel) -> float

        Get the resistance value.
        Returns:
            float: Resistance value.
        """
    def topo(self, *args, **kwargs):
        """topo(self: pyuipc.core.ContactModel) -> Eigen::Matrix<int,2,1,0,2,1>

        Get the topology type.
        Returns:
            str: Topology type string.
        """

class ContactModelCollection:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def create(self, name: str, default_value: object) -> pyuipc.geometry.IAttributeSlot:
        """create(self: pyuipc.core.ContactModelCollection, name: str, default_value: object) -> pyuipc.geometry.IAttributeSlot

        Create a contact model attribute.
        Args:
            name: Attribute name.
            default_value: Default value for the attribute.
        Returns:
            AttributeSlot: Created attribute slot.
        """
    def find(self, name: str) -> pyuipc.geometry.IAttributeSlot:
        """find(self: pyuipc.core.ContactModelCollection, name: str) -> pyuipc.geometry.IAttributeSlot

        Find a contact model attribute by name.
        Args:
            name: Attribute name.
        Returns:
            AttributeSlot or None: Attribute slot if found, None otherwise.
        """

class ContactSystemFeature(Feature):
    FeatureName: ClassVar[str] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def contact_energy(self, prim_type: str, energy: pyuipc.geometry.Geometry) -> None:
        """contact_energy(*args, **kwargs)
        Overloaded function.

        1. contact_energy(self: pyuipc.core.ContactSystemFeature, prim_type: str, energy: pyuipc.geometry.Geometry) -> None

        Compute contact energy for a primitive type.
        Args:
            prim_type: Primitive type string.
            energy: Geometry to store energy values (modified in-place).

        2. contact_energy(self: pyuipc.core.ContactSystemFeature, constitution: pyuipc.constitution.IConstitution, prims: pyuipc.geometry.Geometry) -> None

        Compute contact energy for a constitution.
        Args:
            constitution: Constitution to compute energy for.
            prims: Geometry containing primitives (modified in-place with energy values).
        """
    @overload
    def contact_energy(self, constitution: pyuipc.constitution.IConstitution, prims: pyuipc.geometry.Geometry) -> None:
        """contact_energy(*args, **kwargs)
        Overloaded function.

        1. contact_energy(self: pyuipc.core.ContactSystemFeature, prim_type: str, energy: pyuipc.geometry.Geometry) -> None

        Compute contact energy for a primitive type.
        Args:
            prim_type: Primitive type string.
            energy: Geometry to store energy values (modified in-place).

        2. contact_energy(self: pyuipc.core.ContactSystemFeature, constitution: pyuipc.constitution.IConstitution, prims: pyuipc.geometry.Geometry) -> None

        Compute contact energy for a constitution.
        Args:
            constitution: Constitution to compute energy for.
            prims: Geometry containing primitives (modified in-place with energy values).
        """
    @overload
    def contact_gradient(self, prim_type: str, vert_grad: pyuipc.geometry.Geometry) -> None:
        """contact_gradient(*args, **kwargs)
        Overloaded function.

        1. contact_gradient(self: pyuipc.core.ContactSystemFeature, prim_type: str, vert_grad: pyuipc.geometry.Geometry) -> None

        Compute contact gradient for a primitive type.
        Args:
            prim_type: Primitive type string.
            vert_grad: Geometry to store vertex gradients (modified in-place).

        2. contact_gradient(self: pyuipc.core.ContactSystemFeature, constitution: pyuipc.constitution.IConstitution, vert_grad: pyuipc.geometry.Geometry) -> None

        Compute contact gradient for a constitution.
        Args:
            constitution: Constitution to compute gradient for.
            vert_grad: Geometry to store vertex gradients (modified in-place).
        """
    @overload
    def contact_gradient(self, constitution: pyuipc.constitution.IConstitution, vert_grad: pyuipc.geometry.Geometry) -> None:
        """contact_gradient(*args, **kwargs)
        Overloaded function.

        1. contact_gradient(self: pyuipc.core.ContactSystemFeature, prim_type: str, vert_grad: pyuipc.geometry.Geometry) -> None

        Compute contact gradient for a primitive type.
        Args:
            prim_type: Primitive type string.
            vert_grad: Geometry to store vertex gradients (modified in-place).

        2. contact_gradient(self: pyuipc.core.ContactSystemFeature, constitution: pyuipc.constitution.IConstitution, vert_grad: pyuipc.geometry.Geometry) -> None

        Compute contact gradient for a constitution.
        Args:
            constitution: Constitution to compute gradient for.
            vert_grad: Geometry to store vertex gradients (modified in-place).
        """
    @overload
    def contact_hessian(self, prim_type: str, vert_hess: pyuipc.geometry.Geometry) -> None:
        """contact_hessian(*args, **kwargs)
        Overloaded function.

        1. contact_hessian(self: pyuipc.core.ContactSystemFeature, prim_type: str, vert_hess: pyuipc.geometry.Geometry) -> None

        Compute contact Hessian for a primitive type.
        Args:
            prim_type: Primitive type string.
            vert_hess: Geometry to store vertex Hessians (modified in-place).

        2. contact_hessian(self: pyuipc.core.ContactSystemFeature, constitution: pyuipc.constitution.IConstitution, vert_hess: pyuipc.geometry.Geometry) -> None

        Compute contact Hessian for a constitution.
        Args:
            constitution: Constitution to compute Hessian for.
            vert_hess: Geometry to store vertex Hessians (modified in-place).
        """
    @overload
    def contact_hessian(self, constitution: pyuipc.constitution.IConstitution, vert_hess: pyuipc.geometry.Geometry) -> None:
        """contact_hessian(*args, **kwargs)
        Overloaded function.

        1. contact_hessian(self: pyuipc.core.ContactSystemFeature, prim_type: str, vert_hess: pyuipc.geometry.Geometry) -> None

        Compute contact Hessian for a primitive type.
        Args:
            prim_type: Primitive type string.
            vert_hess: Geometry to store vertex Hessians (modified in-place).

        2. contact_hessian(self: pyuipc.core.ContactSystemFeature, constitution: pyuipc.constitution.IConstitution, vert_hess: pyuipc.geometry.Geometry) -> None

        Compute contact Hessian for a constitution.
        Args:
            constitution: Constitution to compute Hessian for.
            vert_hess: Geometry to store vertex Hessians (modified in-place).
        """
    def contact_primitive_types(self) -> list[str]:
        """contact_primitive_types(self: pyuipc.core.ContactSystemFeature) -> list[str]

        Get the list of supported primitive types.
        Returns:
            list: List of primitive type strings.
        """

class ContactTabular:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def at(self, i: typing.SupportsInt, j: typing.SupportsInt) -> ContactModel:
        """at(self: pyuipc.core.ContactTabular, i: typing.SupportsInt, j: typing.SupportsInt) -> pyuipc.core.ContactModel

        Get the contact model between two elements.
        Args:
            i: First element ID.
            j: Second element ID.
        Returns:
            ContactModel: Contact model between the two elements.
        """
    def contact_models(self) -> ContactModelCollection:
        """contact_models(self: pyuipc.core.ContactTabular) -> pyuipc.core.ContactModelCollection

        Get the contact model collection.
        Returns:
            ContactModelCollection: Collection of contact models.
        """
    def create(self, name: str = ...) -> ContactElement:
        """create(self: pyuipc.core.ContactTabular, name: str = '') -> pyuipc.core.ContactElement

        Create a new contact element.
        Args:
            name: Element name (optional).
        Returns:
            ContactElement: Created contact element.
        """
    def default_element(self) -> ContactElement:
        """default_element(self: pyuipc.core.ContactTabular) -> pyuipc.core.ContactElement

        Get the default contact element.
        Returns:
            ContactElement: Reference to default element.
        """
    @overload
    def default_model(self, friction_rate: typing.SupportsFloat, resistance: typing.SupportsFloat, enable: bool = ..., config: json = ...) -> None:
        """default_model(*args, **kwargs)
        Overloaded function.

        1. default_model(self: pyuipc.core.ContactTabular, friction_rate: typing.SupportsFloat, resistance: typing.SupportsFloat, enable: bool = True, config: json = {}) -> None

        Set the default contact model parameters.
        Args:
            friction_rate: Default friction rate.
            resistance: Default resistance value.
            enable: Whether contacts are enabled by default (default: True).
            config: Default configuration dictionary (default: empty).

        2. default_model(self: pyuipc.core.ContactTabular) -> pyuipc.core.ContactModel

        Get the default contact model.
        Returns:
            ContactModel: Default contact model.
        """
    @overload
    def default_model(self) -> ContactModel:
        """default_model(*args, **kwargs)
        Overloaded function.

        1. default_model(self: pyuipc.core.ContactTabular, friction_rate: typing.SupportsFloat, resistance: typing.SupportsFloat, enable: bool = True, config: json = {}) -> None

        Set the default contact model parameters.
        Args:
            friction_rate: Default friction rate.
            resistance: Default resistance value.
            enable: Whether contacts are enabled by default (default: True).
            config: Default configuration dictionary (default: empty).

        2. default_model(self: pyuipc.core.ContactTabular) -> pyuipc.core.ContactModel

        Get the default contact model.
        Returns:
            ContactModel: Default contact model.
        """
    def element_count(self) -> int:
        """element_count(self: pyuipc.core.ContactTabular) -> int

        Get the number of contact elements.
        Returns:
            int: Number of elements.
        """
    def insert(self, L: ContactElement, R: ContactElement, friction_rate: typing.SupportsFloat, resistance: typing.SupportsFloat, enable: bool = ..., config: json = ...) -> int:
        """insert(self: pyuipc.core.ContactTabular, L: pyuipc.core.ContactElement, R: pyuipc.core.ContactElement, friction_rate: typing.SupportsFloat, resistance: typing.SupportsFloat, enable: bool = True, config: json = {}) -> int

        Insert a contact model between two elements.
        Args:
            L: Left element ID.
            R: Right element ID.
            friction_rate: Friction rate value.
            resistance: Resistance value.
            enable: Whether the contact is enabled (default: True).
            config: Additional configuration dictionary (default: empty).
        """

class DiffSim:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def parameters(self) -> pyuipc.diff_sim.ParameterCollection:
        """parameters(self: pyuipc.core.DiffSim) -> pyuipc.diff_sim.ParameterCollection

        Get the parameter collection.
        Returns:
            ParameterCollection: Reference to the parameter collection.
        """

class Engine:
    @overload
    def __init__(self, backend_name: str, workspace: str = ..., config: json = ...) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: pyuipc.core.Engine, backend_name: str, workspace: str = './', config: json = {'extras': {'gui': {'enable': True}}, 'gpu': {'device': 0}}) -> None

        Create an engine with a backend.
        Args:
            backend_name: Name of the backend to use (e.g., 'cuda', 'none').
            workspace: Workspace directory path (default: './').
            config: Configuration dictionary (optional, uses default if not provided).

        2. __init__(self: pyuipc.core.Engine, backend_name: str, overrider: pyuipc.core.IEngine, workspace: str = './', config: json = {'extras': {'gui': {'enable': True}}, 'gpu': {'device': 0}}) -> None

        Create an engine with a custom engine overrider.
        Args:
            backend_name: Name of the backend to use.
            overrider: Custom IEngine implementation.
            workspace: Workspace directory path (default: './').
            config: Configuration dictionary (optional, uses default if not provided).
        """
    @overload
    def __init__(self, backend_name: str, overrider: IEngine, workspace: str = ..., config: json = ...) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: pyuipc.core.Engine, backend_name: str, workspace: str = './', config: json = {'extras': {'gui': {'enable': True}}, 'gpu': {'device': 0}}) -> None

        Create an engine with a backend.
        Args:
            backend_name: Name of the backend to use (e.g., 'cuda', 'none').
            workspace: Workspace directory path (default: './').
            config: Configuration dictionary (optional, uses default if not provided).

        2. __init__(self: pyuipc.core.Engine, backend_name: str, overrider: pyuipc.core.IEngine, workspace: str = './', config: json = {'extras': {'gui': {'enable': True}}, 'gpu': {'device': 0}}) -> None

        Create an engine with a custom engine overrider.
        Args:
            backend_name: Name of the backend to use.
            overrider: Custom IEngine implementation.
            workspace: Workspace directory path (default: './').
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def backend_name(self) -> str:
        """backend_name(self: pyuipc.core.Engine) -> str

        Get the backend name.
        Returns:
            str: Backend name.
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default engine configuration.
        Returns:
            dict: Default configuration dictionary.
        """
    def features(self) -> FeatureCollection:
        """features(self: pyuipc.core.Engine) -> pyuipc.core.FeatureCollection

        Get the feature collection.
        Returns:
            FeatureCollection: Reference to feature collection.
        """
    def workspace(self) -> str:
        """workspace(self: pyuipc.core.Engine) -> str

        Get the workspace directory.
        Returns:
            str: Workspace directory path.
        """

class EngineStatus:
    class Type:
        __members__: ClassVar[dict] = ...  # read-only
        Error: ClassVar[EngineStatus.Type] = ...
        Info: ClassVar[EngineStatus.Type] = ...
        Warning: ClassVar[EngineStatus.Type] = ...
        __entries: ClassVar[dict] = ...
        def __init__(self, value: typing.SupportsInt) -> None:
            """__init__(self: pyuipc.core.EngineStatus.Type, value: typing.SupportsInt) -> None"""
        def __eq__(self, other: object) -> bool:
            """__eq__(self: object, other: object, /) -> bool"""
        def __hash__(self) -> int:
            """__hash__(self: object, /) -> int"""
        def __index__(self) -> int:
            """__index__(self: pyuipc.core.EngineStatus.Type, /) -> int"""
        def __int__(self) -> int:
            """__int__(self: pyuipc.core.EngineStatus.Type, /) -> int"""
        def __ne__(self, other: object) -> bool:
            """__ne__(self: object, other: object, /) -> bool"""
        @property
        def name(self): ...
        @property
        def value(self) -> int: ...
    Error: ClassVar[EngineStatus.Type] = ...
    Info: ClassVar[EngineStatus.Type] = ...
    Warning: ClassVar[EngineStatus.Type] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def error(msg: str) -> EngineStatus:
        """error(msg: str) -> pyuipc.core.EngineStatus

        Create an error status.
        Args:
            msg: Error message.
        Returns:
            EngineStatus: Error status object.
        """
    @staticmethod
    def info(msg: str) -> EngineStatus:
        """info(msg: str) -> pyuipc.core.EngineStatus

        Create an info status.
        Args:
            msg: Info message.
        Returns:
            EngineStatus: Info status object.
        """
    def type(self, _None, Info, Warning, orError) -> Any:
        """type(self: pyuipc.core.EngineStatus) -> uipc::core::EngineStatus::Type

        Get the status type.
        Returns:
            Type: Status type (None, Info, Warning, or Error).
        """
    @staticmethod
    def warning(msg: str) -> EngineStatus:
        """warning(msg: str) -> pyuipc.core.EngineStatus

        Create a warning status.
        Args:
            msg: Warning message.
        Returns:
            EngineStatus: Warning status object.
        """
    def what(self) -> str:
        """what(self: pyuipc.core.EngineStatus) -> str

        Get the status message.
        Returns:
            str: Status message string.
        """

class EngineStatusCollection:
    def __init__(self) -> None:
        """__init__(self: pyuipc.core.EngineStatusCollection) -> None

        Create an empty status collection.
        """
    def has_error(self) -> bool:
        """has_error(self: pyuipc.core.EngineStatusCollection) -> bool

        Check if the collection contains any errors.
        Returns:
            bool: True if any error status exists, False otherwise.
        """
    def push_back(self, status: EngineStatus) -> None:
        """push_back(self: pyuipc.core.EngineStatusCollection, status: pyuipc.core.EngineStatus) -> None

        Add a status to the collection.
        Args:
            status: EngineStatus to add.
        """
    def to_json(self) -> json:
        """to_json(self: pyuipc.core.EngineStatusCollection) -> json

        Convert status collection to JSON.
        Returns:
            dict: JSON representation of the status collection.
        """

class Feature:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def name(self) -> str:
        """name(self: pyuipc.core.Feature) -> str

        Get the feature name.
        Returns:
            str: Feature name.
        """
    def type_name(self) -> str:
        """type_name(self: pyuipc.core.Feature) -> str

        Get the feature type name.
        Returns:
            str: Feature type name.
        """

class FeatureCollection:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def find(self, name: str) -> Feature:
        """find(*args, **kwargs)
        Overloaded function.

        1. find(self: pyuipc.core.FeatureCollection, name: str) -> pyuipc.core.Feature

        Find a feature by name.
        Args:
            name: Feature name.
        Returns:
            Feature or None: Feature if found, None otherwise.

        2. find(self: pyuipc.core.FeatureCollection, type: type) -> pyuipc.core.Feature

        Find a feature by type.
        Args:
            type: Feature type (must have FeatureName attribute).
        Returns:
            Feature or None: Feature if found, None otherwise.
        """
    @overload
    def find(self, type: type) -> Feature:
        """find(*args, **kwargs)
        Overloaded function.

        1. find(self: pyuipc.core.FeatureCollection, name: str) -> pyuipc.core.Feature

        Find a feature by name.
        Args:
            name: Feature name.
        Returns:
            Feature or None: Feature if found, None otherwise.

        2. find(self: pyuipc.core.FeatureCollection, type: type) -> pyuipc.core.Feature

        Find a feature by type.
        Args:
            type: Feature type (must have FeatureName attribute).
        Returns:
            Feature or None: Feature if found, None otherwise.
        """
    def to_json(self) -> json:
        """to_json(self: pyuipc.core.FeatureCollection) -> json

        Convert feature collection to JSON.
        Returns:
            dict: JSON representation of the feature collection.
        """

class FiniteElementStateAccessorFeature(Feature):
    FeatureName: ClassVar[str] = ...
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def copy_from(self, state_geo: pyuipc.geometry.SimplicialComplex) -> None:
        """copy_from(self: pyuipc.core.FiniteElementStateAccessorFeature, state_geo: pyuipc.geometry.SimplicialComplex) -> None

        Copy state from geometry.
        Args:
            state_geo: Geometry containing state data to copy from.
        """
    def copy_to(self, state_geo: pyuipc.geometry.SimplicialComplex) -> None:
        """copy_to(self: pyuipc.core.FiniteElementStateAccessorFeature, state_geo: pyuipc.geometry.SimplicialComplex) -> None

        Copy state to geometry.
        Args:
            state_geo: Geometry to copy state data to.
        """
    def create_geometry(self, vertex_offset: typing.SupportsInt = ..., vertex_count: typing.SupportsInt = ...) -> pyuipc.geometry.SimplicialComplex:
        """create_geometry(self: pyuipc.core.FiniteElementStateAccessorFeature, vertex_offset: typing.SupportsInt = 0, vertex_count: typing.SupportsInt = 18446744073709551615) -> pyuipc.geometry.SimplicialComplex

        Create geometry from state data.
        Args:
            vertex_offset: Starting vertex index (default: 0).
            vertex_count: Number of vertices to include (default: all).
        Returns:
            Geometry: Geometry created from state data.
        """
    def vertex_count(self) -> int:
        """vertex_count(self: pyuipc.core.FiniteElementStateAccessorFeature) -> int

        Get the number of vertices.
        Returns:
            int: Number of vertices.
        """

class IEngine:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class Object:
    class Geometries:
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        @overload
        def create(self, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot]:
            """create(*args, **kwargs)
            Overloaded function.

            1. create(self: pyuipc.core.Object.Geometries, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot]

            Create geometry from a simplicial complex (rest geometry is auto-generated).
            Args:
                simplicial_complex: SimplicialComplex to create geometry from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            2. create(self: pyuipc.core.Object.Geometries, simplicial_complex: pyuipc.geometry.SimplicialComplex, rest_simplicial_complex: pyuipc.geometry.SimplicialComplex) -> tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot]

            Create geometry from a simplicial complex with explicit rest geometry.
            Args:
                simplicial_complex: SimplicialComplex to create geometry from.
                rest_simplicial_complex: Rest state simplicial complex.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            3. create(self: pyuipc.core.Object.Geometries, implicit_geometry: pyuipc.geometry.ImplicitGeometry) -> tuple[pyuipc.geometry.ImplicitGeometrySlot, pyuipc.geometry.ImplicitGeometrySlot]

            Create geometry from an implicit geometry (rest geometry is auto-generated).
            Args:
                implicit_geometry: ImplicitGeometry to create geometry from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            4. create(self: pyuipc.core.Object.Geometries, implicit_geometry: pyuipc.geometry.ImplicitGeometry, rest_implicit_geometry: pyuipc.geometry.ImplicitGeometry) -> tuple[pyuipc.geometry.ImplicitGeometrySlot, pyuipc.geometry.ImplicitGeometrySlot]

            Create geometry from an implicit geometry with explicit rest geometry.
            Args:
                implicit_geometry: ImplicitGeometry to create geometry from.
                rest_implicit_geometry: Rest state implicit geometry.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            5. create(self: pyuipc.core.Object.Geometries, geometry: pyuipc.geometry.Geometry) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]

            Create geometry from an existing geometry (rest geometry is auto-generated).
            Args:
                geometry: Geometry to create from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            6. create(self: pyuipc.core.Object.Geometries, geometry: pyuipc.geometry.Geometry, rest_geometry: pyuipc.geometry.Geometry) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]

            Create geometry from an existing geometry with explicit rest geometry.
            Args:
                geometry: Geometry to create from.
                rest_geometry: Rest state geometry.
            Returns:
                tuple: Pair of (geometry, rest_geometry).
            """
        @overload
        def create(self, simplicial_complex: pyuipc.geometry.SimplicialComplex, rest_simplicial_complex: pyuipc.geometry.SimplicialComplex) -> tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot]:
            """create(*args, **kwargs)
            Overloaded function.

            1. create(self: pyuipc.core.Object.Geometries, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot]

            Create geometry from a simplicial complex (rest geometry is auto-generated).
            Args:
                simplicial_complex: SimplicialComplex to create geometry from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            2. create(self: pyuipc.core.Object.Geometries, simplicial_complex: pyuipc.geometry.SimplicialComplex, rest_simplicial_complex: pyuipc.geometry.SimplicialComplex) -> tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot]

            Create geometry from a simplicial complex with explicit rest geometry.
            Args:
                simplicial_complex: SimplicialComplex to create geometry from.
                rest_simplicial_complex: Rest state simplicial complex.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            3. create(self: pyuipc.core.Object.Geometries, implicit_geometry: pyuipc.geometry.ImplicitGeometry) -> tuple[pyuipc.geometry.ImplicitGeometrySlot, pyuipc.geometry.ImplicitGeometrySlot]

            Create geometry from an implicit geometry (rest geometry is auto-generated).
            Args:
                implicit_geometry: ImplicitGeometry to create geometry from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            4. create(self: pyuipc.core.Object.Geometries, implicit_geometry: pyuipc.geometry.ImplicitGeometry, rest_implicit_geometry: pyuipc.geometry.ImplicitGeometry) -> tuple[pyuipc.geometry.ImplicitGeometrySlot, pyuipc.geometry.ImplicitGeometrySlot]

            Create geometry from an implicit geometry with explicit rest geometry.
            Args:
                implicit_geometry: ImplicitGeometry to create geometry from.
                rest_implicit_geometry: Rest state implicit geometry.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            5. create(self: pyuipc.core.Object.Geometries, geometry: pyuipc.geometry.Geometry) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]

            Create geometry from an existing geometry (rest geometry is auto-generated).
            Args:
                geometry: Geometry to create from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            6. create(self: pyuipc.core.Object.Geometries, geometry: pyuipc.geometry.Geometry, rest_geometry: pyuipc.geometry.Geometry) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]

            Create geometry from an existing geometry with explicit rest geometry.
            Args:
                geometry: Geometry to create from.
                rest_geometry: Rest state geometry.
            Returns:
                tuple: Pair of (geometry, rest_geometry).
            """
        @overload
        def create(self, implicit_geometry: pyuipc.geometry.ImplicitGeometry) -> tuple[pyuipc.geometry.ImplicitGeometrySlot, pyuipc.geometry.ImplicitGeometrySlot]:
            """create(*args, **kwargs)
            Overloaded function.

            1. create(self: pyuipc.core.Object.Geometries, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot]

            Create geometry from a simplicial complex (rest geometry is auto-generated).
            Args:
                simplicial_complex: SimplicialComplex to create geometry from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            2. create(self: pyuipc.core.Object.Geometries, simplicial_complex: pyuipc.geometry.SimplicialComplex, rest_simplicial_complex: pyuipc.geometry.SimplicialComplex) -> tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot]

            Create geometry from a simplicial complex with explicit rest geometry.
            Args:
                simplicial_complex: SimplicialComplex to create geometry from.
                rest_simplicial_complex: Rest state simplicial complex.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            3. create(self: pyuipc.core.Object.Geometries, implicit_geometry: pyuipc.geometry.ImplicitGeometry) -> tuple[pyuipc.geometry.ImplicitGeometrySlot, pyuipc.geometry.ImplicitGeometrySlot]

            Create geometry from an implicit geometry (rest geometry is auto-generated).
            Args:
                implicit_geometry: ImplicitGeometry to create geometry from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            4. create(self: pyuipc.core.Object.Geometries, implicit_geometry: pyuipc.geometry.ImplicitGeometry, rest_implicit_geometry: pyuipc.geometry.ImplicitGeometry) -> tuple[pyuipc.geometry.ImplicitGeometrySlot, pyuipc.geometry.ImplicitGeometrySlot]

            Create geometry from an implicit geometry with explicit rest geometry.
            Args:
                implicit_geometry: ImplicitGeometry to create geometry from.
                rest_implicit_geometry: Rest state implicit geometry.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            5. create(self: pyuipc.core.Object.Geometries, geometry: pyuipc.geometry.Geometry) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]

            Create geometry from an existing geometry (rest geometry is auto-generated).
            Args:
                geometry: Geometry to create from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            6. create(self: pyuipc.core.Object.Geometries, geometry: pyuipc.geometry.Geometry, rest_geometry: pyuipc.geometry.Geometry) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]

            Create geometry from an existing geometry with explicit rest geometry.
            Args:
                geometry: Geometry to create from.
                rest_geometry: Rest state geometry.
            Returns:
                tuple: Pair of (geometry, rest_geometry).
            """
        @overload
        def create(self, implicit_geometry: pyuipc.geometry.ImplicitGeometry, rest_implicit_geometry: pyuipc.geometry.ImplicitGeometry) -> tuple[pyuipc.geometry.ImplicitGeometrySlot, pyuipc.geometry.ImplicitGeometrySlot]:
            """create(*args, **kwargs)
            Overloaded function.

            1. create(self: pyuipc.core.Object.Geometries, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot]

            Create geometry from a simplicial complex (rest geometry is auto-generated).
            Args:
                simplicial_complex: SimplicialComplex to create geometry from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            2. create(self: pyuipc.core.Object.Geometries, simplicial_complex: pyuipc.geometry.SimplicialComplex, rest_simplicial_complex: pyuipc.geometry.SimplicialComplex) -> tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot]

            Create geometry from a simplicial complex with explicit rest geometry.
            Args:
                simplicial_complex: SimplicialComplex to create geometry from.
                rest_simplicial_complex: Rest state simplicial complex.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            3. create(self: pyuipc.core.Object.Geometries, implicit_geometry: pyuipc.geometry.ImplicitGeometry) -> tuple[pyuipc.geometry.ImplicitGeometrySlot, pyuipc.geometry.ImplicitGeometrySlot]

            Create geometry from an implicit geometry (rest geometry is auto-generated).
            Args:
                implicit_geometry: ImplicitGeometry to create geometry from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            4. create(self: pyuipc.core.Object.Geometries, implicit_geometry: pyuipc.geometry.ImplicitGeometry, rest_implicit_geometry: pyuipc.geometry.ImplicitGeometry) -> tuple[pyuipc.geometry.ImplicitGeometrySlot, pyuipc.geometry.ImplicitGeometrySlot]

            Create geometry from an implicit geometry with explicit rest geometry.
            Args:
                implicit_geometry: ImplicitGeometry to create geometry from.
                rest_implicit_geometry: Rest state implicit geometry.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            5. create(self: pyuipc.core.Object.Geometries, geometry: pyuipc.geometry.Geometry) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]

            Create geometry from an existing geometry (rest geometry is auto-generated).
            Args:
                geometry: Geometry to create from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            6. create(self: pyuipc.core.Object.Geometries, geometry: pyuipc.geometry.Geometry, rest_geometry: pyuipc.geometry.Geometry) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]

            Create geometry from an existing geometry with explicit rest geometry.
            Args:
                geometry: Geometry to create from.
                rest_geometry: Rest state geometry.
            Returns:
                tuple: Pair of (geometry, rest_geometry).
            """
        @overload
        def create(self, geometry: pyuipc.geometry.Geometry) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]:
            """create(*args, **kwargs)
            Overloaded function.

            1. create(self: pyuipc.core.Object.Geometries, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot]

            Create geometry from a simplicial complex (rest geometry is auto-generated).
            Args:
                simplicial_complex: SimplicialComplex to create geometry from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            2. create(self: pyuipc.core.Object.Geometries, simplicial_complex: pyuipc.geometry.SimplicialComplex, rest_simplicial_complex: pyuipc.geometry.SimplicialComplex) -> tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot]

            Create geometry from a simplicial complex with explicit rest geometry.
            Args:
                simplicial_complex: SimplicialComplex to create geometry from.
                rest_simplicial_complex: Rest state simplicial complex.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            3. create(self: pyuipc.core.Object.Geometries, implicit_geometry: pyuipc.geometry.ImplicitGeometry) -> tuple[pyuipc.geometry.ImplicitGeometrySlot, pyuipc.geometry.ImplicitGeometrySlot]

            Create geometry from an implicit geometry (rest geometry is auto-generated).
            Args:
                implicit_geometry: ImplicitGeometry to create geometry from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            4. create(self: pyuipc.core.Object.Geometries, implicit_geometry: pyuipc.geometry.ImplicitGeometry, rest_implicit_geometry: pyuipc.geometry.ImplicitGeometry) -> tuple[pyuipc.geometry.ImplicitGeometrySlot, pyuipc.geometry.ImplicitGeometrySlot]

            Create geometry from an implicit geometry with explicit rest geometry.
            Args:
                implicit_geometry: ImplicitGeometry to create geometry from.
                rest_implicit_geometry: Rest state implicit geometry.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            5. create(self: pyuipc.core.Object.Geometries, geometry: pyuipc.geometry.Geometry) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]

            Create geometry from an existing geometry (rest geometry is auto-generated).
            Args:
                geometry: Geometry to create from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            6. create(self: pyuipc.core.Object.Geometries, geometry: pyuipc.geometry.Geometry, rest_geometry: pyuipc.geometry.Geometry) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]

            Create geometry from an existing geometry with explicit rest geometry.
            Args:
                geometry: Geometry to create from.
                rest_geometry: Rest state geometry.
            Returns:
                tuple: Pair of (geometry, rest_geometry).
            """
        @overload
        def create(self, geometry: pyuipc.geometry.Geometry, rest_geometry: pyuipc.geometry.Geometry) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]:
            """create(*args, **kwargs)
            Overloaded function.

            1. create(self: pyuipc.core.Object.Geometries, simplicial_complex: pyuipc.geometry.SimplicialComplex) -> tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot]

            Create geometry from a simplicial complex (rest geometry is auto-generated).
            Args:
                simplicial_complex: SimplicialComplex to create geometry from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            2. create(self: pyuipc.core.Object.Geometries, simplicial_complex: pyuipc.geometry.SimplicialComplex, rest_simplicial_complex: pyuipc.geometry.SimplicialComplex) -> tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot]

            Create geometry from a simplicial complex with explicit rest geometry.
            Args:
                simplicial_complex: SimplicialComplex to create geometry from.
                rest_simplicial_complex: Rest state simplicial complex.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            3. create(self: pyuipc.core.Object.Geometries, implicit_geometry: pyuipc.geometry.ImplicitGeometry) -> tuple[pyuipc.geometry.ImplicitGeometrySlot, pyuipc.geometry.ImplicitGeometrySlot]

            Create geometry from an implicit geometry (rest geometry is auto-generated).
            Args:
                implicit_geometry: ImplicitGeometry to create geometry from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            4. create(self: pyuipc.core.Object.Geometries, implicit_geometry: pyuipc.geometry.ImplicitGeometry, rest_implicit_geometry: pyuipc.geometry.ImplicitGeometry) -> tuple[pyuipc.geometry.ImplicitGeometrySlot, pyuipc.geometry.ImplicitGeometrySlot]

            Create geometry from an implicit geometry with explicit rest geometry.
            Args:
                implicit_geometry: ImplicitGeometry to create geometry from.
                rest_implicit_geometry: Rest state implicit geometry.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            5. create(self: pyuipc.core.Object.Geometries, geometry: pyuipc.geometry.Geometry) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]

            Create geometry from an existing geometry (rest geometry is auto-generated).
            Args:
                geometry: Geometry to create from.
            Returns:
                tuple: Pair of (geometry, rest_geometry).

            6. create(self: pyuipc.core.Object.Geometries, geometry: pyuipc.geometry.Geometry, rest_geometry: pyuipc.geometry.Geometry) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]

            Create geometry from an existing geometry with explicit rest geometry.
            Args:
                geometry: Geometry to create from.
                rest_geometry: Rest state geometry.
            Returns:
                tuple: Pair of (geometry, rest_geometry).
            """
        def ids(self) -> numpy.typing.NDArray[numpy.int32]:
            """ids(self: pyuipc.core.Object.Geometries) -> numpy.typing.NDArray[numpy.int32]

            Get the IDs of all geometries in the collection.
            Returns:
                numpy.ndarray: Array of geometry IDs.
            """
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def geometries(self) -> Object.Geometries:
        """geometries(self: pyuipc.core.Object) -> pyuipc.core.Object.Geometries

        Get the geometries collection.
        Returns:
            Geometries: Collection of geometries.
        """
    def id(self) -> int:
        """id(self: pyuipc.core.Object) -> int

        Get the object ID.
        Returns:
            int: Object ID.
        """
    def name(self) -> str:
        """name(self: pyuipc.core.Object) -> str

        Get the object name.
        Returns:
            str: Object name.
        """

class PyIEngine(IEngine):
    def __init__(self) -> None:
        """__init__(self: pyuipc.core.PyIEngine) -> None

        Create a new PyIEngine instance.
        """
    def do_advance(self) -> None:
        """do_advance(self: pyuipc.core.PyIEngine) -> None

        Advance the simulation by one step.
        """
    def do_dump(self) -> bool:
        """do_dump(self: pyuipc.core.PyIEngine) -> bool

        Dump engine state.
        """
    def do_init(self) -> None:
        """do_init(self: pyuipc.core.PyIEngine) -> None

        Initialize the engine.
        """
    def do_recover(self, dst_frame: typing.SupportsInt) -> bool:
        """do_recover(self: pyuipc.core.PyIEngine, dst_frame: typing.SupportsInt) -> bool

        Recover engine state to a specific frame.
        Args:
            dst_frame: Target frame number.
        """
    def do_retrieve(self) -> None:
        """do_retrieve(self: pyuipc.core.PyIEngine) -> None

        Retrieve engine state.
        """
    def do_sync(self) -> None:
        """do_sync(self: pyuipc.core.PyIEngine) -> None

        Synchronize engine state.
        """
    def do_to_json(self) -> json:
        """do_to_json(self: pyuipc.core.PyIEngine) -> json

        Convert engine state to JSON.
        Returns:
            dict: JSON representation of engine state.
        """
    def features(self) -> FeatureCollection:
        """features(self: pyuipc.core.PyIEngine) -> pyuipc.core.FeatureCollection

        Get engine features.
        Returns:
            FeatureCollection: Reference to feature collection.
        """
    def get_frame(self) -> int:
        """get_frame(self: pyuipc.core.PyIEngine) -> int

        Get the current frame number.
        Returns:
            int: Current frame number.
        """
    def status(self) -> EngineStatusCollection:
        """status(self: pyuipc.core.PyIEngine) -> pyuipc.core.EngineStatusCollection

        Get engine status collection.
        Returns:
            EngineStatusCollection: Reference to status collection.
        """
    def world(self, *args, **kwargs):
        """world(self: pyuipc.core.PyIEngine) -> uipc::core::World

        Get the world.
        Returns:
            World: Reference to the world.
        """

class SanityCheckMessage:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def geometries(self) -> dict[str, pyuipc.geometry.Geometry]:
        """geometries(self: pyuipc.core.SanityCheckMessage) -> dict[str, pyuipc.geometry.Geometry]

        Get the list of geometry IDs associated with this message.
        Returns:
            list: List of geometry IDs.
        """
    def id(self) -> int:
        """id(self: pyuipc.core.SanityCheckMessage) -> int

        Get the message ID.
        Returns:
            int: Message ID.
        """
    def is_empty(self) -> bool:
        """is_empty(self: pyuipc.core.SanityCheckMessage) -> bool

        Check if the message is empty.
        Returns:
            bool: True if empty, False otherwise.
        """
    def message(self) -> str:
        """message(self: pyuipc.core.SanityCheckMessage) -> str

        Get the message text.
        Returns:
            str: Message text.
        """
    def name(self) -> str:
        """name(self: pyuipc.core.SanityCheckMessage) -> str

        Get the message name.
        Returns:
            str: Message name.
        """
    def result(self) -> SanityCheckResult:
        """result(self: pyuipc.core.SanityCheckMessage) -> pyuipc.core.SanityCheckResult

        Get the check result.
        Returns:
            SanityCheckResult: Result of the check.
        """

class SanityCheckResult:
    __members__: ClassVar[dict] = ...  # read-only
    Error: ClassVar[SanityCheckResult] = ...
    Success: ClassVar[SanityCheckResult] = ...
    Warning: ClassVar[SanityCheckResult] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: typing.SupportsInt) -> None:
        """__init__(self: pyuipc.core.SanityCheckResult, value: typing.SupportsInt) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object, /) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object, /) -> int"""
    def __index__(self) -> int:
        """__index__(self: pyuipc.core.SanityCheckResult, /) -> int"""
    def __int__(self) -> int:
        """__int__(self: pyuipc.core.SanityCheckResult, /) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object, /) -> bool"""
    @property
    def name(self): ...
    @property
    def value(self) -> int: ...

class SanityChecker:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def check(self) -> SanityCheckResult:
        """check(self: pyuipc.core.SanityChecker) -> pyuipc.core.SanityCheckResult"""
    def clear(self) -> None:
        """clear(self: pyuipc.core.SanityChecker) -> None

        Clear all sanity check messages.
        """
    def errors(self) -> dict[int, SanityCheckMessage]:
        """errors(self: pyuipc.core.SanityChecker) -> dict[int, pyuipc.core.SanityCheckMessage]

        Get all error messages.
        Returns:
            list: List of error SanityCheckMessage objects.
        """
    def infos(self) -> dict[int, SanityCheckMessage]:
        """infos(self: pyuipc.core.SanityChecker) -> dict[int, pyuipc.core.SanityCheckMessage]

        Get all info messages.
        Returns:
            list: List of info SanityCheckMessage objects.
        """
    def report(self) -> None:
        """report(self: pyuipc.core.SanityChecker) -> None

        Get a report of all sanity check messages.
        Returns:
            list: List of SanityCheckMessage objects.
        """
    def warns(self) -> dict[int, SanityCheckMessage]:
        """warns(self: pyuipc.core.SanityChecker) -> dict[int, pyuipc.core.SanityCheckMessage]

        Get all warning messages.
        Returns:
            list: List of warning SanityCheckMessage objects.
        """

class Scene:
    class ConfigAttributes:
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def create(self, name: str, object: object) -> pyuipc.geometry.IAttributeSlot:
            """create(self: pyuipc.core.Scene.ConfigAttributes, name: str, object: object) -> pyuipc.geometry.IAttributeSlot

            Create a new attribute from a Python object.
            Args:
                name: Name for the new attribute.
                object: Python object to create attribute from (can be scalar, array, or numpy array).
            Returns:
                AttributeSlot: The created attribute slot.
            """
        def destroy(self, name: str) -> None:
            """destroy(self: pyuipc.core.Scene.ConfigAttributes, name: str) -> None

            Destroy an attribute by name.
            Args:
                name: Name of the attribute to destroy.
            """
        def find(self, name: str) -> pyuipc.geometry.IAttributeSlot:
            """find(self: pyuipc.core.Scene.ConfigAttributes, name: str) -> pyuipc.geometry.IAttributeSlot

            Find an attribute by name.
            Args:
                name: Name of the attribute to find.
            Returns:
                AttributeSlot or None: The attribute slot if found, None otherwise.
            """
        def share(self, name: str, attribute: pyuipc.geometry.IAttributeSlot) -> None:
            """share(self: pyuipc.core.Scene.ConfigAttributes, name: str, attribute: pyuipc.geometry.IAttributeSlot) -> None

            Share an existing attribute slot with a new name.
            Args:
                name: New name for the shared attribute.
                attribute: Attribute slot to share.
            """
        def to_json(self) -> json:
            """to_json(self: pyuipc.core.Scene.ConfigAttributes) -> json

            Convert attributes to JSON representation.
            Returns:
                dict: JSON dictionary representation of the attributes.
            """

    class Geometries:
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def find(self, id: typing.SupportsInt) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]:
            """find(self: pyuipc.core.Scene.Geometries, id: typing.SupportsInt) -> tuple[pyuipc.geometry.GeometrySlot, pyuipc.geometry.GeometrySlot]

            Find geometry and rest geometry by ID.
            Args:
                id: Geometry ID.
            Returns:
                tuple: Pair of (geometry, rest_geometry).
            """

    class Objects:
        def __init__(self, *args, **kwargs) -> None:
            """Initialize self.  See help(type(self)) for accurate signature."""
        def create(self, name: str) -> Object:
            """create(self: pyuipc.core.Scene.Objects, name: str) -> pyuipc.core.Object

            Create a new object with the given name.
            Args:
                name: Name of the object to create.
            Returns:
                Object: The created object.
            """
        def destroy(self, id: typing.SupportsInt) -> None:
            """destroy(self: pyuipc.core.Scene.Objects, id: typing.SupportsInt) -> None

            Destroy an object by ID.
            Args:
                id: Object ID to destroy.
            """
        @overload
        def find(self, id: typing.SupportsInt) -> Object:
            """find(*args, **kwargs)
            Overloaded function.

            1. find(self: pyuipc.core.Scene.Objects, id: typing.SupportsInt) -> pyuipc.core.Object

            Find an object by ID.
            Args:
                id: Object ID.
            Returns:
                Object or None: The object if found, None otherwise.

            2. find(self: pyuipc.core.Scene.Objects, name: str) -> list

            Find all objects with the given name.
            Args:
                name: Object name.
            Returns:
                list: List of objects with the given name.
            """
        @overload
        def find(self, name: str) -> list:
            """find(*args, **kwargs)
            Overloaded function.

            1. find(self: pyuipc.core.Scene.Objects, id: typing.SupportsInt) -> pyuipc.core.Object

            Find an object by ID.
            Args:
                id: Object ID.
            Returns:
                Object or None: The object if found, None otherwise.

            2. find(self: pyuipc.core.Scene.Objects, name: str) -> list

            Find all objects with the given name.
            Args:
                name: Object name.
            Returns:
                list: List of objects with the given name.
            """
        def size(self) -> int:
            """size(self: pyuipc.core.Scene.Objects) -> int

            Get the number of objects in the collection.
            Returns:
                int: Number of objects.
            """
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.core.Scene, config: json = {'cfl': {'enable': 0}, 'collision_detection': {'method': 'linear_bvh'}, 'contact': {'constitution': 'ipc', 'd_hat': 0.01, 'enable': 1, 'eps_velocity': 0.01, 'friction': {'enable': 1}}, 'diff_sim': {'enable': 0}, 'dt': 0.01, 'extras': {'debug': {'dump_surface': 0}, 'strict_mode': {'enable': 0}}, 'gravity': [[0.0], [-9.8], [0.0]], 'integrator': {'type': 'bdf1'}, 'line_search': {'max_iter': 8, 'report_energy': 0}, 'linear_system': {'solver': 'linear_pcg', 'tol_rate': 0.001}, 'newton': {'ccd_tol': 1.0, 'max_iter': 1024, 'min_iter': 1, 'transrate_tol': 0.1, 'use_adaptive_tol': 0, 'velocity_tol': 0.05}, 'sanity_check': {'enable': 1, 'mode': 'normal'}}) -> None

        Create a new scene.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def animator(self) -> Animator:
        """animator(self: pyuipc.core.Scene) -> pyuipc.core.Animator

        Get the animator for the scene.
        Returns:
            Animator: Reference to the scene animator.
        """
    def config(self) -> Scene.ConfigAttributes:
        """config(self: pyuipc.core.Scene) -> pyuipc.core.Scene.ConfigAttributes

        Get the scene configuration.
        Returns:
            dict: Configuration dictionary.
        """
    def constitution_tabular(self) -> ConstitutionTabular:
        """constitution_tabular(self: pyuipc.core.Scene) -> pyuipc.core.ConstitutionTabular

        Get the constitution tabular (constitution configuration).
        Returns:
            ConstitutionTabular: Reference to the constitution tabular.
        """
    def contact_tabular(self) -> ContactTabular:
        """contact_tabular(self: pyuipc.core.Scene) -> pyuipc.core.ContactTabular

        Get the contact tabular (contact system configuration).
        Returns:
            ContactTabular: Reference to the contact tabular.
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default scene configuration.
        Returns:
            dict: Default configuration dictionary.
        """
    def diff_sim(self) -> DiffSim:
        """diff_sim(self: pyuipc.core.Scene) -> pyuipc.core.DiffSim

        Get the differential simulator for the scene.
        Returns:
            DiffSim: Reference to the differential simulator.
        """
    def geometries(self) -> Scene.Geometries:
        """geometries(self: pyuipc.core.Scene) -> pyuipc.core.Scene.Geometries

        Get the geometries collection.
        Returns:
            Geometries: Collection of geometries in the scene.
        """
    def objects(self) -> Scene.Objects:
        """objects(self: pyuipc.core.Scene) -> pyuipc.core.Scene.Objects

        Get the objects collection.
        Returns:
            Objects: Collection of objects in the scene.
        """
    def subscene_tabular(self) -> SubsceneTabular:
        """subscene_tabular(self: pyuipc.core.Scene) -> pyuipc.core.SubsceneTabular

        Get the subscene tabular (subscene configuration).
        Returns:
            SubsceneTabular: Reference to the subscene tabular.
        """

class SceneFactory:
    def __init__(self) -> None:
        """__init__(self: pyuipc.core.SceneFactory) -> None

        Create a new SceneFactory instance.
        """
    def commit_from_json(self, json: json) -> SceneSnapshotCommit:
        """commit_from_json(self: pyuipc.core.SceneFactory, json: json) -> pyuipc.core.SceneSnapshotCommit

        Apply commit changes from JSON.
        Args:
            json: JSON dictionary containing commit changes.
        """
    def commit_to_json(self, reference: SceneSnapshotCommit) -> json:
        """commit_to_json(self: pyuipc.core.SceneFactory, reference: pyuipc.core.SceneSnapshotCommit) -> json

        Generate commit JSON from a reference snapshot.
        Args:
            reference: Reference SceneSnapshot.
        Returns:
            dict: JSON dictionary containing commit changes.
        """
    def from_json(self, json: json) -> SceneSnapshot:
        """from_json(self: pyuipc.core.SceneFactory, json: json) -> pyuipc.core.SceneSnapshot

        Create a scene from JSON.
        Args:
            json: JSON dictionary representing the scene.
        Returns:
            Scene: Created scene.
        """
    def from_snapshot(self, snapshot: SceneSnapshot) -> Scene:
        """from_snapshot(self: pyuipc.core.SceneFactory, snapshot: pyuipc.core.SceneSnapshot) -> pyuipc.core.Scene

        Create a scene from a scene snapshot.
        Args:
            snapshot: SceneSnapshot to create scene from.
        Returns:
            Scene: Created scene.
        """
    def to_json(self, arg0: SceneSnapshot) -> json:
        """to_json(self: pyuipc.core.SceneFactory, arg0: pyuipc.core.SceneSnapshot) -> json

        Convert factory state to JSON.
        Returns:
            dict: JSON representation of the factory state.
        """

class SceneIO:
    def __init__(self, scene: Scene) -> None:
        """__init__(self: pyuipc.core.SceneIO, scene: pyuipc.core.Scene) -> None

        Create a SceneIO instance for a scene.
        Args:
            scene: Scene to perform I/O operations on.
        """
    def commit(self, last: SceneSnapshot, name: str) -> None:
        """commit(self: pyuipc.core.SceneIO, last: pyuipc.core.SceneSnapshot, name: str) -> None

        Commit scene changes to a file.
        Args:
            last: Last scene snapshot for comparison.
            name: Output file path.
        Returns:
            SceneSnapshot: New scene snapshot after commit.
        """
    def commit_to_json(self, reference: SceneSnapshot) -> json:
        """commit_to_json(self: pyuipc.core.SceneIO, reference: pyuipc.core.SceneSnapshot) -> json

        Commit scene changes to JSON.
        Args:
            reference: Reference scene snapshot for comparison.
        Returns:
            dict: JSON dictionary containing committed changes.
        """
    @staticmethod
    def from_json(json: json) -> Scene:
        """from_json(json: json) -> pyuipc.core.Scene

        Create a scene from JSON.
        Args:
            json: JSON dictionary representing the scene.
        Returns:
            Scene: Scene created from JSON.
        """
    @staticmethod
    def load(filename: str) -> Scene:
        """load(filename: str) -> pyuipc.core.Scene

        Load a scene from a file.
        Args:
            filename: Input file path.
        Returns:
            Scene: Loaded scene.
        """
    def save(self, filename: str) -> None:
        """save(self: pyuipc.core.SceneIO, filename: str) -> None

        Save the scene to a file.
        Args:
            filename: Output file path.
        """
    def simplicial_surface(self, dim: typing.SupportsInt = ...) -> pyuipc.geometry.SimplicialComplex:
        """simplicial_surface(self: pyuipc.core.SceneIO, dim: typing.SupportsInt = -1) -> pyuipc.geometry.SimplicialComplex

        Get simplicial surface geometry.
        Args:
            dim: Dimension of simplices to extract (-1 for all dimensions).
        Returns:
            SimplicialComplex: Simplicial surface geometry.
        """
    def to_json(self) -> json:
        """to_json(self: pyuipc.core.SceneIO) -> json

        Convert scene to JSON representation.
        Returns:
            dict: JSON representation of the scene.
        """
    def update(self, filename: str) -> None:
        """update(self: pyuipc.core.SceneIO, filename: str) -> None

        Update scene from a file.
        Args:
            filename: Input file path.
        Returns:
            SceneSnapshot: Scene snapshot after update.
        """
    def update_from_json(self, commit_json: json) -> None:
        """update_from_json(self: pyuipc.core.SceneIO, commit_json: json) -> None

        Update scene from JSON.
        Args:
            commit_json: JSON dictionary containing scene updates.
        Returns:
            SceneSnapshot: Scene snapshot after update.
        """
    def write_surface(self, filename: str) -> None:
        """write_surface(self: pyuipc.core.SceneIO, filename: str) -> None

        Write surface geometry to a file.
        Args:
            filename: Output file path.
        """

class SceneSnapshot:
    def __init__(self, scene: Scene) -> None:
        """__init__(self: pyuipc.core.SceneSnapshot, scene: pyuipc.core.Scene) -> None

        Create a snapshot from a scene.
        Args:
            scene: Scene to snapshot.
        """
    def __sub__(self, other: SceneSnapshot) -> SceneSnapshotCommit:
        """__sub__(self: pyuipc.core.SceneSnapshot, other: pyuipc.core.SceneSnapshot) -> pyuipc.core.SceneSnapshotCommit

        Compute the difference between two snapshots (commit).
        Args:
            other: Other snapshot to compare with.
        Returns:
            SceneSnapshotCommit: Commit representing the difference.
        """

class SceneSnapshotCommit:
    def __init__(self, dst: SceneSnapshot, src: SceneSnapshot) -> None:
        """__init__(self: pyuipc.core.SceneSnapshotCommit, dst: pyuipc.core.SceneSnapshot, src: pyuipc.core.SceneSnapshot) -> None

        Create a commit from two snapshots.
        Args:
            dst: Destination snapshot.
            src: Source snapshot.
        """

class SubsceneElement:
    def __init__(self, id: typing.SupportsInt, name: str) -> None:
        """__init__(self: pyuipc.core.SubsceneElement, id: typing.SupportsInt, name: str) -> None

        Create a subscene element.
        Args:
            id: Element ID.
            name: Element name.
        """
    def apply_to(self, object: pyuipc.geometry.Geometry) -> pyuipc.geometry.AttributeSlotI32:
        """apply_to(self: pyuipc.core.SubsceneElement, object: pyuipc.geometry.Geometry) -> pyuipc.geometry.AttributeSlotI32

        Apply this subscene element to an object.
        Args:
            object: Object to apply to.
        """
    def id(self) -> int:
        """id(self: pyuipc.core.SubsceneElement) -> int

        Get the element ID.
        Returns:
            int: Element ID.
        """
    def name(self) -> str:
        """name(self: pyuipc.core.SubsceneElement) -> str

        Get the element name.
        Returns:
            str: Element name.
        """

class SubsceneModel:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def config(self) -> json:
        """config(self: pyuipc.core.SubsceneModel) -> json

        Get the configuration dictionary.
        Returns:
            dict: Configuration dictionary.
        """
    def is_enabled(self) -> bool:
        """is_enabled(self: pyuipc.core.SubsceneModel) -> bool

        Check if the subscene model is enabled.
        Returns:
            bool: True if enabled, False otherwise.
        """
    def topo(self, *args, **kwargs):
        """topo(self: pyuipc.core.SubsceneModel) -> Eigen::Matrix<int,2,1,0,2,1>

        Get the topology type.
        Returns:
            str: Topology type string.
        """

class SubsceneModelCollection:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def create(self, name: str, default_value: object) -> pyuipc.geometry.IAttributeSlot:
        """create(self: pyuipc.core.SubsceneModelCollection, name: str, default_value: object) -> pyuipc.geometry.IAttributeSlot

        Create a subscene model attribute.
        Args:
            name: Attribute name.
            default_value: Default value for the attribute.
        Returns:
            AttributeSlot: Created attribute slot.
        """
    def find(self, name: str) -> pyuipc.geometry.IAttributeSlot:
        """find(self: pyuipc.core.SubsceneModelCollection, name: str) -> pyuipc.geometry.IAttributeSlot

        Find a subscene model attribute by name.
        Args:
            name: Attribute name.
        Returns:
            AttributeSlot or None: Attribute slot if found, None otherwise.
        """

class SubsceneTabular:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def at(self, i: typing.SupportsInt, j: typing.SupportsInt) -> SubsceneModel:
        """at(self: pyuipc.core.SubsceneTabular, i: typing.SupportsInt, j: typing.SupportsInt) -> pyuipc.core.SubsceneModel

        Get the subscene model between two elements.
        Args:
            i: First element ID.
            j: Second element ID.
        Returns:
            SubsceneModel: Subscene model between the two elements.
        """
    def create(self, name: str = ...) -> SubsceneElement:
        """create(self: pyuipc.core.SubsceneTabular, name: str = '') -> pyuipc.core.SubsceneElement

        Create a new subscene element.
        Args:
            name: Element name (optional).
        Returns:
            SubsceneElement: Created subscene element.
        """
    def default_element(self) -> SubsceneElement:
        """default_element(self: pyuipc.core.SubsceneTabular) -> pyuipc.core.SubsceneElement

        Get the default subscene element.
        Returns:
            SubsceneElement: Reference to default element.
        """
    def element_count(self) -> int:
        """element_count(self: pyuipc.core.SubsceneTabular) -> int

        Get the number of subscene elements.
        Returns:
            int: Number of elements.
        """
    def insert(self, L: SubsceneElement, R: SubsceneElement, enable: bool = ..., config: json = ...) -> int:
        """insert(self: pyuipc.core.SubsceneTabular, L: pyuipc.core.SubsceneElement, R: pyuipc.core.SubsceneElement, enable: bool = False, config: json = {}) -> int

        Insert a subscene model between two elements.
        Args:
            L: Left element ID.
            R: Right element ID.
            enable: Whether the subscene is enabled (default: False).
            config: Additional configuration dictionary (default: empty).
        """
    def subscene_models(self) -> SubsceneModelCollection:
        """subscene_models(self: pyuipc.core.SubsceneTabular) -> pyuipc.core.SubsceneModelCollection

        Get the subscene model collection.
        Returns:
            SubsceneModelCollection: Collection of subscene models.
        """

class World:
    def __init__(self, engine: Engine) -> None:
        """__init__(self: pyuipc.core.World, engine: pyuipc.core.Engine) -> None

        Create a world with an engine.
        Args:
            engine: Engine instance to use.
        """
    def advance(self) -> None:
        """advance(self: pyuipc.core.World) -> None

        Advance the simulation by one step.
        """
    def dump(self) -> bool:
        """dump(self: pyuipc.core.World) -> bool

        Dump the world state.
        """
    def features(self) -> FeatureCollection:
        """features(self: pyuipc.core.World) -> pyuipc.core.FeatureCollection

        Get the feature collection.
        Returns:
            FeatureCollection: Reference to feature collection.
        """
    def frame(self) -> int:
        """frame(self: pyuipc.core.World) -> int

        Get the current frame number.
        Returns:
            int: Current frame number.
        """
    def init(self, scene: Scene) -> None:
        """init(self: pyuipc.core.World, scene: pyuipc.core.Scene) -> None

        Initialize the world with a scene.
        Args:
            scene: Scene to initialize the world with.
        """
    def is_valid(self) -> bool:
        """is_valid(self: pyuipc.core.World) -> bool

        Check if the world is in a valid state.
        Returns:
            bool: True if valid, False otherwise.
        """
    def recover(self, dst_frame: typing.SupportsInt = ...) -> bool:
        """recover(self: pyuipc.core.World, dst_frame: typing.SupportsInt = 18446744073709551615) -> bool

        Recover the world state to a specific frame.
        Args:
            dst_frame: Target frame number (default: maximum frame).
        """
    def retrieve(self) -> None:
        """retrieve(self: pyuipc.core.World) -> None

        Retrieve the world state.
        """
    def sanity_checker(self) -> SanityChecker:
        """sanity_checker(self: pyuipc.core.World) -> pyuipc.core.SanityChecker

        Get the sanity checker.
        Returns:
            SanityChecker: Reference to sanity checker.
        """
    def sync(self) -> None:
        """sync(self: pyuipc.core.World) -> None

        Synchronize the world state.
        """
