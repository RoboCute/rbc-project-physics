import numpy
import numpy.typing
import pyuipc.geometry
import typing
from typing import overload

class ARAP(FiniteElementConstitution):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.ARAP, config: json = {}) -> None

        Create an ARAP constitution.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, kappa: typing.SupportsFloat = ..., mass_density: typing.SupportsFloat = ...) -> None:
        """apply_to(self: pyuipc.constitution.ARAP, sc: pyuipc.geometry.SimplicialComplex, kappa: typing.SupportsFloat = 1000000.0, mass_density: typing.SupportsFloat = 1000.0) -> None

        Apply ARAP constitution to a simplicial complex.
        Args:
            sc: SimplicialComplex to apply to.
            kappa: Stiffness parameter in MPa (default: 1.0 MPa).
            mass_density: Mass density (default: 1000.0).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default ARAP configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class AffineBodyConstitution(IConstitution):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.AffineBodyConstitution, config: json = {'name': 'OrthoPotential'}) -> None

        Create an AffineBodyConstitution.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, kappa: typing.SupportsFloat, mass_density: typing.SupportsFloat = ...) -> None:
        """apply_to(self: pyuipc.constitution.AffineBodyConstitution, sc: pyuipc.geometry.SimplicialComplex, kappa: typing.SupportsFloat, mass_density: typing.SupportsFloat = 1000.0) -> None

        Apply AffineBodyConstitution to a simplicial complex.
        Args:
            sc: SimplicialComplex to apply to.
            kappa: Stiffness parameter.
            mass_density: Mass density (default: 1000.0).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default AffineBodyConstitution configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class AffineBodyExternalBodyForce(IConstitution):
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: pyuipc.constitution.AffineBodyExternalBodyForce) -> None

        2. __init__(self: pyuipc.constitution.AffineBodyExternalBodyForce, config: nlohmann::json_abi_v3_12_0::basic_json<std::map,std::vector,std::basic_string<char,std::char_traits<char>,std::allocator<char> >,bool,__int64,unsigned __int64,double,std::allocator,nlohmann::json_abi_v3_12_0::adl_serializer,std::vector<unsigned char,std::allocator<unsigned char> >,void>) -> None
        """
    @overload
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, force: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, '[12, 1]']) -> None:
        '''apply_to(*args, **kwargs)
        Overloaded function.

        1. apply_to(self: pyuipc.constitution.AffineBodyExternalBodyForce, sc: pyuipc.geometry.SimplicialComplex, force: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[12, 1]"]) -> None

        Apply external force (12D generalized force) to affine body instances.

                     Args:
                         sc: SimplicialComplex representing affine body geometry
                         force: 12D generalized force vector [fx, fy, fz, f_a11, f_a12, ..., f_a33]
                                where f is 3D translational force and f_a is 9D affine force
             

        2. apply_to(self: pyuipc.constitution.AffineBodyExternalBodyForce, sc: pyuipc.geometry.SimplicialComplex, force: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> None

        Apply external translational force to affine body instances.

                     Args:
                         sc: SimplicialComplex representing affine body geometry
                         force: 3D translational force vector (affine force = 0)
             
        '''
    @overload
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, force: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, '[3, 1]']) -> None:
        '''apply_to(*args, **kwargs)
        Overloaded function.

        1. apply_to(self: pyuipc.constitution.AffineBodyExternalBodyForce, sc: pyuipc.geometry.SimplicialComplex, force: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[12, 1]"]) -> None

        Apply external force (12D generalized force) to affine body instances.

                     Args:
                         sc: SimplicialComplex representing affine body geometry
                         force: 12D generalized force vector [fx, fy, fz, f_a11, f_a12, ..., f_a33]
                                where f is 3D translational force and f_a is 9D affine force
             

        2. apply_to(self: pyuipc.constitution.AffineBodyExternalBodyForce, sc: pyuipc.geometry.SimplicialComplex, force: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> None

        Apply external translational force to affine body instances.

                     Args:
                         sc: SimplicialComplex representing affine body geometry
                         force: 3D translational force vector (affine force = 0)
             
        '''
    @staticmethod
    def default_config(*args, **kwargs):
        """default_config() -> nlohmann::json_abi_v3_12_0::basic_json<std::map,std::vector,std::basic_string<char,std::char_traits<char>,std::allocator<char> >,bool,__int64,unsigned __int64,double,std::allocator,nlohmann::json_abi_v3_12_0::adl_serializer,std::vector<unsigned char,std::allocator<unsigned char> >,void>

        Get the default configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class AffineBodyRevoluteJoint(InterAffineBodyConstitution):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.AffineBodyRevoluteJoint, config: json = {}) -> None

        Create an AffineBodyRevoluteJoint.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, geo_slot_tuples: list, strength_ratio: typing.SupportsFloat = ...) -> None:
        """apply_to(self: pyuipc.constitution.AffineBodyRevoluteJoint, sc: pyuipc.geometry.SimplicialComplex, geo_slot_tuples: list, strength_ratio: typing.SupportsFloat = 100.0) -> None

        Create joint between two affine bodies.
        sc: Every edge in the simplicial complex is treated as a joint axis.
        geo_slot_tuples: A list of tuples, each containing two SimplicialComplexSlot objects, telling who are linked by the joint.
        strength_ratio: Stiffness = strength_ratio * (BodyMassA + BodyMassB)
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default AffineBodyRevoluteJoint configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class Constraint(IConstitution):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class DiscreteShellBending(FiniteElementExtraConstitution):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.DiscreteShellBending, config: json = {}) -> None

        Create a DiscreteShellBending constitution.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, bending_stiffness: typing.SupportsFloat = ...) -> None:
        """apply_to(self: pyuipc.constitution.DiscreteShellBending, sc: pyuipc.geometry.SimplicialComplex, bending_stiffness: typing.SupportsFloat = 100000.0) -> None

        Apply DiscreteShellBending constitution to a simplicial complex.
        Args:
            sc: SimplicialComplex to apply to.
            bending_stiffness: Bending stiffness in kPa (default: 100.0 kPa).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default DiscreteShellBending configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class ElasticModuli:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def lame(_lambda: typing.SupportsFloat, mu: typing.SupportsFloat) -> ElasticModuli:
        """lame(lambda: typing.SupportsFloat, mu: typing.SupportsFloat) -> pyuipc.constitution.ElasticModuli

        Create elastic moduli from Lame parameters.
        Args:
            lambda: First Lame parameter.
            mu: Second Lame parameter (shear modulus).
        Returns:
            ElasticModuli: Elastic moduli object.
        """
    def mu(self) -> float:
        """mu(self: pyuipc.constitution.ElasticModuli) -> float

        Get the second Lame parameter (mu, shear modulus).
        Returns:
            float: Second Lame parameter.
        """
    @staticmethod
    def youngs_poisson(E: typing.SupportsFloat, nu: typing.SupportsFloat) -> ElasticModuli:
        """youngs_poisson(E: typing.SupportsFloat, nu: typing.SupportsFloat) -> pyuipc.constitution.ElasticModuli

        Create elastic moduli from Young's modulus and Poisson's ratio.
        Args:
            E: Young's modulus.
            nu: Poisson's ratio.
        Returns:
            ElasticModuli: Elastic moduli object.
        """
    @staticmethod
    def youngs_shear(E: typing.SupportsFloat, G: typing.SupportsFloat) -> ElasticModuli:
        """youngs_shear(E: typing.SupportsFloat, G: typing.SupportsFloat) -> pyuipc.constitution.ElasticModuli

        Create elastic moduli from Young's modulus and shear modulus.
        Args:
            E: Young's modulus.
            G: Shear modulus.
        Returns:
            ElasticModuli: Elastic moduli object.
        """

class Empty(FiniteElementConstitution):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.Empty, config: json = {}) -> None

        Create an Empty constitution.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, mass_density: typing.SupportsFloat = ..., thickness: typing.SupportsFloat = ...) -> None:
        """apply_to(self: pyuipc.constitution.Empty, sc: pyuipc.geometry.SimplicialComplex, mass_density: typing.SupportsFloat = 1000.0, thickness: typing.SupportsFloat = 0.01) -> None

        Apply Empty constitution to a simplicial complex.
        Args:
            sc: SimplicialComplex to apply to.
            mass_density: Mass density (default: 1000.0).
            thickness: Thickness in meters (default: 0.01 m).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default Empty configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class FiniteElementConstitution(IConstitution):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class FiniteElementExtraConstitution(IConstitution):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class HookeanSpring(FiniteElementConstitution):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.HookeanSpring, config: json = {}) -> None

        Create a HookeanSpring constitution.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, moduli: typing.SupportsFloat = ..., mass_density: typing.SupportsFloat = ..., thickness: typing.SupportsFloat = ...) -> None:
        """apply_to(self: pyuipc.constitution.HookeanSpring, sc: pyuipc.geometry.SimplicialComplex, moduli: typing.SupportsFloat = 40000000.0, mass_density: typing.SupportsFloat = 1000.0, thickness: typing.SupportsFloat = 0.01) -> None

        Apply HookeanSpring constitution to a simplicial complex.
        Args:
            sc: SimplicialComplex to apply to.
            moduli: Elastic moduli in MPa (default: 40.0 MPa).
            mass_density: Mass density (default: 1000.0).
            thickness: Thickness in meters (default: 0.01 m).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default HookeanSpring configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class IConstitution:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def name(self) -> str:
        """name(self: pyuipc.constitution.IConstitution) -> str

        Get the constitution name.
        Returns:
            str: Constitution name.
        """
    def type(self) -> str:
        """type(self: pyuipc.constitution.IConstitution) -> str

        Get the constitution type name.
        Returns:
            str: Constitution type name.
        """
    def uid(self) -> int:
        """uid(self: pyuipc.constitution.IConstitution) -> int

        Get the constitution UID (unique identifier).
        Returns:
            int: Constitution UID.
        """

class InterAffineBodyConstitution(IConstitution):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class InterPrimitiveConstitution(IConstitution):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class KirchhoffRodBending(FiniteElementExtraConstitution):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.KirchhoffRodBending, config: json = {}) -> None

        Create a KirchhoffRodBending constitution.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, E: typing.SupportsFloat = ...) -> None:
        """apply_to(self: pyuipc.constitution.KirchhoffRodBending, sc: pyuipc.geometry.SimplicialComplex, E: typing.SupportsFloat = 10000000.0) -> None

        Apply KirchhoffRodBending constitution to a simplicial complex.
        Args:
            sc: SimplicialComplex to apply to.
            E: Young's modulus in MPa (default: 10.0 MPa).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default KirchhoffRodBending configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class LinearMotor(Constraint):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.LinearMotor, config: json = {}) -> None

        Create a LinearMotor.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    @staticmethod
    def animate(sc: pyuipc.geometry.SimplicialComplex, dt: typing.SupportsFloat) -> None:
        """animate(sc: pyuipc.geometry.SimplicialComplex, dt: typing.SupportsFloat) -> None

        Animate the linear motor for a time step.
        Args:
            sc: SimplicialComplex to animate.
            dt: Time step.
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, strength: typing.SupportsFloat = ..., motor_axis: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] = ..., motor_vel: typing.SupportsFloat = ...) -> None:
        """apply_to(self: pyuipc.constitution.LinearMotor, sc: pyuipc.geometry.SimplicialComplex, strength: typing.SupportsFloat = 100.0, motor_axis: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] = array([[-0.], [-0.], [-1.]]), motor_vel: typing.SupportsFloat = 1.0) -> None

        Apply linear motor to a simplicial complex.
        Args:
            sc: SimplicialComplex to apply to.
            strength: Motor strength (default: 100.0).
            motor_axis: Motion direction vector (default: -Z-axis).
            motor_vel: Linear velocity (default: 1.0).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default LinearMotor configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class NeoHookeanShell(FiniteElementConstitution):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.NeoHookeanShell, config: json = {}) -> None

        Create a NeoHookeanShell constitution.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, moduli: ElasticModuli = ..., mass_density: typing.SupportsFloat = ..., thickness: typing.SupportsFloat = ...) -> None:
        """apply_to(self: pyuipc.constitution.NeoHookeanShell, sc: pyuipc.geometry.SimplicialComplex, moduli: pyuipc.constitution.ElasticModuli = <pyuipc.constitution.ElasticModuli object at 0x00000211789A8230>, mass_density: typing.SupportsFloat = 1000.0, thickness: typing.SupportsFloat = 0.01) -> None

        Apply NeoHookeanShell constitution to a simplicial complex.
        Args:
            sc: SimplicialComplex to apply to.
            moduli: Elastic moduli (default: Young's modulus 10.0 MPa, Poisson's ratio 0.49).
            mass_density: Mass density (default: 1000.0).
            thickness: Shell thickness in meters (default: 0.01 m).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default NeoHookeanShell configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class Particle(FiniteElementConstitution):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.Particle, config: json = {}) -> None

        Create a Particle constitution.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, mass_density: typing.SupportsFloat = ..., thickness: typing.SupportsFloat = ...) -> None:
        """apply_to(self: pyuipc.constitution.Particle, sc: pyuipc.geometry.SimplicialComplex, mass_density: typing.SupportsFloat = 1000.0, thickness: typing.SupportsFloat = 0.01) -> None

        Apply Particle constitution to a simplicial complex.
        Args:
            sc: SimplicialComplex to apply to.
            mass_density: Mass density (default: 1000.0).
            thickness: Thickness in meters (default: 0.01 m).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default Particle configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class RotatingMotor(Constraint):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.RotatingMotor, config: json = {}) -> None

        Create a RotatingMotor.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    @staticmethod
    def animate(sc: pyuipc.geometry.SimplicialComplex, dt: typing.SupportsFloat) -> None:
        """animate(sc: pyuipc.geometry.SimplicialComplex, dt: typing.SupportsFloat) -> None

        Animate the rotating motor for a time step.
        Args:
            sc: SimplicialComplex to animate.
            dt: Time step.
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, strength: typing.SupportsFloat = ..., motor_axis: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] = ..., motor_rot_vel: typing.SupportsFloat = ...) -> None:
        """apply_to(self: pyuipc.constitution.RotatingMotor, sc: pyuipc.geometry.SimplicialComplex, strength: typing.SupportsFloat = 100.0, motor_axis: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] = array([[1.], [0.], [0.]]), motor_rot_vel: typing.SupportsFloat = 6.283185307179586) -> None

        Apply rotating motor to a simplicial complex.
        Args:
            sc: SimplicialComplex to apply to.
            strength: Motor strength (default: 100.0).
            motor_axis: Rotation axis vector (default: X-axis).
            motor_rot_vel: Rotational velocity in rad/s (default: 2π).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default RotatingMotor configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class SoftPositionConstraint(Constraint):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.SoftPositionConstraint, config: json = {}) -> None

        Create a SoftPositionConstraint.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, strength_rate: typing.SupportsFloat = ...) -> None:
        """apply_to(self: pyuipc.constitution.SoftPositionConstraint, sc: pyuipc.geometry.SimplicialComplex, strength_rate: typing.SupportsFloat = 100.0) -> None

        Apply soft position constraint to a simplicial complex.
        Args:
            sc: SimplicialComplex to apply to.
            strength_rate: Constraint strength rate (default: 100.0).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default SoftPositionConstraint configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class SoftTransformConstraint(Constraint):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.SoftTransformConstraint, config: json = {}) -> None

        Create a SoftTransformConstraint.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, strength_rate: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] = ...) -> None:
        """apply_to(self: pyuipc.constitution.SoftTransformConstraint, sc: pyuipc.geometry.SimplicialComplex, strength_rate: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] = array([[100.], [100.]])) -> None

        Apply soft transform constraint to a simplicial complex.
        Args:
            sc: SimplicialComplex to apply to.
            strength_rate: 2D vector [translation_strength, rotation_strength] (default: [100.0, 100]).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default SoftTransformConstraint configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class SoftVertexStitch(InterPrimitiveConstitution):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.SoftVertexStitch, config: json = {}) -> None

        Create a SoftVertexStitch constitution.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    @overload
    def create_geometry(self, aim_geo_slots: tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot], stitched_vert_ids: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], kappa: typing.SupportsFloat = ...) -> pyuipc.geometry.Geometry:
        """create_geometry(*args, **kwargs)
        Overloaded function.

        1. create_geometry(self: pyuipc.constitution.SoftVertexStitch, aim_geo_slots: tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot], stitched_vert_ids: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], kappa: typing.SupportsFloat = 1000000.0) -> pyuipc.geometry.Geometry

        Create geometry for vertex stitching.
        Args:
            aim_geo_slots: Tuple of geometry slots to stitch.
            stitched_vert_ids: Array of vertex ID pairs [geometry0_vert, geometry1_vert] to stitch.
            kappa: Stitching stiffness (default: 1e6).
        Returns:
            Geometry: Created stitching geometry.

        2. create_geometry(self: pyuipc.constitution.SoftVertexStitch, aim_geo_slots: tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot], stitched_vert_ids: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], contact_elements: tuple[uipc::core::ContactElement, uipc::core::ContactElement], kappa: typing.SupportsFloat = 1000000.0) -> pyuipc.geometry.Geometry

        Create geometry for vertex stitching with contact elements.
        Args:
            aim_geo_slots: Tuple of geometry slots to stitch.
            stitched_vert_ids: Array of vertex ID pairs [geometry0_vert, geometry1_vert] to stitch.
            contact_elements: Tuple of contact elements.
            kappa: Stitching stiffness (default: 1e6).
        Returns:
            Geometry: Created stitching geometry.
        """
    @overload
    def create_geometry(self, aim_geo_slots: tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot], stitched_vert_ids: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], contact_elements, kappa: typing.SupportsFloat = ...) -> pyuipc.geometry.Geometry:
        """create_geometry(*args, **kwargs)
        Overloaded function.

        1. create_geometry(self: pyuipc.constitution.SoftVertexStitch, aim_geo_slots: tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot], stitched_vert_ids: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], kappa: typing.SupportsFloat = 1000000.0) -> pyuipc.geometry.Geometry

        Create geometry for vertex stitching.
        Args:
            aim_geo_slots: Tuple of geometry slots to stitch.
            stitched_vert_ids: Array of vertex ID pairs [geometry0_vert, geometry1_vert] to stitch.
            kappa: Stitching stiffness (default: 1e6).
        Returns:
            Geometry: Created stitching geometry.

        2. create_geometry(self: pyuipc.constitution.SoftVertexStitch, aim_geo_slots: tuple[pyuipc.geometry.SimplicialComplexSlot, pyuipc.geometry.SimplicialComplexSlot], stitched_vert_ids: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], contact_elements: tuple[uipc::core::ContactElement, uipc::core::ContactElement], kappa: typing.SupportsFloat = 1000000.0) -> pyuipc.geometry.Geometry

        Create geometry for vertex stitching with contact elements.
        Args:
            aim_geo_slots: Tuple of geometry slots to stitch.
            stitched_vert_ids: Array of vertex ID pairs [geometry0_vert, geometry1_vert] to stitch.
            contact_elements: Tuple of contact elements.
            kappa: Stitching stiffness (default: 1e6).
        Returns:
            Geometry: Created stitching geometry.
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default SoftVertexStitch configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class StableNeoHookean(FiniteElementConstitution):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.StableNeoHookean, config: json = {}) -> None

        Create a StableNeoHookean constitution.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, moduli: ElasticModuli = ..., mass_density: typing.SupportsFloat = ...) -> None:
        """apply_to(self: pyuipc.constitution.StableNeoHookean, sc: pyuipc.geometry.SimplicialComplex, moduli: pyuipc.constitution.ElasticModuli = <pyuipc.constitution.ElasticModuli object at 0x00000211789A8370>, mass_density: typing.SupportsFloat = 1000.0) -> None

        Apply StableNeoHookean constitution to a simplicial complex.
        Args:
            sc: SimplicialComplex to apply to.
            moduli: Elastic moduli (default: Young's modulus 20.0 kPa, Poisson's ratio 0.49).
            mass_density: Mass density (default: 1000.0).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default StableNeoHookean configuration.
        Returns:
            dict: Default configuration dictionary.
        """

class StrainLimitingBaraffWitkinShell(FiniteElementConstitution):
    def __init__(self, config: json = ...) -> None:
        """__init__(self: pyuipc.constitution.StrainLimitingBaraffWitkinShell, config: json = {}) -> None

        Create a StrainLimitingBaraffWitkinShell constitution.
        Args:
            config: Configuration dictionary (optional, uses default if not provided).
        """
    def apply_to(self, sc: pyuipc.geometry.SimplicialComplex, moduli: ElasticModuli = ..., mass_density: typing.SupportsFloat = ..., thickness: typing.SupportsFloat = ...) -> None:
        """apply_to(self: pyuipc.constitution.StrainLimitingBaraffWitkinShell, sc: pyuipc.geometry.SimplicialComplex, moduli: pyuipc.constitution.ElasticModuli = <pyuipc.constitution.ElasticModuli object at 0x0000021176314470>, mass_density: typing.SupportsFloat = 200.0, thickness: typing.SupportsFloat = 0.001) -> None

        Apply StrainLimitingBaraffWitkinShell constitution to a simplicial complex.
        Args:
            sc: SimplicialComplex to apply to.
            moduli: Elastic moduli (default: Young's modulus 1.0 MPa, Poisson's ratio 0.49).
            mass_density: Mass density (default: 200.0).
            thickness: Shell thickness in meters (default: 0.001 m).
        """
    @staticmethod
    def default_config() -> json:
        """default_config() -> json

        Get the default StrainLimitingBaraffWitkinShell configuration.
        Returns:
            dict: Default configuration dictionary.
        """
