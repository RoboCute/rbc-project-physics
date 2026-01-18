import typing

aim_position: str
aim_transform: str
backend_abd_body_offset: str
backend_fem_vertex_offset: str
constitution_uid: str
constraint_uids: str
contact_element_id: str
d_hat: str
dof_count: str
dof_offset: str
external_kinetic: str
extra_constitution_uids: str
global_vertex_offset: str
gravity: str
implicit_geometry_uid: str
is_constrained: str
is_dynamic: str
is_facet: str
is_fixed: str
is_surf: str
mass_density: str
orient: str
parent_id: str
position: str
self_collision: str
subscene_element_id: str
thickness: str
topo: str
transform: str
velocity: str
volume: str

class ConstitutionUIDCollection(UIDRegister):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def instance() -> ConstitutionUIDCollection:
        """instance() -> pyuipc.builtin.ConstitutionUIDCollection

        Get the singleton instance of the constitution UID collection.
        Returns:
            ConstitutionUIDCollection: Singleton instance.
        """

class ImplicitGeometryUIDCollection(UIDRegister):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @staticmethod
    def instance() -> ImplicitGeometryUIDCollection:
        """instance() -> pyuipc.builtin.ImplicitGeometryUIDCollection

        Get the singleton instance of the implicit geometry UID collection.
        Returns:
            ImplicitGeometryUIDCollection: Singleton instance.
        """

class UIDRegister:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def exists(self, uid: typing.SupportsInt) -> bool:
        """exists(self: pyuipc.builtin.UIDRegister, uid: typing.SupportsInt) -> bool

        Check if a UID exists in the register.
        Args:
            uid: UID to check.
        Returns:
            bool: True if UID exists, False otherwise.
        """
    def find(self, *args, **kwargs):
        """find(self: pyuipc.builtin.UIDRegister, uid: typing.SupportsInt) -> uipc::builtin::UIDInfo

        Find a UID in the register.
        Args:
            uid: UID to find.
        Returns:
            str or None: Name associated with UID if found, None otherwise.
        """
    def to_json(self) -> json:
        """to_json(self: pyuipc.builtin.UIDRegister) -> json

        Convert register to JSON representation.
        Returns:
            dict: JSON representation of the register.
        """
