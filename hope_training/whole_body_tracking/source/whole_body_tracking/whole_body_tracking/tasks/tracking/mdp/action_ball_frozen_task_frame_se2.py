"""Typed, dependency-light accepted-shot SE(2) transform.

Validation belongs to construction/restore boundaries.  Runtime consumers use
the pure apply methods and never synchronize the device merely to re-prove an
immutable accepted transform.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenTaskFrameSE2:
    yaw_wxyz: object
    translation_xyz: object

    @classmethod
    def derive(cls, torch, *, source_root_xyz, source_yaw_wxyz,
               target_root_xyz, target_yaw_wxyz, quat_apply, quat_mul):
        inverse = source_yaw_wxyz.clone()
        inverse[..., 1:].neg_()
        yaw = quat_mul(target_yaw_wxyz, inverse)
        translation = target_root_xyz - quat_apply(yaw, source_root_xyz)
        translation = translation.clone()
        translation[..., 2] = 0.0
        return cls(yaw.contiguous(), translation.contiguous())

    def validate_async(self, torch, *, valid=None, atol=1.0e-6):
        yaw = self.yaw_wxyz
        translation = self.translation_xyz
        ok = (
            torch.isfinite(yaw).all(dim=-1)
            & torch.isfinite(translation).all(dim=-1)
            & torch.isclose(
                torch.linalg.vector_norm(yaw, dim=-1),
                torch.ones_like(yaw[..., 0]),
                rtol=0.0,
                atol=1.0e-5,
            )
            & yaw[..., 1].abs().le(atol)
            & yaw[..., 2].abs().le(atol)
            & translation[..., 2].abs().le(atol)
        )
        if valid is not None:
            ok = ~valid | ok
        torch._assert_async(ok.all())
        return self

    def apply_point(self, torch, point, quat_apply):
        return quat_apply(self.yaw_wxyz, point) + self.translation_xyz

    def apply_vector(self, torch, vector, quat_apply):
        return quat_apply(self.yaw_wxyz, vector)

    def apply_quat(self, torch, quat, quat_mul):
        return quat_mul(self.yaw_wxyz, quat)
