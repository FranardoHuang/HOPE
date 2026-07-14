"""Stable input/output file handles for the lateral full-scene probe.

This module deliberately has no Isaac, Torch or project imports so the path-race and no-clobber
contract can be red-teamed in a dependency-light process.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path


def _nofollow_flag() -> int:
    return int(getattr(os, "O_NOFOLLOW", 0))


def _cloexec_flag() -> int:
    return int(getattr(os, "O_CLOEXEC", 0))


def _require_absolute_no_symlink_components(path: Path, *, include_leaf: bool, label: str) -> None:
    if not path.is_absolute():
        raise RuntimeError(f"{label} path must be absolute: {path}")
    parts = path.parts[1:] if include_leaf else path.parent.parts[1:]
    current = Path(path.anchor)
    for part in parts:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if include_leaf or current != path:
                raise RuntimeError(f"{label} path component is absent: {current}")
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{label} path contains a symlink: {current}")


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(fd, 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


@dataclass
class StableInputFile:
    """A regular input kept open and re-bound to its public path before publication."""

    path: Path
    label: str
    fd: int
    identity: tuple[int, int, int, int, int, int]
    sha256: str

    @classmethod
    def open(cls, path_text: str | os.PathLike[str], *, label: str) -> "StableInputFile":
        path = Path(path_text).expanduser()
        _require_absolute_no_symlink_components(path, include_leaf=True, label=label)
        expected = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(expected.st_mode):
            raise RuntimeError(f"{label} must be a regular file: {path}")
        flags = os.O_RDONLY | _cloexec_flag() | _nofollow_flag()
        fd = os.open(path, flags)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"{label} must be a regular file: {path}")
            if _identity(metadata) != _identity(expected):
                raise RuntimeError(f"{label} path raced during stable open")
            return cls(
                path=path,
                label=label,
                fd=fd,
                identity=_identity(metadata),
                sha256=_sha256_fd(fd),
            )
        except BaseException:
            os.close(fd)
            raise

    def verify_path_unchanged(self) -> None:
        """Reject unlink/replace/symlink/swap, even if replacement bytes hash the same."""

        _require_absolute_no_symlink_components(self.path, include_leaf=True, label=self.label)
        flags = os.O_RDONLY | _cloexec_flag() | _nofollow_flag()
        check_fd = os.open(self.path, flags)
        try:
            current = os.fstat(check_fd)
            if not stat.S_ISREG(current.st_mode):
                raise RuntimeError(f"{self.label} path is no longer a regular file")
            if _identity(current) != self.identity:
                raise RuntimeError(f"{self.label} path identity changed during probe")
            if _sha256_fd(check_fd) != self.sha256:
                raise RuntimeError(f"{self.label} bytes changed during probe")
            if _identity(os.fstat(self.fd)) != self.identity or _sha256_fd(self.fd) != self.sha256:
                raise RuntimeError(f"{self.label} stable descriptor changed during probe")
        finally:
            os.close(check_fd)

    def runtime_path(self) -> str:
        """Return a kernel fd path that re-opens this exact inode for MotionLoader."""

        if self.fd < 0:
            raise RuntimeError(f"{self.label} stable descriptor is closed")
        for root in (Path("/proc/self/fd"), Path("/dev/fd")):
            candidate = root / str(self.fd)
            try:
                check_fd = os.open(candidate, os.O_RDONLY | _cloexec_flag())
            except OSError:
                continue
            try:
                metadata = os.fstat(check_fd)
                if int(metadata.st_size) != self.identity[3] or _sha256_fd(check_fd) != self.sha256:
                    raise RuntimeError(f"{self.label} kernel fd path bytes mismatch")
            finally:
                os.close(check_fd)
            return str(candidate)
        raise RuntimeError(f"{self.label} has no usable kernel fd path")

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "StableInputFile":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass
class StableOutputDirectory:
    """A stable parent dirfd used for one ``openat(O_EXCL|O_NOFOLLOW)`` receipt."""

    requested_path: Path
    parent_path: Path
    basename: str
    dirfd: int
    parent_identity: tuple[int, int, int]

    @classmethod
    def open(
        cls,
        path_text: str | os.PathLike[str],
        *,
        forbidden_roots: tuple[Path, ...] = (),
    ) -> "StableOutputDirectory":
        requested = Path(path_text).expanduser()
        if not requested.is_absolute():
            raise RuntimeError(f"output path must be absolute: {requested}")
        if requested.name in ("", ".", "..") or "/" in requested.name:
            raise RuntimeError("output basename is invalid")
        _require_absolute_no_symlink_components(requested, include_leaf=False, label="output")
        if os.path.lexists(requested):
            raise RuntimeError(f"refusing to clobber output: {requested}")
        resolved_parent = requested.parent.resolve(strict=True)
        expected_parent = os.stat(resolved_parent, follow_symlinks=False)
        if not stat.S_ISDIR(expected_parent.st_mode):
            raise RuntimeError("output parent is not a directory")
        for forbidden in forbidden_roots:
            try:
                (resolved_parent / requested.name).relative_to(forbidden.resolve(strict=True))
            except ValueError:
                continue
            raise RuntimeError(f"output must be outside reviewed checkout: {forbidden}")
        flags = os.O_RDONLY | _cloexec_flag() | _nofollow_flag()
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        dirfd = os.open(requested.parent, flags)
        try:
            metadata = os.fstat(dirfd)
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError("output parent descriptor is not a directory")
            expected_identity = (
                int(expected_parent.st_dev),
                int(expected_parent.st_ino),
                int(expected_parent.st_mode),
            )
            actual_identity = (int(metadata.st_dev), int(metadata.st_ino), int(metadata.st_mode))
            if actual_identity != expected_identity:
                raise RuntimeError("output parent path raced during stable open")
            return cls(
                requested_path=requested,
                parent_path=requested.parent,
                basename=requested.name,
                dirfd=dirfd,
                parent_identity=actual_identity,
            )
        except BaseException:
            os.close(dirfd)
            raise

    def verify_parent_path_unchanged(self) -> None:
        _require_absolute_no_symlink_components(
            self.requested_path, include_leaf=False, label="output"
        )
        flags = os.O_RDONLY | _cloexec_flag() | _nofollow_flag()
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        check_fd = os.open(self.parent_path, flags)
        try:
            metadata = os.fstat(check_fd)
            identity = (int(metadata.st_dev), int(metadata.st_ino), int(metadata.st_mode))
            if identity != self.parent_identity:
                raise RuntimeError("output parent path identity changed during probe")
            stable = os.fstat(self.dirfd)
            stable_identity = (int(stable.st_dev), int(stable.st_ino), int(stable.st_mode))
            if stable_identity != self.parent_identity:
                raise RuntimeError("stable output parent descriptor changed during probe")
            if os.path.lexists(self.requested_path):
                raise RuntimeError(f"refusing to clobber output: {self.requested_path}")
        finally:
            os.close(check_fd)

    def write_no_clobber(self, payload: bytes, *, mode: int = 0o600) -> None:
        if type(payload) is not bytes:
            raise TypeError("output payload must be bytes")
        self.verify_parent_path_unchanged()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _cloexec_flag() | _nofollow_flag()
        fd = os.open(self.basename, flags, mode, dir_fd=self.dirfd)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("created receipt is not a regular file")
            remaining = memoryview(payload)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("short write while persisting runtime probe receipt")
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(self.dirfd)
        self.verify_parent_path_unchanged_after_create()

    def verify_parent_path_unchanged_after_create(self) -> None:
        """Post-write parent binding check; target existence is required here."""

        _require_absolute_no_symlink_components(
            self.requested_path, include_leaf=False, label="output"
        )
        metadata = os.stat(self.parent_path, follow_symlinks=False)
        identity = (int(metadata.st_dev), int(metadata.st_ino), int(metadata.st_mode))
        if identity != self.parent_identity:
            raise RuntimeError("output parent path changed during receipt creation")
        target = os.stat(self.basename, dir_fd=self.dirfd, follow_symlinks=False)
        if not stat.S_ISREG(target.st_mode):
            raise RuntimeError("persisted receipt is not a regular file")

    def close(self) -> None:
        if self.dirfd >= 0:
            os.close(self.dirfd)
            self.dirfd = -1

    def __enter__(self) -> "StableOutputDirectory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["StableInputFile", "StableOutputDirectory"]
