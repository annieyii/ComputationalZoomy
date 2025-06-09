"""
ulr_renderer.py  –  v0.2
=============================================
* Stand-alone ModernGL wrapper for the original
  **Unstructured-Lumigraph Rendering (ULR)** shaders.

Update v0.2
-----------
* **shader_dir** is now a separate parameter (defaults to the folder
  where this file lives).  The GLSL files *vertex.glsl* and *fragment.glsl*
  are looked up there instead of *dataset_dir*.
* Clearer error messages if the shader files can’t be found.
* Minor refactor: `_load_shaders()` helper.

You hit *FileNotFoundError* because you pointed **dataset_dir** to your
`sunglassgirl` folder (which has COLMAP data but no GLSL).  With this
version you can simply:

```python
ulr = ULRRenderer(
    dataset_dir=Path('ulr/sunglassgirl'),
    shader_dir = Path('ulr')            # where vertex.glsl lives
)
```

and it will work.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Tuple

import moderngl
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Helper dataclass to describe one source camera from COLMAP / ULR
# ---------------------------------------------------------------------------
@dataclass
class Camera:
    position: Tuple[float, float, float]
    mvp:      np.ndarray                # (4,4) column-major float32
    color:    Tuple[float, float, float] = (1.0, 1.0, 1.0)


class ULRRenderer:
    """Load original GLSL shaders then offer ``render()`` API → numpy RGB."""

    def __init__(self,
                 dataset_dir: str | Path,
                 *,
                 width: int = 1024,
                 height: int = 768,
                 shader_dir: str | Path | None = None):
        self.dataset_dir = Path(dataset_dir)
        self.shader_dir  = Path(shader_dir) if shader_dir else Path(__file__).resolve().parent
        self.width, self.height = width, height

        # ---------------------------------------------------------------
        # 1.  Create headless ModernGL context & FBO
        # ---------------------------------------------------------------
        self.ctx = moderngl.create_standalone_context(require=330)
        self.fbo = self.ctx.simple_framebuffer((self.width, self.height))
        self.fbo.use()

        # ---------------------------------------------------------------
        # 2.  Compile shaders
        # ---------------------------------------------------------------
        self.prog = self._load_shaders()

        # ---------------------------------------------------------------
        # 3.  Build a simple proxy plane (replace with OBJ if needed)
        # ---------------------------------------------------------------
        self._build_proxy_plane()

        # Place-holders to be filled by user calls
        self.texture_array = None   # set by upload_images()
        self.camera_ubo    = None   # set by upload_cameras()

    # ===================================================================
    # API (called from CZ)
    # ===================================================================
    def upload_images(self, image_paths: Sequence[Path]):
        imgs = [np.asarray(Image.open(p).convert('RGB')) for p in image_paths]
        h, w, _ = imgs[0].shape
        layers  = np.stack(imgs).astype(np.uint8)
        tex = self.ctx.texture_array((w, h, len(imgs)), 3, layers.tobytes())
        tex.build_mipmaps()
        tex.use(location=0)
        self.prog['images'].value = 0
        self.texture_array = tex

    def upload_cameras(self, cams: Sequence[Camera]):
        # TODO: pack std140 properly; using dummy 16-byte per cam for now
        buf = bytearray(16 * len(cams))
        self.camera_ubo = self.ctx.buffer(buf)
        self.camera_ubo.bind_to_uniform_block(self.prog, 'cameras')
        try:
            self.prog['CAMERA_COUNT'].value = len(cams)
        except KeyError:
            pass  # shader may use #define instead

    def render(self, quat: Tuple[float, float, float, float],
               pos: Tuple[float, float, float],
               fov_y: float) -> np.ndarray:
        # TODO: set view/proj uniforms if shader expects them
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render()
        rgb = np.frombuffer(self.fbo.read(components=3, alignment=1), np.uint8)
        return rgb.reshape(self.height, self.width, 3)[::-1]

    # ===================================================================
    # internal helpers
    # ===================================================================
    def _load_shaders(self):
        v_path = self.shader_dir / 'vertex.glsl'
        f_path = self.shader_dir / 'fragment.glsl'
        if not v_path.exists() or not f_path.exists():
            raise FileNotFoundError(
                f"Shader files not found. Expected at:\n  {v_path}\n  {f_path}\n" +
                "Pass shader_dir=Path('path/to/unstructured-lumigraph') in ULRRenderer().")
        vert_src = v_path.read_text()
        frag_src = f_path.read_text()

        if not vert_src.lstrip().startswith('#version'):
            vert_src = '#version 330 core\\n' + vert_src
        if not frag_src.lstrip().startswith('#version'):
            frag_src = '#version 330 core\\n' + frag_src
        return self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

    def _build_proxy_plane(self):
        verts = np.array([
            -1, -1, 0, 0, 0,
            +1, -1, 0, 1, 0,
            +1, +1, 0, 1, 1,
            -1, +1, 0, 0, 1,
        ], dtype='f4')
        idx   = np.array([0, 1, 2, 2, 3, 0], dtype='i4')
        vbo   = self.ctx.buffer(verts.tobytes())
        ibo   = self.ctx.buffer(idx.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(vbo, '3f 2f', 'in_pos', 'in_uv')], ibo)
